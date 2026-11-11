from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def vr_avatar_module(monkeypatch):
    # vr_avatar's device lifecycle is pure Python, but its visualization imports
    # are normally supplied by Kit.  Lightweight stubs keep this focused test
    # runnable in the same non-GUI test job as the explicit-feedback FSM.
    pxr = types.ModuleType("pxr")
    pxr.Gf = types.SimpleNamespace()
    pxr.Usd = types.SimpleNamespace()
    pxr.UsdGeom = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    geometry = types.ModuleType("end_effector_safety_geometry")
    geometry.DEFAULT_THRESHOLDS = types.SimpleNamespace(hand_radius_m=0.035)
    monkeypatch.setitem(sys.modules, "end_effector_safety_geometry", geometry)

    path = Path(__file__).resolve().parents[1] / "v3_chan" / "vr_avatar.py"
    name = "_test_v3_vr_avatar_device_refresh"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class _Matrix:
    def __init__(self, position=(0.0, 0.0, 0.0)) -> None:
        self.position = tuple(position)

    def ExtractTranslation(self):
        return self.position


class _Device:
    def __init__(self, position=(0.0, 0.0, 0.0)) -> None:
        self.matrix = _Matrix(position)

    def get_pose_names(self):
        return ("grip",)

    def get_pose(self, _name=""):
        return self.matrix

    def get_raw_pose(self, _name=""):
        return self.matrix

    def get_virtual_world_pose(self, _name=""):
        return self.matrix


class _CountingCore:
    def __init__(self, devices) -> None:
        self.devices = dict(devices)
        self.has_calls = []
        self.get_calls = []

    def has_input_device(self, path):
        self.has_calls.append(path)
        return path in self.devices

    def get_input_device(self, path):
        self.get_calls.append(path)
        return self.devices.get(path)


def _avatar(module, core):
    avatar = module.VRAvatar()
    avatar._xr_core = core
    return avatar


def test_visual_update_refreshes_each_physical_controller_exactly_once(
    vr_avatar_module, monkeypatch
) -> None:
    module = vr_avatar_module
    left = _Device()
    right = _Device()
    core = _CountingCore(
        {
            "/user/hand/left": left,
            "/user/hand/right": right,
        }
    )
    avatar = _avatar(module, core)
    monkeypatch.setattr(module, "XR_CONTROLLER_POSE_MODE", "visual")
    monkeypatch.setattr(module, "XR_DRAW_ARM_LINES", False)
    monkeypatch.setattr(module, "HAND_PROXY_ENABLED", False)
    monkeypatch.setattr(
        avatar,
        "get_hmd_pose_sample",
        lambda: module.PoseSample.invalid(acquisition_monotonic_ns=1),
    )

    # A successful stage-visual pose used to bypass XRInputDevice completely.
    monkeypatch.setattr(
        avatar,
        "_stage_visual_hand_pos",
        lambda hand: np.array(
            [0.85, 0.2 if hand == "left" else -0.2, 1.25], dtype=float
        ),
    )
    avatar.update_with_status()

    assert core.has_calls == ["/user/hand/left", "/user/hand/right"]
    assert core.get_calls == ["/user/hand/left", "/user/hand/right"]
    assert avatar.get_cached_xr_device("/user/hand/left") is left
    assert avatar.get_cached_xr_device("/user/hand/right") is right


def test_controller_pose_readers_reuse_refreshed_wrapper(vr_avatar_module) -> None:
    module = vr_avatar_module
    path = "/user/hand/left"
    device = _Device((0.2, 0.1, 1.2))
    core = _CountingCore({path: device})
    avatar = _avatar(module, core)

    avatar._refresh_controller_devices()
    get_calls_after_refresh = list(core.get_calls)
    assert avatar._raw_pos(path, ("grip",)) is not None
    assert avatar._virtual_world_pos(path, ("grip",), force=True) is not None
    assert avatar._raw_pos(path, ("grip",)) is not None

    # Physical, raw, and virtual fallback reads all share the one frame cache.
    assert core.get_calls == get_calls_after_refresh
    assert core.get_calls.count(path) == 1


def test_refresh_clears_stale_wrapper_on_disconnect(vr_avatar_module) -> None:
    module = vr_avatar_module
    left_path = "/user/hand/left"
    right_path = "/user/hand/right"
    core = _CountingCore(
        {left_path: _Device(), right_path: _Device()}
    )
    avatar = _avatar(module, core)
    avatar._refresh_controller_devices()
    assert avatar.get_cached_xr_device(left_path) is not None

    del core.devices[left_path]
    avatar._refresh_controller_devices()
    assert avatar.get_cached_xr_device(left_path) is None
    assert avatar.get_cached_xr_device(right_path) is not None


def test_logical_hand_swap_selects_opposite_refreshed_physical_cache(
    vr_avatar_module, monkeypatch
) -> None:
    module = vr_avatar_module
    physical_left = _Device()
    physical_right = _Device()
    core = _CountingCore(
        {
            "/user/hand/left": physical_left,
            "/user/hand/right": physical_right,
        }
    )
    avatar = _avatar(module, core)
    monkeypatch.setattr(module, "XR_SWAP_HANDS", True)
    avatar._refresh_controller_devices()

    assert avatar.get_cached_xr_device_for_logical_hand("left") is physical_right
    assert avatar.get_cached_xr_device_for_logical_hand("right") is physical_left
    assert core.get_calls == ["/user/hand/left", "/user/hand/right"]

