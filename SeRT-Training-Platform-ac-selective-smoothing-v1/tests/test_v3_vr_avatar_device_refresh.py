from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def vr_avatar_module(monkeypatch):
    # The lifecycle logic is pure Python; Kit visualization imports are stubbed
    # so this regression test can run in the non-GUI test job.
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
    def __init__(self, position=(0.2, 0.8, 0.1)) -> None:
        self.matrix = _Matrix(position)

    def get_pose_names(self):
        return ("grip",)

    def get_pose(self, _name=""):
        return self.matrix

    def get_raw_pose(self, _name=""):
        return self.matrix

    def get_virtual_world_pose(self, _name=""):
        return self.matrix


class _Prim:
    def __init__(self, path) -> None:
        self.path = str(path)
        self.valid = True

    def IsValid(self):
        return self.valid

    def GetPath(self):
        return self.path


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


def test_visual_update_refreshes_each_physical_controller_once(
    vr_avatar_module, monkeypatch
) -> None:
    module = vr_avatar_module
    left = _Device()
    right = _Device()
    core = _CountingCore(
        {"/user/hand/left": left, "/user/hand/right": right}
    )
    avatar = _avatar(module, core)
    avatar._last_stage_visual_path = {
        "left": "/_xr/gui/controllers/user/hand/left",
        "right": "/_xr/gui/controllers/user/hand/right",
    }
    monkeypatch.setattr(module, "XR_CONTROLLER_POSE_MODE", "visual")
    monkeypatch.setattr(module, "XR_DRAW_ARM_LINES", False)
    monkeypatch.setattr(module, "HAND_PROXY_ENABLED", False)
    monkeypatch.setattr(
        avatar,
        "get_hmd_pose_sample",
        lambda: module.PoseSample.invalid(acquisition_monotonic_ns=1),
    )
    monkeypatch.setattr(
        avatar,
        "_stage_visual_hand_pos",
        lambda hand: np.array(
            [0.85, 0.2 if hand == "left" else -0.2, 1.25], dtype=float
        ),
    )

    _head, left_sample, right_sample = avatar.update_with_status()

    assert core.has_calls == ["/user/hand/left", "/user/hand/right"]
    assert core.get_calls == ["/user/hand/left", "/user/hand/right"]
    assert avatar.get_cached_xr_device("/user/hand/left") is left
    assert avatar.get_cached_xr_device("/user/hand/right") is right
    assert left_sample.source_name == "xr_stage_visual"
    assert right_sample.source_name == "xr_stage_visual"
    assert left_sample.source_path.endswith("/hand/left")
    assert right_sample.source_path.endswith("/hand/right")


def test_stage_visual_is_rejected_without_current_physical_pose(
    vr_avatar_module, monkeypatch
) -> None:
    module = vr_avatar_module
    path = "/user/hand/left"
    avatar = _avatar(module, _CountingCore({path: _Device((0.0, 0.0, 0.0))}))
    monkeypatch.setattr(module, "XR_CONTROLLER_POSE_MODE", "visual")
    monkeypatch.setattr(module, "XR_VIRTUAL_WORLD_POSE_FALLBACK", True)
    monkeypatch.setattr(module, "XR_STAGE_VISUAL_FALLBACK", True)
    monkeypatch.setattr(
        avatar,
        "_stage_visual_hand_pos",
        lambda _hand: np.array([0.8, 0.2, 1.2], dtype=float),
    )
    avatar._refresh_controller_devices()

    sample = avatar.get_controller_pose_sample("left")

    assert not sample.pose_valid
    assert sample.source_name == "missing"


def test_stage_visual_prim_stays_latched_until_it_is_invalid(
    vr_avatar_module, monkeypatch
) -> None:
    module = vr_avatar_module
    avatar = _avatar(module, _CountingCore({}))
    root = _Prim("/_xr/gui/controllers/user/hand/left/Root")
    model_geo = _Prim("/_xr/gui/controllers/user/hand/left/Root/model_geo")
    replacement = _Prim("/_xr/gui/controllers/user/hand/left/replacement")
    scans = []

    def candidates(_hand):
        scans.append(True)
        return [root, model_geo] if len(scans) == 1 else [replacement]

    positions = {
        root: (np.array([0.80, 0.20, 1.20]), 2.0),
        model_geo: (np.array([0.95, 0.25, 1.10]), 1.0),
        replacement: (np.array([0.75, 0.18, 1.25]), 3.0),
    }
    monkeypatch.setattr(avatar, "_find_stage_visual_candidates", candidates)
    monkeypatch.setattr(
        avatar,
        "_prim_world_bbox_center",
        lambda prim: positions.get(prim),
    )

    first = avatar._stage_visual_hand_pos("left")
    positions[root] = (np.array([0.82, 0.22, 1.23]), 0.5)
    positions[model_geo] = (np.array([0.60, 0.30, 1.40]), 4.0)
    second = avatar._stage_visual_hand_pos("left")

    assert np.allclose(first, [0.80, 0.20, 1.20])
    assert np.allclose(second, positions[root][0])
    assert avatar._last_stage_visual_path["left"] == str(root.GetPath())
    assert len(scans) == 1

    root.valid = False
    recovered = avatar._stage_visual_hand_pos("left")

    assert np.allclose(recovered, positions[replacement][0])
    assert avatar._last_stage_visual_path["left"] == str(replacement.GetPath())
    assert len(scans) == 2


def test_pose_readers_reuse_the_current_frame_wrapper(vr_avatar_module) -> None:
    path = "/user/hand/left"
    device = _Device((0.2, 0.1, 1.2))
    core = _CountingCore({path: device})
    avatar = _avatar(vr_avatar_module, core)

    avatar._refresh_controller_devices()
    get_calls_after_refresh = list(core.get_calls)
    assert avatar._raw_pos(path, ("grip",)) is not None
    assert avatar._virtual_world_pos(path, ("grip",), force=True) is not None
    assert avatar._raw_pos(path, ("grip",)) is not None

    assert core.get_calls == get_calls_after_refresh
    assert core.get_calls.count(path) == 1


def test_refresh_clears_stale_wrapper_on_disconnect(vr_avatar_module) -> None:
    left_path = "/user/hand/left"
    right_path = "/user/hand/right"
    core = _CountingCore({left_path: _Device(), right_path: _Device()})
    avatar = _avatar(vr_avatar_module, core)
    avatar._refresh_controller_devices()
    assert avatar.get_cached_xr_device(left_path) is not None

    del core.devices[left_path]
    avatar._refresh_controller_devices()

    assert avatar.get_cached_xr_device(left_path) is None
    assert avatar.get_cached_xr_device(right_path) is not None


def test_logical_hand_swap_uses_opposite_physical_cache(
    vr_avatar_module, monkeypatch
) -> None:
    left = _Device()
    right = _Device()
    core = _CountingCore(
        {"/user/hand/left": left, "/user/hand/right": right}
    )
    avatar = _avatar(vr_avatar_module, core)
    monkeypatch.setattr(vr_avatar_module, "XR_SWAP_HANDS", True)
    avatar._refresh_controller_devices()

    assert avatar.get_cached_xr_device_for_logical_hand("left") is right
    assert avatar.get_cached_xr_device_for_logical_hand("right") is left
    assert core.get_calls == ["/user/hand/left", "/user/hand/right"]
