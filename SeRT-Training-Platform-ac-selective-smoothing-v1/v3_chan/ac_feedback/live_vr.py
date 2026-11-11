"""Live VR tracking adapter preserving PoseSample validity and provenance."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .tracking import TrackingDropout, TrackingSnapshot, TrackingWatchdog


_DEFAULT_KOREAN_UI_FONT = Path(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
)
_KOREAN_UI_GLYPH_REGION = (
    Path(__file__).resolve().parent / "assets" / "korean_ui_chars.txt"
)
_FEEDBACK_XR_EXTENSION_ID = "omni.sert.xr.feedback_input"
_FEEDBACK_XR_EXTENSION_ROOT = Path(__file__).resolve().parent / "xr_exts"
_FEEDBACK_XR_EXTENSION_CONFIG = (
    _FEEDBACK_XR_EXTENSION_ROOT
    / _FEEDBACK_XR_EXTENSION_ID
    / "config"
    / "extension.toml"
)
_FEEDBACK_ACTION_MAP_DOMINANT_HANDS = {
    "SeRT Feedback Oculus (Left)": "left",
    "SeRT Feedback Oculus (Right)": "right",
}


def _korean_ui_font_args() -> list[str]:
    """Return fail-closed Kit font-atlas settings for mandatory Korean UI."""

    font_path = Path(
        os.environ.get("XR_UI_KOREAN_FONT_PATH", str(_DEFAULT_KOREAN_UI_FONT))
    ).expanduser()
    if not font_path.is_file():
        raise RuntimeError(
            "Korean XR UI font is missing: "
            f"{font_path}. Set XR_UI_KOREAN_FONT_PATH to a Hangul-capable font."
        )
    if not _KOREAN_UI_GLYPH_REGION.is_file():
        raise RuntimeError(
            f"Korean XR UI glyph region is missing: {_KOREAN_UI_GLYPH_REGION}"
        )
    return [
        f"--/app/font/customFontPath={font_path.resolve()}",
        "--/app/font/customRegionFiles="
        f"[{_KOREAN_UI_GLYPH_REGION.resolve()}]",
    ]


def xr_prestartup_args(*, xr_mode: str, xr_backend: str) -> list[str]:
    """Return XR settings that must exist before the XR extensions start.

    The stock VR experience enables the profile while ``SimulationApp`` is
    starting.  Applying these settings later is too late: the viewport may
    already have scheduled an active-camera/user-height teleport or selected
    an incompatible controller action map.
    """

    mode = str(xr_mode).strip().lower()
    profile_name = "ar" if mode == "openxr" else "vr"
    profile_root = f"/xr/profile/{profile_name}"
    if not _FEEDBACK_XR_EXTENSION_CONFIG.is_file():
        raise RuntimeError(
            "feedback-only XR action-map extension is missing: "
            f"{_FEEDBACK_XR_EXTENSION_CONFIG}"
        )
    return [
        # Register the collector-owned action map before XRCore scans manifest
        # directories.  It retains the stock ``vr`` layout (and therefore the
        # physical controller wrappers) but enables no built-in Kit tools.
        # This prevents X/A/B from also toggling selection beams or menus.
        "--ext-folder",
        str(_FEEDBACK_XR_EXTENSION_ROOT),
        "--enable",
        _FEEDBACK_XR_EXTENSION_ID,
        # Do not inherit or mutate the operator's global XR profile.  This also
        # makes XRProfile use the non-persistent /xr/profile/... settings below.
        "--/xr/persistence/enabled=false",
        # Match the proven pick-and-place collector.  That runtime starts from
        # the active-camera frame and then places the HMD with
        # XRCore.schedule_set_camera().  Forcing scene-origin mode here puts
        # the physical OpenXR origin near the Panda base/table instead.
        f"--{profile_root}/anchorMode=active camera",
        f"--{profile_root}/adjustForUserHeight=false",
        f"--{profile_root}/system/display={str(xr_backend).strip()}",
        # Keep Isaac Sim 4.5's stock VR layout.  The layout also selects the
        # Oculus/OpenXR action map; an empty value produces ``no actionmap`` on
        # real ALVR/SteamVR sessions and can invalidate controller wrappers.
        f"--{profile_root}/tools/layout=vr",
        "--/xr/ui/enabled=true",
    ] + _korean_ui_font_args()


def _xr_token_text(value: Any) -> str:
    """Return the readable name from either a string or Kit's XRToken."""

    if isinstance(value, str):
        return value
    text = str(value)
    match = re.fullmatch(r"XRToken\(name:\s*([\"'])(.*?)\1\)", text)
    return match.group(2) if match is not None else text


def verify_feedback_only_action_map(
    xr_core: Any | None = None, *, emit_success: bool = True
) -> dict[str, Any]:
    """Fail closed unless the collector-owned, tool-free map is active.

    An empty ``tools/layout`` makes real OpenXR controller wrappers disappear
    on Isaac Sim 4.5.  Conversely, the stock ``vr`` map binds the same face
    buttons used for feedback to selection beams and the Kit menu.  The local
    high-priority map keeps a valid layout while deliberately containing no
    tool actions.
    """

    if xr_core is None:
        from omni.kit.xr.core import XRCore

        xr_core = XRCore.get_singleton()
    action_map = xr_core.get_action_map()
    if action_map is None:
        raise RuntimeError("XR feedback action map is unavailable")
    name = _xr_token_text(action_map.get_name())
    layout = _xr_token_text(action_map.get_tool_layout())
    dominant_hand = _xr_token_text(action_map.get_dominant_hand())
    tools = tuple(action_map.get_tool_list())
    actions = tuple(action_map.get_action_map())
    failures = []
    expected_dominant_hand = _FEEDBACK_ACTION_MAP_DOMINANT_HANDS.get(name)
    if expected_dominant_hand is None:
        failures.append(f"selected map is {name!r}")
    if layout != "vr":
        failures.append(f"tools layout is {layout!r}")
    if (
        expected_dominant_hand is not None
        and dominant_hand != expected_dominant_hand
    ):
        failures.append(
            f"dominant hand is {dominant_hand!r}, expected "
            f"{expected_dominant_hand!r}"
        )
    if tools:
        failures.append(
            "built-in tools are enabled: "
            + ", ".join(_xr_token_text(value) for value in tools)
        )
    if actions:
        failures.append(f"built-in tool actions are mapped ({len(actions)})")
    if failures:
        raise RuntimeError(
            "feedback-only XR action-map isolation failed: "
            + "; ".join(failures)
        )
    result = {
        "name": name,
        "tools_layout": layout,
        "dominant_hand": dominant_hand,
        "tool_count": 0,
        "action_count": 0,
    }
    if emit_success:
        print(
            "[ExplicitFeedback] feedback-only XR action map verified: "
            f"{name}; dominant-hand={dominant_hand}; built-in-tools=off",
            flush=True,
        )
    return result


def _configure_xr_profile_settings(
    settings: Any, *, profile_name: str, xr_backend: str
) -> None:
    """Reassert the isolated session settings before requesting a profile."""

    for profile_root in (
        f"/defaults/xr/profile/{profile_name}",
        f"/xr/profile/{profile_name}",
    ):
        settings.set(f"{profile_root}/anchorMode", "active camera")
        settings.set(f"{profile_root}/adjustForUserHeight", False)
        settings.set(f"{profile_root}/system/display", xr_backend)
        settings.set(f"{profile_root}/tools/layout", "vr")
    settings.set("/xr/ui/enabled", True)


class LiveVRTrackingProvider:
    """Bridge VRAvatar pose samples into PickPlaceEnv human-state payloads."""

    def __init__(self, watchdog: TrackingWatchdog) -> None:
        self.watchdog = watchdog
        self.avatar = None
        self._anchor_applied = False
        self._state: dict[str, Any] = {}
        self.observation_snapshot: TrackingSnapshot | None = None
        self.latest_snapshot: TrackingSnapshot | None = None
        self._simulation_app = None
        self._view_target_world: np.ndarray | None = None
        self._placement_method = "not_applied"
        self._feedback_action_map_guard_armed = False

    @property
    def anchor_applied(self) -> bool:
        return self._anchor_applied

    @property
    def placement_method(self) -> str:
        return self._placement_method

    def setup(
        self,
        world,
        simulation_app,
        *,
        view_target_world: np.ndarray | None = None,
    ) -> None:
        start_live_vr_profile(simulation_app)
        try:
            from v3_chan.vr_avatar import VRAvatar
        except ImportError:
            from vr_avatar import VRAvatar

        self.avatar = VRAvatar()
        self.avatar.setup(world)
        self._simulation_app = simulation_app
        self._view_target_world = (
            None
            if view_target_world is None
            else np.asarray(view_target_world, dtype=np.float64).reshape(3).copy()
        )

    def sample(self) -> TrackingSnapshot:
        if self.avatar is None:
            raise RuntimeError("LiveVRTrackingProvider.setup() has not been called")
        if self._feedback_action_map_guard_armed:
            # Action maps may be reselected when a controller reconnects or
            # the dominant hand changes.  Recheck on every tracking sample so
            # a stock beam/menu binding cannot silently return mid-session.
            verify_feedback_only_action_map(emit_success=False)
        samples = self.avatar.update_with_status()
        if not self._anchor_applied:
            room_hmd_pos = self.avatar.capture_initial_hmd_pos()
            if room_hmd_pos is not None:
                placed = False
                if (
                    self._simulation_app is not None
                    and self._view_target_world is not None
                ):
                    placed = schedule_live_vr_camera_view(
                        simulation_app=self._simulation_app,
                        target_world=self._view_target_world,
                    )
                    if placed:
                        self._placement_method = "xr_camera_teleport"
                if not placed and set_live_vr_anchor(room_hmd_pos):
                    placed = True
                    self._placement_method = "xr_anchor_fallback"
                if placed:
                    self.avatar.notify_anchor_applied()
                    self._anchor_applied = True
                    print(
                        "[ExplicitFeedback] HMD/avatar frame aligned via "
                        f"{self._placement_method}",
                        flush=True,
                    )
                    samples = self.avatar.update_with_status()
        snapshot = self.watchdog.evaluate(samples)
        self.latest_snapshot = snapshot
        return snapshot

    def arm_feedback_action_map_guard(self) -> dict[str, Any]:
        """Verify the tool-free map once, then enforce it for every sample."""

        audit = verify_feedback_only_action_map()
        self._feedback_action_map_guard_armed = True
        return audit

    def refresh_observation_state(self) -> TrackingSnapshot:
        snapshot = self.sample()
        self.observation_snapshot = snapshot
        self._state = snapshot.state_payload()
        return snapshot

    def __call__(self) -> dict[str, Any]:
        # PickPlaceEnv calls this while building the state that the next policy
        # action will consume.  Preserve that exact snapshot separately from a
        # newer pre-apply watchdog sample.
        self.refresh_observation_state()
        return dict(self._state)

    def wait_until_ready(
        self,
        simulation_app,
        *,
        timeout_s: float,
        stable_frames: int = 6,
        stop_requested: Callable[[], Any] | None = None,
    ) -> TrackingSnapshot:
        started = time.monotonic()
        stable = 0
        last: TrackingSnapshot | None = None
        while stable < max(1, int(stable_frames)):
            if stop_requested is not None:
                stop_requested()
            is_running = getattr(simulation_app, "is_running", None)
            if callable(is_running) and not bool(is_running()):
                raise RuntimeError("SimulationApp stopped during tracking readiness")
            last = self.refresh_observation_state()
            try:
                self.watchdog.require_valid(last)
            except TrackingDropout:
                stable = 0
            else:
                stable += 1
            simulation_app.update()
            if stop_requested is not None:
                stop_requested()
            if timeout_s > 0.0 and time.monotonic() - started >= timeout_s:
                reasons = () if last is None else last.invalid_reasons
                raise TimeoutError(
                    "live VR head and both hands did not become stably valid: "
                    + ", ".join(reasons)
                )
            time.sleep(0.01)
        if last is None:
            raise RuntimeError("tracking readiness loop produced no sample")
        if not self._anchor_applied:
            raise RuntimeError("XR anchor was not applied")
        return last

    def wait_until_hmd_ready(
        self,
        simulation_app,
        *,
        timeout_s: float,
        stable_frames: int = 2,
        stop_requested: Callable[[], Any] | None = None,
    ) -> TrackingSnapshot:
        """Apply the XR camera placement without gating on controller poses.

        The full hand gate must run only after the in-HMD UI exists; otherwise
        an occluded or source-switching controller leaves the participant in a
        blank, paused world with no recovery instruction.  The robot remains
        in controlled hold throughout this HMD-only startup gate.
        """

        started = time.monotonic()
        stable = 0
        last: TrackingSnapshot | None = None
        required_stable = max(1, int(stable_frames))
        while stable < required_stable:
            if stop_requested is not None:
                stop_requested()
            is_running = getattr(simulation_app, "is_running", None)
            if callable(is_running) and not bool(is_running()):
                raise RuntimeError("SimulationApp stopped during HMD readiness")
            last = self.refresh_observation_state()
            hmd_ready = bool(self._anchor_applied and last.valid_mask[0])
            stable = stable + 1 if hmd_ready else 0
            simulation_app.update()
            if stop_requested is not None:
                stop_requested()
            if timeout_s > 0.0 and time.monotonic() - started >= timeout_s:
                reasons = (
                    []
                    if last is None
                    else [
                        reason
                        for reason in last.invalid_reasons
                        if str(reason).startswith("head:")
                    ]
                )
                if not self._anchor_applied:
                    reasons.append("xr_anchor_not_applied")
                raise TimeoutError(
                    "live VR HMD did not become stably valid: "
                    + ", ".join(reasons or ["unknown"])
                )
            time.sleep(0.01)
        if last is None:
            raise RuntimeError("HMD readiness loop produced no sample")
        print(
            "[ExplicitFeedback] HMD anchor ready; controller qualification "
            "continues in the in-HMD UI",
            flush=True,
        )
        return last


def start_live_vr_profile(simulation_app) -> None:
    """Enable the same XR profile used by the frozen evaluator."""

    import carb
    from omni.isaac.core.utils.extensions import enable_extension

    xr_mode = os.environ.get("ISAAC_XR_MODE", "vr").strip().lower()
    xr_backend = os.environ.get("ISAAC_XR_BACKEND", "OpenXR").strip()
    if xr_mode == "openxr":
        extension_ids = (
            "omni.kit.xr.system.openxr",
            "omni.kit.xr.profile.ar",
            "isaacsim.xr.openxr",
        )
        profile_name = "ar"
    elif xr_backend.lower() == "openxr":
        extension_ids = (
            "omni.kit.xr.system.openxr",
            "omni.kit.xr.profile.vr",
        )
        profile_name = "vr"
    else:
        extension_ids = (
            "omni.kit.xr.system.steamvr",
            "omni.kit.xr.profile.vr",
        )
        profile_name = "vr"
    settings = carb.settings.get_settings()
    if settings.get("/xr/persistence/enabled") is not False:
        raise RuntimeError(
            "XR session isolation was not applied before extension startup"
        )
    # Reassert all values before enabling/requesting the profile.  The active
    # camera plus disabled height adjustment is the same configuration as the
    # proven slow/fast/mid pick-and-place collector.  Once a real HMD sample is
    # available, LiveVRTrackingProvider schedules the exact same XR camera
    # placement used there and only falls back to a direct xrAnchor transform.
    _configure_xr_profile_settings(
        settings, profile_name=profile_name, xr_backend=xr_backend
    )
    for key in (
        "/xr/profile/vr/enableControllerPhysics",
        "/xr/profile/vr/controllerPhysicsEnabled",
        "/xr/profile/vr/enablePhysicsInteraction",
        "/xr/profile/vr/pickAndPlace/enabled",
    ):
        settings.set(key, False)

    enabled: list[str] = []
    for extension_id in extension_ids:
        try:
            enable_extension(extension_id)
            enabled.append(extension_id)
        except Exception:
            continue
    if not enabled:
        raise RuntimeError("No XR extension could be enabled for live VR collection")
    for _ in range(5):
        simulation_app.update()

    from omni.kit.xr.core import XRCore

    XRCore.request_enable_profile(profile_name)
    for _ in range(10):
        simulation_app.update()
    print(
        f"[ExplicitFeedback] requested XR profile={profile_name} "
        f"backend={xr_backend} extensions={','.join(enabled)} "
        f"anchor-mode=active-camera "
        f"tools-layout=vr",
        flush=True,
    )


def schedule_live_vr_camera_view(
    *,
    simulation_app: Any,
    target_world: np.ndarray,
) -> bool:
    """Place the HMD with the proven pick-and-place XR camera path."""

    try:
        from pxr import Gf
        from omni.kit.xr.core import XRCore

        try:
            from v3_chan.vr_avatar import AVATAR_EYE_POS
        except ImportError:
            from vr_avatar import AVATAR_EYE_POS

        eye = np.asarray(AVATAR_EYE_POS, dtype=np.float64).reshape(3)
        target = np.asarray(target_world, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(eye)) or not np.all(np.isfinite(target)):
            raise ValueError("XR camera eye/target must be finite")
        core = XRCore.get_singleton()
        try:
            up = core.get_coordinate_system().get_up_vector()
        except Exception:
            up = Gf.Vec3d(0.0, 0.0, 1.0)
        eye_gf = Gf.Vec3d(*(float(value) for value in eye))
        target_gf = Gf.Vec3d(*(float(value) for value in target))
        pose = Gf.Matrix4d().SetLookAt(eye_gf, target_gf, up).GetInverse()
        core.schedule_set_camera(pose)
        # Mirror the established collector: allow three Kit updates for the
        # scheduled XR camera command to reach the compositor before poses are
        # interpreted in the aligned avatar frame.
        for _ in range(3):
            simulation_app.update()
        print(
            "[ExplicitFeedback] XR camera placement scheduled: "
            f"eye={np.round(eye, 3)} target={np.round(target, 3)}",
            flush=True,
        )
        return True
    except Exception as error:
        print(
            "[ExplicitFeedback] XR camera placement failed; "
            f"using xrAnchor fallback: {type(error).__name__}: {error}",
            flush=True,
        )
        return False


def set_live_vr_anchor(room_hmd_pos: np.ndarray) -> bool:
    import omni.usd
    from pxr import Gf, UsdGeom

    try:
        from v3_chan.vr_avatar import (
            AVATAR_EYE_POS,
            ROOM_TO_WORLD_MATRIX_ROWS,
            room_to_world_point,
        )
    except ImportError:
        from vr_avatar import (
            AVATAR_EYE_POS,
            ROOM_TO_WORLD_MATRIX_ROWS,
            room_to_world_point,
        )

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/_xr/stage/xrAnchor")
    if not prim.IsValid():
        return False
    try:
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        translation = AVATAR_EYE_POS - room_to_world_point(room_hmd_pos)
        rows = [list(row) for row in ROOM_TO_WORLD_MATRIX_ROWS]
        rows[3][0:3] = [float(value) for value in translation]
        matrix = Gf.Matrix4d(*[value for row in rows for value in row])
        matrix_ops = [
            op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTransform
        ]
        if matrix_ops:
            matrix_ops[0].Set(matrix)
            xformable.SetXformOpOrder([matrix_ops[0]])
        else:
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp().Set(matrix)
        return True
    except Exception:
        return False
