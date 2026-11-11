# main.py — Isaac Sim 4.5 전용
#
# 실행 방법:
#   isaac ~/isaac_vr_project/v2/main.py
#
import sys; print("[main.py] Python started:", sys.version, flush=True); del sys
# =============================================================================

# ── 반드시 가장 먼저 임포트 (표준 라이브러리 제외) ────────────────────────────
import os
import signal

# SteamVR OpenXR 런타임 경로 자동 설정
_initial_xr_mode = os.environ.get("ISAAC_XR_MODE", "vr").strip().lower()
_initial_xr_enabled = _initial_xr_mode not in (
    "",
    "0",
    "false",
    "none",
    "off",
    "disabled",
)
if _initial_xr_enabled and "XR_RUNTIME_JSON" not in os.environ:
    _known_paths = [
        os.path.expanduser("~/.steam/debian-installation/steamapps/common/SteamVR/steamxr_linux64.json"),
        os.path.expanduser("~/.steam/steam/steamapps/common/SteamVR/steamxr_linux64.json"),
        os.path.expanduser("~/.local/share/Steam/steamapps/common/SteamVR/steamxr_linux64.json"),
        "/usr/share/steam/steamapps/common/SteamVR/steamxr_linux64.json",
    ]
    _found = next((p for p in _known_paths if os.path.exists(p)), None)
    if _found:
        os.environ["XR_RUNTIME_JSON"] = _found
        print(f"[VR] XR_RUNTIME_JSON={_found}", flush=True)
    else:
        print("[VR] Warning: steamxr_linux64.json not found — xrCreateInstance may fail.", flush=True)

from omni.isaac.kit import SimulationApp

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[Boot] Invalid {name}={value!r}; using {default}.", flush=True)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[Boot] Invalid {name}={value!r}; using {default}.", flush=True)
        return default


_isaac_root = os.environ.get("ISAACSIM_ROOT", os.getcwd())
_enable_vr_boot = _env_bool("ENABLE_VR", False)
_xr_mode = _initial_xr_mode
_xr_enabled = _initial_xr_enabled
_xr_backend = os.environ.get("ISAAC_XR_BACKEND", "").strip()
if not _xr_backend:
    _xr_backend = "OpenXR"
_xr_openxr_experience = os.path.join(_isaac_root, "apps", "isaacsim.exp.base.xr.openxr.kit")
_xr_vr_experience = os.path.join(_isaac_root, "apps", "isaacsim.exp.base.xr.vr.kit")
if not _xr_enabled:
    _xr_experience = None
elif _xr_mode == "openxr" and os.path.exists(_xr_openxr_experience):
    _xr_experience = _xr_openxr_experience
elif os.path.exists(_xr_vr_experience):
    _xr_mode = "vr"
    _xr_experience = _xr_vr_experience
else:
    _xr_experience = None
_use_xr_experience = _xr_experience is not None

_sim_config = {
    "headless": _env_bool("ISAAC_HEADLESS", False),
    "width": _env_int("ISAAC_WIDTH", 1280),
    "height": _env_int("ISAAC_HEIGHT", 720),
    "active_gpu": 0,
    "physics_gpu": 0,
    "multi_gpu": False,
    "max_gpu_count": 1,
}
if _use_xr_experience:
    _sim_config["experience"] = _xr_experience

print("[Boot] Creating SimulationApp...", flush=True)
simulation_app = SimulationApp(_sim_config)
print("[Boot] SimulationApp ready.", flush=True)

# ── SimulationApp 초기화 후에 나머지 임포트 ───────────────────────────────
import sys
import numpy as np
import omni.kit.app
import carb
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.viewports import set_camera_view
XRCore = None
if _xr_enabled:
    try:
        from omni.kit.xr.core import XRCore
    except Exception as exc:
        print(f"[VR] XRCore import failed: {exc}")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from app_settings.app_config import (
    BHAPTICS_NOTEBOOK_IP,
    BHAPTICS_UDP_PORT,
    ENABLE_GRIPPER_CAMERA,
    ENABLE_GRIPPER_CAMERA_RECORDING,
    ENABLE_GRIPPER_CAMERA_VIEWPORT,
    ERRP_MARKERS_PATH,
    GRIPPER_CAMERA_PRIM_PATH,
    GRIPPER_CAMERA_RECORD_DIR,
    GRIPPER_CAMERA_RECORD_INTERVAL_STEPS,
    GRIPPER_CAMERA_RECORD_RESOLUTION,
    HAND_TRACKING_UDP_HOST,
    HAND_TRACKING_UDP_PORT,
    SAMPLE_LOG_INTERVAL_STEPS,
    SESSION_SAMPLES_PATH,
)
from simulation.scene.scene_setup import create_world, randomize_cubes, setup_scene
from logging_tools.event_logger import EventLogger
from simulation.camera.gripper_camera import GripperCamera
from vr.hand_tracking import HandTrackingReceiver
from haptics.haptics_udp import HapticsUdpClient
from simulation.robot.panda_robot import add_panda, print_robot_info
from simulation.task.pick_controller import create_pick_controller, run_pick_place
from simulation.task.handover_task import HandoverTaskConfig, HandoverTaskManager
from learning.handover_dataset_recorder import (
    DEFAULT_DATASET_DIR,
    HandoverDatasetRecorder,
    build_flat_observation,
    observation_load_error,
    safe_joint_positions,
    safe_vec3,
    task_phase_from_sample,
)
from vr.vr_grab import VRGrabManager
from vr.vr_avatar import (
    VRAvatar,
    AVATAR_HEAD_INIT,
    AVATAR_EYE_POS,
    ROOM_TO_WORLD_MATRIX_ROWS,
    room_to_world_point,
)


def _enable_vr_extensions() -> None:
    global XRCore
    if not _xr_enabled:
        print("[VR] Disabled. Set ENABLE_VR=true or ISAAC_XR_MODE=vr/openxr to enable XR.")
        return
    if _xr_mode == "openxr":
        candidates = [
            "omni.kit.xr.system.openxr",
            "omni.kit.xr.profile.ar",
            "isaacsim.xr.openxr",
        ]
    elif _xr_backend.lower() == "openxr":
        candidates = [
            "omni.kit.xr.system.openxr",
            "omni.kit.xr.profile.vr",
        ]
    else:
        candidates = [
            "omni.kit.xr.system.steamvr",
            "omni.kit.xr.profile.vr",
        ]
    enabled = []
    for ext_id in candidates:
        try:
            enable_extension(ext_id)
            enabled.append(ext_id)
        except Exception:
            continue
    if enabled:
        print(f"[VR] Enabled extensions: {', '.join(enabled)}")
        for _ in range(5):
            simulation_app.update()
        if XRCore is None:
            try:
                from omni.kit.xr.core import XRCore as _XRCore

                XRCore = _XRCore
            except Exception as exc:
                print(f"[VR] XRCore import after extension startup failed: {exc}")
    else:
        print("[VR] No VR extensions found/enabled. Check Extensions window.")


def _request_vr_start(profile_name: str | None = None) -> None:
    if not _xr_enabled:
        return
    if XRCore is None:
        print("[VR] XRCore unavailable; skipping XR profile start.")
        return
    try:
        if profile_name is None:
            profile_name = "ar" if _xr_mode == "openxr" else "vr"
        settings = carb.settings.get_settings()
        settings.set("/xr/profile/" + profile_name + "/adjustForUserHeight", False)
        settings.set("/defaults/xr/profile/" + profile_name + "/adjustForUserHeight", False)
        settings.set("/xr/profile/" + profile_name + "/system/display", _xr_backend)
        settings.set("/defaults/xr/profile/" + profile_name + "/system/display", _xr_backend)
        settings.set("/xr/ui/enabled", False)
        if profile_name == "ar":
            settings.set("/xrstage/profile/ar/anchorMode", "scene origin")
        xr_builtin_controller_physics = _env_bool("XR_BUILTIN_CONTROLLER_PHYSICS", False)
        # XR 내장 controller physics는 Isaac/PhysX에서 불안정할 수 있어 기본 OFF.
        # 실제 손 충돌은 VRAvatar의 kinematic physical hand proxy가 담당한다.
        for key in (
            "/xr/profile/vr/enableControllerPhysics",
            "/xr/profile/vr/controllerPhysicsEnabled",
            "/xr/profile/vr/enablePhysicsInteraction",
            "/xr/profile/vr/pickAndPlace/enabled",
        ):
            try:
                settings.set(key, xr_builtin_controller_physics)
            except Exception:
                pass
        xr_show_controllers = os.environ.get("XR_SHOW_CONTROLLERS", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        # 손 프록시를 직접 그릴 때만 XR 기본 컨트롤러 모델을 숨길 수 있다.
        # Isaac/Kit 버전에 따라 실제 키가 다를 수 있어, 없는 키는 조용히 무시된다.
        for key in (
            "/xr/profile/vr/showControllers",
            "/xr/profile/vr/renderControllers",
            "/xr/profile/vr/controllerModel/enabled",
            "/xr/profile/vr/controllers/visible",
            "/xr/profile/vr/controllerVisualization/enabled",
            "/defaults/xr/profile/vr/showControllers",
            "/defaults/xr/profile/vr/renderControllers",
            "/defaults/xr/profile/vr/controllerModel/enabled",
            "/defaults/xr/profile/vr/controllers/visible",
            "/defaults/xr/profile/vr/controllerVisualization/enabled",
        ):
            try:
                settings.set(key, xr_show_controllers)
            except Exception:
                pass
        XRCore.request_enable_profile(profile_name)
        for _ in range(10):
            simulation_app.update()
        print(
            f"[VR] Requested XR profile start: {profile_name} backend={_xr_backend} "
            f"builtin_controller_physics={xr_builtin_controller_physics}"
        )
    except Exception as exc:
        print(f"[VR] Failed to request XR profile '{profile_name}': {exc}")


def _set_xr_anchor_once(room_hmd_pos: np.ndarray) -> bool:
    """
    xrAnchor 를 SteamVR room space → Isaac world space 변환으로 1회 설정.
    SteamVR room: x=right, y=up, -z=forward
    Isaac world: z=up, user faces the table along -x
    """
    import omni.usd
    from pxr import UsdGeom, Gf
    XR_ANCHOR = "/_xr/stage/xrAnchor"
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(XR_ANCHOR)
    if not prim.IsValid():
        print(f"[XR] {XR_ANCHOR} 없음 — xrCreateInstance 실패로 생성 안 됐을 수 있음")
        return False
    try:
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        op_names = [op.GetOpName() for op in ops]
        print(f"[XR] xrAnchor existing xformOps: {op_names}")

        translation = AVATAR_EYE_POS - room_to_world_point(room_hmd_pos)
        rows = [list(row) for row in ROOM_TO_WORLD_MATRIX_ROWS]
        rows[3][0] = float(translation[0])
        rows[3][1] = float(translation[1])
        rows[3][2] = float(translation[2])
        mat = Gf.Matrix4d(*[value for row in rows for value in row])

        # xrAnchor는 좌표계 변환 루트라 기존 transform op가 있으면 재사용한다.
        matrix_ops = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTransform]
        if matrix_ops:
            matrix_ops[0].Set(mat)
            xformable.SetXformOpOrder([matrix_ops[0]])
        else:
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp().Set(mat)
        print(f"[XR] xrAnchor room→world transform set, t={np.round(translation, 3)}")
        return True
    except Exception as e:
        print(f"[XR] xrAnchor 이동 실패: {e}")
        return False


def _schedule_xr_camera_view(eye: np.ndarray, target: np.ndarray) -> bool:
    """
    XRCore에 직접 현재 HMD view를 원하는 월드 카메라 pose로 teleport 요청.
    set_camera_view()는 데스크톱 viewport만 만지고, VR HMD view는 이 API가 필요하다.
    """
    try:
        if not _xr_enabled or XRCore is None:
            return False
        from pxr import Gf

        core = XRCore.get_singleton()
        try:
            up_vec = core.get_coordinate_system().get_up_vector()
        except Exception:
            up_vec = Gf.Vec3d(0.0, 0.0, 1.0)

        eye_v = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
        target_v = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
        pose = Gf.Matrix4d().SetLookAt(eye_v, target_v, up_vec).GetInverse()
        core.schedule_set_camera(pose)
        for _ in range(3):
            simulation_app.update()
        print(
            "[XR] XRCore camera teleport scheduled: "
            f"eye={np.round(eye, 3)} target={np.round(target, 3)}"
        )
        return True
    except Exception as exc:
        print(f"[XR] XRCore camera teleport failed: {exc}")
        return False


def _distance_or_none(a: "np.ndarray | None", b: "np.ndarray | None") -> "float | None":
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _min_optional(*values: "float | None") -> "float | None":
    valid = [v for v in values if v is not None]
    return min(valid) if valid else None


GRIPPER_HAPTIC_LINK_NAMES = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
HAPTICS_ROBOT_COLLISION_LINK_FILTER = os.environ.get(
    "HAPTICS_ROBOT_COLLISION_LINK_FILTER", "gripper"
).strip().lower()


def _should_pulse_haptics_for_hits(hit_prims: list[str]) -> bool:
    if not hit_prims:
        return False
    if HAPTICS_ROBOT_COLLISION_LINK_FILTER in ("all", "any", "*"):
        return True
    if HAPTICS_ROBOT_COLLISION_LINK_FILTER in ("off", "none", "disabled"):
        return False
    return any(
        gripper_name in hit
        for hit in hit_prims
        for gripper_name in GRIPPER_HAPTIC_LINK_NAMES
    )


def _gripper_center_from_fingers(robot) -> "np.ndarray | None":
    try:
        left_pos, _ = robot.gripper._left_finger.get_world_pose()
        right_pos, _ = robot.gripper._right_finger.get_world_pose()
        return (np.asarray(left_pos, dtype=float) + np.asarray(right_pos, dtype=float)) * 0.5
    except Exception:
        return None


def _world_position_or_none(obj) -> "np.ndarray | None":
    if obj is None:
        return None
    try:
        pos, _ = obj.get_world_pose()
    except Exception:
        return None
    try:
        arr = np.asarray(pos, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


# =============================================================================
# 설정
# =============================================================================
ENABLE_PICK_PLACE = os.environ.get("ENABLE_PICK_PLACE", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ENABLE_HANDOVER_TASK = os.environ.get("ENABLE_HANDOVER_TASK", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
WAIT_FOR_VR_TRACKING = os.environ.get("WAIT_FOR_VR_TRACKING", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MIRROR_VIEW_TO_HMD = os.environ.get("MIRROR_VIEW_TO_HMD", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
XR_CAMERA_TELEPORT = os.environ.get("XR_CAMERA_TELEPORT", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SIM_MAX_STEPS = _env_int("SIM_MAX_STEPS", 0)
ENABLE_HANDOVER_DATASET_RECORDING = _env_bool(
    "ENABLE_HANDOVER_DATASET_RECORDING",
    ENABLE_HANDOVER_TASK,
)
_handover_dataset_dir_raw = os.environ.get("HANDOVER_DATASET_DIR", DEFAULT_DATASET_DIR)
HANDOVER_DATASET_DIR = (
    _handover_dataset_dir_raw
    if os.path.isabs(_handover_dataset_dir_raw)
    else os.path.join(PROJECT_DIR, _handover_dataset_dir_raw)
)
HANDOVER_DATASET_OVERWRITE = _env_bool("HANDOVER_DATASET_OVERWRITE", False)
HANDOVER_DATASET_RECORD_FAILED = _env_bool("HANDOVER_DATASET_RECORD_FAILED", True)
HANDOVER_DATASET_MIN_STEPS = _env_int("HANDOVER_DATASET_MIN_STEPS", 20)
HANDOVER_DATASET_RECORD_IMAGES = _env_bool("HANDOVER_DATASET_RECORD_IMAGES", False)
HANDOVER_DATASET_EXIT_AFTER_EPISODES = max(
    0,
    _env_int("HANDOVER_DATASET_EXIT_AFTER_EPISODES", 0),
)
HANDOVER_DATASET_SAVE_ON_INTERRUPT = _env_bool("HANDOVER_DATASET_SAVE_ON_INTERRUPT", True)
HANDOVER_DATASET_AUTO_COMPLETE_ON_PLACE = _env_bool(
    "HANDOVER_DATASET_AUTO_COMPLETE_ON_PLACE",
    True,
)
HANDOVER_DATASET_AUTO_COMPLETE_STABLE_STEPS = max(
    1,
    _env_int("HANDOVER_DATASET_AUTO_COMPLETE_STABLE_STEPS", 8),
)
HANDOVER_DATASET_PLACE_SUCCESS_DIST_M = max(
    0.005,
    _env_float("HANDOVER_DATASET_PLACE_SUCCESS_DIST_M", 0.08),
)
HANDOVER_DATASET_VALIDATE_SUCCESS_POSE = _env_bool(
    "HANDOVER_DATASET_VALIDATE_SUCCESS_POSE",
    True,
)


# =============================================================================
# 메인
# =============================================================================
def main():
    print("[Main] Entered main().", flush=True)
    if _use_xr_experience:
        print(f"[VR] Using XR experience ({_xr_mode}): {_xr_experience}")
    else:
        print("[VR] XR experience disabled; starting standard Isaac Sim app.")
    print(f"[VR] Backend={_xr_backend}")
    print(f"[VR] WAIT_FOR_VR_TRACKING={WAIT_FOR_VR_TRACKING}")
    print(f"[Main] ENABLE_PICK_PLACE={ENABLE_PICK_PLACE}")
    print(f"[Main] ENABLE_HANDOVER_TASK={ENABLE_HANDOVER_TASK}")
    print(f"[Dataset] ENABLE_HANDOVER_DATASET_RECORDING={ENABLE_HANDOVER_DATASET_RECORDING}")
    print(f"[XR] XR_CAMERA_TELEPORT={XR_CAMERA_TELEPORT}")
    if SIM_MAX_STEPS > 0:
        print(f"[Main] SIM_MAX_STEPS={SIM_MAX_STEPS}")
    if _xr_enabled:
        _enable_vr_extensions()
        _request_vr_start()

    # 1. 월드 생성
    print("[Main] Creating world...", flush=True)
    world = create_world()
    print("[Main] World created.", flush=True)

    # 2. 씬 구성
    print("[Main] Setting up scene...", flush=True)
    (
        cubes,
        place_target,
        table_top_z,
        cube_size,
        table_xy,
        table_size,
        stack_base_xy,
    ) = setup_scene(world, cube_count=6)
    print("[Main] Scene ready.", flush=True)
    pick_targets = cubes[:3]
    green_indices = list(range(3, 6))
    green_cubes = [cubes[i] for i in green_indices if i < len(cubes)]
    cube_half = cube_size / 2.0
    cube_center_z = table_top_z + cube_half
    viewport_target = np.array([table_xy[0], table_xy[1], table_top_z])
    log_path = ERRP_MARKERS_PATH

    # 3. Panda 추가
    print("[Main] Adding Panda...", flush=True)
    panda = add_panda(world, base_z=table_top_z)
    print("[Main] Panda added.", flush=True)

    # 4. VR 아바타 프림 생성 (world.reset() 전에 추가해야 함)
    avatar = None
    if _xr_enabled:
        avatar = VRAvatar()
        avatar.setup(world)
    else:
        print("[VR] Avatar disabled for non-XR run.")

    # 5. 리셋 (물리 핸들 초기화 — 반드시 필요)
    print("[Main] Resetting world...", flush=True)
    world.reset()
    world.play()
    print("[Main] World reset complete.", flush=True)

    place_target.set_world_pose(
        position=np.array([stack_base_xy[0], stack_base_xy[1], table_top_z + cube_half])
    )

    # 6. 기본 정보 출력
    print_robot_info(panda)

    # 7. Pick-and-Place 컨트롤러
    controller = None
    approach_height = table_top_z + 0.2
    if ENABLE_PICK_PLACE or ENABLE_HANDOVER_TASK:
        controller = create_pick_controller(panda, end_effector_initial_height=approach_height)
        print("[Phase 2] Pick-and-Place 컨트롤러 활성화")
    else:
        print("[Phase 1] 씬 확인 모드")

    # 8. 그리퍼 카메라 (가림 여부 판단 기반)
    gripper_camera = GripperCamera(
        prim_path=GRIPPER_CAMERA_PRIM_PATH,
        enabled=ENABLE_GRIPPER_CAMERA,
        show_viewport=ENABLE_GRIPPER_CAMERA_VIEWPORT,
        record_enabled=ENABLE_GRIPPER_CAMERA_RECORDING,
        record_dir=GRIPPER_CAMERA_RECORD_DIR,
        record_resolution=GRIPPER_CAMERA_RECORD_RESOLUTION,
        record_interval_steps=GRIPPER_CAMERA_RECORD_INTERVAL_STEPS,
    )
    gripper_camera.setup()

    # 9. VR/UDP 그랩 매니저 (avatar가 있으면 XRCore 재사용, 없으면 UDP hand tracking 사용)
    # Handover mode also lets the user choose the object naturally: whichever
    # table cube is grabbed first becomes the active handover cube.
    grab_cubes = cubes
    grab_manager = (
        VRGrabManager(grab_cubes, avatar=avatar, cube_size=cube_size)
        if (_xr_enabled or ENABLE_HANDOVER_TASK)
        else None
    )

    # 10. 시뮬레이션 루프
    xr_anchor_done = False   # xrAnchor 1회 이동 완료 플래그
    step = 0
    sim_time = 0.0
    task_done = False
    current_pick_idx = 0
    completed_picks = 0
    cycle_reset_requested = False
    dataset_auto_complete_steps = 0
    saved_dataset_episodes = 0
    logger = EventLogger(
        log_path=log_path,
        cube_size=cube_size,
        speed_threshold=0.6,
        collision_dist=cube_size * 0.9,
        stack_drop_threshold=0.03,
        max_human_collisions=1000,
        sample_path=SESSION_SAMPLES_PATH,
        sample_interval_steps=SAMPLE_LOG_INTERVAL_STEPS,
    )
    haptics = HapticsUdpClient(BHAPTICS_NOTEBOOK_IP, BHAPTICS_UDP_PORT)
    hand_tracking = HandTrackingReceiver(HAND_TRACKING_UDP_HOST, HAND_TRACKING_UDP_PORT)
    handover_manager = None
    if ENABLE_HANDOVER_TASK and controller is not None:
        if grab_manager is None:
            print("[Handover] VR grab manager unavailable; task will wait for XR/hand input.")
        handover_manager = HandoverTaskManager(
            cubes=cubes,
            pick_targets=pick_targets,
            place_target=place_target,
            table_top_z=table_top_z,
            cube_size=cube_size,
            stack_base_xy=stack_base_xy,
            controller=controller,
            robot=panda,
            approach_height=approach_height,
            grab_manager=grab_manager,
            haptics=haptics,
            logger=logger,
            config=HandoverTaskConfig.from_env(),
        )
    cubes_by_name = {getattr(cube, "name", f"cube_{idx}"): cube for idx, cube in enumerate(cubes)}
    handover_dataset_recorder = None
    if ENABLE_HANDOVER_DATASET_RECORDING:
        handover_dataset_recorder = HandoverDatasetRecorder(
            HANDOVER_DATASET_DIR,
            overwrite=HANDOVER_DATASET_OVERWRITE,
            record_failed=HANDOVER_DATASET_RECORD_FAILED,
            min_steps=HANDOVER_DATASET_MIN_STEPS,
            record_images=HANDOVER_DATASET_RECORD_IMAGES,
        )
        obs_error = observation_load_error()
        if obs_error:
            print(f"[Dataset] observation builder unavailable: {obs_error}", flush=True)
        print(
            "[Dataset] handover recorder ready "
            f"dir={HANDOVER_DATASET_DIR} record_failed={HANDOVER_DATASET_RECORD_FAILED} "
            f"min_steps={HANDOVER_DATASET_MIN_STEPS} record_images={HANDOVER_DATASET_RECORD_IMAGES} "
            f"exit_after_episodes={HANDOVER_DATASET_EXIT_AFTER_EPISODES} "
            f"save_on_interrupt={HANDOVER_DATASET_SAVE_ON_INTERRUPT} "
            f"place_success_dist={HANDOVER_DATASET_PLACE_SUCCESS_DIST_M:.3f} "
            f"validate_success_pose={HANDOVER_DATASET_VALIDATE_SUCCESS_POSE}",
            flush=True,
        )
    elif ENABLE_HANDOVER_TASK:
        print(
            "[Dataset] WARNING: handover dataset recording is disabled. "
            "Set ENABLE_HANDOVER_DATASET_RECORDING=1 to save trajectories.",
            flush=True,
        )

    def _dataset_sample_cube(sample: dict | None):
        if sample is None:
            return None
        active_cube_name = str(sample.get("active_cube") or "")
        cube = cubes_by_name.get(active_cube_name)
        if cube is not None:
            return cube
        if handover_manager is not None:
            try:
                return handover_manager.active_cube
            except Exception:
                return None
        return None

    def _dataset_place_pos(sample: dict | None) -> "np.ndarray | None":
        if sample is not None:
            pos = safe_vec3(sample.get("place_pos"))
            if np.all(np.isfinite(pos)):
                return np.asarray(pos, dtype=float)
        if handover_manager is not None:
            try:
                return np.asarray(handover_manager.active_place_pos, dtype=float)
            except Exception:
                pass
        return _world_position_or_none(place_target)

    def _dataset_cube_place_dist(sample: dict | None) -> "float | None":
        cube = _dataset_sample_cube(sample)
        cube_pos = _world_position_or_none(cube)
        place_pos = _dataset_place_pos(sample)
        if cube_pos is None or place_pos is None:
            return None
        return float(np.linalg.norm(np.asarray(cube_pos, dtype=float) - np.asarray(place_pos, dtype=float)))

    def _dataset_pose_success(sample: dict | None, *, require_release_stage: bool) -> tuple[bool, str]:
        if not sample:
            return False, ""
        dist = _dataset_cube_place_dist(sample)
        if dist is None or dist > HANDOVER_DATASET_PLACE_SUCCESS_DIST_M:
            return False, ""
        carry_stage = str(sample.get("physical_carry_stage") or "")
        gripper_command = str(sample.get("gripper_command") or "")
        release_like = carry_stage == "release" or gripper_command == "open"
        if require_release_stage and not release_like:
            return False, ""
        if not bool(sample.get("robot_has_taken_over", False)) and not release_like:
            return False, ""
        return True, f"cube_at_place_dist={dist:.3f}"

    def _finish_active_dataset_episode(reason: str) -> None:
        if handover_dataset_recorder is None or not handover_dataset_recorder.active:
            return
        if not HANDOVER_DATASET_SAVE_ON_INTERRUPT and reason in (
            "keyboard_interrupt",
            "simulation_shutdown",
        ):
            handover_dataset_recorder.abort_episode(reason)
            if handover_manager is not None:
                handover_manager.clear_learning_terminal_event()
            return
        sample = handover_manager.get_learning_sample() if handover_manager is not None else {}
        success_event = str(sample.get("success_event") or "")
        failure_reason = str(sample.get("failure_reason") or "")
        success = bool(success_event)
        terminal_event = success_event or failure_reason or reason
        if not success:
            pose_success, pose_event = _dataset_pose_success(sample, require_release_stage=False)
            if pose_success:
                success = True
                terminal_event = f"{reason}:{pose_event}"
                failure_reason = ""
        handover_dataset_recorder.end_episode(
            success=success,
            failure_reason="" if success else (failure_reason or reason),
            metadata={
                "end_step": step,
                "end_sim_time": sim_time,
                "terminal_event": terminal_event,
            },
        )
        if handover_manager is not None:
            handover_manager.clear_learning_terminal_event()

    def _sim_time(world_obj, step_idx: int) -> float:
        if hasattr(world_obj, "current_time"):
            return float(world_obj.current_time)
        if hasattr(world_obj, "get_physics_dt"):
            return float(step_idx) * float(world_obj.get_physics_dt())
        return float(step_idx) * (1.0 / 60.0)

    print("[Main] Entering simulation loop...", flush=True)
    waiting_for_vr_logged = False
    shutdown_requested = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def _request_shutdown(signum, frame) -> None:
        nonlocal shutdown_requested
        if not shutdown_requested:
            print(
                "[Main] Ctrl-C received; will save active dataset episode before closing.",
                flush=True,
            )
        shutdown_requested = True

    signal.signal(signal.SIGINT, _request_shutdown)
    while simulation_app.is_running():
        try:
            world.step(render=True)
        except KeyboardInterrupt:
            print("[Main] KeyboardInterrupt received; saving active dataset episode and closing.")
            _finish_active_dataset_episode("keyboard_interrupt")
            break

        if world.is_playing():
            if world.current_time_step_index == 0:
                if grab_manager is not None:
                    grab_manager.release_all()
                world.reset()
                if controller is not None:
                    controller.reset(end_effector_initial_height=approach_height)
                if handover_manager is not None:
                    handover_manager.reset_cycle()
                if handover_dataset_recorder is not None and handover_dataset_recorder.active:
                    _finish_active_dataset_episode("world_reset")
                place_target.set_world_pose(
                    position=np.array([stack_base_xy[0], stack_base_xy[1], table_top_z + cube_half])
                )
                task_done = False
                step = 0
                cycle_reset_requested = False

            step += 1
            sim_time = _sim_time(world, step)
            logger.update_context(step, sim_time)
            logger.ensure_episode_started()

            hand_tracking.poll()
            pinch_points = hand_tracking.get_pinch_points()
            tracked_hand_positions = hand_tracking.get_hand_positions()

            # ── xrAnchor 1회 이동: HMD 첫 감지 시점에 눈 위치로 고정 ────────
            # VR HMD view는 데스크톱 viewport와 별개라 XRCore camera teleport가 필요함.
            if _xr_enabled and avatar is not None and not xr_anchor_done:
                p_hmd0 = avatar.capture_initial_hmd_pos()
                if p_hmd0 is not None:
                    camera_ok = False
                    if XR_CAMERA_TELEPORT:
                        camera_ok = _schedule_xr_camera_view(AVATAR_EYE_POS, viewport_target)
                    if camera_ok or _set_xr_anchor_once(p_hmd0):
                        avatar.notify_anchor_applied()
                    xr_anchor_done = True  # 성공 여부와 무관하게 재시도 금지

            # ── VR 아바타 갱신 (머리·손 프림 위치 + 팔 DebugDraw) ──────────
            if avatar is not None:
                head_pos, left_pos, right_pos = avatar.update()
            else:
                head_pos, left_pos, right_pos = None, None, None
            if left_pos is None:
                left_pos = tracked_hand_positions.get("left")
            if right_pos is None:
                right_pos = tracked_hand_positions.get("right")
            if MIRROR_VIEW_TO_HMD and head_pos is not None and step % 2 == 0:
                try:
                    hmd_forward = avatar.get_hmd_forward()
                    view_target = (
                        head_pos + hmd_forward
                        if hmd_forward is not None
                        else viewport_target
                    )
                    set_camera_view(eye=head_pos, target=view_target)
                except Exception:
                    pass

            vr_tracking_ready = (
                (head_pos is not None and (left_pos is not None or right_pos is not None))
                or (not _xr_enabled and (left_pos is not None or right_pos is not None))
            )
            if WAIT_FOR_VR_TRACKING and not vr_tracking_ready:
                if not waiting_for_vr_logged or step % 300 == 0:
                    print(
                        "[VR] Waiting for real XR tracking. "
                        "Start SteamVR/ALVR, connect the headset, and make sure SteamVR is the OpenXR runtime."
                    )
                    waiting_for_vr_logged = True
                continue
            waiting_for_vr_logged = False

            # 팔↔로봇 충돌 → ErrP
            ee_pos, _ = panda.end_effector.get_world_pose()
            camera_cube = (
                handover_manager.active_cube
                if handover_manager is not None
                else pick_targets[current_pick_idx]
            )
            current_cube_pos_for_camera, _ = camera_cube.get_world_pose()
            gripper_camera.update(
                ee_pos,
                target_pos=current_cube_pos_for_camera,
                mount_pos=_gripper_center_from_fingers(panda),
            )
            arm_robot_collision_active = False
            for hand, ctrl_pos in (("left", left_pos), ("right", right_pos)):
                hits = (
                    avatar.check_robot_collision(hand, head_pos, ctrl_pos)
                    if avatar is not None
                    else []
                )
                logger.check_arm_robot_collision(hand, hits)
                if _should_pulse_haptics_for_hits(hits):
                    arm_robot_collision_active = True
                    haptics.pulse(100, hand=hand, event="gripper_robot_collision")

            left_hand_gripper_dist = _distance_or_none(left_pos, ee_pos)
            right_hand_gripper_dist = _distance_or_none(right_pos, ee_pos)
            logger.log_sample(
                left_hand_gripper_dist=left_hand_gripper_dist,
                right_hand_gripper_dist=right_hand_gripper_dist,
                min_hand_gripper_dist=_min_optional(
                    left_hand_gripper_dist,
                    right_hand_gripper_dist,
                ),
                human_robot_collision=arm_robot_collision_active,
            )

            # VR 그랩 (hand tracking pinch 우선, 없으면 컨트롤러 grip 폴백)
            if grab_manager is not None:
                handover_phase = handover_manager.phase if handover_manager is not None else None
                if handover_manager is not None:
                    grab_manager.set_active_target(
                        handover_manager.grab_target_cube,
                        hand=handover_manager.active_hand,
                    )
                else:
                    grab_manager.set_active_target(None)
                grab_allowed = handover_phase not in ("robot_takeover",)
                if grab_allowed:
                    grab_manager.update(left_pos, right_pos, pinch_points=pinch_points)

            if handover_manager is not None:
                result = handover_manager.update(
                    hand_positions={"left": left_pos, "right": right_pos},
                    step=step,
                    sim_time=sim_time,
                )
                current_pick_idx = result.current_pick_idx
                completed_picks = result.completed_picks
                if handover_dataset_recorder is not None:
                    sample = handover_manager.get_learning_sample()
                    should_start_dataset_episode = bool(
                        sample.get("episode_active")
                    ) or str(result.message).startswith("robot_takeover_started")
                    if should_start_dataset_episode and not handover_dataset_recorder.active:
                        active_hand = str(sample.get("active_hand") or "hand")
                        instruction = (
                            f"Receive the cube from the {active_hand} hand "
                            "and place it on the target."
                        )
                        handover_dataset_recorder.start_episode(
                            instruction=instruction,
                            metadata={
                                "start_step": step,
                                "start_sim_time": sim_time,
                                "active_hand": sample.get("active_hand", ""),
                                "active_cube": sample.get("active_cube", ""),
                            },
                        )
                        dataset_auto_complete_steps = 0
                    if handover_dataset_recorder.active:
                        active_cube_name = str(sample.get("active_cube") or "")
                        sample_cube = cubes_by_name.get(active_cube_name) or result.active_cube
                        if sample_cube is None:
                            sample_cube = handover_manager.active_cube
                        sample_place_pos = safe_vec3(sample.get("place_pos"))
                        if not np.all(np.isfinite(sample_place_pos)):
                            sample_place_pos = safe_vec3(_world_position_or_none(place_target))
                        place_for_observation = (
                            sample_place_pos
                            if np.all(np.isfinite(sample_place_pos))
                            else place_target
                        )
                        gripper_center_pos = _gripper_center_from_fingers(panda)
                        obs_state = build_flat_observation(
                            robot=panda,
                            cube=sample_cube,
                            place_target=place_for_observation,
                            human_head_pos=head_pos,
                            human_left_hand_pos=left_pos,
                            human_right_hand_pos=right_pos,
                            gripper_center_pos=gripper_center_pos,
                            human_robot_collision=arm_robot_collision_active,
                            has_grasped_cube=bool(sample.get("robot_has_taken_over", False)),
                            task_phase=task_phase_from_sample(sample),
                            controller_event=sample.get("controller_event"),
                            controller_t=float(sample.get("controller_t", 0.0)),
                            min_hand_gripper_dist_override=_min_optional(
                                left_hand_gripper_dist,
                                right_hand_gripper_dist,
                            ),
                        )
                        handover_dataset_recorder.add_step(
                            obs_state=obs_state,
                            sample=sample,
                            joint_pos=safe_joint_positions(panda),
                            cube_pos=_world_position_or_none(sample_cube),
                            place_pos=sample_place_pos,
                            left_hand_pos=left_pos,
                            right_hand_pos=right_pos,
                            step=step,
                            sim_time=sim_time,
                            image_path="",
                        )
                        success_event = str(sample.get("success_event") or "")
                        failure_reason = str(sample.get("failure_reason") or "")
                        if (
                            not success_event
                            and not failure_reason
                            and HANDOVER_DATASET_AUTO_COMPLETE_ON_PLACE
                        ):
                            pose_success, pose_event = _dataset_pose_success(
                                sample,
                                require_release_stage=True,
                            )
                            if pose_success:
                                dataset_auto_complete_steps += 1
                            else:
                                dataset_auto_complete_steps = 0
                            if (
                                dataset_auto_complete_steps
                                >= HANDOVER_DATASET_AUTO_COMPLETE_STABLE_STEPS
                            ):
                                success_event = f"auto_place_complete:{pose_event}"
                                print(
                                    "[Dataset] auto place complete "
                                    f"stable={dataset_auto_complete_steps}/"
                                    f"{HANDOVER_DATASET_AUTO_COMPLETE_STABLE_STEPS} "
                                    f"{pose_event}",
                                    flush=True,
                                )
                        if success_event or failure_reason:
                            if success_event and HANDOVER_DATASET_VALIDATE_SUCCESS_POSE:
                                pose_success, pose_event = _dataset_pose_success(
                                    sample,
                                    require_release_stage=False,
                                )
                                if not pose_success:
                                    dist = _dataset_cube_place_dist(sample)
                                    failure_reason = (
                                        "success_event_pose_mismatch:"
                                        f"event={success_event},"
                                        f"dist={dist if dist is not None else 'nan'},"
                                        f"limit={HANDOVER_DATASET_PLACE_SUCCESS_DIST_M:.3f}"
                                    )
                                    print(
                                        "[Dataset] success event rejected by pose validation "
                                        f"{failure_reason}",
                                        flush=True,
                                    )
                                    success_event = ""
                                else:
                                    success_event = f"{success_event}:{pose_event}"
                            manifest = handover_dataset_recorder.end_episode(
                                success=bool(success_event),
                                failure_reason=failure_reason,
                                metadata={
                                    "end_step": step,
                                    "end_sim_time": sim_time,
                                    "terminal_event": success_event or failure_reason,
                                },
                            )
                            handover_manager.clear_learning_terminal_event()
                            dataset_auto_complete_steps = 0
                            if manifest is not None:
                                saved_dataset_episodes += 1
                                if (
                                    HANDOVER_DATASET_EXIT_AFTER_EPISODES > 0
                                    and saved_dataset_episodes
                                    >= HANDOVER_DATASET_EXIT_AFTER_EPISODES
                                ):
                                    print(
                                        "[Dataset] exit_after_episodes reached "
                                        f"({saved_dataset_episodes}/"
                                        f"{HANDOVER_DATASET_EXIT_AFTER_EPISODES}); closing.",
                                        flush=True,
                                    )
                                    break
                    elif sample.get("success_event") or sample.get("failure_reason"):
                        print(
                            "[Dataset] WARNING: handover terminal event arrived but "
                            "no dataset episode was active; nothing was saved. "
                            f"success_event={sample.get('success_event')!r} "
                            f"failure_reason={sample.get('failure_reason')!r} "
                            f"message={result.message!r}",
                            flush=True,
                        )
                if result.completed_cycle:
                    logger.end_episode(
                        f"reason=completed_handover_cycle,picks={handover_manager.cycle_length}"
                    )
                    logger.reset_cycle()
                    logger.start_episode("reason=handover_cycle_complete")

            # 아바타 위치 디버그 (처음 10초간, 1초마다)
            if step % 60 == 0 and step <= 600:
                cube0_pos, _ = cubes[0].get_world_pose()
                print(
                    f"[DBG] head={np.round(head_pos, 3) if head_pos is not None else 'None'}"
                    f"  L={np.round(left_pos, 3) if left_pos is not None else 'None'}"
                    f"  R={np.round(right_pos, 3) if right_pos is not None else 'None'}"
                    f"  cube0={np.round(cube0_pos, 3)}"
                )

            # 약 2초마다 상태 출력
            if step % 120 == 0:
                cube_pos, _ = pick_targets[current_pick_idx].get_world_pose()
                stack_height = (
                    table_top_z + cube_half
                    + (completed_picks % len(pick_targets)) * (cube_size + 0.002)
                )
                place_pos = np.array([stack_base_xy[0], stack_base_xy[1], stack_height])
                ee_pos, _ = panda.end_effector.get_world_pose()
                gripper_pos = panda.gripper.get_joint_positions()
                print(
                    f"[Step {step:5d}] "
                    f"Cube: {np.round(cube_pos, 3)} | "
                    f"Place: {np.round(place_pos, 3)} | "
                    f"EE: {np.round(ee_pos, 3)} | "
                    f"Gripper: {np.round(gripper_pos, 4)}"
                )

            # ── Phase 2: 로봇 Pick-and-Place ─────────────────────────────────
            if (
                ENABLE_PICK_PLACE
                and handover_manager is None
                and controller is not None
                and not task_done
            ):
                current_pick_pos, _ = pick_targets[current_pick_idx].get_world_pose()
                stack_height = (
                    table_top_z + cube_half
                    + (completed_picks % len(pick_targets)) * (cube_size + 0.002)
                )
                place_pos = np.array([stack_base_xy[0], stack_base_xy[1], stack_height])
                task_done = run_pick_place(
                    controller=controller,
                    robot=panda,
                    pick_position=current_pick_pos,
                    place_position=place_pos,
                )
                if task_done:
                    placed_cube = pick_targets[current_pick_idx]
                    logger.record_stack_expected(placed_cube.name, stack_height)
                    completed_picks += 1
                    current_pick_idx = (current_pick_idx + 1) % len(pick_targets)
                    controller.reset(end_effector_initial_height=approach_height)
                    task_done = False
                    logger.reset_pick_miss()
                    if completed_picks % len(pick_targets) == 0:
                        logger.end_episode(
                            f"reason=completed_pick_cycle,picks={len(pick_targets)}"
                        )
                        randomize_cubes(
                            cubes, table_xy, table_size, cube_center_z,
                            cube_size, forbidden_xy=stack_base_xy,
                        )
                        completed_picks = 0
                        current_pick_idx = 0
                        controller.reset(end_effector_initial_height=approach_height)
                        cycle_reset_requested = True
                        logger.reset_cycle()
                        logger.start_episode("reason=cubes_randomized")
                        print(f"\n✅ [Step {step}] 3개 Pick-and-Place 완료! 새 배치로 재시작")

            if cycle_reset_requested:
                if grab_manager is not None:
                    grab_manager.release_all()
                world.reset()
                if controller is not None:
                    controller.reset(end_effector_initial_height=approach_height)
                if handover_manager is not None:
                    handover_manager.reset_cycle()
                if handover_dataset_recorder is not None and handover_dataset_recorder.active:
                    _finish_active_dataset_episode("cycle_reset_requested")
                place_target.set_world_pose(
                    position=np.array([stack_base_xy[0], stack_base_xy[1], table_top_z + cube_half])
                )
                task_done = False
                step = 0
                cycle_reset_requested = False

            # ── ErrP 후보 이벤트 감지 ─────────────────────────────────────────
            gripper_pos = panda.gripper.get_joint_positions()
            gripper_closed = np.all(np.array(gripper_pos) < 0.01)
            ee_pos, _ = panda.end_effector.get_world_pose()
            current_cube = (
                handover_manager.active_cube
                if handover_manager is not None
                else pick_targets[current_pick_idx]
            )
            logger.update_contact(ee_pos, pick_targets)
            logger.check_pick_miss(gripper_closed, ee_pos, current_cube)
            logger.check_drop_throw(pick_targets)
            logger.check_collision_green(pick_targets, green_cubes)
            logger.check_human_collision(
                ee_pos,
                avatar.get_avatar_prims() if avatar is not None else [],
            )
            logger.check_stack_failure(pick_targets)

            if SIM_MAX_STEPS > 0 and step >= SIM_MAX_STEPS:
                print(f"[Main] SIM_MAX_STEPS reached ({SIM_MAX_STEPS}); closing.")
                break
            if shutdown_requested:
                _finish_active_dataset_episode("keyboard_interrupt")
                break

    signal.signal(signal.SIGINT, previous_sigint_handler)
    if handover_dataset_recorder is not None and handover_dataset_recorder.active:
        _finish_active_dataset_episode("simulation_shutdown")
    hand_tracking.close()
    haptics.close()
    gripper_camera.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
