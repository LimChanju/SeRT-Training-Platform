# main.py — Isaac Sim 4.5 전용
#
# 실행 방법:
#   isaac ~/isaac_vr_project/v2/main.py
#
import sys; print("[main.py] Python started:", sys.version, flush=True); del sys
# =============================================================================

# ── 반드시 가장 먼저 임포트 (표준 라이브러리 제외) ────────────────────────────
import os
import time

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_config import (
    BHAPTICS_NOTEBOOK_IP,
    BHAPTICS_UDP_PORT,
    ENABLE_GRIPPER_CAMERA,
    ENABLE_GRIPPER_CAMERA_RECORDING,
    ENABLE_GRIPPER_CAMERA_VIEWPORT,
    ENABLE_HRI_TRAJECTORY_RECORDING,
    ERRP_MARKERS_PATH,
    GRIPPER_CAMERA_PRIM_PATH,
    GRIPPER_CAMERA_RECORD_DIR,
    GRIPPER_CAMERA_RECORD_INTERVAL_STEPS,
    GRIPPER_CAMERA_RECORD_RESOLUTION,
    HAND_TRACKING_UDP_HOST,
    HAND_TRACKING_UDP_PORT,
    HRI_TRAJECTORY_MAX_EPISODES,
    HRI_TRAJECTORY_OVERWRITE,
    HRI_TRAJECTORY_PATH,
    SAMPLE_LOG_INTERVAL_STEPS,
    SESSION_SAMPLES_PATH,
)
from scene_setup import create_world, randomize_cubes, setup_scene
from event_logger import EventLogger
from end_effector_safety_runtime import PandaEndEffectorSafetyRuntime
from gripper_camera import GripperCamera
from hand_tracking import HandTrackingReceiver
from haptics_udp import HapticsUdpClient
from hri_obs_recorder import HRIObsRecorder, build_observation
from panda_robot import add_panda, print_robot_info
from pick_controller import create_pick_controller, run_pick_place
from vr_grab import VRGrabManager
from vr_avatar import (
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
        # 컨트롤러 빌트인 물리 충돌 비활성화 (큐브와 충돌 시 크래시 방지)
        for key in (
            "/xr/profile/vr/enableControllerPhysics",
            "/xr/profile/vr/controllerPhysicsEnabled",
            "/xr/profile/vr/enablePhysicsInteraction",
            "/xr/profile/vr/pickAndPlace/enabled",
        ):
            try:
                settings.set(key, False)
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
        print(f"[VR] Requested XR profile start: {profile_name} backend={_xr_backend}")
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


DEBUG_HAPTICS_COLLISION = os.environ.get("DEBUG_HAPTICS_COLLISION", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HAPTICS_CONTACT_MIN_STEPS = max(1, int(os.environ.get("HAPTICS_CONTACT_MIN_STEPS", "1")))


def _gripper_center_from_fingers(robot) -> "np.ndarray | None":
    try:
        left_pos, _ = robot.gripper._left_finger.get_world_pose()
        right_pos, _ = robot.gripper._right_finger.get_world_pose()
        return (np.asarray(left_pos, dtype=float) + np.asarray(right_pos, dtype=float)) * 0.5
    except Exception:
        return None


def _safe_controller_event(controller) -> "int | None":
    if controller is None:
        return None
    for attr in ("_event", "_current_event", "event", "current_event"):
        try:
            value = getattr(controller, attr)
        except Exception:
            continue
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            pass
    return None


def _task_phase_from_event_or_state(
    event: "int | None",
    *,
    ee_pos: np.ndarray,
    cube_pos: np.ndarray,
    place_pos: np.ndarray,
    has_grasped_cube: bool,
) -> str:
    if event is not None:
        if event <= 2:
            return "approach_cube" if event < 2 else "grasp_cube"
        if event in (3,):
            return "grasp_cube"
        if event in (4, 5):
            return "move_to_target"
        if event >= 6:
            return "release_cube"
    ee_cube_dist = float(np.linalg.norm(cube_pos - ee_pos))
    cube_target_dist = float(np.linalg.norm(place_pos - cube_pos))
    if cube_target_dist < 0.08:
        return "release_cube"
    if has_grasped_cube:
        return "move_to_target"
    if ee_cube_dist < 0.10:
        return "grasp_cube"
    return "approach_cube"


def _has_grasped_cube(robot, cube, gripper_center: "np.ndarray | None") -> bool:
    if gripper_center is None:
        return False
    try:
        cube_pos, _ = cube.get_world_pose()
        gripper_width = float(np.sum(robot.gripper.get_joint_positions()))
    except Exception:
        return False
    return bool(np.linalg.norm(np.asarray(cube_pos, dtype=float) - gripper_center) < 0.075 and gripper_width < 0.045)


def _build_runtime_observation(
    *,
    panda,
    cube,
    place_pos: np.ndarray,
    head_pos: "np.ndarray | None",
    left_pos: "np.ndarray | None",
    right_pos: "np.ndarray | None",
    gripper_center: "np.ndarray | None",
    safety_result,
    controller,
) -> dict:
    ee_pos, _ = panda.end_effector.get_world_pose()
    cube_pos, _ = cube.get_world_pose()
    has_grasped = _has_grasped_cube(panda, cube, gripper_center)
    event = _safe_controller_event(controller)
    task_phase = _task_phase_from_event_or_state(
        event,
        ee_pos=np.asarray(ee_pos, dtype=float),
        cube_pos=np.asarray(cube_pos, dtype=float),
        place_pos=np.asarray(place_pos, dtype=float),
        has_grasped_cube=has_grasped,
    )
    return build_observation(
        robot=panda,
        cube=cube,
        place_target=np.asarray(place_pos, dtype=float),
        human_head_pos=head_pos,
        human_left_hand_pos=left_pos,
        human_right_hand_pos=right_pos,
        gripper_center_pos=gripper_center,
        human_robot_collision=safety_result.collision,
        near_human=safety_result.near,
        near_miss=safety_result.near_miss,
        has_grasped_cube=has_grasped,
        task_phase=task_phase,
        controller_event=event,
        controller_t=0.0,
        min_hand_gripper_dist_override=_min_optional(
            _distance_or_none(left_pos, gripper_center),
            _distance_or_none(right_pos, gripper_center),
        ),
        min_hand_gripper_surface_gap_override=safety_result.min_surface_gap_m,
        left_hand_surface_gap_override=safety_result.left.surface_gap_m,
        right_hand_surface_gap_override=safety_result.right.surface_gap_m,
        left_hand_contact=safety_result.left.contact,
        right_hand_contact=safety_result.right.contact,
        distance_gate_override=safety_result.distance_gate,
        geometry_valid_override=safety_result.geometry_valid,
    )


# =============================================================================
# 설정
# =============================================================================
ENABLE_PICK_PLACE = os.environ.get("ENABLE_PICK_PLACE", "1").lower() in (
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
    print(f"[Main] ENABLE_HRI_TRAJECTORY_RECORDING={ENABLE_HRI_TRAJECTORY_RECORDING}")
    print(f"[XR] XR_CAMERA_TELEPORT={XR_CAMERA_TELEPORT}")
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

    # One authoritative geometry source for collision, proximity, observation,
    # recording, and haptics. Initialization fails loudly if Panda colliders are
    # unavailable; falling back to the old hand-authored capsules would silently
    # change dataset semantics.
    safety_geometry = PandaEndEffectorSafetyRuntime(robot_prim_path="/World/Franka")

    place_target.set_world_pose(
        position=np.array([stack_base_xy[0], stack_base_xy[1], table_top_z + cube_half])
    )

    # 6. 기본 정보 출력
    print_robot_info(panda)

    # 7. Pick-and-Place 컨트롤러
    controller = None
    approach_height = table_top_z + 0.2
    if ENABLE_PICK_PLACE:
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

    # 9. VR 그랩 매니저 (avatar 전달 → XRCore 재사용)
    grab_manager = VRGrabManager(cubes, avatar=avatar) if _xr_enabled else None

    # 10. 시뮬레이션 루프
    xr_anchor_done = False   # xrAnchor 1회 이동 완료 플래그
    step = 0
    task_done = False
    current_pick_idx = 0
    completed_picks = 0
    cycle_reset_requested = False
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
    haptics = HapticsUdpClient(
        BHAPTICS_NOTEBOOK_IP,
        BHAPTICS_UDP_PORT,
        min_interval=float(os.environ.get("BHAPTICS_MIN_INTERVAL", "0.08")),
    )
    hand_tracking = HandTrackingReceiver(HAND_TRACKING_UDP_HOST, HAND_TRACKING_UDP_PORT)
    hri_recorder = None
    if ENABLE_HRI_TRAJECTORY_RECORDING:
        try:
            hri_recorder = HRIObsRecorder(
                HRI_TRAJECTORY_PATH,
                overwrite=HRI_TRAJECTORY_OVERWRITE,
                sample_interval_steps=SAMPLE_LOG_INTERVAL_STEPS,
                file_metadata=safety_geometry.metadata(),
            )
            if (
                HRI_TRAJECTORY_MAX_EPISODES > 0
                and hri_recorder.num_episodes >= HRI_TRAJECTORY_MAX_EPISODES
            ):
                hri_recorder.close()
                root, ext = os.path.splitext(HRI_TRAJECTORY_PATH)
                continued_path = f"{root}_continued_{time.time_ns()}{ext or '.hdf5'}"
                print(
                    "[HRI] requested file already reached its episode limit; "
                    f"switching to {continued_path}",
                    flush=True,
                )
                hri_recorder = HRIObsRecorder(
                    continued_path,
                    overwrite=False,
                    sample_interval_steps=SAMPLE_LOG_INTERVAL_STEPS,
                    file_metadata=safety_geometry.metadata(),
                )
            if (
                HRI_TRAJECTORY_MAX_EPISODES <= 0
                or hri_recorder.num_episodes < HRI_TRAJECTORY_MAX_EPISODES
            ):
                hri_recorder.start_episode(
                    {"reason": "run_start", "mode": "vr_pick_place", "proxy": "sphere"}
                )
            print(f"[HRI] recording obs dataset: {hri_recorder.path}")
        except Exception as exc:
            print(f"[HRI] recorder disabled: {exc}")
            hri_recorder = None

    def _sim_time(world_obj, step_idx: int) -> float:
        if hasattr(world_obj, "current_time"):
            return float(world_obj.current_time)
        if hasattr(world_obj, "get_physics_dt"):
            return float(step_idx) * float(world_obj.get_physics_dt())
        return float(step_idx) * (1.0 / 60.0)

    print("[Main] Entering simulation loop...", flush=True)
    waiting_for_vr_logged = False
    haptic_contact_steps = {"left": 0, "right": 0}
    last_robot_action = None
    while simulation_app.is_running():
        world.step(render=True)

        if world.is_playing():
            if world.current_time_step_index == 0:
                grab_manager.release_all()
                world.reset()
                if controller is not None:
                    controller.reset(end_effector_initial_height=approach_height)
                place_target.set_world_pose(
                    position=np.array([stack_base_xy[0], stack_base_xy[1], table_top_z + cube_half])
                )
                task_done = False
                step = 0
                last_robot_action = None
                cycle_reset_requested = False
                if (
                    hri_recorder is not None
                    and not hri_recorder.is_open
                    and (
                        HRI_TRAJECTORY_MAX_EPISODES <= 0
                        or hri_recorder.num_episodes < HRI_TRAJECTORY_MAX_EPISODES
                    )
                ):
                    hri_recorder.start_episode(
                        {
                            "reason": "world_time_reset",
                            "mode": "vr_pick_place",
                            "proxy": "sphere",
                        }
                    )

            step += 1
            sim_time = _sim_time(world, step)
            logger.update_context(step, sim_time)
            logger.ensure_episode_started()

            hand_tracking.poll()
            pinch_points = hand_tracking.get_pinch_points()
            hand_joint_positions = hand_tracking.get_hand_joint_positions()

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
                avatar.set_external_hand_joints(hand_joint_positions)
                head_pos, left_pos, right_pos = avatar.update()
            else:
                head_pos, left_pos, right_pos = None, None, None
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

            vr_tracking_ready = head_pos is not None and (left_pos is not None or right_pos is not None)
            if WAIT_FOR_VR_TRACKING and not vr_tracking_ready:
                if not waiting_for_vr_logged or step % 300 == 0:
                    print(
                        "[VR] Waiting for real XR tracking. "
                        "Start SteamVR/ALVR, connect the headset, and make sure SteamVR is the OpenXR runtime."
                    )
                    waiting_for_vr_logged = True
                continue
            waiting_for_vr_logged = False

            # Tracked hand spheres against the built-in distal Panda colliders.
            ee_pos, _ = panda.end_effector.get_world_pose()
            current_cube_pos_for_camera, _ = pick_targets[current_pick_idx].get_world_pose()
            gripper_camera.update(
                ee_pos,
                target_pos=current_cube_pos_for_camera,
                mount_pos=_gripper_center_from_fingers(panda),
            )
            safety_result = safety_geometry.evaluate(left_pos, right_pos)
            human_robot_collision_active = safety_result.collision
            haptic_pulse_by_hand = {"left": False, "right": False}
            for hand, hand_result in (
                ("left", safety_result.left),
                ("right", safety_result.right),
            ):
                closest_hit = (
                    [hand_result.closest_collider_path]
                    if hand_result.closest_collider_path
                    else []
                )
                proximity_hits = closest_hit if hand_result.near else []
                collision_hits = closest_hit if hand_result.collision else []
                logger.check_arm_robot_proximity(
                    hand, proximity_hits, hand_result.surface_gap_m
                )
                logger.check_arm_robot_collision(
                    hand, collision_hits, hand_result.surface_gap_m
                )
                if hand_result.contact:
                    haptic_contact_steps[hand] = haptic_contact_steps.get(hand, 0) + 1
                else:
                    haptic_contact_steps[hand] = 0
                haptic_active = (
                    haptic_contact_steps[hand] >= HAPTICS_CONTACT_MIN_STEPS
                )
                if DEBUG_HAPTICS_COLLISION and (hand_result.near or hand_result.contact):
                    print(
                        "[HapticsDBG] builtin-physx "
                        f"hand={hand} contact={hand_result.contact} "
                        f"steps={haptic_contact_steps[hand]}/{HAPTICS_CONTACT_MIN_STEPS} "
                        f"gap={hand_result.surface_gap_m:.4f} "
                        f"penetration={hand_result.penetration_m:.4f} "
                        f"near={hand_result.near} gate={hand_result.distance_gate:.3f} "
                        f"link={hand_result.closest_link} "
                        f"collider={hand_result.closest_collider_path}"
                    )
                if haptic_active:
                    haptic_pulse_by_hand[hand] = haptics.pulse(
                        100,
                        hand=hand,
                        event="panda_distal_surface_contact",
                    )

            gripper_center_for_distance = _gripper_center_from_fingers(panda)
            if gripper_center_for_distance is None:
                gripper_center_for_distance = ee_pos
            left_hand_gripper_dist = _distance_or_none(
                left_pos, gripper_center_for_distance
            )
            right_hand_gripper_dist = _distance_or_none(
                right_pos, gripper_center_for_distance
            )
            logger.log_sample(
                left_hand_gripper_dist=left_hand_gripper_dist,
                right_hand_gripper_dist=right_hand_gripper_dist,
                min_hand_gripper_dist=_min_optional(
                    left_hand_gripper_dist,
                    right_hand_gripper_dist,
                ),
                human_robot_collision=human_robot_collision_active,
            )
            if hri_recorder is not None and hri_recorder.is_open:
                stack_height = (
                    table_top_z + cube_half
                    + (completed_picks % len(pick_targets)) * (cube_size + 0.002)
                )
                current_place_pos = np.array(
                    [stack_base_xy[0], stack_base_xy[1], stack_height],
                    dtype=float,
                )
                obs = _build_runtime_observation(
                    panda=panda,
                    cube=pick_targets[current_pick_idx],
                    place_pos=current_place_pos,
                    head_pos=head_pos,
                    left_pos=left_pos,
                    right_pos=right_pos,
                    gripper_center=_gripper_center_from_fingers(panda),
                    safety_result=safety_result,
                    controller=controller,
                )
                hri_recorder.add_sample(
                    step=step,
                    sim_time=sim_time,
                    obs=obs,
                    current_pick_idx=current_pick_idx,
                    completed_picks=completed_picks,
                    safety={
                        "left_hand_gripper_dist_m": (
                            left_hand_gripper_dist
                            if left_hand_gripper_dist is not None
                            else np.nan
                        ),
                        "right_hand_gripper_dist_m": (
                            right_hand_gripper_dist
                            if right_hand_gripper_dist is not None
                            else np.nan
                        ),
                        "min_hand_gripper_dist_m": float(
                            np.asarray(obs["min_hand_gripper_dist"]).reshape(-1)[0]
                        ),
                        "min_hand_gripper_center_dist_m": float(
                            np.asarray(obs["min_hand_gripper_center_dist"]).reshape(-1)[0]
                        ),
                        "min_hand_gripper_surface_gap_m": float(
                            np.asarray(obs["min_hand_gripper_surface_gap"]).reshape(-1)[0]
                        ),
                        "near_human": float(
                            np.asarray(obs["near_human"]).reshape(-1)[0]
                        ),
                        "near_miss": float(np.asarray(obs["near_miss"]).reshape(-1)[0]),
                        "human_robot_collision": float(
                            np.asarray(obs["human_robot_collision"]).reshape(-1)[0]
                        ),
                        "haptic_pulse_left": 1.0 if haptic_pulse_by_hand["left"] else 0.0,
                        "haptic_pulse_right": 1.0 if haptic_pulse_by_hand["right"] else 0.0,
                        # Legacy names remain as exact aliases for readers that
                        # consumed the v3 safety group.
                        "gripper_gap_left_m": safety_result.left.surface_gap_m,
                        "gripper_gap_right_m": safety_result.right.surface_gap_m,
                        "gripper_penetration_left_m": safety_result.left.penetration_m,
                        "gripper_penetration_right_m": safety_result.right.penetration_m,
                        "left_hand_surface_gap_m": safety_result.left.surface_gap_m,
                        "right_hand_surface_gap_m": safety_result.right.surface_gap_m,
                        "min_hand_end_effector_surface_gap_m": safety_result.min_surface_gap_m,
                        "left_end_effector_surface_gap_m": safety_result.left.surface_gap_m,
                        "right_end_effector_surface_gap_m": safety_result.right.surface_gap_m,
                        "end_effector_surface_gap_m": safety_result.min_surface_gap_m,
                        "closest_human_hand_id": safety_result.closest_human_hand_id,
                        "closest_robot_link_id": safety_result.closest_robot_link_id,
                        "closest_collider_id": safety_result.closest_collider_id,
                        "closest_link_left_id": safety_result.left.closest_link_id,
                        "closest_link_right_id": safety_result.right.closest_link_id,
                        "closest_collider_left_id": safety_result.left.closest_collider_id,
                        "closest_collider_right_id": safety_result.right.closest_collider_id,
                        "contact_left": float(safety_result.left.contact),
                        "contact_right": float(safety_result.right.contact),
                        "contact_active": float(safety_result.contact),
                        "contact_force_left_n": safety_result.left.contact_force_n,
                        "contact_force_right_n": safety_result.right.contact_force_n,
                        "contact_force_n": max(
                            safety_result.left.contact_force_n,
                            safety_result.right.contact_force_n,
                        ),
                        "contact_force_valid_left": float(
                            safety_result.left.contact_force_valid
                        ),
                        "contact_force_valid_right": float(
                            safety_result.right.contact_force_valid
                        ),
                        "penetration_left_m": safety_result.left.penetration_m,
                        "penetration_right_m": safety_result.right.penetration_m,
                        "penetration_depth_m": safety_result.penetration_depth_m,
                        "near_left": float(safety_result.left.near),
                        "near_right": float(safety_result.right.near),
                        "near_miss_left": float(safety_result.left.near_miss),
                        "near_miss_right": float(safety_result.right.near_miss),
                        "distance_gate_left": safety_result.left.distance_gate,
                        "distance_gate_right": safety_result.right.distance_gate,
                        "distance_gate": safety_result.distance_gate,
                        "geometry_valid_left": float(safety_result.left.geometry_valid),
                        "geometry_valid_right": float(safety_result.right.geometry_valid),
                        "geometry_valid": float(safety_result.geometry_valid),
                        "query_time_left_ms": safety_result.left.query_time_ms,
                        "query_time_right_ms": safety_result.right.query_time_ms,
                        "query_count_left": safety_result.left.query_count,
                        "query_count_right": safety_result.right.query_count,
                    },
                    action=last_robot_action,
                )

            # VR 그랩 (hand tracking pinch 우선, 없으면 컨트롤러 grip 폴백)
            if grab_manager is not None:
                grab_manager.update(left_pos, right_pos, pinch_points=pinch_points)

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
            if ENABLE_PICK_PLACE and controller is not None and not task_done:
                current_pick_pos, _ = pick_targets[current_pick_idx].get_world_pose()
                stack_height = (
                    table_top_z + cube_half
                    + (completed_picks % len(pick_targets)) * (cube_size + 0.002)
                )
                place_pos = np.array([stack_base_xy[0], stack_base_xy[1], stack_height])
                task_done, last_robot_action = run_pick_place(
                    controller=controller,
                    robot=panda,
                    pick_position=current_pick_pos,
                    place_position=place_pos,
                    return_action=True,
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
                        if hri_recorder is not None and hri_recorder.is_open:
                            episode_path = hri_recorder.end_episode(
                                success=True,
                                metadata={
                                    "reason": "completed_pick_cycle",
                                    "picks": len(pick_targets),
                                },
                            )
                            if episode_path:
                                print(f"[HRI] saved episode {episode_path}")
                            if (
                                HRI_TRAJECTORY_MAX_EPISODES > 0
                                and hri_recorder.num_episodes
                                >= HRI_TRAJECTORY_MAX_EPISODES
                            ):
                                print(
                                    "[HRI] collection complete: "
                                    f"{hri_recorder.num_episodes}/"
                                    f"{HRI_TRAJECTORY_MAX_EPISODES} episodes. "
                                    "Closing Isaac Sim.",
                                    flush=True,
                                )
                                break
                            if (
                                HRI_TRAJECTORY_MAX_EPISODES <= 0
                                or hri_recorder.num_episodes < HRI_TRAJECTORY_MAX_EPISODES
                            ):
                                hri_recorder.start_episode(
                                    {
                                        "reason": "cubes_randomized",
                                        "mode": "vr_pick_place",
                                        "proxy": "sphere",
                                    }
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
            current_cube = pick_targets[current_pick_idx]
            logger.update_contact(ee_pos, pick_targets)
            logger.check_pick_miss(gripper_closed, ee_pos, current_cube)
            logger.check_drop_throw(pick_targets)
            logger.check_collision_green(pick_targets, green_cubes)
            logger.check_stack_failure(pick_targets)

    hand_tracking.close()
    haptics.close()
    gripper_camera.close()
    if hri_recorder is not None:
        saved_path = None
        if hri_recorder.is_open:
            saved_path = hri_recorder.end_episode(
                success=False,
                metadata={"reason": "simulation_stopped"},
            )
        hri_recorder.close()
        if saved_path:
            print(f"[HRI] saved final episode {saved_path}")
    print(
        "[SafetyGeometry] session mean PhysX query time="
        f"{safety_geometry.mean_query_time_ms:.4f} ms",
        flush=True,
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
