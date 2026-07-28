import os
import time

try:
    from v3_chan.scene_randomization import resolve_session_seed
except ImportError:
    from scene_randomization import resolve_session_seed
try:
    from v3_chan.collection_provenance import resolve_code_version
except ImportError:
    from collection_provenance import resolve_code_version

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
HRI_PRODUCTION_MODE = os.environ.get("HRI_PRODUCTION_MODE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HRI_SESSION_ID = os.environ.get("HRI_SESSION_ID", "").strip() or (
    f"local_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
)
HRI_SESSION_SEED = resolve_session_seed(os.environ.get("HRI_SESSION_SEED"))
(
    HRI_CODE_VERSION,
    HRI_CODE_VERSION_SOURCE,
    HRI_SOURCE_TREE_SHA256,
) = resolve_code_version(PROJECT_DIR, os.environ.get("HRI_CODE_VERSION"))
HRI_PARTICIPANT_ID = os.environ.get("HRI_PARTICIPANT_ID", "unspecified").strip()
HRI_PROTOCOL_VERSION = os.environ.get(
    "HRI_PROTOCOL_VERSION", "surface_gap_dynamic_multispeed_dualclock_v4"
).strip()
HRI_SPEED_ORDER_INDEX = int(os.environ.get("HRI_SPEED_ORDER_INDEX", "0"))
HRI_SPEED_PROFILE_ORDER = os.environ.get(
    "HRI_SPEED_PROFILE_ORDER", "slow,medium,fast"
).strip()
HRI_ROOM_CALIBRATION_ID = os.environ.get(
    "HRI_ROOM_CALIBRATION_ID", "room_to_world_default_v1"
).strip()
_log_dir = os.environ.get("HRI_LOG_DIR", os.path.join(BASE_DIR, "logs"))
HRI_LOG_DIR = (
    _log_dir
    if os.path.isabs(_log_dir)
    else os.path.abspath(os.path.join(PROJECT_DIR, _log_dir))
)
ERRP_MARKERS_PATH = os.environ.get(
    "ERRP_MARKERS_PATH",
    os.path.join(HRI_LOG_DIR, f"errp_markers_{HRI_SESSION_ID}.csv"),
)
SESSION_SAMPLES_PATH = os.environ.get(
    "SESSION_SAMPLES_PATH",
    os.path.join(HRI_LOG_DIR, f"session_samples_{HRI_SESSION_ID}.csv"),
)
SAMPLE_LOG_INTERVAL_STEPS = int(os.environ.get("SAMPLE_LOG_INTERVAL_STEPS", "1"))
ENABLE_HRI_TRAJECTORY_RECORDING = os.environ.get(
    "ENABLE_HRI_TRAJECTORY_RECORDING", "0"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_hri_trajectory_path = os.environ.get(
    "HRI_TRAJECTORY_PATH",
    os.path.join(BASE_DIR, "trajectories", "hri_vr_expert_v0.hdf5"),
)
HRI_TRAJECTORY_PATH = (
    _hri_trajectory_path
    if os.path.isabs(_hri_trajectory_path)
    else os.path.abspath(os.path.join(PROJECT_DIR, _hri_trajectory_path))
)
HRI_TRAJECTORY_OVERWRITE = os.environ.get("HRI_TRAJECTORY_OVERWRITE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HRI_TRAJECTORY_MAX_EPISODES = int(os.environ.get("HRI_TRAJECTORY_MAX_EPISODES", "0"))
PICK_PLACE_SUCCESS_XY_TOLERANCE_M = float(
    os.environ.get("PICK_PLACE_SUCCESS_XY_TOLERANCE_M", "0.04")
)
PICK_PLACE_SUCCESS_Z_TOLERANCE_M = float(
    os.environ.get("PICK_PLACE_SUCCESS_Z_TOLERANCE_M", "0.03")
)
PICK_PLACE_SUCCESS_MAX_SPEED_MPS = float(
    os.environ.get("PICK_PLACE_SUCCESS_MAX_SPEED_MPS", "0.05")
)
PICK_PLACE_SUCCESS_MIN_LIFT_M = float(
    os.environ.get("PICK_PLACE_SUCCESS_MIN_LIFT_M", "0.05")
)
PICK_PLACE_MISS_RECENT_STEPS = int(
    os.environ.get("PICK_PLACE_MISS_RECENT_STEPS", "30")
)
ENABLE_GRIPPER_CAMERA = os.environ.get("ENABLE_GRIPPER_CAMERA", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
GRIPPER_CAMERA_PRIM_PATH = os.environ.get("GRIPPER_CAMERA_PRIM_PATH", "/World/GripperCamera")
ENABLE_GRIPPER_CAMERA_VIEWPORT = os.environ.get(
    "ENABLE_GRIPPER_CAMERA_VIEWPORT", "1"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ENABLE_GRIPPER_CAMERA_RECORDING = os.environ.get(
    "ENABLE_GRIPPER_CAMERA_RECORDING", "0"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_record_dir = os.environ.get(
    "GRIPPER_CAMERA_RECORD_DIR",
    os.path.join(BASE_DIR, "gripper_camera_recording"),
)
GRIPPER_CAMERA_RECORD_DIR = (
    _record_dir
    if os.path.isabs(_record_dir)
    else os.path.abspath(os.path.join(PROJECT_DIR, _record_dir))
)
GRIPPER_CAMERA_RECORD_RESOLUTION = os.environ.get(
    "GRIPPER_CAMERA_RECORD_RESOLUTION", "640,480"
)
GRIPPER_CAMERA_RECORD_INTERVAL_STEPS = int(
    os.environ.get("GRIPPER_CAMERA_RECORD_INTERVAL_STEPS", "5")
)

ENABLE_HRI_VIDEO_RECORDING = os.environ.get(
    "ENABLE_HRI_VIDEO_RECORDING", "0"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HRI_VIDEO_PRIM_PATH = os.environ.get("HRI_VIDEO_PRIM_PATH", "/World/HRIOverviewCamera")
_hri_video_record_dir = os.environ.get(
    "HRI_VIDEO_RECORD_DIR",
    os.path.join(BASE_DIR, "videos", "latest"),
)
HRI_VIDEO_RECORD_DIR = (
    _hri_video_record_dir
    if os.path.isabs(_hri_video_record_dir)
    else os.path.abspath(os.path.join(PROJECT_DIR, _hri_video_record_dir))
)
HRI_VIDEO_RECORD_RESOLUTION = os.environ.get("HRI_VIDEO_RECORD_RESOLUTION", "1280,720")
HRI_VIDEO_RECORD_INTERVAL_STEPS = int(
    os.environ.get("HRI_VIDEO_RECORD_INTERVAL_STEPS", "3")
)
HRI_VIDEO_FPS = int(os.environ.get("HRI_VIDEO_FPS", "20"))
HRI_VIDEO_EYE = os.environ.get("HRI_VIDEO_EYE", "1.35,-1.15,1.75")
HRI_VIDEO_TARGET = os.environ.get("HRI_VIDEO_TARGET", "0.45,0.0,1.05")
HRI_VIDEO_MP4_PATH = os.environ.get("HRI_VIDEO_MP4_PATH", "").strip()

BHAPTICS_NOTEBOOK_IP = os.environ.get("BHAPTICS_NOTEBOOK_IP", "").strip()
BHAPTICS_UDP_PORT = int(os.environ.get("BHAPTICS_UDP_PORT", "5005"))

HAND_TRACKING_UDP_HOST = os.environ.get("HAND_TRACKING_UDP_HOST", "0.0.0.0")
HAND_TRACKING_UDP_PORT = int(os.environ.get("HAND_TRACKING_UDP_PORT", "5555"))
