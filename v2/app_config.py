import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ERRP_MARKERS_PATH = os.path.join(BASE_DIR, "errp_markers.csv")
SESSION_SAMPLES_PATH = os.path.join(BASE_DIR, "session_samples.csv")
SAMPLE_LOG_INTERVAL_STEPS = int(os.environ.get("SAMPLE_LOG_INTERVAL_STEPS", "1"))
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

BHAPTICS_NOTEBOOK_IP = os.environ.get("BHAPTICS_NOTEBOOK_IP", "").strip()
BHAPTICS_UDP_PORT = int(os.environ.get("BHAPTICS_UDP_PORT", "5005"))

HAND_TRACKING_UDP_HOST = os.environ.get("HAND_TRACKING_UDP_HOST", "0.0.0.0")
HAND_TRACKING_UDP_PORT = int(os.environ.get("HAND_TRACKING_UDP_PORT", "5555"))
