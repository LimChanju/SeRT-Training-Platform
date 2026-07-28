#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/railab/Desktop/Isaac_HRC"
cd "$PROJECT_ROOT"

HRI_SESSION_ID="$(date +%Y%m%d_%H%M%S_%N)_$$"
export HRI_SESSION_ID
export HRI_PARTICIPANT_ID="${HRI_PARTICIPANT_ID:-P01}"
export HRI_PROTOCOL_VERSION="${HRI_PROTOCOL_VERSION:-surface_gap_dynamic_multispeed_dualclock_v4}"
export HRI_ROOM_CALIBRATION_ID="${HRI_ROOM_CALIBRATION_ID:-room_to_world_default_v1}"
HRI_COLLECTION_TEST_MODE="${HRI_COLLECTION_TEST_MODE:-0}"
if [[ "$HRI_COLLECTION_TEST_MODE" == "1" ]]; then
    export HRI_PRODUCTION_MODE=0
else
    export HRI_PRODUCTION_MODE=1
fi
if [[ "$HRI_PRODUCTION_MODE" == "1" ]] && [[ -z "$HRI_PARTICIPANT_ID" || "$HRI_PARTICIPANT_ID" == "unspecified" || "$HRI_PARTICIPANT_ID" == "unknown" ]]; then
    printf '[Collect] production collection requires a valid HRI_PARTICIPANT_ID\n' >&2
    exit 2
fi

mkdir -p "$PROJECT_ROOT/v3_chan/logs" "$PROJECT_ROOT/v3_chan/trajectories"

# Rotate the three episode orders across production sessions. Set
# HRI_SPEED_ORDER_INDEX=0,1,2 to reproduce a specific order without consuming
# the automatic counter. Test mode never advances the counter.
_speed_order_auto=0
if [[ -n "${HRI_SPEED_ORDER_INDEX:-}" ]]; then
    if [[ ! "$HRI_SPEED_ORDER_INDEX" =~ ^[0-9]+$ ]]; then
        printf '[Collect] invalid HRI_SPEED_ORDER_INDEX=%s (expected non-negative integer)\n' \
            "$HRI_SPEED_ORDER_INDEX" >&2
        exit 2
    fi
    _speed_order_index=$((10#$HRI_SPEED_ORDER_INDEX % 3))
elif [[ "$HRI_COLLECTION_TEST_MODE" == "1" ]]; then
    _speed_order_index=0
else
    _speed_order_auto=1
    _speed_order_state_path="${HRI_SPEED_ORDER_STATE_PATH:-$PROJECT_ROOT/v3_chan/logs/.speed_order_counter}"
    exec 9>"${_speed_order_state_path}.lock"
    flock 9
    _speed_order_counter=0
    if [[ -f "$_speed_order_state_path" ]]; then
        read -r _speed_order_counter < "$_speed_order_state_path" || _speed_order_counter=0
    fi
    if [[ ! "$_speed_order_counter" =~ ^[0-9]+$ ]]; then
        _speed_order_counter=0
    fi
    _speed_order_index=$((10#$_speed_order_counter % 3))
fi

case "$_speed_order_index" in
    0) HRI_SPEED_PROFILE_ORDER="slow,medium,fast" ;;
    1) HRI_SPEED_PROFILE_ORDER="medium,fast,slow" ;;
    2) HRI_SPEED_PROFILE_ORDER="fast,slow,medium" ;;
esac
export HRI_SPEED_ORDER_INDEX="$_speed_order_index"
export HRI_SPEED_PROFILE_ORDER

export BHAPTICS_NOTEBOOK_IP=10.3.129.185
export BHAPTICS_UDP_PORT=5005
export BHAPTICS_MIN_INTERVAL=0.08
export ISAAC_XR_MODE=vr
export ISAAC_XR_BACKEND=OpenXR
export WAIT_FOR_VR_TRACKING=1
export DEBUG_HAPTICS_UDP=1
export DEBUG_HAPTICS_COLLISION=0

export HRI_TRAJECTORY_PATH="$PROJECT_ROOT/v3_chan/trajectories/hri_vr_sphere_obs_${HRI_SESSION_ID}.hdf5"
export HRI_TRAJECTORY_OVERWRITE=0
if [[ "$HRI_COLLECTION_TEST_MODE" == "1" ]]; then
    export ENABLE_HRI_TRAJECTORY_RECORDING="${HRI_TEST_RECORDING:-0}"
    export HRI_TRAJECTORY_MAX_EPISODES=0
else
    export ENABLE_HRI_TRAJECTORY_RECORDING=1
    export HRI_TRAJECTORY_MAX_EPISODES=3
fi
# Keep collection labels identical across terminals and sessions. Debug-only
# settings below remain caller-overridable, but dataset semantics do not.
export HRI_NEAR_HUMAN_SURFACE_GAP_M=0.05
export HRI_NEAR_MISS_SURFACE_GAP_M=0.02
export HRI_COLLISION_SURFACE_GAP_M=0.0
export HRI_HAND_PROXY_RADIUS_M=0.035
export HRI_DISTANCE_GATE_FULL_GAP_M=0.05
export HRI_DISTANCE_GATE_START_GAP_M=0.13
export HRI_GEOMETRY_MAX_QUERY_GAP_M=2.0
export HRI_GEOMETRY_QUERY_TOLERANCE_M=0.00025
export HRI_GEOMETRY_QUERY_ITERATIONS=14
if [[ "$HRI_COLLECTION_TEST_MODE" == "1" ]]; then
    HRI_COLLECTION_DEBUG_VISUALIZATION=1
else
    HRI_COLLECTION_DEBUG_VISUALIZATION=0
fi
export HRI_SHOW_PHYSX_COLLIDERS="$HRI_COLLECTION_DEBUG_VISUALIZATION"
export HRI_DEBUG_SAFETY_VISUALIZATION="$HRI_COLLECTION_DEBUG_VISUALIZATION"
export DEBUG_HRI_SAFETY_GEOMETRY="$HRI_COLLECTION_DEBUG_VISUALIZATION"
export HRI_SAFETY_DEBUG_PRINT_EVERY="${HRI_SAFETY_DEBUG_PRINT_EVERY:-30}"

export PICK_PLACE_SUCCESS_XY_TOLERANCE_M=0.04
export PICK_PLACE_SUCCESS_Z_TOLERANCE_M=0.03
export PICK_PLACE_SUCCESS_MAX_SPEED_MPS=0.05
export PICK_PLACE_SUCCESS_MIN_LIFT_M=0.05
export PICK_PLACE_MISS_RECENT_STEPS=30
export PICK_PLACE_CUBE_X_MIN=0.30
export PICK_PLACE_CUBE_X_MAX=0.65

export ENABLE_HRI_VIDEO_RECORDING=0
export ENABLE_GRIPPER_CAMERA=0
export ENABLE_GRIPPER_CAMERA_VIEWPORT=0
export ENABLE_GRIPPER_CAMERA_RECORDING=0

export HAND_TRACKING_UDP_HOST=0.0.0.0
export HAND_TRACKING_UDP_PORT=5555
export XR_EXTERNAL_HAND_TRACKING=0
export XR_HAND_PROXY_ENABLED=0
export XR_HAND_SPHERE_ENABLED=1
export XR_HAND_HAPTIC_POINT_MODE=sphere
export XR_SHOW_CONTROLLERS=1
export XR_CONTROLLER_POSE_MODE=visual
export XR_CONTROLLER_WORKSPACE_GUARD=1
export XR_CONTROLLER_MAX_HEAD_DIST_M=1.85
export XR_CONTROLLER_MIN_Z_M=0.35
export XR_CONTROLLER_MAX_Z_M=2.25
export XR_VIRTUAL_WORLD_POSE_FALLBACK=1
export XR_STAGE_VISUAL_FALLBACK=1
export XR_STAGE_VISUAL_SEARCH_INTERVAL_STEPS=30

export HAPTICS_CONTACT_MIN_STEPS=1

LOG_PATH="$PROJECT_ROOT/v3_chan/logs/pick_place_${HRI_SESSION_ID}.log"
export HRI_LOG_DIR="$PROJECT_ROOT/v3_chan/logs"
export ERRP_MARKERS_PATH="$HRI_LOG_DIR/errp_markers_${HRI_SESSION_ID}.csv"
export SESSION_SAMPLES_PATH="$HRI_LOG_DIR/session_samples_${HRI_SESSION_ID}.csv"

printf '[Collect] session=%s\n' "$HRI_SESSION_ID"
printf '[Collect] hdf5=%s\n' "$HRI_TRAJECTORY_PATH"
printf '[Collect] log=%s\n' "$LOG_PATH"
printf '[Collect] markers=%s\n' "$ERRP_MARKERS_PATH"
printf '[Collect] samples=%s\n' "$SESSION_SAMPLES_PATH"
printf '[Collect] participant=%s protocol=%s calibration=%s\n' \
    "$HRI_PARTICIPANT_ID" "$HRI_PROTOCOL_VERSION" "$HRI_ROOM_CALIBRATION_ID"
printf '[Collect] production=%s session_seed=%s code_version=%s\n' \
    "$HRI_PRODUCTION_MODE" \
    "${HRI_SESSION_SEED:-auto}" \
    "${HRI_CODE_VERSION:-auto-source-tree-sha256}"
printf '[Collect] speed_order_index=%s episode_speed_schedule=%s motion_scale=slow:1.0,medium:1.5,fast:2.0\n' \
    "$HRI_SPEED_ORDER_INDEX" "$HRI_SPEED_PROFILE_ORDER"
printf '[Collect] test_mode=%s episodes=%s recording=%s debug_visualization=%s\n' \
    "$HRI_COLLECTION_TEST_MODE" \
    "$HRI_TRAJECTORY_MAX_EPISODES" \
    "$ENABLE_HRI_TRAJECTORY_RECORDING" \
    "$HRI_COLLECTION_DEBUG_VISUALIZATION"
printf '[Collect] safety_gap_m collision=%s near_miss=%s near=%s gate_full=%s gate_start=%s\n' \
    "$HRI_COLLISION_SURFACE_GAP_M" \
    "$HRI_NEAR_MISS_SURFACE_GAP_M" \
    "$HRI_NEAR_HUMAN_SURFACE_GAP_M" \
    "$HRI_DISTANCE_GATE_FULL_GAP_M" \
    "$HRI_DISTANCE_GATE_START_GAP_M"

# Keep the log pipe alive after Ctrl+C so main.py can flush an interrupted
# episode before Isaac Sim exits.
./launch_isaac.sh "$PROJECT_ROOT/v3_chan/main.py" 2>&1 | tee -i "$LOG_PATH"

if [[ "$_speed_order_auto" == "1" ]]; then
    if grep -Fq '[HRI] collection complete:' "$LOG_PATH"; then
        printf '%s\n' "$(((_speed_order_index + 1) % 3))" > "$_speed_order_state_path"
        printf '[Collect] completed; next_speed_order_index=%s\n' \
            "$(((_speed_order_index + 1) % 3))"
    else
        printf '[Collect] incomplete session; speed order counter was not advanced\n'
    fi
    flock -u 9
    exec 9>&-
fi
