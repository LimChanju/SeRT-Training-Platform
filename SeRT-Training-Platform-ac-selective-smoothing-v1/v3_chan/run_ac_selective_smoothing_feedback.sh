#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_isaac_python="${ISAACSIM_PYTHON:-/home/railab/isaac-sim-4.5.0/python.sh}"

if [[ ! -x "$_isaac_python" ]]; then
    printf '[ACFeedback] Isaac Python is not executable: %s\n' "$_isaac_python" >&2
    printf '[ACFeedback] Set ISAACSIM_PYTHON to the absolute path of python.sh.\n' >&2
    exit 2
fi

# Haptics are outside this study and are unconditionally disabled for practice,
# pilot, and production collection.  Caller-provided values cannot override it.
export BHAPTICS_ENABLED=0
export HRI_HAPTIC_CONDITION=off
export HRI_HAPTICS_ENABLED=0
export HRI_HAPTIC_FEEDBACK=0
export PYTHONUNBUFFERED=1

# Use the controller transform that drives Kit's visible Quest model, matching
# the proven slow/medium/fast VR collector.  VRAvatar additionally requires a
# valid physical controller pose in the same frame before accepting this
# visual transform for safety geometry or logging.
export ISAAC_XR_MODE="${ISAAC_XR_MODE:-vr}"
export ISAAC_XR_BACKEND="${ISAAC_XR_BACKEND:-OpenXR}"
export WAIT_FOR_VR_TRACKING="${WAIT_FOR_VR_TRACKING:-1}"
export XR_EXTERNAL_HAND_TRACKING=0
export XR_HAND_PROXY_ENABLED=0
export XR_HAND_SPHERE_ENABLED=1
export XR_HAND_HAPTIC_POINT_MODE=sphere
export XR_SHOW_CONTROLLERS="${XR_SHOW_CONTROLLERS:-1}"
export XR_CONTROLLER_POSE_MODE="${XR_CONTROLLER_POSE_MODE:-visual}"
export XR_CONTROLLER_WORKSPACE_GUARD="${XR_CONTROLLER_WORKSPACE_GUARD:-1}"
export XR_CONTROLLER_MAX_HEAD_DIST_M="${XR_CONTROLLER_MAX_HEAD_DIST_M:-1.85}"
export XR_CONTROLLER_MIN_Z_M="${XR_CONTROLLER_MIN_Z_M:-0.35}"
export XR_CONTROLLER_MAX_Z_M="${XR_CONTROLLER_MAX_Z_M:-2.25}"
export XR_VIRTUAL_WORLD_POSE_FALLBACK="${XR_VIRTUAL_WORLD_POSE_FALLBACK:-1}"
export XR_STAGE_VISUAL_FALLBACK="${XR_STAGE_VISUAL_FALLBACK:-1}"
export XR_STAGE_VISUAL_SEARCH_INTERVAL_STEPS="${XR_STAGE_VISUAL_SEARCH_INTERVAL_STEPS:-30}"
export XR_SWAP_HANDS="${XR_SWAP_HANDS:-0}"

exec "$_isaac_python" \
    "$_script_dir/collect_ac_selective_smoothing_feedback.py" \
    "$@"
