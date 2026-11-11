#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

participant_id="${HRI_PARTICIPANT_ID:-}"
session_id="${HRI_SESSION_ID:-}"

# Known-good SteamVR/OpenXR defaults from the existing VR collection path.
# Every tracking/display choice remains caller-overridable.  Explicit-feedback
# v1 does not use either controller output or the external bHaptics transport.
export ISAAC_XR_MODE="${ISAAC_XR_MODE:-vr}"
export ISAAC_XR_BACKEND="${ISAAC_XR_BACKEND:-OpenXR}"
export WAIT_FOR_VR_TRACKING="${WAIT_FOR_VR_TRACKING:-1}"
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

# This protocol is visual/controller-input only.  Force all feedback-output
# transports off even when a parent shell exported settings for another study.
export BHAPTICS_ENABLED=0
export HRI_HAPTIC_CONDITION=off
export BHAPTICS_NOTEBOOK_IP=""
export DEBUG_HAPTICS_UDP=0
export DEBUG_HAPTICS_COLLISION=0

if [[ $# -gt 0 ]]; then
    exec "$PROJECT_ROOT/launch_isaac.sh" \
        "$SCRIPT_DIR/collect_bc_cbf_explicit_feedback.py" "$@"
fi

if [[ -z "$participant_id" || -z "$session_id" ]]; then
    echo "Set HRI_PARTICIPANT_ID and HRI_SESSION_ID, or pass collector arguments directly." >&2
    echo "Example: HRI_PARTICIPANT_ID=P001 HRI_SESSION_ID=P001_S01 $0" >&2
    exit 2
fi

exec "$PROJECT_ROOT/launch_isaac.sh" \
    "$SCRIPT_DIR/collect_bc_cbf_explicit_feedback.py" \
    --participant-id "$participant_id" \
    --session-id "$session_id"
