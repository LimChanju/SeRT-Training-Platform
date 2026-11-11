#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTROLLER="${1:-none}"
MANIFEST="${2:-${SCRIPT_DIR}/eval_results/physical_safety/physical_safety_source_aligned_fold01_v1/visual/top3_task_only_collision_manifest.json}"
DELAY_SEC="${VISUAL_DELAY_SEC:-0.01}"
SEED="${VISUAL_SEED:-11}"

case "${CONTROLLER}" in
  none|rmpflow|cbf|rmpflow_cbf)
    ;;
  *)
    printf 'Unknown controller: %s\n' "${CONTROLLER}" >&2
    printf 'Choose one of: none, rmpflow, cbf, rmpflow_cbf\n' >&2
    exit 2
    ;;
esac

if [[ ! -f "${MANIFEST}" ]]; then
  printf 'Visual encounter manifest not found: %s\n' "${MANIFEST}" >&2
  exit 2
fi

OUTPUT_DIR="${SCRIPT_DIR}/eval_results/physical_safety/physical_safety_source_aligned_fold01_v1/visual/${CONTROLLER}"

printf '[PhysicalSafetyVisual] controller=%s manifest=%s delay=%ss\n' \
  "${CONTROLLER}" "${MANIFEST}" "${DELAY_SEC}"

ISAAC_SKIP_VR_WAIT=1 ISAAC_SKIP_XR_RUNTIME_SEARCH=1 \
  "${PROJECT_DIR}/launch_isaac.sh" \
  "${SCRIPT_DIR}/evaluate_rollout_policy.py" \
  --checkpoint "${SCRIPT_DIR}/policies/ppo_pick_place_v7_residual_rewardv4_strict_best.pt" \
  --encounter-manifest "${MANIFEST}" \
  --encounter-policy cycle \
  --encounter-timebase recorded \
  --episodes 0 \
  --max-steps 1200 \
  --seed "${SEED}" \
  --device cuda \
  --mask-human-obs-for-policy \
  --no-pseudo-errp \
  --physical-safety-controller "${CONTROLLER}" \
  --rmpflow-human-safety-margin-m 0.05 \
  --cbf-safe-gap-m 0.05 \
  --cbf-activation-gap-m 0.13 \
  --cbf-gamma-per-s 8.0 \
  --cbf-prediction-horizon-s 0.15 \
  --cbf-max-prediction-buffer-m 0.08 \
  --cbf-max-joint-speed-rad-s 2.0 \
  --render \
  --render-step-delay-sec "${DELAY_SEC}" \
  --visualize-human-replay \
  --visualize-physical-safety \
  --output-json "${OUTPUT_DIR}/seed_${SEED}.json" \
  --output-csv "${OUTPUT_DIR}/seed_${SEED}.csv" \
  --output-step-csv "${OUTPUT_DIR}/seed_${SEED}_steps.csv" \
  --log-every 1
