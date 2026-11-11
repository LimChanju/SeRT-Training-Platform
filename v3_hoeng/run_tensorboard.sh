#!/usr/bin/env bash
set -euo pipefail

cd /home/railab/Desktop/Isaac_HRC/v3_hoeng

DATASET_DIR="${1:-data/trajectories/handover_dataset_v0}"
LOG_DIR="${2:-runs/handover_dataset_v0}"
PORT="${3:-6006}"

if ! /home/railab/isaac-sim-4.5.0/python.sh - <<'PY'
from torch.utils.tensorboard import SummaryWriter
import tensorboard
PY
then
  echo
  echo "[TensorBoard] Missing tensorboard package. Install once:"
  echo "  /home/railab/isaac-sim-4.5.0/python.sh -m pip install -r requirements.txt"
  exit 1
fi

rm -rf "${LOG_DIR}"

/home/railab/isaac-sim-4.5.0/python.sh \
  learning/export_handover_tensorboard.py \
  "${DATASET_DIR}" \
  --log-dir "${LOG_DIR}"

exec /home/railab/isaac-sim-4.5.0/python.sh -m tensorboard.main \
  --logdir "${LOG_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}"
