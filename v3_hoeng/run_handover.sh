#!/usr/bin/env bash
set -euo pipefail

cd /home/railab/Desktop/Isaac_HRC/v3_hoeng
exec /home/railab/isaac-sim-4.5.0/python.sh run_handover_dataset.py "$@"
