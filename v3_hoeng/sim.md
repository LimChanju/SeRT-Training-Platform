
cd /home/railab/Desktop/Isaac_HRC/v3_hoeng

./run_handover.sh \
  --episodes 10 \
  --dataset-dir data/trajectories/handover_dataset_v0 \
  --min-steps 20 \
  --place-success-dist 0.08 \
  --auto-complete-stable-steps 8 \
  --release-hold-steps 30
