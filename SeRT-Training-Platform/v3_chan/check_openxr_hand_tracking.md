```bash
cd /home/railab/Desktop/Isaac_HRC

export ISAACSIM_ROOT=/home/railab/isaac-sim-4.5.0
export EXP_PATH="$ISAACSIM_ROOT/apps"
export ISAAC_XR_MODE=openxr
export ISAAC_XR_BACKEND=OpenXR
export OPENXR_HAND_CHECK_FRAMES=900
export OPENXR_HAND_CHECK_PRINT_EVERY=60

mkdir -p "$PWD/v3_chan/logs"
./launch_isaac.sh "$PWD/v3_chan/check_openxr_hand_tracking.py" 2>&1 | tee "$PWD/v3_chan/logs/latest_openxr_hand_check.log"
```

```bash
cd /home/railab/Desktop/Isaac_HRC

export ISAACSIM_ROOT=/home/railab/isaac-sim-4.5.0
export EXP_PATH="$ISAACSIM_ROOT/apps"
export ISAAC_XR_MODE=openxr
export ISAAC_XR_BACKEND=OpenXR

./launch_isaac.sh "$ISAACSIM_ROOT/standalone_examples/api/isaacsim.xr.openxr/hand_tracking/hand_tracking_sample.py"
```
