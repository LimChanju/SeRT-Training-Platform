#!/bin/bash
# Isaac Sim launcher — conda isaac_vr 환경에서 충돌 없이 실행

echo "[Launch] Step 1: sourcing conda..."
source "$HOME/anaconda3/etc/profile.d/conda.sh"

echo "[Launch] Step 2: activating isaac_vr..."
conda activate isaac_vr 2>/dev/null
echo "[Launch] conda env: $CONDA_DEFAULT_ENV, python: $(which python)"

echo "[Launch] Step 3: clearing LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64"

echo "[Launch] Step 4: sourcing setup_conda_env.sh..."
cd "$HOME/isaac-sim-4.5.0"
source ./setup_conda_env.sh
echo "[Launch] Step 4 done."

_ros2_lib="$HOME/isaac-sim-4.5.0/exts/omni.isaac.ros2_bridge/humble/lib"
if [ -d "$_ros2_lib" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$_ros2_lib"
fi
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "[Launch] Step 5: testing python import..."
python -c "import sys; print('[Launch] Python OK:', sys.version)" || { echo "[Launch] Python failed!"; exit 1; }

echo "[Launch] Step 6: running main.py..."
export PYTHONFAULTHANDLER=1
export LD_PRELOAD="$HOME/isaac-sim-4.5.0/kit/libcarb.so"
export RESOURCE_NAME="IsaacSim"
export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
if [ -z "$VK_ICD_FILENAMES" ] && [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

# XR_RUNTIME_JSON 미리 탐색 (Python glob 재귀 탐색으로 인한 hang 방지)
if [ -z "$XR_RUNTIME_JSON" ]; then
    _xr_json=$(find "$HOME/.steam" "$HOME/.local/share/Steam" /usr/share/steam \
        -maxdepth 8 -name "steamxr_linux64.json" 2>/dev/null | head -1)
    if [ -n "$_xr_json" ]; then
        export XR_RUNTIME_JSON="$_xr_json"
        echo "[Launch] XR_RUNTIME_JSON=$_xr_json"
    else
        echo "[Launch] Warning: steamxr_linux64.json not found"
    fi
fi

echo "[Launch] XR_RUNTIME_JSON=${XR_RUNTIME_JSON:-<unset>}"
echo "[Launch] VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>}"
echo "[Launch] ISAAC_XR_MODE=${ISAAC_XR_MODE:-vr}"
echo "[Launch] ISAAC_XR_BACKEND=${ISAAC_XR_BACKEND:-SteamVR}"

# SteamVR VR 세션이 완전히 열릴 때까지 대기
echo "[Launch] Waiting 8s for SteamVR VR session to be ready..."
sleep 8

exec python -u "$@"
