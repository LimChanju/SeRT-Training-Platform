#!/bin/bash
# Isaac Sim launcher — Isaac bundled Python by default, conda optional

set -e

echo "[Launch] Step 1: selecting Python runtime..."
if [ -z "$CONDA_ROOT" ]; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_ROOT="$HOME/miniconda3"
    else
        CONDA_ROOT="$HOME/anaconda3"
    fi
fi
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaac-sim-4.5.0}"
export ISAACSIM_ROOT
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaac_vr}"
USE_CONDA_PYTHON="${USE_CONDA_PYTHON:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$USE_CONDA_PYTHON" = "1" ]; then
    echo "[Launch] Using conda Python from env '${CONDA_ENV_NAME}'"
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        source "$CONDA_ROOT/etc/profile.d/conda.sh"
    else
        echo "[Launch] Warning: conda.sh not found at $CONDA_ROOT/etc/profile.d/conda.sh"
    fi
    if command -v conda >/dev/null 2>&1; then
        conda activate "$CONDA_ENV_NAME" 2>/dev/null || echo "[Launch] Warning: conda env '${CONDA_ENV_NAME}' not activated"
    else
        echo "[Launch] Warning: conda command not available"
    fi
    PYTHON_RUNNER=(python)
else
    echo "[Launch] Using Isaac Sim bundled Python"
    unset CONDA_PREFIX
    unset CONDA_DEFAULT_ENV
    unset CONDA_PROMPT_MODIFIER
    unset PYTHONHOME
    unset PYTHONPATH
    PYTHON_RUNNER=("$ISAACSIM_ROOT/python.sh")
fi
echo "[Launch] python runner: ${PYTHON_RUNNER[*]}"

echo "[Launch] Step 3: clearing LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64"

if [ "$USE_CONDA_PYTHON" = "1" ]; then
    echo "[Launch] Step 4: sourcing setup_conda_env.sh..."
    cd "$ISAACSIM_ROOT"
    source ./setup_conda_env.sh
    echo "[Launch] Step 4 done."
else
    echo "[Launch] Step 4: skipping setup_conda_env.sh for bundled Python"
fi

for _ros2_lib in \
    "$ISAACSIM_ROOT/exts/isaacsim.ros2.bridge/humble/lib" \
    "$ISAACSIM_ROOT/exts/omni.isaac.ros2_bridge/humble/lib"
do
    if [ -d "$_ros2_lib" ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$_ros2_lib"
    fi
done
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "[Launch] Step 5: testing python import..."
"${PYTHON_RUNNER[@]}" -c "import sys; print('[Launch] Python OK:', sys.version)" || { echo "[Launch] Python failed!"; exit 1; }

echo "[Launch] Step 6: running main.py..."
export PYTHONFAULTHANDLER=1
export LD_PRELOAD="$ISAACSIM_ROOT/kit/libcarb.so"
export RESOURCE_NAME="IsaacSim"
export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
if [ -z "$VK_ICD_FILENAMES" ] && [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

_xr_mode="${ISAAC_XR_MODE:-vr}"
_xr_enabled=1
case "$_xr_mode" in
    ""|"0"|"false"|"none"|"off"|"disabled")
        _xr_enabled=0
        ;;
esac

# XR_RUNTIME_JSON 미리 탐색 (Python glob 재귀 탐색으로 인한 hang 방지)
if [ "$_xr_enabled" = "1" ] && [ -z "$XR_RUNTIME_JSON" ]; then
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
echo "[Launch] ISAAC_XR_BACKEND=${ISAAC_XR_BACKEND:-OpenXR}"

if [ "$_xr_enabled" = "0" ]; then
        echo "[Launch] XR disabled for this run; skipping SteamVR wait."
else
    # SteamVR VR 세션이 완전히 열릴 때까지 대기
    echo "[Launch] Waiting 8s for SteamVR VR session to be ready..."
    sleep 8
fi

if [ "$#" -eq 0 ]; then
    set -- "$PROJECT_ROOT/v2/main.py"
fi

exec "${PYTHON_RUNNER[@]}" -u "$@"
