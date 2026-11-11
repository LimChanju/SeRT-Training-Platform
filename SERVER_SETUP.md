# Isaac_HRC Server Setup

This workspace contains `LimChanju/SeRT-Training-Platform` imported into:

```bash
/home/railab/Desktop/Isaac_HRC
```

## Current Server Notes

- `git` is not installed on this server.
- `python3` is available, but `pip` and `pytest` are not installed in the system Python.
- Ubuntu is `22.04.5 LTS`.
- `nvidia-smi` exists, but currently cannot communicate with the NVIDIA driver.
- The GPU is detected on PCI as an NVIDIA device, and Ubuntu recommends `nvidia-driver-595-open`.
- The platform source was imported from the GitHub `main` branch as a ZIP archive.

## Target Versions

| Item | Version |
| --- | --- |
| Isaac Sim | NVIDIA Isaac Sim 4.5.0 |
| Python for Isaac | Python 3.10.15 |
| Conda environment | `isaac_vr` |
| CUDA | CUDA 11.8 |
| PyTorch | `torch==2.5.1+cu118` |
| numpy | `numpy>=1.24` |
| Analysis | `pandas`, `scipy`, `matplotlib`, `seaborn` |
| HDF5 | `h5py` |
| Training logs | `tensorboard`, `tensorboardX` |
| GPU | NVIDIA RTX, previous experiments used RTX 4090 |
| OS | Linux/Ubuntu |

## Setup Order

### 1. Fix NVIDIA Driver First

Isaac Sim will not run until `nvidia-smi` works. On this server, Ubuntu reports `nvidia-driver-595-open` as the recommended driver.

```bash
sudo apt update
sudo apt install nvidia-driver-595-open
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

Continue only when the GPU and driver version are shown correctly.

### 2. Install Conda

Install Miniconda or Anaconda, then create the project environment:

```bash
cd /home/railab/Desktop/Isaac_HRC
conda env create -f environment.yml
conda activate isaac_vr
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected:

```text
Python 3.10.15
2.5.1+cu118 True
```

### 3. Install Isaac Sim 4.5.0

Install NVIDIA Isaac Sim 4.5.0 and place or symlink it at:

```bash
$HOME/isaac-sim-4.5.0
```

If it is installed elsewhere, pass the location when launching:

```bash
ISAACSIM_ROOT=/path/to/isaac-sim-4.5.0 ./launch_isaac.sh
```

## Isaac Sim Runtime

The main simulation entrypoint is:

```bash
v2/main.py
```

The project targets NVIDIA Isaac Sim 4.5. By default, the launcher expects Isaac Sim at:

```bash
$HOME/isaac-sim-4.5.0
```

Override it when needed:

```bash
ISAACSIM_ROOT=/path/to/isaac-sim-4.5.0 ./launch_isaac.sh
```

The launcher runs `v2/main.py` automatically when no script argument is provided. To run a different script:

```bash
./launch_isaac.sh v2/evaluate_rollout_policy.py
```

By default, the launcher uses Isaac Sim's bundled Python via `$HOME/isaac-sim-4.5.0/python.sh`. Keep the `isaac_vr` conda environment for local tests, analysis, notebooks, and non-Isaac Python tooling. Running Kit UI with the conda Python can crash because conda packages such as PySide/Qt, SciPy, and PyTorch are loaded into the Isaac process.

Only use conda Python for Isaac when explicitly debugging that path:

```bash
USE_CONDA_PYTHON=1 ./launch_isaac.sh
```

## Environment Flags

Load the default flags before running Isaac:

```bash
set -a
source flags.env
set +a
./launch_isaac.sh
```

If Isaac Sim Full crashes while enabling ROS2 Bridge with `ROS_DISTRO env var not found` or `AMENT_PREFIX_PATH is not set`, launch through this project's `launch_isaac.sh` or set the bundled ROS2 bridge library path first:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/isaac-sim-4.5.0/exts/isaacsim.ros2.bridge/humble/lib"
```

Hand tracking is received over UDP on port `5555` by default. The relevant settings are in `v2/app_config.py` and can be overridden with environment variables.

## Local Python Checks

Once `pip` is available, install the lightweight test dependencies and run:

```bash
python3 -m pip install pytest pytest-cov numpy
python3 -m pytest tests --ignore=tests/e2e -q
```

Isaac Sim itself should be launched with Isaac's Python environment, not the system Python.
