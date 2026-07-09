# 다른 서버 이식 요구사항

이 문서는 Isaac VR 데이터 수집 플랫폼을 다른 컴퓨터나 서버로 옮겨 실행할 때
맞춰야 하는 런타임, 드라이버, Python 패키지, VR 장비, 데이터 파일 조건을
정리한다.

## 핵심 버전

| 항목 | 권장/확인 버전 |
| --- | --- |
| Isaac Sim | NVIDIA Isaac Sim 4.5.0 |
| Python, Isaac 실행용 | Python 3.10.15 |
| Conda environment | `isaac_vr` |
| CUDA | CUDA 11.8 |
| PyTorch, Isaac/RL 기준 | `torch==2.5.1+cu118` |
| numpy | `numpy>=1.24` |
| HDF5 저장/재생 | `h5py` |
| GPU | NVIDIA RTX 계열 권장, 기존 실험은 RTX 4090 기준 |
| OS | Linux/Ubuntu 권장 |

주의할 점은 conda 환경의 torch 버전보다 Isaac Sim Python에서 import되는
torch 버전이 더 중요하다는 것이다. RL 학습과 rollout 평가는 Isaac Sim
런타임에서 실행되므로, 아래 명령의 결과를 기준으로 확인한다.

```bash
~/isaac-sim-4.5.0/python.sh -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

기대값은 다음과 같다.

```text
2.5.1+cu118
11.8
```

필요하면 PyTorch CUDA 11.8 wheel을 설치한다.

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

## 시스템 요구사항

새 서버에서는 먼저 NVIDIA driver와 GPU 접근이 정상인지 확인한다.

```bash
nvidia-smi
```

`nvidia-smi`가 실패하면 Isaac Sim, CUDA, PyTorch CUDA 실행도 정상 동작하기
어렵다. CUDA 11.8 호환 driver가 필요하며, Isaac Sim/RTX 렌더링까지 고려하면
너무 낮은 driver 버전은 피한다.

Vulkan ICD 파일도 확인한다.

```bash
ls /usr/share/vulkan/icd.d/nvidia_icd.json
```

이 프로젝트의 launcher는 해당 파일이 있으면 `VK_ICD_FILENAMES`에 사용한다.

## 경로와 환경

현재 launcher는 다음 경로를 기본으로 가정한다.

| 항목 | 기본 경로/값 |
| --- | --- |
| Isaac Sim 설치 경로 | `~/isaac-sim-4.5.0` |
| Isaac setup script | `~/isaac-sim-4.5.0/setup_conda_env.sh` |
| Conda 설치 경로 | `~/anaconda3` |
| Conda environment | `isaac_vr` |
| CUDA library path | `/usr/local/cuda-11.8/lib64` |

다른 경로에 설치했다면 `launch_isaac.sh`의 경로를 수정하거나, 새 서버에서
위 경로에 맞춰 설치한다.

기본 확인 명령은 다음과 같다.

```bash
ls ~/isaac-sim-4.5.0/setup_conda_env.sh
conda activate isaac_vr
python -V
```

## VR/OpenXR 요구사항

VR 데이터 수집을 하려면 SteamVR 또는 OpenXR runtime이 필요하다.

확인할 항목은 다음과 같다.

- SteamVR이 HMD와 controller를 인식하는지 확인한다.
- SteamVR을 OpenXR runtime으로 설정한다.
- `XR_RUNTIME_JSON`이 `steamxr_linux64.json`을 가리키는지 확인한다.
- Quest/ALVR 또는 다른 hand tracking 송신 경로가 UDP `5555`로 데이터를
  보낼 수 있어야 한다.

`XR_RUNTIME_JSON`이 자동으로 잡히지 않으면 직접 지정한다.

```bash
export XR_RUNTIME_JSON=/path/to/steamxr_linux64.json
```

## VR 연동 버전

SteamVR과 ALVR은 자동 업데이트나 headset/PC streamer 버전 차이로 문제가
생길 수 있으므로, 이식 서버에서는 아래 조합을 기준으로 맞춘다.

| 항목 | 권장/확인 버전 |
| --- | --- |
| SteamVR | SteamVR 2.x stable |
| SteamVR AppID | `250820` |
| SteamVR 확인 build | `22542555` |
| SteamVR beta channel | `previous` |
| OpenXR runtime | SteamVR OpenXR runtime |
| OpenXR runtime JSON | `steamxr_linux64.json` |
| ALVR | `v20.14.1` |
| ALVR PC streamer | `alvr_streamer_linux` `v20.14.1` |
| ALVR headset app/APK | PC streamer와 같은 `v20.14.1` |
| Headset | Meta Quest 3 또는 Quest 계열 |
| Hand tracking input | UDP `5555` |

SteamVR은 PyTorch처럼 정확한 semantic version으로 고정하기보다 Steam app
manifest의 build id와 OpenXR runtime 설정을 확인하는 방식이 현실적이다.

```bash
cat ~/.steam/debian-installation/steamapps/appmanifest_250820.acf
cat ~/.steam/debian-installation/steamapps/common/SteamVR/steamxr_linux64.json
```

`steamxr_linux64.json`은 SteamVR runtime을 가리켜야 한다.

```json
{
  "runtime": {
    "name": "SteamVR",
    "library_path": "bin/linux64/vrclient.so"
  }
}
```

ALVR은 PC streamer와 headset app/APK 버전을 반드시 맞춘다. 현재 이 프로젝트의
로컬 진단 스크립트는 아래 ALVR 설치 경로를 기준으로 작성되어 있다.

```text
~/.local/share/ALVR-Launcher/installations/v20.14.1/
```

SteamVR 업데이트가 `vrcompositor`를 덮어쓰면 ALVR compositor wrapper 연결이
깨질 수 있다. 이 경우 아래 스크립트로 현재 상태를 확인한다.

```bash
python scripts/steamvr_compositor_switch.py --status
```

필요하면 ALVR wrapper를 다시 설치한다.

```bash
python scripts/steamvr_compositor_switch.py --install-alvr-wrapper
```

## UDP 포트

| 용도 | 포트 |
| --- | --- |
| VR/hand tracking JSON 입력 | UDP `5555` |
| bHaptics bridge | UDP `5005` |

방화벽이나 Docker/원격 서버 네트워크 설정 때문에 UDP packet이 막히지 않는지
확인해야 한다.

## Python 패키지

일반 테스트와 API 실행에는 최소한 다음이 필요하다.

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

trajectory 저장/재생과 RL 학습에는 다음이 필요하다.

```bash
pip install h5py
```

TensorBoard 변환을 사용할 경우:

```bash
pip install tensorboard tensorboardX
```

bHaptics 장갑 연동을 사용할 경우에는 별도 bHaptics SDK/Python package와
아래 환경변수가 필요하다.

```bash
export BHAPTICS_APP_ID=...
export BHAPTICS_API_KEY=...
```

## 옮겨야 하는 데이터와 산출물

코드만 옮기면 학습/평가 재현이 되지 않을 수 있다. 필요에 따라 아래 파일도
같이 옮긴다.

| 경로 | 설명 |
| --- | --- |
| `v2/policies/*.pt` | BC/PPO policy checkpoint |
| `v2/policies/*_history.json` | 학습 history |
| `v2/trajectories/*.hdf5` | expert/HRI trajectory dataset |
| `v2/eval_results/*.json` | rollout 평가 요약 |
| `v2/eval_results/*.csv` | episode별 rollout 평가 결과 |
| `v2/errp_markers.csv` | safety/event marker 로그 |
| `v2/session_samples.csv` | session sample 로그 |

새 서버에서 새로 수집한다면 `v2/trajectories/`, `v2/eval_results/`, `v2/` 아래
CSV 파일 생성 경로에 쓰기 권한이 있어야 한다.

## 주요 환경변수

| 환경변수 | 용도 |
| --- | --- |
| `ENABLE_VR` | VR mode 활성화 |
| `VR_SUBJECT_IDS` | 특정 피험자에게만 VR 활성화 |
| `ENABLE_HRI_TRAJECTORY_RECORDING` | HRI trajectory HDF5 기록 활성화 |
| `HRI_TRAJECTORY_PATH` | HRI trajectory 저장 경로 |
| `ISAAC_SKIP_VR_WAIT` | RL/평가처럼 VR 대기가 필요 없는 실행에서 사용 |
| `XR_RUNTIME_JSON` | SteamVR/OpenXR runtime JSON 경로 |
| `ISAAC_XR_MODE` | Isaac XR mode |
| `ISAAC_XR_BACKEND` | SteamVR/OpenXR backend |
| `BHAPTICS_NOTEBOOK_IP` | bHaptics bridge 대상 IP |
| `BHAPTICS_UDP_PORT` | bHaptics UDP 포트, 기본 `5005` |

## 최소 이식 검증 순서

먼저 GPU와 런타임을 확인한다.

```bash
nvidia-smi
ls /usr/share/vulkan/icd.d/nvidia_icd.json
ls ~/isaac-sim-4.5.0/setup_conda_env.sh
```

Python과 torch를 확인한다.

```bash
conda activate isaac_vr
python -V
~/isaac-sim-4.5.0/python.sh -c "import torch; print(torch.__version__, torch.version.cuda)"
```

일반 테스트와 API를 확인한다.

```bash
python -m pytest tests/ -v
python api/app.py
```

다른 터미널에서 확인한다.

```bash
curl http://localhost:8080/health
```

Isaac Sim 메인 런타임을 실행한다.

```bash
./launch_isaac.sh "$PWD/v2/main.py"
```

HRI trajectory 수집을 확인한다.

```bash
ENABLE_HRI_TRAJECTORY_RECORDING=1 \
HRI_TRAJECTORY_PATH=v2/trajectories/hri_vr_expert_v0.hdf5 \
./launch_isaac.sh "$PWD/v2/main.py"
```

실행 후 아래 파일이 생성되면 기본 이식은 성공한 것으로 볼 수 있다.

```text
v2/session_samples.csv
v2/errp_markers.csv
v2/trajectories/hri_vr_expert_v0.hdf5
```

## 자주 발생하는 문제

- Isaac Sim 설치 경로가 `~/isaac-sim-4.5.0`이 아니어서 launcher가 실패한다.
- conda environment 이름이 `isaac_vr`이 아니어서 launcher가 잘못된 Python을 쓴다.
- CUDA 11.8은 설치했지만 NVIDIA driver 또는 Vulkan 설정이 잡히지 않았다.
- SteamVR은 켰지만 OpenXR runtime이 SteamVR로 설정되지 않았다.
- `XR_RUNTIME_JSON`이 비어 있거나 잘못된 `steamxr_linux64.json`을 가리킨다.
- hand tracking UDP `5555` packet이 방화벽이나 네트워크 설정 때문에 들어오지 않는다.
- 코드만 옮기고 `v2/policies/*.pt`, `v2/trajectories/*.hdf5`를 옮기지 않았다.
- trajectory 저장 경로에 쓰기 권한이 없다.
