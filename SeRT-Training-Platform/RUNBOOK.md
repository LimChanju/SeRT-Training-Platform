# Runbook

## 목적

이 문서는 VR/Isaac Sim 기반 인간-로봇 협업 데이터 수집 플랫폼의 실행,
로그 확인, 데이터 수집, RL 학습/평가, 장애 대응 절차를 정리한다.

핵심 기능은 다음과 같다.

- VR/hand tracking 기반 사람 상태 수집
- Franka Panda pick-and-place simulation
- 사람-로봇 근접/충돌 및 safety marker 로깅
- expert/HRI trajectory HDF5 기록
- BC/PPO 정책 학습 및 rollout 평가
- 선택적 bHaptics tactile feedback 연동

## 주요 구성 요소

| 구성 요소 | 경로 | 설명 |
| --- | --- | --- |
| Isaac Sim runtime | `v2/main.py` | VR 입력, 로봇 제어, 충돌/근접 이벤트 로깅 |
| HRI safety experiment | `v3_chan/` | 새 HRI safety experiment runtime |
| AI policy training | `v2/train_rl.py` | PPO 기반 정책 학습 |
| Rollout evaluation | `v2/evaluate_rollout_policy.py` | 학습 정책 평가 및 JSON/CSV 결과 저장 |
| Safety feedback labeling | `v2/rl/pseudo_errp.py` | 근접/충돌 이벤트를 safety feedback signal로 변환 |
| Haptics bridge | `scripts/bhaptics_udp_bridge.py` | Isaac collision event를 bHaptics glove로 전달 |
| Artifact manifest | `artifacts/MANIFEST.md` | 큰 모델/trajectory artifact 목록 |

## 로컬 실행

### 1. 저장소 준비

```bash
git clone https://github.com/LimChanju/SeRT-Training-Platform.git
cd SeRT-Training-Platform
```

### 2. Python 단위 테스트

```bash
python -m pytest tests/ -v
```

### 3. Isaac Sim 데이터 수집 실행

Isaac Sim이 설치된 환경에서 실행한다.

```bash
./launch_isaac.sh "$PWD/v2/main.py"
```

또는 shell alias가 설정되어 있다면:

```bash
isaac "$PWD/v2/main.py"
```

VR/hand tracking 데이터는 UDP `5555`로 JSON 형태로 전달된다.

### 4. HRI trajectory 기록

```bash
ENABLE_HRI_TRAJECTORY_RECORDING=1 \
HRI_TRAJECTORY_PATH=v2/trajectories/hri_vr_expert_v0.hdf5 \
./launch_isaac.sh "$PWD/v2/main.py"
```

## 로그 및 산출물 확인

| 산출물 | 경로 | 설명 |
| --- | --- | --- |
| Safety/event markers | `v2/errp_markers.csv` | episode, collision, safety feedback event |
| Session samples | `v2/session_samples.csv` | 거리, 충돌 여부 등 frame/session sample |
| HRI trajectories | `v2/trajectories/*.hdf5` | 사람 head/hand replay와 trajectory dataset |
| Rollout eval results | `v2/eval_results/*.json` | 정책 평가 요약 |
| Rollout eval CSV | `v2/eval_results/*.csv` | episode별 평가 결과 |
| Policy checkpoints | `v2/policies/*.pt` | BC/PPO checkpoint |

## Expert/RL 실행

Robot-only expert trajectory 수집:

```bash
ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/collect_expert_trajectories.py" \
  --episodes 10 \
  --overwrite
```

HRI replay를 포함한 PPO 학습:

```bash
ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/train_rl.py" \
  --human-replay-data v2/trajectories/hri_vr_expert_v0.hdf5 \
  --human-replay-mode step \
  --human-replay-episode-policy cycle
```

Rollout 평가:

```bash
ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/evaluate_rollout_policy.py" \
  --checkpoint v2/policies/ppo_pick_place_v7_residual_rewardv4_strict_best.pt \
  --human-replay-data v2/trajectories/hri_vr_expert_v0.hdf5
```

## bHaptics 확인

Bridge 실행:

```bash
BHAPTICS_APP_ID=... \
BHAPTICS_API_KEY=... \
python scripts/bhaptics_udp_bridge.py
```

테스트 pulse:

```bash
python scripts/test_haptics_udp.py
```

## 장애 대응

### Isaac Sim 실행 실패

1. `~/isaac-sim-4.5.0/setup_conda_env.sh`가 존재하는지 확인한다.
2. `conda activate isaac_vr` 후 Python 버전을 확인한다.
3. `nvidia-smi`와 Vulkan ICD 설정을 확인한다.
4. `XR_RUNTIME_JSON`이 SteamVR runtime JSON을 가리키는지 확인한다.
5. `docs/porting_requirements.md`의 버전 요구사항을 다시 확인한다.

### VR tracking 미수신

1. SteamVR에서 headset/controller가 인식되는지 확인한다.
2. SteamVR이 OpenXR runtime으로 설정되어 있는지 확인한다.
3. Quest/ALVR hand tracking 송신이 Isaac 서버의 UDP `5555`로 향하는지 확인한다.
4. `python scripts/inspect_hand_tracking_udp.py`로 raw packet 수신을 확인한다.

### bHaptics 미동작

1. bHaptics Player와 glove 연결 상태를 확인한다.
2. `BHAPTICS_APP_ID`, `BHAPTICS_API_KEY` 환경변수를 확인한다.
3. Bridge가 UDP `5005`를 listen 중인지 확인한다.
4. `python scripts/test_haptics_udp.py`로 test pulse를 보낸다.

### Rollout 평가 실패

1. checkpoint 경로가 올바른지 확인한다.
2. Isaac Sim 실행 환경과 GPU/CPU device 설정을 확인한다.
3. `v2/evaluate_rollout_policy.py`의 입력 인자와 output 경로를 확인한다.
4. 실패한 episode seed가 있으면 `v2/debug_rollout_seeds.py`로 재현한다.

## 롤백

문제가 발생한 변경이 main에 병합된 경우, 직전 안정 커밋 또는 태그로
되돌린다.

```bash
git log --oneline
git revert <problem_commit_sha>
git push origin main
```

큰 모델/trajectory artifact는 Git history가 아니라 release artifact 또는 별도
artifact store 기준으로 복구한다. 현재 robot-only baseline artifact 목록은
`artifacts/MANIFEST.md`를 참고한다.
