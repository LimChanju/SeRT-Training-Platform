# Isaac VR Project

VR/Isaac Sim 기반 인간-로봇 협업 데이터 수집 및 robot learning
파이프라인입니다.

이 저장소는 NVIDIA Isaac Sim 환경에서 Franka Panda pick-and-place 작업을
실행하고, VR/hand tracking 기반 사람 상태, 사람-로봇 근접/충돌 이벤트,
안전 관련 피드백 라벨, trajectory 데이터를 수집하는 연구용 플랫폼입니다.
수집된 expert/HRI trajectory는 BC/PPO 정책 학습과 rollout 평가에 사용됩니다.

## 주요 구성

| 항목 | 경로 |
| --- | --- |
| Isaac Sim VR runtime | `v2/main.py`, `v3_chan/main.py` |
| Robot-only RL baseline | `v2/train_rl.py`, `v2/evaluate_rollout_policy.py`, `v2/rl/` |
| HRI safety experiment | `v3_chan/` |
| Trajectory schema | `docs/rl_trajectory_schema.md` |
| RL progress notes | `docs/rl_progress.md` |
| System pipeline | `docs/pipeline.md` |
| Porting requirements | `docs/porting_requirements.md` |
| Artifact manifest | `artifacts/MANIFEST.md` |

## Installation

1. NVIDIA Isaac Sim 4.5.0을 설치한다.
2. CUDA 11.8, Isaac Sim Python, PyTorch `2.5.1+cu118` 조합을 맞춘다.
3. SteamVR/ALVR/OpenXR 환경을 준비한다.
4. 저장소를 clone한다.

```bash
git clone https://github.com/LimChanju/SeRT-Training-Platform.git
cd SeRT-Training-Platform
```

자세한 이식 요구사항은 `docs/porting_requirements.md`를 참고한다.

## Runtime

메인 Isaac Sim 환경은 launcher를 통해 실행한다.

```bash
./launch_isaac.sh "$PWD/v2/main.py"
```

또는 로컬 shell alias가 설정되어 있다면:

```bash
isaac "$PWD/v2/main.py"
```

VR/hand tracking 데이터는 UDP `5555`로 JSON packet을 받는다. bHaptics 연동을
사용하는 경우 haptic bridge는 UDP `5005`를 사용한다.

## HRI Trajectory Recording

VR 협업 episode를 HDF5 trajectory로 저장하려면:

```bash
ENABLE_HRI_TRAJECTORY_RECORDING=1 \
HRI_TRAJECTORY_PATH=v2/trajectories/hri_vr_expert_v0.hdf5 \
./launch_isaac.sh "$PWD/v2/main.py"
```

주요 산출물:

| 경로 | 설명 |
| --- | --- |
| `v2/session_samples.csv` | frame/session sample 로그 |
| `v2/errp_markers.csv` | safety/event marker 로그 |
| `v2/trajectories/*.hdf5` | expert/HRI trajectory dataset |
| `v2/eval_results/*.json` | rollout 평가 요약 |
| `v2/eval_results/*.csv` | episode별 rollout 평가 결과 |

## Expert/RL Pipeline

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

## Haptics

bHaptics bridge 실행:

```bash
BHAPTICS_APP_ID=... \
BHAPTICS_API_KEY=... \
python scripts/bhaptics_udp_bridge.py
```

테스트 pulse 전송:

```bash
python scripts/test_haptics_udp.py
```

## Tests

일반 Python 단위 테스트:

```bash
pip install pytest pytest-cov numpy h5py
python -m pytest tests/ -v
```

Isaac Sim 의존 runtime, rollout, PPO 학습은 Isaac Sim launcher를 통해 별도로
검증한다.

## Artifacts

큰 모델/trajectory artifact는 일반 Git history에 넣지 않고 release artifact
또는 별도 artifact store에 보관한다. 현재 robot-only baseline artifact 목록은
`artifacts/MANIFEST.md`에 정리되어 있다.

## License

이 프로젝트는 [MIT License](LICENSE)에 따라 배포됩니다.
