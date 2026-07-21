# v3_chan Personal Server Handoff

이 문서는 현재 Isaac HRC 서버에서 진행한 `v3_chan` HRI pick-and-place 실험 상태와, 개인 서버에서 RL 학습/평가를 이어가기 위해 옮겨야 할 산출물을 정리한 것이다.

## 현재 상태

### 실험 목적

- 사람과 로봇 팔이 함께 pick-and-place 작업을 수행하는 HRI 환경을 만든다.
- 로봇은 기존 robot-only pick-and-place 성공 정책을 유지한다.
- 사람 손/로봇 근접, 충돌, 햅틱 이벤트를 관측으로 추가한다.
- 최종 연구 목적은 ErrP EEG를 safety reward shaping에 사용해서, 사람의 인지적 안전을 보호하는 정책을 학습하는 것이다.

### 현재 구현 기준

- 현재 안정적으로 동작하는 손 입력은 Quest controller/visual pose 기반 구 프록시이다.
- OpenXR hand joint tracking은 현재 Isaac/SteamVR 경유에서 유효한 joint가 들어오지 않아 보류한다.
- 따라서 현재 수집 데이터는 `sphere proxy` 기반 HRI 데이터이다.

현재 실행 기준:

```text
XR_EXTERNAL_HAND_TRACKING=0
XR_HAND_PROXY_ENABLED=0
XR_HAND_SPHERE_ENABLED=1
XR_HAND_HAPTIC_POINT_MODE=sphere
```

### 현재 HRI 데이터

현재 수집된 대표 파일:

```text
v3_chan/trajectories/hri_vr_sphere_obs.hdf5
```

과거 v1 schema에서 확인된 첫 episode 예시:

```text
episode_000000
success = True
completed_picks = 3
episode_length = 2751
obs_policy shape = (2751, 84)
hri_obs_policy shape = (2751, 74)
safety/near_human sum = 409
safety/human_robot_collision sum = 66
safety/haptic_pulse_left sum = 29
safety/haptic_pulse_right sum = 37
```

`hri_obs_v2_verified_place_retry`부터는 실패한 pick-place attempt를 같은 episode 안에서 재시도하므로 `episode_length`와 observation shape의 첫 번째 차원은 가변적이다. Episode의 `success=True`는 세 cube가 실제 grasp, lift, place 검증을 모두 통과한 경우에만 기록된다.

`hri_obs_v4_builtin_panda_collision_geometry`부터 `obs_policy`는 기존 checkpoint 호환을 위해 84차원을 유지하고, `hri_obs_policy`도 built-in Panda collider 기반 보조 변수를 포함한 84차원이다. 기존 v1/v2의 74차원 및 v3의 77차원 HRI vector와 v4를 같은 batch에 바로 섞지 말고 schema별로 변환하거나 분리한다.

세부 요약은 아래 문서에 있다.

```text
v3_chan/hri_episode_000000_summary.md
v3_chan/hri_data_variables.md
```

## 개인 서버에서의 권장 구조

개인 서버에서는 `v3_chan`을 새 실험 기준으로 사용해도 된다.

다만 `v2`는 삭제하지 말고 robot-only baseline 보존용으로 남긴다.

```text
v2
  robot-only baseline 보존용

v3_chan
  HRI obs + haptic + safety residual RL 실험용
```

기존 task policy는 robot-only 학습으로 이미 성공률이 높으므로 건드리지 않는 것을 권장한다.

```text
pi_task
  기존 robot-only residual PPO policy
  input: obs_policy, 84 dim
  output: action, 5 dim
  frozen

pi_safe
  새로 학습할 safety residual policy
  input: hri_obs_policy 또는 safety obs
  output: action residual, 5 dim
  trainable
```

최종 action 구성:

```text
a_task = pi_task(obs_policy)
a_safe = pi_safe(hri_obs_policy)
gate = distance_gate(min_hand_gripper_dist)
alpha = fixed residual scale

a_final = a_task + gate * alpha * a_safe
```

## Gate와 Alpha

`gate`는 사람 손과 그리퍼 사이 거리를 기준으로 켜지는 safety policy 활성화 값이다.

예시:

```text
gate = clip((0.13 - min_hand_end_effector_surface_gap) / (0.13 - 0.05), 0, 1)
```

해석:

```text
min_hand_end_effector_surface_gap >= 0.13 m  -> gate = 0
min_hand_end_effector_surface_gap <= 0.05 m  -> gate = 1
중간 거리                      -> 선형 보간
```

`alpha`는 safety residual action의 최대 영향력을 제한하는 고정 상수로 둔다.

초기 추천:

```text
alpha = 0.1 ~ 0.3
```

ErrP는 `alpha`를 직접 바꾸는 데 쓰지 않고, `pi_safe`의 reward shaping에 사용하는 것을 권장한다.

현재 코드의 선택형 gate 인자는 다음과 같다.

```text
--policy-mode residual --residual-gate-mode distance --residual-scale 0.1
```

기존 robot-only residual checkpoint는 `residual_gate_mode` metadata가 없으므로 `none`으로 평가되어 과거 동작을 유지한다. 단, 현재 `pi_task`가 `BC + task residual PPO`의 합성 policy라면 safety trainer의 base loader도 그 두 policy를 함께 평가해야 한다. task residual actor의 `model_state_dict`만 `pi_task`로 사용하면 안 된다.

## ErrP Replay 계획

현재 live EEG 장비가 없기 때문에 처음에는 event-aligned ErrP replay를 사용한다.

### 1단계: event-aligned replay

HRI event를 기준으로 숫자 feedback을 만든다.

예시:

```text
if safety/human_robot_collision == 1:
    errp_feedback = 1.0
elif safety/haptic_pulse_left == 1 or safety/haptic_pulse_right == 1:
    errp_feedback = 1.0
elif safety/near_human == 1:
    errp_feedback = 0.3
else:
    errp_feedback = 0.0
```

reward 예시:

```text
r_total = r_task + r_safety + r_errp
r_errp = -lambda_errp * errp_feedback
```

### 2단계: classifier replay

나중에 ErrP EEG dataset과 classifier가 준비되면, 위의 `errp_feedback` 생성부만 classifier output으로 바꾼다.

```text
event-aligned replay:
  event -> errp_feedback

classifier replay:
  eeg_epoch -> errp_classifier -> errp_feedback
```

RL 코드 입장에서는 둘 다 같은 `errp_feedback` 인터페이스를 사용하게 만든다.

## 개인 서버로 옮길 산출물

### 필수

| 경로 | 크기 | 용도 |
|---|---:|---|
| `v3_chan/trajectories/hri_vr_sphere_obs.hdf5` | 2.5M | 현재 수집한 HRI trajectory dataset |
| `v3_chan/policies/ppo_pick_place_v7_residual_rewardv4_strict_best.pt` | 1.1M | frozen `pi_task`로 사용할 robot-only residual PPO policy |
| `v3_chan/policies/bc_pick_place_v1_100eps.pt` | 360K | PPO residual policy의 BC base/reference |
| `v3_chan/policies/ppo_pick_place_v7_residual_rewardv4_strict_history.json` | 24K | PPO 학습 이력 확인용 |
| `v3_chan/policies/bc_pick_place_v1_100eps_history.json` | 24K | BC 학습 이력 확인용 |
| `v3_chan/hri_data_variables.md` | - | HDF5 변수 설명 |
| `v3_chan/hri_episode_000000_summary.md` | - | 첫 HRI episode sanity check 결과 |
| `v3_chan/personal_server_handoff.md` | - | 이 문서 |

### 강력 추천

| 경로 | 크기 | 용도 |
|---|---:|---|
| `v3_chan/policies/expert_pick_place_v1.hdf5` | 62M | robot-only expert data, BC 재학습/검증/비교용 |
| `v3_chan/errp_markers.csv` | 248K | event marker 확인 및 ErrP alignment 실험용 |
| `v3_chan/session_samples.csv` | 17M | 디버깅/시계열 빠른 확인용 CSV |

### 코드도 같이 옮길 경우

개인 서버에 GitHub 최신 코드가 없다면 `v3_chan` 전체를 옮기는 것이 가장 편하다.

특히 아래 파일들은 HRI obs와 recording에 직접 관련된다.

```text
v3_chan/main.py
v3_chan/hri_obs_recorder.py
v3_chan/rl/observations.py
v3_chan/rl/human_replay.py
v3_chan/rl/pseudo_errp.py
v3_chan/run_pick_place.md
v3_chan/run_pick_place_hand_tracking.md
v3_chan/hri_data_variables.md
```

## 전송 예시

개인 서버에서 코드가 이미 최신이라면 산출물만 옮긴다.

```bash
rsync -avh \
  v3_chan/trajectories/hri_vr_sphere_obs.hdf5 \
  v3_chan/policies/ \
  v3_chan/errp_markers.csv \
  v3_chan/session_samples.csv \
  v3_chan/hri_data_variables.md \
  v3_chan/hri_episode_000000_summary.md \
  v3_chan/personal_server_handoff.md \
  USER@PERSONAL_SERVER:/path/to/Isaac_HRC/v3_chan/
```

개인 서버 코드까지 한 번에 맞추고 싶다면 `v3_chan` 전체를 압축해서 옮긴다.

```bash
tar -czf v3_chan_handoff.tar.gz \
  v3_chan \
  --exclude='v3_chan/__pycache__' \
  --exclude='v3_chan/logs'

scp v3_chan_handoff.tar.gz USER@PERSONAL_SERVER:/path/to/Isaac_HRC/
```

## 개인 서버에서 먼저 확인할 것

### 1. Policy checkpoint 확인

기존 robot-only policy는 다음 스키마를 유지해야 한다.

```text
obs_dim = 84
action_dim = 5
observation_version = obs_v1_state_controller_phase
action_version = action_v1_controller_target_delta
```

`pi_task`에는 `hri_obs_policy`를 넣지 않는다.

### 2. HRI dataset 확인

HDF5에서 최소한 아래 dataset이 읽혀야 한다.

```text
episode_000000/obs_policy
episode_000000/hri_obs_policy
episode_000000/safety/min_hand_gripper_dist_m
episode_000000/safety/near_human
episode_000000/safety/human_robot_collision
episode_000000/safety/haptic_pulse_left
episode_000000/safety/haptic_pulse_right
episode_000000/errp/feedback
episode_000000/rewards/total
```

### 3. 학습 순서

추천 순서:

```text
1. pi_task checkpoint load 확인
2. HRI dataset loader 작성
3. event-aligned errp_feedback 생성
4. frozen pi_task + trainable pi_safe residual 구조 구현
5. offline replay 또는 simulated replay로 pi_safe pretrain
6. Isaac 환경에서 residual PPO fine-tuning
7. task success rate와 safety metric을 동시에 평가
```

## 평가 지표

task 성능:

```text
success_rate
episode_length
completed_picks
cube_place_error
```

safety 성능:

```text
safety/near_human count
safety/human_robot_collision count
safety/haptic_pulse_left count
safety/haptic_pulse_right count
safety/min_hand_gripper_dist_m
errp/feedback sum
```

정책 비교는 최소한 아래 세 가지로 한다.

```text
1. pi_task only
2. pi_task + distance gate safety residual without ErrP reward
3. pi_task + distance gate safety residual with ErrP reward
```

## 주의점

- `v3_chan/run_pick_place.md`는 현재 `HRI_TRAJECTORY_OVERWRITE=1`이다.
- 같은 파일에 여러 session을 누적하고 싶으면 `HRI_TRAJECTORY_OVERWRITE=0`으로 바꾼다.
- restart마다 파일을 분리하고 싶으면 `HRI_TRAJECTORY_PATH`에 날짜/회차를 넣는다.

예시:

```bash
export HRI_TRAJECTORY_PATH="$PWD/v3_chan/trajectories/hri_vr_sphere_obs_session_001.hdf5"
```

현재 방향의 핵심은 다음과 같다.

```text
robot-only task policy는 freeze한다.
HRI 정보는 safety residual policy에만 준다.
ErrP는 alpha 조절이 아니라 safety reward shaping에 쓴다.
```
