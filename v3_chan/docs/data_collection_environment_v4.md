# HRI Data Collection Environment v4

## 변경 목적

기존 수동 capsule, AABB, 로봇 링크 중심거리 기반 판정에서 발생하던 충돌 오검출과 접촉 누락을 줄이기 위해 안전 geometry를 변경했다.

## 주요 변경 사항

- 로봇 안전 판정은 Isaac Sim Franka asset의 built-in PhysX collider를 사용한다.
- 감시 범위는 distal end-effector 영역인 `panda_link6`, `panda_link7`, `panda_link8`, `panda_hand`, `panda_leftfinger`, `panda_rightfinger`이다.
- Isaac Sim 4.5의 `panda_link8`에는 독립 collider가 없으므로, 같은 위치의 built-in `panda_hand` collider가 flange 영역을 담당한다.
- 사람의 왼손과 오른손은 VR tracking 위치에 놓인 반지름 `0.035 m` sphere로 계산한다.
- collision, near, distance gate, 햅틱이 모두 동일한 PhysX surface-gap 결과를 사용한다.
- 햅틱은 손 sphere와 distal collider가 실제 overlap한 손에만 전달한다.
- tracking loss 또는 query 실패는 `geometry_valid=0`으로 기록하며 collision으로 처리하지 않는다.

## 기본 판정 기준

```text
collision: contact 또는 surface_gap <= 0.00 m
near-miss: 0.00 m < surface_gap <= 0.02 m
near: surface_gap <= 0.05 m
distance_gate: clip((0.13 - surface_gap) / (0.13 - 0.05), 0, 1)
```

## 수집 데이터

- HDF5 schema: `hri_obs_v9_cartesian_speed_aligned`
- 기존 84차원 `obs_policy`는 robot-only checkpoint 호환을 위해 유지한다.
- safety residual용 `hri_obs_policy`는 gripper 중심거리 field를 제외한 83차원으로 고정한다.
- 양손 surface gap, closest link/collider ID, contact, penetration, near, near-miss, gate, haptic pulse, geometry validity를 매 step 저장한다.
- 양손의 world-frame velocity, exact closest surface point, link linear/angular velocity, 회전 보정 속도, 최종 surface-point velocity, relative velocity, surface-gap rate, closing speed, TTC도 매 step 저장한다.
- 손·gap·TTC 동적 값은 실제 XR pose 획득 monotonic time을 기준으로 하는 wall-time canonical 값과 simulation-time 진단값(`dynamic_sim/*`)을 분리해 저장한다. link origin 선속도는 두 timebase 각각의 pose finite difference를 사용하고, closest point 회전 성분은 world-frame angular velocity로 보정한다. 손 속도와 gap rate에는 `0.1 s` dt 기반 EMA를 적용한다. TTC cap은 `10 s`이며, tracking/query invalid와 closest collider 변경 frame은 별도 valid flag로 표시한다.
- 기존 HDF5는 덮어쓰지 않고 session별 새 파일을 생성한다.
- HDF5 root에는 session/participant/protocol/git/Isaac/physics dt/room calibration metadata를 저장한다.
- 각 step에는 simulation time, monotonic time, Unix epoch time을 함께 저장해 이후 EEG marker와 정렬할 수 있게 한다.
- marker와 sample CSV도 session별 파일로 저장하고 `session_id`, `episode_index`를 포함한다.
- 한 session의 세 episode는 각각 `slow`, `medium`, `fast` profile을 하나씩 사용한다. session마다 순서를 `slow -> medium -> fast`, `medium -> fast -> slow`, `fast -> slow -> medium`으로 순환해 order effect를 counterbalance한다.
- 속도 profile은 `events_dt` 배율이 아니라 Lula FK로 검증한 end-effector Cartesian 변위 상한으로 적용한다. `transport`는 slow/medium/fast 각각 `0.30/0.40/0.50 m/s`, 접근·배치는 `0.10/0.15/0.20 m/s`, 사람 근접 full gate는 `0.05/0.10/0.15 m/s`이다. 0.13~0.05 m surface-gap 구간에서는 기존 distance gate로 정상 단계 속도에서 근접 속도로 선형 blending한다. 자세한 제어·기록·검증 규칙은 `cartesian_speed_control.md`를 따른다.
- `PickPlaceController`의 phase timer가 실제 arm tracking보다 빨라 공중에서 gripper를 닫지 않도록 event 2에서 controller가 제어하는 `panda_rightfinger` end-effector와 cube center를 확인한다. finger-link 원점 평균은 grasp gate 기준으로 사용하지 않는다. 기본값은 `0.075 m` 이내에서 3 physics step 연속이며, 3초 안에 충족하지 못하면 해당 attempt를 실패 처리한다. 이 값은 기존 grasp 관측의 `PICK_PLACE_GRASP_MAX_DISTANCE_M=0.075`와 동일하며, Panda pose의 sub-millimeter tracking 편차가 정상 grasp를 막지 않도록 한다. cube별 최대 attempt는 3회이고, 초과하면 production session을 실패로 종료한다.
- 현재 기본 `HRI_COLLECTION_LABEL`은 `surface_twist_v9_cartesian_speed_v1`이다. events_dt 배율 기반 v8 및 이전 grasp-gate dataset과 동일 speed condition으로 직접 섞지 않는다.

## 본수집 Experiment Metadata

본수집 HDF5 root attribute에는 참가자 가명, 실험 조건, block, protocol/calibration 버전, XR hand/proxy 설정, task 성공 threshold, 햅틱 설정을 저장한다. `haptic_pulse_left/right`는 UDP `sendto()` 성공만 의미하며 Windows bridge 수신이나 장갑의 실제 진동을 확인하지는 않는다.

본수집에서는 다음 값이 비어 있거나 `unspecified`이면 실행을 중단한다.

```text
HRI_PARTICIPANT_ID
HRI_PARTICIPANT_SESSION_INDEX
HRI_PARTICIPANT_HANDEDNESS
HRI_IS_PRACTICE
HRI_EXPERIMENT_CONDITION
HRI_EXPERIMENT_BLOCK_ID
HRI_HAPTIC_CONDITION
HRI_PROTOCOL_VERSION
HRI_ROOM_CALIBRATION_ID
```

`HRI_IS_PRACTICE`는 반드시 `0/1`, `HRI_HAPTIC_CONDITION`은 반드시 `on/off`로 지정한다. 햅틱 조건 `on/off`는 실제 `BHAPTICS_ENABLED=1/0` 및 UDP endpoint 설정과 일치해야 한다. 빠른 개발 테스트만 `HRI_COLLECTION_TEST_MODE=1`로 이 검증을 건너뛸 수 있다. 본수집에서는 HDF5의 `haptics_intensity`, `haptics_min_interval_s`, `haptics_contact_min_steps`, `haptics_pulse_flag_semantics`를 함께 확인한다.

## 이전 데이터 보관

2026-07-21 이전 수집 HDF5와 새 환경 도입 전 로그는 다음 위치로 이동했다.

```text
v3_chan/archive/pre_builtin_physx_20260721/trajectories/
v3_chan/archive/pre_builtin_physx_20260721/logs/
```

과거 HDF5의 손 trajectory는 replay에 사용할 수 있지만, 저장된 collision/near/gate label은 새 학습과 평가에 직접 사용하지 않는다. replay된 손 위치와 현재 Panda collider로 안전 값을 다시 계산한다.

## 수집 실행

```bash
cd /home/railab/Desktop/Isaac_HRC
HRI_PARTICIPANT_ID=P01 \
HRI_PARTICIPANT_SESSION_INDEX=1 \
HRI_PARTICIPANT_HANDEDNESS=right \
HRI_IS_PRACTICE=0 \
HRI_EXPERIMENT_CONDITION=haptic_on_contact_multispeed_v1 \
HRI_EXPERIMENT_BLOCK_ID=block_01 \
HRI_HAPTIC_CONDITION=on \
BHAPTICS_ENABLED=1 \
HRI_PROTOCOL_VERSION=errp_hri_collection_multispeed_v1 \
HRI_ROOM_CALIBRATION_ID=vr_room_to_isaac_world_v1 \
bash v3_chan/run_pick_place.sh
```

한 번 실행하면 최대 3 episode를 수집하고 session별 HDF5와 로그를 새로 저장한다. 정상 완료 뒤 `validate_hri_collection.py`가 schema, metadata, timestamp, layout, speed-condition, 양손별 pose valid fraction, RTF valid fraction, XR anchor 상태를 검사한다. 기본 production 하한은 각 손 pose valid `0.90`, RTF valid `0.95`이며 각각 `HRI_VALIDATOR_MIN_HAND_POSE_VALID_FRACTION`, `HRI_VALIDATOR_MIN_RTF_VALID_FRACTION`으로 조정할 수 있다. validator가 통과한 경우에만 speed order counter가 다음 순서로 진행하며, 불완전 session은 nonzero exit code로 종료한다. 테스트 모드는 counter를 바꾸지 않으며, 특정 순서를 재현할 때는 `HRI_SPEED_ORDER_INDEX=0`, `1`, `2`를 지정한다.
