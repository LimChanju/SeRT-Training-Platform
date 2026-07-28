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

- HDF5 schema: `hri_obs_v8_dual_clock_tracked_action_aligned`
- 기존 84차원 `obs_policy`는 robot-only checkpoint 호환을 위해 유지한다.
- safety residual용 `hri_obs_policy`는 gripper 중심거리 field를 제외한 83차원으로 고정한다.
- 양손 surface gap, closest link/collider ID, contact, penetration, near, near-miss, gate, haptic pulse, geometry validity를 매 step 저장한다.
- 양손의 world-frame velocity, exact closest surface point, link linear/angular velocity, 회전 보정 속도, 최종 surface-point velocity, relative velocity, surface-gap rate, closing speed, TTC도 매 step 저장한다.
- canonical 손/로봇 표면/상대 속도는 `time.monotonic_ns()`를 기준으로 계산한다. simulation-time 버전은 `dynamic_sim/*`에 진단용으로 병렬 저장한다. Unix epoch time은 EEG 및 외부 stream 정렬에만 사용한다. 손 속도와 gap rate에는 `0.1 s` dt 기반 EMA를 적용하고 validity를 단계별로 분리한다.
- 기존 HDF5는 덮어쓰지 않고 session별 새 파일을 생성한다.
- HDF5 root에는 session/participant/protocol/code version/source-tree hash/Isaac/physics dt/room calibration metadata를 저장한다. production mode에서 participant 또는 code provenance가 placeholder면 실행이 중단된다.
- 각 step에는 simulation time, monotonic time, Unix epoch time을 함께 저장해 이후 EEG marker와 정렬할 수 있게 한다.
- marker와 sample CSV도 session별 파일로 저장하고 `session_id`, `episode_index`를 포함한다.
- 한 session의 세 episode 속도 순서는 session마다 `slow -> medium -> fast`, `medium -> fast -> slow`, `fast -> slow -> medium`으로 순환한다. 세 속도는 동일한 초기 cube layout을 공유하고 session 간에만 layout을 바꾼다. 이동 phase의 controller progress만 각각 `1.0x`, `1.5x`, `2.0x`로 조절하고, grasp/release 및 안정화 phase timing은 동일하게 유지한다.
- HDF5 root에는 해당 session의 speed schedule과 counterbalance order index를, 각 episode attribute에는 `controller_speed_profile`, 실제 `events_dt`, motion scale, nominal cycle step/time을 저장한다.

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
bash v3_chan/run_pick_place.sh
```

한 번 실행하면 최대 3 episode를 수집하고 session별 HDF5와 로그를 새로 저장한다. 정상 수집 실행마다 speed order counter가 자동으로 진행하며 테스트 모드는 counter를 변경하지 않는다. 특정 순서를 재현할 때는 `HRI_SPEED_ORDER_INDEX=0`, `1`, `2`를 지정한다.

수집 후 학습 데이터로 이동하기 전에 다음 validator로 v8 schema, provenance, 세 speed, 공통 layout, row alignment를 확인한다.

```bash
python v3_chan/validate_hri_collection.py v3_chan/trajectories/<session>.hdf5
```
