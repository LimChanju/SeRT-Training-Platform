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

- HDF5 schema: `hri_obs_v4_builtin_panda_collision_geometry`
- 기존 84차원 `obs_policy`는 robot-only checkpoint 호환을 위해 유지한다.
- 양손 surface gap, closest link/collider ID, contact, penetration, near, near-miss, gate, haptic pulse, geometry validity를 매 step 저장한다.
- 기존 HDF5는 덮어쓰지 않고 session별 새 파일을 생성한다.

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

한 번 실행하면 최대 3 episode를 수집하고 session별 HDF5와 로그를 새로 저장한다.
