# Built-in Panda Safety Geometry

## 목적

VR 데이터 수집, human replay, RL 학습 환경, rollout 평가가 같은 사람-로봇 안전 geometry를 사용하도록 통합했다. 로봇 링크 중심거리, visual mesh, 수동 capsule은 최종 안전 판정에 사용하지 않는다.

## 실제 Stage 조사 결과

Isaac Sim 4.5의 `omni.isaac.franka.Franka`가 `/World/Franka`에 생성한 composed Stage를 `UsdPhysics.CollisionAPI`로 순회해 확인했다.

| 논리 링크 | 실제 collider prim | PhysX approximation |
|---|---|---|
| `panda_link6` | `/World/Franka/panda_link6/geometry/panda_link6` | `convexHull` |
| `panda_link7` | `/World/Franka/panda_link7/geometry/panda_link7` | `convexHull` |
| `panda_link8` | 없음 | 없음 |
| `panda_hand` | `/World/Franka/panda_hand/geometry/panda_hand` | `convexHull` |
| `panda_leftfinger` | `/World/Franka/panda_leftfinger/geometry/panda_leftfinger` | `convexDecomposition` |
| `panda_rightfinger` | `/World/Franka/panda_rightfinger/geometry/panda_rightfinger` | `convexDecomposition` |

`/World/Franka/panda_link8` prim 자체는 존재하지만 이 asset에는 `CollisionAPI`가 적용된 자체 geometry가 없다. 원본 asset을 바꾸거나 임의 형상을 추가하지 않았으며, 실행 로그와 HDF5의 `safety_missing_links_json`에 이를 명시한다. 실제 flange/gripper body의 물리 표면은 asset의 인접한 `panda_hand` collider로 판정된다. 따라서 이 asset에서는 접촉 라벨이 `panda_link8`이 아니라 `panda_hand`로 기록될 수 있다.

실행 시 모든 collider path, shape, contact/rest offset과 누락 링크를 한 번 출력한다.

이 asset에서 `panda_link8`과 `panda_hand`의 world transform은 동일한 것도 Stage smoke test로 확인한다. 따라서 link8 위치의 실제 mounting/flange 표면은 `panda_hand` collider로 감지되며, 이를 억지로 `panda_link8`이라고 다시 라벨링하지 않는다.

## 손과 Surface Gap

- 왼손과 오른손은 VR tracking 위치의 구로 표현한다.
- 시각 구와 query 구는 모두 `HRI_HAND_PROXY_RADIUS_M`을 사용한다.
- 기본 반지름은 `0.035 m`이다.
- 로봇 쪽은 위의 실제 PhysX collider만 사용한다.

PhysX `overlap_sphere`의 query 반지름을 이분 탐색해 손 중심에서 가장 가까운 collider 표면 거리 `r_nearest`를 얻는다.

```text
surface_gap = r_nearest - hand_radius
```

- `surface_gap > 0`: 분리
- `surface_gap == 0`: 접촉 경계
- `surface_gap < 0`: overlap/penetration

손 tracking이 없거나 query가 실패하면 `geometry_valid=0`이며 collision으로 처리하지 않는다. 기본 탐색 범위는 `HRI_GEOMETRY_MAX_QUERY_GAP_M=2.0`이다.

기본 14회 이분 탐색과 `0.00025 m` tolerance를 사용한다. hard contact는 손 반지름 그대로의 `overlap_sphere` 결과가 authoritative하며, 거리 탐색 오차로 contact가 빠지지 않는다.

현재 손 구는 로봇을 밀지 않는 non-physical query sphere이다. 따라서 `contact_active`는 손 반지름의 실제 PhysX overlap으로 판정하지만 물리 contact force는 생성되지 않는다. HDF5에는 `contact_force_n=0`, `contact_force_valid_*=0`으로 정직하게 기록한다.

## 공통 Threshold

| 판정 | 기본값 |
|---|---:|
| hard collision | `contact` 또는 `surface_gap <= 0.0 m` |
| near-miss | `0.0 m < surface_gap <= 0.02 m` |
| near | `surface_gap <= 0.05 m` |
| gate 시작 | `surface_gap < 0.13 m` |
| gate 최대 | `surface_gap <= 0.05 m` |

```text
distance_gate = clip((0.13 - surface_gap) / (0.13 - 0.05), 0, 1)
```

기준은 `end_effector_safety_geometry.py` 한 곳에서 관리하며 환경변수로 변경할 수 있다.

## 동일 데이터 흐름

`PandaEndEffectorSafetyRuntime.evaluate(left_pos, right_pos)`의 한 결과를 다음 경로가 함께 사용한다.

1. `main.py`의 near/collision event 로그
2. 왼손/오른손 bHaptics UDP pulse
3. 84차원 호환 `obs_policy`의 기존 safety field
4. 확장된 `hri_obs_policy`와 HDF5 `safety/*`
5. replay된 손 위치와 현재 rollout Panda pose의 동적 재계산
6. rollout step CSV와 episode JSON/CSV 평가 지표

수집 당시의 `recorded_human_robot_collision`과 `recorded_near_human`은 replay metadata로만 보존되고 현재 rollout 라벨로 재사용되지 않는다.

## HDF5 v4

새 schema는 `hri_obs_v4_builtin_panda_collision_geometry`이다.

- `obs_policy`: 기존 robot-only checkpoint 호환을 위해 `84`차원 유지
- `hri_obs_policy`: 새 geometry 보조 field를 포함해 `84`차원
- 기존 v1/v2/v3 HDF5는 수정하지 않음
- 기존 schema 파일 경로를 재사용하면 v4용 새 파일로 자동 분기

주요 신규 `safety/*` dataset:

```text
end_effector_surface_gap_m
left_end_effector_surface_gap_m
right_end_effector_surface_gap_m
closest_human_hand_id
closest_robot_link_id
closest_collider_id
contact_active
contact_force_n
contact_force_valid_left
contact_force_valid_right
penetration_depth_m
near_human
near_miss
human_robot_collision
distance_gate
haptic_pulse_left
haptic_pulse_right
geometry_valid
```

링크·collider·손 ID 매핑과 실제 collider property는 HDF5 root attribute의 JSON metadata에 저장된다.

## Debug 및 검증

실제 collider 구조와 smoke/performance test:

```bash
cd /home/railab/Desktop/Isaac_HRC
ISAAC_XR_MODE=off ISAAC_HEADLESS=1 \
  ./launch_isaac.sh "$PWD/v3_chan/inspect_panda_collision_geometry.py"
```

수집 화면에서 PhysX collider visualization 활성화:

```bash
export HRI_SHOW_PHYSX_COLLIDERS=1
export DEBUG_HRI_SAFETY_GEOMETRY=1
export HRI_DEBUG_SAFETY_VISUALIZATION=1
bash v3_chan/run_pick_place.sh
```

`HRI_SHOW_PHYSX_COLLIDERS`는 계산에 쓰이는 정확한 PhysX collider를 표시한다. `HRI_DEBUG_SAFETY_VISUALIZATION`은 손 상태를 collision/near/gate 색으로 덧그리고, 손에서 현재 closest collider의 extent center까지 association line과 marker를 표시한다. 정확한 gap과 상태는 throttled `SafetyGeometryDBG` 로그에서 함께 확인한다.

smoke test에서 link6, link7, hand, left finger, right finger collider 중심에 놓인 query sphere가 모두 해당 링크의 contact로 검출됐다. tracking loss도 `geometry_valid=0`, `collision=0`, `near=0`으로 검증했다. 14회 탐색 기준 contact 위치 5개의 평균 판정 시간은 `0.0908 ms`, 일반적인 두 손 위치의 geometry 계산은 `0.288 ms/frame`, 개별 PhysX overlap query 평균은 `0.0071 ms`였다. 같은 headless physics benchmark는 safety query 없이 `0.172 ms/frame`(`5811.5 FPS`), query 포함 `0.438 ms/frame`(`2281.3 FPS`)였다. 이는 렌더링/SteamVR을 포함하지 않은 동일 실행 내 geometry overhead 비교값이며, 실제 collider 방식을 유지하기에 충분하다.

## 데이터 호환성

과거 dataset은 robot-only task policy 비교 및 사람 raw trajectory replay에는 사용할 수 있다. 다만 v4 safety label과 직접 섞지 않는다. 새 policy/geometry를 평가할 때는 과거 파일의 손 위치만 replay하고 collision, near, gate는 현재 Panda collider로 다시 계산한다. 실제 ErrP/햅틱 실험용 safety label은 v4로 다시 수집해야 한다.

현재 수집 PC checkout의 `v3_chan/rl`에는 개인 학습 서버에만 있는 `actions.py`, `rewards.py`, `policies.py`, `pick_place_phase.py`가 포함되어 있지 않다. 따라서 이 PC에서는 full RL rollout smoke를 실행하지 못했으며, 개인 서버에서 해당 모듈을 유지한 채 이번 변경의 `end_effector_safety_geometry.py`, `end_effector_safety_runtime.py`, `rl/observations.py`, `rl/pick_place_env.py`, `rl/human_replay.py`, `evaluate_rollout_policy.py`를 함께 반영해야 한다.

`train_rl.py`에는 기존 robot-only residual을 바꾸지 않는 `--residual-gate-mode none`과 공통 surface-gap gate를 action에 곱하는 `--residual-gate-mode distance`를 추가했다. 기존 checkpoint는 gate metadata가 없으므로 평가 시 자동으로 `none`이 된다. `distance` 모드는 현재 trainer에서 direct 또는 미리 합성된 frozen task-policy checkpoint를 base로 받을 때만 사용한다. 기존 `BC + task residual PPO`를 다시 safety residual의 base로 쌓으려면 개인 서버에서 두 단계 task policy 전체를 하나의 frozen `pi_task`로 평가해야 하며, task residual actor 하나만 base로 넣으면 안 된다.
