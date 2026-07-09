# HRI Data Variables

## 1. `v3_chan/errp_markers.csv`

이 파일은 이벤트가 발생한 시점만 기록하는 marker 로그이다.

| 변수명 | 의미 |
|---|---|
| `sim_time` | 이벤트가 발생한 시뮬레이션 시간 |
| `event` | 이벤트 이름 |
| `details` | 이벤트 상세 정보 |

| `event` 값 | 의미 |
|---|---|
| `episode_start` | episode 시작 |
| `episode_end` | episode 종료 |
| `arm_robot_proximity` | 사람 손/팔 proxy가 로봇 링크의 근접 거리 안에 들어옴 |
| `arm_robot_collision` | 사람 손/팔 proxy가 로봇 링크의 접촉 거리 안에 들어옴 |
| `pick_miss` | 로봇 gripper가 cube grasp에 실패한 후보 이벤트 |
| `drop_throw` | cube가 떨어지거나 튕겨나간 후보 이벤트 |
| `collision_green` | 보호 대상 green cube와 충돌한 후보 이벤트 |
| `human_collision` | 기존 human proxy 기준 충돌 후보 이벤트 |
| `stack_failure` | 쌓은 cube가 목표 높이 아래로 떨어진 후보 이벤트 |

## 2. `v3_chan/session_samples.csv`

이 파일은 매 simulation step마다 손과 gripper 사이의 거리 및 충돌 flag를 저장하는 가벼운 시계열 로그이다.

| 변수명 | 의미 |
|---|---|
| `sim_time` | 샘플이 기록된 시뮬레이션 시간 |
| `step` | simulation step index |
| `left_hand_gripper_dist_m` | 왼손 sphere proxy와 gripper 사이 거리 |
| `right_hand_gripper_dist_m` | 오른손 sphere proxy와 gripper 사이 거리 |
| `min_hand_gripper_dist_m` | 양손 중 gripper와 더 가까운 거리 |
| `human_robot_collision` | gripper contact/haptic trigger 기준 충돌 flag |

## 3. `v3_chan/trajectories/hri_vr_sphere_obs.hdf5`

이 파일은 학습에 사용할 수 있는 HDF5 episode dataset이다. 기존 전체 observation인 `obs_policy`와, HRI cognitive safety 연구용 핵심 observation인 `hri_obs_policy`를 함께 저장한다.

### Root Attributes

| 변수명 | 의미 |
|---|---|
| `schema_version` | HDF5 schema 이름 |
| `observation_version` | 원본 observation schema 이름 |
| `observation_dim` | `obs_policy` 차원 |
| `hri_observation_dim` | `hri_obs_policy` 차원 |
| `hri_observation_fields` | `hri_obs_policy`를 구성하는 field 목록 |
| `sample_interval_steps` | 몇 step마다 샘플을 저장했는지 |

### Episode-Level Datasets

경로 형식은 `episodes/episode_000000/...` 이다.

| 변수명 | 의미 |
|---|---|
| `sim_time` | 각 sample의 시뮬레이션 시간 |
| `step` | 각 sample의 simulation step index |
| `obs_policy` | 기존 전체 observation을 flatten한 vector |
| `hri_obs_policy` | HRI cognitive safety 연구용 핵심 observation을 flatten한 vector |
| `human_valid_mask` | `human_head_pos`, `human_left_hand_pos`, `human_right_hand_pos` 유효 여부 |
| `current_pick_idx` | 현재 pick 대상 cube index |
| `completed_picks` | 현재 episode 안에서 완료한 pick 개수 |

### `hri_obs/*`

`hri_obs_policy`에 포함되는 핵심 변수이다.

| 변수명 | 의미 |
|---|---|
| `robot_joint_pos` | Panda arm joint position |
| `robot_joint_vel` | Panda arm joint velocity |
| `gripper_width` | Franka gripper finger joint position 합 |
| `ee_pos` | end-effector world position |
| `ee_quat` | end-effector world quaternion |
| `cube_pos` | 현재 pick cube world position |
| `cube_quat` | 현재 pick cube world quaternion |
| `place_target_pos` | place target world position |
| `ee_to_cube` | `cube_pos - ee_pos` |
| `cube_to_place_target` | `place_target_pos - cube_pos` |
| `ee_to_place_target` | `place_target_pos - ee_pos` |
| `human_head_pos` | HMD/head world position |
| `human_left_hand_pos` | 왼손 sphere proxy world position |
| `human_right_hand_pos` | 오른손 sphere proxy world position |
| `ee_to_left_hand` | `human_left_hand_pos - ee_pos` |
| `ee_to_right_hand` | `human_right_hand_pos - ee_pos` |
| `min_hand_gripper_dist` | 양손 중 gripper와 가장 가까운 거리 |
| `human_robot_collision` | 사람 손 proxy와 gripper/robot 충돌 flag |
| `near_human` | 손이 인지적 안전 근접 거리 안에 있는지 |
| `has_grasped_cube` | 현재 cube grasp 추정 flag |
| `task_phase` | task phase one-hot |
| `controller_event` | PickPlaceController event one-hot |

### `obs/*`

기존 전체 observation schema이다. `hri_obs/*`보다 넓은 호환용 field를 포함한다.

| 변수명 | 의미 |
|---|---|
| `robot_joint_pos` | Panda arm joint position |
| `robot_joint_vel` | Panda arm joint velocity |
| `gripper_width` | Franka gripper finger joint position 합 |
| `ee_pos` | end-effector world position |
| `ee_quat` | end-effector world quaternion |
| `cube_pos` | 현재 pick cube world position |
| `cube_quat` | 현재 pick cube world quaternion |
| `cube_lin_vel` | 현재 pick cube linear velocity |
| `cube_ang_vel` | 현재 pick cube angular velocity |
| `place_target_pos` | place target world position |
| `ee_to_cube` | `cube_pos - ee_pos` |
| `cube_to_place_target` | `place_target_pos - cube_pos` |
| `ee_to_place_target` | `place_target_pos - ee_pos` |
| `human_head_pos` | HMD/head world position |
| `human_left_hand_pos` | 왼손 sphere proxy world position |
| `human_right_hand_pos` | 오른손 sphere proxy world position |
| `ee_to_left_hand` | `human_left_hand_pos - ee_pos` |
| `ee_to_right_hand` | `human_right_hand_pos - ee_pos` |
| `min_hand_gripper_dist` | 양손 중 gripper와 가장 가까운 거리 |
| `human_robot_collision` | 사람 손 proxy와 gripper/robot 충돌 flag |
| `near_human` | 손이 인지적 안전 근접 거리 안에 있는지 |
| `collision_green` | green cube 충돌 flag |
| `pick_miss_recent` | 최근 pick miss flag |
| `drop_throw_recent` | 최근 drop/throw flag |
| `has_grasped_cube` | 현재 cube grasp 추정 flag |
| `task_phase` | task phase one-hot |
| `controller_event` | PickPlaceController event one-hot |
| `controller_t` | controller event progress |

### `human/*`

사람 위치 raw trajectory를 따로 모아둔 group이다.

| 변수명 | 의미 |
|---|---|
| `head_pos` | HMD/head world position |
| `left_hand_pos` | 왼손 sphere proxy world position |
| `right_hand_pos` | 오른손 sphere proxy world position |

### `safety/*`

사람의 인지적 안전 및 물리적 안전과 관련된 proxy metric이다.

| 변수명 | 의미 |
|---|---|
| `left_hand_gripper_dist_m` | 왼손 sphere proxy와 gripper 사이 거리 |
| `right_hand_gripper_dist_m` | 오른손 sphere proxy와 gripper 사이 거리 |
| `min_hand_gripper_dist_m` | 양손 중 gripper와 가장 가까운 거리 |
| `near_human` | 손이 인지적 안전 근접 거리 안에 있는지 |
| `human_robot_collision` | 사람 손 proxy와 gripper/robot 충돌 flag |
| `haptic_pulse_left` | 왼손 bHaptics pulse trigger 여부 |
| `haptic_pulse_right` | 오른손 bHaptics pulse trigger 여부 |
| `gripper_gap_left_m` | 왼손 proxy와 gripper haptic proxy 사이 signed gap |
| `gripper_gap_right_m` | 오른손 proxy와 gripper haptic proxy 사이 signed gap |
| `gripper_penetration_left_m` | 왼손 proxy와 gripper haptic proxy의 penetration depth |
| `gripper_penetration_right_m` | 오른손 proxy와 gripper haptic proxy의 penetration depth |

### `errp/*`

ErrP reward shaping을 위한 group이다. 현재는 실시간 EEG 장비가 없으므로 event-aligned replay 또는 classifier replay가 값을 채우기 전까지 placeholder로 저장된다.

| 변수명 | 의미 |
|---|---|
| `label` | ErrP label |
| `feedback` | reward shaping에 사용할 ErrP feedback scalar |
| `uncertainty` | ErrP classifier/replay uncertainty |
| `timestamp` | EEG 또는 replay event timestamp |
| `aligned_step` | ErrP가 align된 simulation step |

### `actions/*`

로봇 controller가 낸 action을 저장하는 group이다.

| 변수명 | 의미 |
|---|---|
| `robot_joint_positions` | controller action의 joint position target |
| `robot_joint_velocities` | controller action의 joint velocity target |
| `robot_joint_efforts` | controller action의 joint effort target |

### `rewards/*`

RL reward shaping을 위한 group이다.

| 변수명 | 의미 |
|---|---|
| `task` | pick-and-place task reward |
| `safety` | 거리/충돌 기반 safety reward 또는 penalty |
| `errp` | ErrP feedback 기반 reward 또는 penalty |
| `total` | `task + safety + errp` |

### `task/*`

작업 진행 상태를 따로 모아둔 group이다.

| 변수명 | 의미 |
|---|---|
| `current_pick_idx` | 현재 pick 대상 cube index |
| `completed_picks` | 현재 episode 안에서 완료한 pick 개수 |
| `has_grasped_cube` | 현재 cube grasp 추정 flag |
| `controller_event` | PickPlaceController event index |
