# HRI Data Variables

## 1. `v3_chan/logs/errp_markers_<session_id>.csv`

이 파일은 이벤트가 발생한 시점만 기록하는 marker 로그이다.

| 변수명 | 의미 |
|---|---|
| `session_id` | 수집 실행 단위의 고유 ID |
| `episode_index` | 해당 session 안의 episode index |
| `sim_time` | 이벤트가 발생한 시뮬레이션 시간 |
| `step` | simulation step index |
| `monotonic_time_ns` | 프로세스 monotonic clock. 내부 event 순서와 시간 간격 계산용 |
| `wall_time_unix_ns` | Unix epoch 기준 wall-clock time. EEG/외부 marker 정렬용 |
| `event` | 이벤트 이름 |
| `details` | 이벤트 상세 정보. 사람-로봇 event에는 hand, collider path와 `surface_gap_m` 포함 |

| `event` 값 | 의미 |
|---|---|
| `episode_start` | episode 시작 |
| `episode_end` | episode 종료 |
| `arm_robot_proximity` | 손 sphere와 distal Panda built-in collider의 surface gap이 `0.05 m` 이하 |
| `arm_robot_collision` | 손 sphere와 distal Panda built-in collider가 실제 overlap/contact |
| `pick_miss` | 로봇 gripper가 cube grasp에 실패한 후보 이벤트 |
| `drop_throw` | cube가 떨어지거나 튕겨나간 후보 이벤트 |
| `collision_green` | 보호 대상 green cube와 충돌한 후보 이벤트 |
| `stack_failure` | 쌓은 cube가 목표 높이 아래로 떨어진 후보 이벤트 |
| `pick_attempt_success` | grasp, lift, 목표 위치 및 안정성 검증을 통과한 pick-place attempt |
| `pick_attempt_failed` | 실제 place 검증에 실패하여 같은 cube를 재시도하는 attempt |

## 2. `v3_chan/logs/session_samples_<session_id>.csv`

이 파일은 매 simulation step마다 손과 gripper 사이의 거리 및 충돌 flag를 저장하는 가벼운 시계열 로그이다.

| 변수명 | 의미 |
|---|---|
| `sim_time` | 샘플이 기록된 시뮬레이션 시간 |
| `step` | simulation step index |
| `session_id` | 수집 실행 단위의 고유 ID |
| `episode_index` | 해당 session 안의 episode index |
| `monotonic_time_ns` | 프로세스 monotonic clock. 내부 sample 순서와 시간 간격 계산용 |
| `wall_time_unix_ns` | Unix epoch 기준 wall-clock time. EEG/외부 stream 정렬용 |
| `left_hand_gripper_dist_m` | 왼손 sphere proxy와 gripper 사이 거리 |
| `right_hand_gripper_dist_m` | 오른손 sphere proxy와 gripper 사이 거리 |
| `min_hand_gripper_dist_m` | 양손 중 gripper와 더 가까운 거리 |
| `human_robot_collision` | 손 sphere와 distal Panda built-in collider의 overlap 기준 충돌 flag |

## 3. `v3_chan/trajectories/hri_vr_sphere_surface_twist_v8_dualclock_tracked_v1_<session_id>.hdf5`

이 파일은 학습에 사용할 수 있는 HDF5 episode dataset이다. 기존 전체 observation인 `obs_policy`와, HRI cognitive safety 연구용 핵심 observation인 `hri_obs_policy`를 함께 저장한다.

### Root Attributes

| 변수명 | 의미 |
|---|---|
| `schema_version` | HDF5 schema 이름 |
| `experiment_metadata_schema_version` | 실험 metadata schema. 현재 `hri_experiment_metadata_v3` |
| `observation_version` | 원본 observation schema 이름 |
| `observation_dim` | `obs_policy` 차원 |
| `hri_observation_version` | 83D safety observation schema 이름 |
| `hri_observation_dim` | `hri_obs_policy` 차원 |
| `hri_observation_fields` | `hri_obs_policy`를 구성하는 field 목록 |
| `sample_interval_steps` | 몇 step마다 샘플을 저장했는지 |
| `controller_speed_schedule` | 해당 session에 적용된 세 episode의 speed 순서 |
| `controller_speed_counterbalance_order_index` | counterbalance 순서 번호. `0`, `1`, `2` |
| `controller_speed_counterbalance_orders_json` | 전체 counterbalance 순서 정의 |
| `controller_speed_profiles_json` | profile별 motion scale, 실제 `events_dt`, nominal cycle 길이 |
| `participant_id` | 참가자 가명. 현재 본수집은 `P01` |
| `participant_session_index` | 해당 참가자의 수집 세션 번호 |
| `participant_handedness` | self-reported handedness |
| `is_practice` | practice session 여부 (`0/1`) |
| `experiment_condition` | 실험 조건 이름 |
| `experiment_block_id` | 참가자 내 실험 block 식별자 |
| `haptic_experiment_condition` | 햅틱 조건 이름 |
| `haptics_enabled` | 실제 햅틱 UDP 전송 활성화 여부 (`0/1`) |
| `haptics_udp_configured` | UDP endpoint 설정 여부 (`0/1`) |
| `haptics_intensity` | UDP bridge로 전송한 햅틱 강도 (`0..100`) |
| `haptics_min_interval_s` | 같은 장갑에 허용하는 최소 pulse 간격 |
| `haptics_contact_min_steps` | pulse 전 필요한 연속 contact step 수 |
| `xr_anchor_status` | XR camera/anchor 적용 결과 또는 실패 상태 |
| `source_tree_sha256` | Git checkout 유무와 독립적인 실행 source tree hash |

production validator는 각 episode에서 왼손과 오른손의 `pose_valid` 비율이 각각 기본 `0.90` 이상이고, `real_time_factor_valid` 비율이 `0.95` 이상인지 확인한다. `xr_anchor_status`는 `xr_anchor` 또는 `xr_camera_teleport`처럼 실제 적용을 나타내야 한다. tracking이나 RTF가 전 구간 무효인 파일, 햅틱 condition과 실제 활성화 상태가 다른 파일은 학습 후보로 승인하지 않는다.

### Episode-Level Datasets

경로 형식은 `episodes/episode_000000/...` 이다.

| 변수명 | 의미 |
|---|---|
| `sim_time` | 각 sample의 시뮬레이션 시간 |
| `monotonic_time_ns` | frame sample의 monotonic clock timestamp |
| `pose_monotonic_time_ns` | XR pose 획득 직후의 monotonic timestamp; 실제 시간 속도와 RTF의 기준 |
| `wall_time_unix_ns` | EEG 및 외부 stream 정렬용 Unix epoch timestamp |
| `real_time_factor` | `delta_sim_time / delta_monotonic_time` |
| `action_command_monotonic_ns` | `next_commanded_action_t`를 controller에 전달한 시각 |
| `step` | 각 sample의 simulation step index |
| `obs_policy` | 기존 전체 observation을 flatten한 vector |
| `hri_obs_policy` | HRI cognitive safety 연구용 핵심 observation을 flatten한 vector |
| `human_valid_mask` | `human_head_pos`, `human_left_hand_pos`, `human_right_hand_pos` 유효 여부 |
| `current_pick_idx` | 현재 pick 대상 cube index |
| `completed_picks` | 현재 episode 안에서 완료한 pick 개수 |

새 수집 데이터에서 episode 길이는 고정값이 아니다. 실제 place 검증에 실패하면 같은 cube를 재시도하므로, 3회보다 많은 pick-and-place controller attempt가 포함될 수 있다.

session별 episode 속도 순서는 `slow -> medium -> fast`, `medium -> fast -> slow`, `fast -> slow -> medium`으로 순환한다. 한 session의 세 속도 조건은 동일한 초기 cube pose와 `layout_id`를 공유하며, session이 바뀔 때만 layout이 바뀐다. 각 episode attribute에는 해당 session schedule과 `controller_speed_profile`, `controller_motion_phase_scale`, `controller_events_dt_json`, `controller_nominal_cycle_steps`, `controller_nominal_cycle_duration_s`를 기록한다. 속도 변화는 이동 phase에만 적용하며 grasp/release와 안정화 phase timing은 바꾸지 않는다.

새 recorder schema는 `hri_obs_v8_dual_clock_tracked_action_aligned`이며, safety observation schema는 `hri_policy_obs_v1_83d_surface_gap`이다. `obs_policy`는 기존 robot-only policy와의 호환을 위해 84차원을 유지하고, `hri_obs_policy`는 gripper 중심거리 field를 제외한 83차원을 저장한다. v8은 policy 차원을 바꾸지 않고 pose provenance, dual-clock dynamics, action alignment, initial scene snapshot을 추가한다. 기존 v1-v7 파일은 과거 데이터로 그대로 유지된다.

`human/*`에는 wall-time 기준 손 속도와 함께 pose valid, `position_tracked`(-1/0/1), tracking-status-known, source ID/name/path, source-switch, 획득 monotonic time을 저장한다. `safety/*`의 동적 값은 wall-time canonical이며, `dynamic_sim/*`에 simulation-time 진단값을 별도로 저장한다. validity는 tracking, gap measurement/rate, robot surface velocity, relative velocity, closing speed, TTC로 분리한다. 이 값들은 policy observation에 포함되지 않는다.

surface 기반 flag의 기본 기준은 다음과 같다.

| 변수명 | 기본 기준 |
|---|---|
| `human_robot_collision` | 실제 PhysX overlap 또는 `min_hand_end_effector_surface_gap <= 0.0 m` |
| `near_miss` | `0.0 m < min_hand_end_effector_surface_gap <= 0.02 m` |
| `near_human` | `min_hand_end_effector_surface_gap <= 0.05 m` |
| `distance_gate` | `clip((0.13 - gap) / (0.13 - 0.05), 0, 1)` |

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
| `min_hand_gripper_dist` | 호환용 canonical signed surface gap. v2 surface observation부터 음수는 겹침을 뜻함 |
| `min_hand_gripper_surface_gap` | 호환용 이름. 손 sphere와 distal end-effector collider 사이 signed surface gap |
| `human_robot_collision` | 사람 손 proxy와 distal end-effector collider 충돌 flag |
| `near_human` | 손이 인지적 안전 근접 거리 안에 있는지 |
| `near_miss` | 접촉하지 않았지만 surface gap이 near-miss 범위 안인지 |
| `left_hand_end_effector_surface_gap` | 왼손 sphere와 distal Panda built-in collider 사이 최소 signed gap |
| `right_hand_end_effector_surface_gap` | 오른손 sphere와 distal Panda built-in collider 사이 최소 signed gap |
| `min_hand_end_effector_surface_gap` | 양손 중 최소 signed gap |
| `left_hand_contact` | 왼손 PhysX overlap/contact flag |
| `right_hand_contact` | 오른손 PhysX overlap/contact flag |
| `distance_gate` | 공통 safety residual gate |
| `geometry_valid` | collider query와 손 tracking 유효 여부 |
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
| `min_hand_gripper_dist` | 호환용 canonical signed surface gap. `obs_policy`의 84차원 위치는 유지됨 |
| `min_hand_gripper_center_dist` | 양손 sphere 중심 중 gripper 중심과 가장 가까운 Euclidean 거리. flatten된 `obs_policy`에는 제외됨 |
| `min_hand_gripper_surface_gap` | 손 sphere와 gripper geometry 사이 signed surface gap. flatten된 `obs_policy`에는 제외됨 |
| `human_robot_collision` | 사람 손 proxy와 gripper/robot 충돌 flag |
| `near_human` | 손이 인지적 안전 근접 거리 안에 있는지 |
| `near_miss` | 접촉하지 않았지만 surface gap이 near-miss 범위 안인지. flatten된 `obs_policy`에는 제외됨 |
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
| `min_hand_gripper_dist_m` | canonical signed surface gap. `min_hand_gripper_surface_gap_m`과 동일한 의미 |
| `min_hand_gripper_center_dist_m` | 양손 sphere 중심 중 gripper 중심과 가장 가까운 거리 |
| `min_hand_gripper_surface_gap_m` | 호환용 이름. 손 sphere와 distal end-effector collider 사이 signed surface gap |
| `near_human` | 손이 인지적 안전 근접 거리 안에 있는지 |
| `near_miss` | 접촉 없이 surface gap이 near-miss 범위 안인지 |
| `human_robot_collision` | 사람 손 proxy와 distal end-effector collider 충돌 flag |
| `haptic_pulse_left` | 왼손 bHaptics pulse trigger 여부 |
| `haptic_pulse_right` | 오른손 bHaptics pulse trigger 여부 |
| `end_effector_surface_gap_m` | 양손과 distal Panda built-in collider 사이 최소 signed gap |
| `left_end_effector_surface_gap_m` | 왼손 최소 signed gap |
| `right_end_effector_surface_gap_m` | 오른손 최소 signed gap |
| `closest_human_hand_id` | closest hand ID (`0=none, 1=left, 2=right`) |
| `closest_robot_link_id` | HDF5 root metadata의 링크 ID |
| `closest_collider_id` | HDF5 root metadata의 collider ID |
| `contact_active` | 양손 중 하나라도 contact인지 |
| `contact_force_n` | available contact magnitude. query sphere에서는 `0` |
| `penetration_depth_m` | 최소 gap 기준 penetration depth |
| `distance_gate` | safety residual gate |
| `geometry_valid` | 손 tracking 및 geometry query 유효 여부 |
| `gripper_gap_left_m` | v3 reader 호환용 왼손 built-in collider signed gap alias |
| `gripper_gap_right_m` | v3 reader 호환용 오른손 built-in collider signed gap alias |

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

각 행의 `state_t`와 시간적으로 정렬된 controller action을 저장하는 group이다.

| 변수명 | 의미 |
|---|---|
| `previous_applied_joint_*` | 현재 `state_t`를 만들기 직전에 적용되어 있던 `action_(t-1)` |
| `previous_applied_valid` | 이전 action 저장 유효 여부 |
| `next_commanded_joint_*` | `state_t`에서 controller가 계산해 다음 physics step에 적용한 `action_t` |
| `next_commanded_valid` | 다음 action 저장 유효 여부 |

### `initial_scene/*`

각 episode 시작 직후의 여섯 cube pose, role, place target pose, session/layout seed, 고정 `layout_id`, 실제 pose hash를 저장한다. 같은 session의 slow/medium/fast episode는 이 값이 같아야 하며 `validate_hri_collection.py`가 이를 검사한다.

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
| `attempt_index` | episode 안에서 현재 수행 중인 전체 attempt 번호. 1부터 시작 |
| `current_cube_attempt` | 현재 cube에 대한 attempt 번호. 실패 후 재시도할 때 증가 |
| `failed_attempts` | 현재 step까지 누적된 실제 place 검증 실패 횟수 |
| `has_grasped_cube` | 현재 cube grasp 추정 flag |
| `controller_event` | PickPlaceController event index |

Episode attribute의 `success=True`는 세 cube가 robot grasp, 5 cm 이상 lift, 목표 위치 및 안정성 검증을 모두 통과했다는 뜻이다. `attempts`와 `failed_attempts`에는 해당 episode의 전체 시도 및 재시도 횟수가 기록된다.
