# Pick-and-Place Cartesian Speed Control

## 적용 범위

이 구현은 Isaac Sim 4.5의 `v3_chan` VR-HRI 데이터 수집 환경 전용이다. 기존 `PickPlaceController`의 10단계 FSM, grasp gate, surface-gap 판정, collision/haptic 처리, slow/medium/fast counterbalancing을 유지하면서 end-effector 이동 명령에 실제 m/s 상한을 적용한다.

84D `obs_policy`, 83D `hri_obs_policy`, ErrP/pseudo-ErrP 로직은 변경하지 않는다. 속도 정보는 HDF5의 별도 `controller/` 그룹에 기록한다.

## FSM 단계와 속도

| Event | Isaac 단계 | 속도 단계 |
|---:|---|---|
| 0 | cube 위로 이동 | `approach_place` |
| 1 | grasp 높이로 하강 | `approach_place` |
| 2 | grasp 전 settling | limiter 비활성, 기존 시간 유지 |
| 3 | gripper close | limiter 비활성, 기존 시간 유지 |
| 4 | grasp 후 lift | `transport` |
| 5 | place target으로 수평 이동 | `transport` |
| 6 | place 높이로 하강 | `approach_place` |
| 7 | gripper open | limiter 비활성, 기존 시간 유지 |
| 8 | 배치 후 retract | `transport` |
| 9 | 안전한 복귀/종료 | `transport` |

| Profile | `transport` | `approach_place` | `near_human` |
|---|---:|---:|---:|
| `slow` | 0.30 m/s | 0.10 m/s | 0.05 m/s |
| `medium` | 0.40 m/s | 0.15 m/s | 0.10 m/s |
| `fast` | 0.50 m/s | 0.20 m/s | 0.15 m/s |

## Distance-Gate Blending

기존 built-in Panda PhysX collider query에서 계산한 양손 surface-gap 기반 `distance_gate`를 그대로 재사용한다. 추가 PhysX closest-point query는 수행하지 않는다.

```text
gate = clip((0.13 - surface_gap) / (0.13 - 0.05), 0, 1)
active_target_limit = (1 - gate) * phase_limit + gate * near_human_limit
```

- `surface_gap >= 0.13 m`: 정상 단계 상한
- `0.05 m < surface_gap < 0.13 m`: 연속 선형 감속
- `surface_gap <= 0.05 m`: profile별 `near_human` 상한
- collision/penetration에서는 기존 gate가 1이므로 near-human 상한을 적용한다.
- tracking/query invalid의 기존 `geometry_valid=0`, `distance_gate=0` 의미는 변경하지 않는다.

## 실제 제어 경로

1. 기존 RMPFlow가 다음 arm joint position/velocity target을 생성한다.
2. 현재 7D arm joint와 후보 target을 Lula FK의 실제 `right_gripper` end-effector frame으로 변환한다.
3. 후보 EE 변위가 `active_speed_limit_mps * simulation_dt`를 넘으면 joint target을 현재 joint 쪽으로 보간한다.
4. 24회 binary search 후 FK 변위를 다시 계산해 상한 충족 여부를 검증한다.
5. joint velocity target도 같은 비율로 축소한다.
6. motion event는 실제 EE가 최종 waypoint 0.005 m 이내에 도달할 때까지 유지한다. 따라서 limiter 때문에 arm이 늦어져도 FSM만 먼저 grasp/release 단계로 넘어가지 않는다.

제어 `dt`는 `world.current_time` 차이만 사용한다. 첫 frame, reset 직후, 역전/중복 timestamp, 기본 범위 `1e-5~0.1 s` 밖의 dt에서는 해당 command를 진행하지 않고 estimator/ramp를 다시 준비한다.

상한 증가에는 기본 acceleration `1.0 m/s^2`, jerk `10.0 m/s^3` ramp를 적용한다. 단계 전환이나 distance gate로 상한이 낮아질 때는 안전 상한을 초과하지 않도록 즉시 낮은 상한을 적용한다. 정상적인 0.13~0.05 m gate 구간은 gate 자체가 연속이므로 감속 상한도 연속이다.

## HDF5

Schema는 `hri_obs_v9_cartesian_speed_aligned`, speed schedule은 `pick_place_cartesian_speed_v1`이다.

주요 root/episode metadata:

- `controller_speed_profile`
- `transport_speed_limit_mps`
- `approach_place_speed_limit_mps`
- `near_human_speed_limit_mps`
- `controller_speed_gate_start_gap_m`, `controller_speed_gate_full_gap_m`
- `controller_speed_gate_blending`
- `controller_speed_acceleration_limit_mps2`, `controller_speed_jerk_limit_mps3`
- `controller_speed_action_limit_method`
- `controller_speed_end_effector_frame_name=right_gripper`
- `controller_speed_schedule_version`, `controller_speed_step_schema_version`

주요 step dataset은 `controller/` 아래에 저장한다.

- `controller_event`, `controller_speed_profile`
- `nominal_phase_mode`, `speed_limit_mode`
- `distance_gate`, `surface_gap_m`, `geometry_valid`
- `phase_speed_limit_mps`, `near_human_speed_limit_mps`
- `blended_speed_limit_mps`, `active_speed_limit_mps`
- `ramp_acceleration_mps2`, `sim_dt_s`, `control_dt_valid`
- `ee_velocity_sim_world_mps`, `ee_speed_sim_mps`, `ee_speed_sim_valid`
- `ee_velocity_wall_world_mps`, `ee_speed_wall_mps`, `ee_speed_wall_valid`
- `command_scale`, `command_predicted_ee_displacement_m`
- `command_full_target_ee_displacement_m`, `command_cartesian_speed_mps`
- `command_limit_valid`, `command_was_limited`
- `waypoint_distance_m`, `waypoint_reached`

`ee_speed_sim_*`는 simulation timestamp 기준 제어/검증 값이다. `ee_speed_wall_*`는 monotonic wall-clock 기준 관측값이며 제어 상한 계산에는 사용하지 않는다.

## 검증

```bash
/home/railab/isaac-sim-4.5.0/python.sh -m pytest -o addopts='' -q tests
/home/railab/isaac-sim-4.5.0/python.sh v3_chan/validate_hri_collection.py <session.hdf5>
/home/railab/isaac-sim-4.5.0/python.sh v3_chan/trajectory_metrics.py <session.hdf5> --aggregate-only
```

validator는 reset/비정상 timestamp frame을 valid flag로 제외한다. FK command에는 절대 허용오차 0.005 m/s, 실제 simulation-time EE 속도에는 `max(0.02 m/s, active_limit의 5%)` 허용오차를 적용한다.

## 제한 및 하드웨어 경고

- 속도 상한은 Isaac의 kinematic target과 simulated articulation을 위한 것이다. safety-rated monitor가 아니다.
- 실제 EE는 PD tracking, 관성, contact impulse로 command target을 일시적으로 초과할 수 있으므로 HDF5 실측값과 validator를 반드시 확인해야 한다.
- `PickPlaceController`가 progress setter를 제공하지 않아 Isaac 4.5에서 확인한 `_event`, `_t` 내부 필드로 최종 waypoint를 유지한다. Isaac 버전을 바꾸면 smoke test를 다시 수행해야 한다.
- `v3_chan/trajectory_metrics.py`는 이번에 추가했지만, 요청 목록의 `v3_chan/rl/safemotions_style.py`는 이 저장소에 존재하지 않으며 이번 수집-only 변경에서는 생성하지 않았다.

이 설정을 실제 Franka 하드웨어에 바로 배포하면 안 된다. 실제 하드웨어 배포 전 별도의 위험성 평가, 안전정격 속도 제한, 정지거리 검증, 사람 검출 실패 대응, 비상정지 회로 검증, 제조사 안전 기능 구성이 필요하다.
