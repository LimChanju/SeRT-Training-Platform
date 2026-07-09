# HRI Episode Summary: `episode_000000`

## Source

| 항목 | 값 |
|---|---|
| HDF5 file | `v3_chan/trajectories/hri_vr_sphere_obs.hdf5` |
| Episode | `episodes/episode_000000` |
| Mode | `vr_pick_place` |
| Proxy | `sphere` |
| Reason | `completed_pick_cycle` |
| Success | `True` |
| Picks | `3` |
| Episode length | `2751` steps |

## Dataset Shapes

| 변수명 | shape | 의미 |
|---|---:|---|
| `obs_policy` | `(2751, 84)` | 기존 전체 observation flatten vector |
| `hri_obs_policy` | `(2751, 74)` | HRI cognitive safety용 핵심 observation flatten vector |
| `sim_time` | `(2751,)` | 시뮬레이션 시간 |
| `step` | `(2751,)` | simulation step index |
| `human_valid_mask` | `(2751, 3)` | `head`, `left_hand`, `right_hand` tracking 유효 여부 |

## Safety Counts

| 변수명 | count | 전체 step 대비 |
|---|---:|---:|
| `safety/near_human` | `409` | `14.87%` |
| `safety/human_robot_collision` | `66` | `2.40%` |
| `safety/haptic_pulse_left` | `29` | `1.05%` |
| `safety/haptic_pulse_right` | `37` | `1.34%` |
| `safety/haptic_pulse_left OR safety/haptic_pulse_right` | `66` | `2.40%` |

## Safety Label Consistency

| 관계 | count | 해석 |
|---|---:|---|
| `human_robot_collision AND near_human` | `66` | 모든 collision frame이 near-human 구간 안에 있음 |
| `haptic_pulse AND human_robot_collision` | `66` | 모든 haptic pulse가 collision frame과 일치 |
| `haptic_pulse AND near_human` | `66` | 모든 haptic pulse가 near-human 구간 안에 있음 |
| `human_robot_collision AND NOT near_human` | `0` | collision이 near-human 밖에서 발생하지 않음 |
| `haptic_pulse AND NOT human_robot_collision` | `0` | haptic이 collision 없이 발생하지 않음 |
| `near_human AND NOT human_robot_collision` | `343` | 근접하지만 접촉/햅틱은 아닌 구간 |

## Distance Statistics

단위는 meter이다.

| 변수명 | min | mean | p05 | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| `safety/min_hand_gripper_dist_m` | `0.0104` | `0.2377` | `0.0693` | `0.2179` | `0.4441` | `0.6082` |
| `safety/left_hand_gripper_dist_m` | `0.0104` | `0.2933` | `0.1066` | `0.2506` | `0.5953` | `0.6883` |
| `safety/right_hand_gripper_dist_m` | `0.0141` | `0.3026` | `0.0834` | `0.3066` | `0.4750` | `0.6082` |

## Conditional Distance Statistics

단위는 meter이며, 모두 `safety/min_hand_gripper_dist_m` 기준이다.

| 조건 | min | mean | p05 | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| `near_human == 1` | `0.0104` | `0.0783` | `0.0305` | `0.0808` | `0.1162` | `0.1200` |
| `human_robot_collision == 1` | `0.0104` | `0.0416` | `0.0202` | `0.0369` | `0.0779` | `0.1024` |

## Collision Step Samples

| 항목 | step |
|---|---|
| First collision steps | `115, 116, 141, 142, 616, 617, 618, 619, 620, 649` |
| Last collision steps | `2451, 2452, 2470, 2471, 2551, 2552, 2725, 2726, 2727, 2728` |

## Validity Notes

- `near_human` is broader than `human_robot_collision`, as intended.
- `human_robot_collision` is a subset of `near_human`.
- `haptic_pulse_left OR haptic_pulse_right` exactly matches `human_robot_collision`.
- `errp/*` is currently placeholder data because no online EEG device is connected.
- This episode is valid as an initial HRI sphere-proxy dataset for event-aligned ErrP replay experiments.

