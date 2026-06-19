# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/bc_pick_place_v2_failure_curriculum_rollout_eval.json`

## Summary

- Episodes: 50
- Successes: 44
- Failures: 6
- Success rate: 0.880
- Success distance: 0.060 m

## Failure Categories

- `released_outside_success_radius`: 6

## Success vs Failure Stats

| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |
|---|---:|---:|---:|---:|---:|
| success | 44 | 623.1 | 0.0597 | 0.0597 | 1.000 |
| failure | 6 | 1200.0 | 0.1254 | 0.0784 | 1.000 |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 1 | 0.059 | 0.0597 |
| cube_1 | 17 | 1 | 0.059 | 0.0608 |
| cube_2 | 16 | 4 | 0.250 | 0.0831 |

## Failed Episodes

| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 2 | 13 | cube_2 | `released_outside_success_radius` | 1200 | 0.0973 | 0.0973 | 1 | 8 |
| 9 | 20 | cube_0 | `released_outside_success_radius` | 1200 | 0.0612 | 0.0612 | 1 | 8 |
| 16 | 27 | cube_1 | `released_outside_success_radius` | 1200 | 0.0788 | 0.0761 | 1 | 8 |
| 17 | 28 | cube_2 | `released_outside_success_radius` | 1200 | 0.0974 | 0.0973 | 1 | 8 |
| 26 | 37 | cube_2 | `released_outside_success_radius` | 1200 | 0.0762 | 0.0759 | 1 | 8 |
| 41 | 52 | cube_2 | `released_outside_success_radius` | 1200 | 0.3417 | 0.0628 | 1 | 8 |

## Recommendations

- Prioritize placement refinement: add reward/termination pressure for final cube-target distance after release.
- Consider a short post-release settle check or a tighter event-6/event-7 release gate before opening the gripper.
- The policy usually grasps but releases outside the success radius; grasp reward is less urgent than placement accuracy.

## Source Rollout Summary

```json
{
  "episodes": 50,
  "successes": 44,
  "success_rate": 0.8799999952316284,
  "truncated_rate": 0.11999999731779099,
  "grasp_rate": 1.0,
  "mean_steps": 692.3200073242188,
  "std_steps": 187.59439086914062,
  "mean_total_reward": -145.8839111328125,
  "std_total_reward": 23.702531814575195,
  "mean_final_cube_target_dist": 0.06755676865577698,
  "std_final_cube_target_dist": 0.03997838497161865,
  "mean_min_cube_target_dist": 0.06191708892583847,
  "std_min_cube_target_dist": 0.00791567750275135
}
```
