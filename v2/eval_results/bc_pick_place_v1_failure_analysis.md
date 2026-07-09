# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/bc_pick_place_v1_rollout_eval.json`

## Summary

- Episodes: 50
- Successes: 45
- Failures: 5
- Success rate: 0.900
- Success distance: 0.060 m

## Failure Categories

- `released_outside_success_radius`: 5

## Success vs Failure Stats

| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |
|---|---:|---:|---:|---:|---:|
| success | 45 | 792.0 | 0.0596 | 0.0596 | 1.000 |
| failure | 5 | 1200.0 | 0.1375 | 0.0900 | 1.000 |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 3 | 0.176 | 0.0737 |
| cube_1 | 17 | 0 | 0.000 | 0.0597 |
| cube_2 | 16 | 2 | 0.125 | 0.0690 |

## Failed Episodes

| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 2 | 13 | cube_2 | `released_outside_success_radius` | 1200 | 0.1068 | 0.0620 | 1 | 10 |
| 24 | 35 | cube_0 | `released_outside_success_radius` | 1200 | 0.1499 | 0.0623 | 1 | 10 |
| 30 | 41 | cube_0 | `released_outside_success_radius` | 1200 | 0.0759 | 0.0681 | 1 | 10 |
| 35 | 46 | cube_2 | `released_outside_success_radius` | 1200 | 0.1652 | 0.1292 | 1 | 10 |
| 42 | 53 | cube_0 | `released_outside_success_radius` | 1200 | 0.1896 | 0.1284 | 1 | 10 |

## Recommendations

- Prioritize placement refinement: add reward/termination pressure for final cube-target distance after release.
- Consider a short post-release settle check or a tighter event-6/event-7 release gate before opening the gripper.
- The policy usually grasps but releases outside the success radius; grasp reward is less urgent than placement accuracy.

## Source Rollout Summary

```json
{
  "episodes": 50,
  "successes": 45,
  "success_rate": 0.8999999761581421,
  "truncated_rate": 0.10000000149011612,
  "grasp_rate": 1.0,
  "mean_steps": 832.8400268554688,
  "std_steps": 148.14822387695312,
  "mean_total_reward": 77.39773559570312,
  "std_total_reward": 10.771554946899414,
  "mean_final_cube_target_dist": 0.06740925461053848,
  "std_final_cube_target_dist": 0.026709698140621185,
  "mean_min_cube_target_dist": 0.06266044825315475,
  "std_min_cube_target_dist": 0.013577728532254696
}
```
