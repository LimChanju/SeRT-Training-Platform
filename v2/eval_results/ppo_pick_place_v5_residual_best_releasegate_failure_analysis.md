# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/ppo_pick_place_v5_residual_best_releasegate_rollout_eval.json`

## Summary

- Episodes: 50
- Successes: 47
- Failures: 3
- Success rate: 0.940
- Success distance: 0.060 m

## Failure Categories

- `released_outside_success_radius`: 3

## Success vs Failure Stats

| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |
|---|---:|---:|---:|---:|---:|
| success | 47 | 646.1 | 0.0598 | 0.0598 | 1.000 |
| failure | 3 | 1200.0 | 0.0726 | 0.0636 | 1.000 |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 1 | 0.059 | 0.0613 |
| cube_1 | 17 | 2 | 0.118 | 0.0605 |
| cube_2 | 16 | 0 | 0.000 | 0.0597 |

## Failed Episodes

| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 13 | 24 | cube_1 | `released_outside_success_radius` | 1200 | 0.0629 | 0.0624 | 1 | 8 |
| 36 | 47 | cube_0 | `released_outside_success_radius` | 1200 | 0.0857 | 0.0671 | 1 | 8 |
| 49 | 60 | cube_1 | `released_outside_success_radius` | 1200 | 0.0693 | 0.0614 | 1 | 8 |

## Recommendations

- Prioritize placement refinement: add reward/termination pressure for final cube-target distance after release.
- Consider a short post-release settle check or a tighter event-6/event-7 release gate before opening the gripper.
- The policy usually grasps but releases outside the success radius; grasp reward is less urgent than placement accuracy.

## Source Rollout Summary

```json
{
  "episodes": 50,
  "successes": 47,
  "success_rate": 0.9399999976158142,
  "truncated_rate": 0.05999999865889549,
  "grasp_rate": 1.0,
  "mean_steps": 679.3400268554688,
  "std_steps": 151.39968872070312,
  "mean_total_reward": -144.01454162597656,
  "std_total_reward": 19.30119514465332,
  "mean_final_cube_target_dist": 0.060534682124853134,
  "std_final_cube_target_dist": 0.0038656452670693398,
  "mean_min_cube_target_dist": 0.059994425624608994,
  "std_min_cube_target_dist": 0.0011172057129442692
}
```
