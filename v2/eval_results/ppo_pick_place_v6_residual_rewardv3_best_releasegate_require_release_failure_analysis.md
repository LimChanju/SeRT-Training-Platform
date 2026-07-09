# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/ppo_pick_place_v6_residual_rewardv3_best_releasegate_require_release_rollout_eval.json`

## Summary

- Episodes: 50
- Successes: 49
- Failures: 1
- Success rate: 0.980
- Success distance: 0.060 m

## Failure Categories

- `reached_target_then_drifted`: 1

## Success vs Failure Stats

| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |
|---|---:|---:|---:|---:|---:|
| success | 49 | 813.6 | 0.0429 | 0.0385 | 1.000 |
| failure | 1 | 1200.0 | 0.0706 | 0.0527 | 1.000 |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 0 | 0.000 | 0.0413 |
| cube_1 | 17 | 1 | 0.059 | 0.0450 |
| cube_2 | 16 | 0 | 0.000 | 0.0440 |

## Failed Episodes

| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 4 | 15 | cube_1 | `reached_target_then_drifted` | 1200 | 0.0706 | 0.0527 | 1 | 8 |

## Recommendations

- Prioritize placement refinement: add reward/termination pressure for final cube-target distance after release.
- Consider a short post-release settle check or a tighter event-6/event-7 release gate before opening the gripper.
- The policy usually grasps but releases outside the success radius; grasp reward is less urgent than placement accuracy.

## Source Rollout Summary

```json
{
  "episodes": 50,
  "successes": 49,
  "success_rate": 0.9800000190734863,
  "truncated_rate": 0.019999999552965164,
  "grasp_rate": 1.0,
  "mean_steps": 821.3200073242188,
  "std_steps": 115.00772857666016,
  "mean_total_reward": -113.90385437011719,
  "std_total_reward": 22.224834442138672,
  "mean_final_cube_target_dist": 0.043433502316474915,
  "std_final_cube_target_dist": 0.01023293100297451,
  "mean_min_cube_target_dist": 0.0388147197663784,
  "std_min_cube_target_dist": 0.010054870508611202
}
```
