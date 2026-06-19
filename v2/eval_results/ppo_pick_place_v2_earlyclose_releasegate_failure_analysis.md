# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/ppo_pick_place_v2_earlyclose_releasegate_rollout_eval.json`

## Summary

- Episodes: 50
- Successes: 46
- Failures: 4
- Success rate: 0.920
- Success distance: 0.060 m

## Failure Categories

- `released_outside_success_radius`: 4

## Success vs Failure Stats

| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |
|---|---:|---:|---:|---:|---:|
| success | 46 | 632.9 | 0.0596 | 0.0596 | 1.000 |
| failure | 4 | 1200.0 | 0.1963 | 0.1282 | 1.000 |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 2 | 0.118 | 0.0824 |
| cube_1 | 17 | 0 | 0.000 | 0.0597 |
| cube_2 | 16 | 2 | 0.125 | 0.0695 |

## Failed Episodes

| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 27 | 38 | cube_0 | `released_outside_success_radius` | 1200 | 0.4302 | 0.1968 | 1 | 8 |
| 32 | 43 | cube_2 | `released_outside_success_radius` | 1200 | 0.1516 | 0.1362 | 1 | 8 |
| 41 | 52 | cube_2 | `released_outside_success_radius` | 1200 | 0.1283 | 0.1094 | 1 | 8 |
| 45 | 56 | cube_0 | `released_outside_success_radius` | 1200 | 0.0748 | 0.0703 | 1 | 8 |

## Recommendations

- Prioritize placement refinement: add reward/termination pressure for final cube-target distance after release.
- Consider a short post-release settle check or a tighter event-6/event-7 release gate before opening the gripper.
- The policy usually grasps but releases outside the success radius; grasp reward is less urgent than placement accuracy.

## Source Rollout Summary

```json
{
  "episodes": 50,
  "successes": 46,
  "success_rate": 0.9200000166893005,
  "truncated_rate": 0.07999999821186066,
  "grasp_rate": 1.0,
  "mean_steps": 678.2999877929688,
  "std_steps": 173.59185791015625,
  "mean_total_reward": -155.60073852539062,
  "std_total_reward": 72.1246109008789,
  "mean_final_cube_target_dist": 0.0705641433596611,
  "std_final_cube_target_dist": 0.0538153313100338,
  "mean_min_cube_target_dist": 0.06511630117893219,
  "std_min_cube_target_dist": 0.02270265854895115
}
```
