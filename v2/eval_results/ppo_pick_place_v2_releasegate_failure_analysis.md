# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/ppo_pick_place_v2_releasegate_rollout_eval.json`

## Summary

- Episodes: 50
- Successes: 48
- Failures: 2
- Success rate: 0.960
- Success distance: 0.060 m

## Failure Categories

- `grasp_never_established`: 2

## Success vs Failure Stats

| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |
|---|---:|---:|---:|---:|---:|
| success | 48 | 695.2 | 0.0594 | 0.0594 | 1.000 |
| failure | 2 | 1200.0 | 0.6231 | 0.5598 | 0.000 |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 2 | 0.118 | 0.1256 |
| cube_1 | 17 | 0 | 0.000 | 0.0594 |
| cube_2 | 16 | 0 | 0.000 | 0.0595 |

## Failed Episodes

| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 27 | 38 | cube_0 | `grasp_never_established` | 1200 | 0.5803 | 0.5625 | 0 | 6 |
| 45 | 56 | cube_0 | `grasp_never_established` | 1200 | 0.6658 | 0.5571 | 0 | 6 |

## Recommendations

- Some failures never grasped the cube; tune close timing or grasp-phase rewards.

## Source Rollout Summary

```json
{
  "episodes": 50,
  "successes": 48,
  "success_rate": 0.9599999785423279,
  "truncated_rate": 0.03999999910593033,
  "grasp_rate": 0.9599999785423279,
  "mean_steps": 715.3800048828125,
  "std_steps": 200.296875,
  "mean_total_reward": -215.58773803710938,
  "std_total_reward": 414.0227966308594,
  "mean_final_cube_target_dist": 0.08194576948881149,
  "std_final_cube_target_dist": 0.1107923611998558,
  "mean_min_cube_target_dist": 0.07941487431526184,
  "std_min_cube_target_dist": 0.09806491434574127
}
```
