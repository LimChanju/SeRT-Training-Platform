# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/ppo_pick_place_v6_residual_rewardv3_best_releasegate_rollout_eval.json`

## Summary

- Episodes: 50
- Successes: 50
- Failures: 0
- Success rate: 1.000
- Success distance: 0.060 m

## Failure Categories

- No failures

## Success vs Failure Stats

| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |
|---|---:|---:|---:|---:|---:|
| success | 50 | 673.5 | 0.0596 | 0.0596 | 1.000 |
| failure | 0 | - | - | - | - |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 0 | 0.000 | 0.0596 |
| cube_1 | 17 | 0 | 0.000 | 0.0597 |
| cube_2 | 16 | 0 | 0.000 | 0.0593 |

## Failed Episodes

| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |
|---:|---:|---|---|---:|---:|---:|---:|---:|

## Recommendations

- No failures observed in this rollout evaluation.

## Source Rollout Summary

```json
{
  "episodes": 50,
  "successes": 50,
  "success_rate": 1.0,
  "truncated_rate": 0.0,
  "grasp_rate": 1.0,
  "mean_steps": 673.52001953125,
  "std_steps": 133.05941772460938,
  "mean_total_reward": -142.75839233398438,
  "std_total_reward": 17.546188354492188,
  "mean_final_cube_target_dist": 0.05957282707095146,
  "std_final_cube_target_dist": 0.0007101304363459349,
  "mean_min_cube_target_dist": 0.05957282707095146,
  "std_min_cube_target_dist": 0.0007101304363459349
}
```
