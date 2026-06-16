# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/ppo_pick_place_v7_residual_rewardv4_strict_best_require_release_rollout_eval.json`

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
| success | 50 | 786.5 | 0.0299 | 0.0273 | 1.000 |
| failure | 0 | - | - | - | - |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 0 | 0.000 | 0.0275 |
| cube_1 | 17 | 0 | 0.000 | 0.0303 |
| cube_2 | 16 | 0 | 0.000 | 0.0321 |

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
  "mean_steps": 786.4600219726562,
  "std_steps": 52.220001220703125,
  "mean_total_reward": -103.40974426269531,
  "std_total_reward": 20.722389221191406,
  "mean_final_cube_target_dist": 0.02992241457104683,
  "std_final_cube_target_dist": 0.00948779284954071,
  "mean_min_cube_target_dist": 0.02726849727332592,
  "std_min_cube_target_dist": 0.007299719378352165
}
```
