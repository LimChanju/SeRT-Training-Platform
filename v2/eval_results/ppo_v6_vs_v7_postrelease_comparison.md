# Rollout Comparison

Baseline: `ppo_v6_rewardv3_strict`
Candidate: `ppo_v7_rewardv4_strict`

## Summary

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `success_rate` | 0.9800 | 1.0000 | 0.0200 |
| `grasp_rate` | 1.0000 | 1.0000 | 0.0000 |
| `mean_steps` | 821.3200 | 786.4600 | -34.8600 |
| `mean_final_cube_target_dist` | 0.0434 | 0.0299 | -0.0135 |
| `mean_min_cube_target_dist` | 0.0388 | 0.0273 | -0.0115 |
| `mean_total_reward` | -113.9039 | -103.4097 | 10.4941 |

## Failure Categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `reached_target_then_drifted` | 1 | 0 | -1 |

## Seed-Level Changes

- Shared seeds: 50
- Both success: 49
- Candidate recoveries: 1
- Candidate regressions: 0
- Both failure: 0
- Mean step delta: -34.86
- Mean final distance delta: -0.0135 m

## Interpretation

- ppo_v7_rewardv4_strict improved overall success rate by 0.020.
- ppo_v7_rewardv4_strict finished successful/failed rollouts faster on average by 34.9 steps.
- Seed-level comparison shows 1 recoveries and 0 regressions.

## Candidate Regressions

- None

## Candidate Recoveries

- seed=15 baseline_dist=0.0706 candidate_dist=0.0551 candidate_grasped=1
