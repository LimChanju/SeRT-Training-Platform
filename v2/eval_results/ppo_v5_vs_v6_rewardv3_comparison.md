# Rollout Comparison

Baseline: `Residual PPO v5`
Candidate: `Residual PPO v6 reward v3`

## Summary

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `success_rate` | 0.9400 | 1.0000 | 0.0600 |
| `grasp_rate` | 1.0000 | 1.0000 | 0.0000 |
| `mean_steps` | 679.3400 | 673.5200 | -5.8200 |
| `mean_final_cube_target_dist` | 0.0605 | 0.0596 | -0.0010 |
| `mean_min_cube_target_dist` | 0.0600 | 0.0596 | -0.0004 |
| `mean_total_reward` | -144.0145 | -142.7584 | 1.2561 |

## Failure Categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `released_outside_success_radius` | 3 | 0 | -3 |

## Seed-Level Changes

- Shared seeds: 50
- Both success: 47
- Candidate recoveries: 3
- Candidate regressions: 0
- Both failure: 0
- Mean step delta: -5.82
- Mean final distance delta: -0.0010 m

## Interpretation

- Residual PPO v6 reward v3 improved overall success rate by 0.060.
- Residual PPO v6 reward v3 finished successful/failed rollouts faster on average by 5.8 steps.
- Residual PPO v6 reward v3 reduced release-outside-target failures.
- Seed-level comparison shows 3 recoveries and 0 regressions.

## Candidate Regressions

- None

## Candidate Recoveries

- seed=24 baseline_dist=0.0629 candidate_dist=0.0599 candidate_grasped=1
- seed=47 baseline_dist=0.0857 candidate_dist=0.0595 candidate_grasped=1
- seed=60 baseline_dist=0.0693 candidate_dist=0.0598 candidate_grasped=1
