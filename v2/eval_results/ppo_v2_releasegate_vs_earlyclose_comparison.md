# Rollout Comparison

Baseline: `PPO-v2-releasegate`
Candidate: `PPO-v2-earlyclose-releasegate`

## Summary

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `success_rate` | 0.9600 | 0.9200 | -0.0400 |
| `grasp_rate` | 0.9600 | 1.0000 | 0.0400 |
| `mean_steps` | 715.3800 | 678.3000 | -37.0800 |
| `mean_final_cube_target_dist` | 0.0819 | 0.0706 | -0.0114 |
| `mean_min_cube_target_dist` | 0.0794 | 0.0651 | -0.0143 |
| `mean_total_reward` | -215.5877 | -155.6007 | 59.9870 |

## Failure Categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `grasp_never_established` | 2 | 0 | -2 |
| `released_outside_success_radius` | 0 | 4 | +4 |

## Seed-Level Changes

- Shared seeds: 50
- Both success: 46
- Candidate recoveries: 0
- Candidate regressions: 2
- Both failure: 2
- Mean step delta: -37.08
- Mean final distance delta: -0.0114 m

## Interpretation

- PPO-v2-earlyclose-releasegate reduced overall success rate by 0.040.
- PPO-v2-earlyclose-releasegate finished successful/failed rollouts faster on average by 37.1 steps.
- Seed-level comparison shows 0 recoveries and 2 regressions.

## Candidate Regressions

- seed=43 baseline_dist=0.0594 candidate_dist=0.1516 candidate_grasped=1
- seed=52 baseline_dist=0.0598 candidate_dist=0.1283 candidate_grasped=1

## Candidate Recoveries

- None
