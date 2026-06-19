# Rollout Comparison

Baseline: `PPO-v1`
Candidate: `PPO-v2`

## Summary

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `success_rate` | 0.9000 | 0.8800 | -0.0200 |
| `grasp_rate` | 0.9200 | 0.9600 | 0.0400 |
| `mean_steps` | 795.5200 | 704.0000 | -91.5200 |
| `mean_final_cube_target_dist` | 0.1185 | 0.1055 | -0.0131 |
| `mean_min_cube_target_dist` | 0.0707 | 0.0800 | 0.0093 |
| `mean_total_reward` | -56.5711 | -216.1763 | -159.6053 |

## Failure Categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `grasp_never_established` | 4 | 2 | -2 |
| `released_outside_success_radius` | 1 | 4 | +3 |

## Seed-Level Changes

- Shared seeds: 50
- Both success: 39
- Candidate recoveries: 5
- Candidate regressions: 6
- Both failure: 0
- Mean step delta: -91.52
- Mean final distance delta: -0.0131 m

## Interpretation

- PPO-v2 reduced overall success rate by 0.020.
- PPO-v2 finished successful/failed rollouts faster on average by 91.5 steps.
- Seed-level comparison shows 5 recoveries and 6 regressions.

## Candidate Regressions

- seed=15 baseline_dist=0.0597 candidate_dist=0.0672 candidate_grasped=1
- seed=31 baseline_dist=0.0597 candidate_dist=0.0621 candidate_grasped=1
- seed=35 baseline_dist=0.0597 candidate_dist=1.2039 candidate_grasped=1
- seed=38 baseline_dist=0.0598 candidate_dist=0.5803 candidate_grasped=0
- seed=56 baseline_dist=0.0599 candidate_dist=0.6658 candidate_grasped=0
- seed=59 baseline_dist=0.0598 candidate_dist=0.0743 candidate_grasped=1

## Candidate Recoveries

- seed=13 baseline_dist=1.2128 candidate_dist=0.0597 candidate_grasped=1
- seed=21 baseline_dist=0.1776 candidate_dist=0.0598 candidate_grasped=1
- seed=36 baseline_dist=0.1424 candidate_dist=0.0581 candidate_grasped=1
- seed=45 baseline_dist=0.1723 candidate_dist=0.0594 candidate_grasped=1
- seed=52 baseline_dist=1.5349 candidate_dist=0.0598 candidate_grasped=1
