# Rollout Comparison

Baseline: `BC`
Candidate: `PPO`

## Summary

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `success_rate` | 0.9000 | 0.9000 | 0.0000 |
| `grasp_rate` | 1.0000 | 0.9200 | -0.0800 |
| `mean_steps` | 832.8400 | 795.5200 | -37.3200 |
| `mean_final_cube_target_dist` | 0.0674 | 0.1185 | 0.0511 |
| `mean_min_cube_target_dist` | 0.0627 | 0.0707 | 0.0080 |
| `mean_total_reward` | 77.3977 | -56.5711 | -133.9688 |

## Failure Categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `grasp_never_established` | 0 | 4 | +4 |
| `released_outside_success_radius` | 5 | 1 | -4 |

## Seed-Level Changes

- Shared seeds: 50
- Both success: 41
- Candidate recoveries: 4
- Candidate regressions: 4
- Both failure: 1
- Mean step delta: -37.32
- Mean final distance delta: 0.0511 m

## Interpretation

- PPO preserved the overall success rate of BC.
- PPO finished successful/failed rollouts faster on average by 37.3 steps.
- PPO reduced release-outside-target failures.
- PPO introduced more grasp-establishment failures.
- Seed-level comparison shows 4 recoveries and 4 regressions.

## Candidate Regressions

- seed=21 baseline_dist=0.0597 candidate_dist=0.1776 candidate_grasped=0
- seed=36 baseline_dist=0.0599 candidate_dist=0.1424 candidate_grasped=0
- seed=45 baseline_dist=0.0596 candidate_dist=0.1723 candidate_grasped=0
- seed=52 baseline_dist=0.0553 candidate_dist=1.5349 candidate_grasped=0

## Candidate Recoveries

- seed=35 baseline_dist=0.1499 candidate_dist=0.0597 candidate_grasped=1
- seed=41 baseline_dist=0.0759 candidate_dist=0.0597 candidate_grasped=1
- seed=46 baseline_dist=0.1652 candidate_dist=0.0598 candidate_grasped=1
- seed=53 baseline_dist=0.1896 candidate_dist=0.0598 candidate_grasped=1
