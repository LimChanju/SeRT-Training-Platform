# Rollout Comparison

Baseline: `PPO-v1`
Candidate: `PPO-v2-releasegate`

## Summary

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `success_rate` | 0.9000 | 0.9600 | 0.0600 |
| `grasp_rate` | 0.9200 | 0.9600 | 0.0400 |
| `mean_steps` | 795.5200 | 715.3800 | -80.1400 |
| `mean_final_cube_target_dist` | 0.1185 | 0.0819 | -0.0366 |
| `mean_min_cube_target_dist` | 0.0707 | 0.0794 | 0.0087 |
| `mean_total_reward` | -56.5711 | -215.5877 | -159.0167 |

## Failure Categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `grasp_never_established` | 4 | 2 | -2 |
| `released_outside_success_radius` | 1 | 0 | -1 |

## Seed-Level Changes

- Shared seeds: 50
- Both success: 43
- Candidate recoveries: 5
- Candidate regressions: 2
- Both failure: 0
- Mean step delta: -80.14
- Mean final distance delta: -0.0366 m

## Interpretation

- PPO-v2-releasegate improved overall success rate by 0.060.
- PPO-v2-releasegate finished successful/failed rollouts faster on average by 80.1 steps.
- PPO-v2-releasegate reduced release-outside-target failures.
- Seed-level comparison shows 5 recoveries and 2 regressions.

## Candidate Regressions

- seed=38 baseline_dist=0.0598 candidate_dist=0.5803 candidate_grasped=0
- seed=56 baseline_dist=0.0599 candidate_dist=0.6658 candidate_grasped=0

## Candidate Recoveries

- seed=13 baseline_dist=1.2128 candidate_dist=0.0597 candidate_grasped=1
- seed=21 baseline_dist=0.1776 candidate_dist=0.0598 candidate_grasped=1
- seed=36 baseline_dist=0.1424 candidate_dist=0.0599 candidate_grasped=1
- seed=45 baseline_dist=0.1723 candidate_dist=0.0594 candidate_grasped=1
- seed=52 baseline_dist=1.5349 candidate_dist=0.0598 candidate_grasped=1
