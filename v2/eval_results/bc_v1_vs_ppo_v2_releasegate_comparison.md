# Rollout Comparison

Baseline: `BC-v1`
Candidate: `PPO-v2-releasegate`

## Summary

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `success_rate` | 0.9000 | 0.9600 | 0.0600 |
| `grasp_rate` | 1.0000 | 0.9600 | -0.0400 |
| `mean_steps` | 832.8400 | 715.3800 | -117.4600 |
| `mean_final_cube_target_dist` | 0.0674 | 0.0819 | 0.0145 |
| `mean_min_cube_target_dist` | 0.0627 | 0.0794 | 0.0168 |
| `mean_total_reward` | 77.3977 | -215.5877 | -292.9855 |

## Failure Categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `grasp_never_established` | 0 | 2 | +2 |
| `released_outside_success_radius` | 5 | 0 | -5 |

## Seed-Level Changes

- Shared seeds: 50
- Both success: 43
- Candidate recoveries: 5
- Candidate regressions: 2
- Both failure: 0
- Mean step delta: -117.46
- Mean final distance delta: 0.0145 m

## Interpretation

- PPO-v2-releasegate improved overall success rate by 0.060.
- PPO-v2-releasegate finished successful/failed rollouts faster on average by 117.5 steps.
- PPO-v2-releasegate reduced release-outside-target failures.
- PPO-v2-releasegate introduced more grasp-establishment failures.
- Seed-level comparison shows 5 recoveries and 2 regressions.

## Candidate Regressions

- seed=38 baseline_dist=0.0598 candidate_dist=0.5803 candidate_grasped=0
- seed=56 baseline_dist=0.0599 candidate_dist=0.6658 candidate_grasped=0

## Candidate Recoveries

- seed=13 baseline_dist=0.1068 candidate_dist=0.0597 candidate_grasped=1
- seed=35 baseline_dist=0.1499 candidate_dist=0.0571 candidate_grasped=1
- seed=41 baseline_dist=0.0759 candidate_dist=0.0598 candidate_grasped=1
- seed=46 baseline_dist=0.1652 candidate_dist=0.0593 candidate_grasped=1
- seed=53 baseline_dist=0.1896 candidate_dist=0.0598 candidate_grasped=1
