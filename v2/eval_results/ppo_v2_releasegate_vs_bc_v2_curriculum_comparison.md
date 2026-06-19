# Rollout Comparison

Baseline: `PPO-v2-releasegate`
Candidate: `BC-v2-curriculum`

## Summary

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `success_rate` | 0.9600 | 0.8800 | -0.0800 |
| `grasp_rate` | 0.9600 | 1.0000 | 0.0400 |
| `mean_steps` | 715.3800 | 692.3200 | -23.0600 |
| `mean_final_cube_target_dist` | 0.0819 | 0.0676 | -0.0144 |
| `mean_min_cube_target_dist` | 0.0794 | 0.0619 | -0.0175 |
| `mean_total_reward` | -215.5877 | -145.8839 | 69.7038 |

## Failure Categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `grasp_never_established` | 2 | 0 | -2 |
| `released_outside_success_radius` | 0 | 6 | +6 |

## Seed-Level Changes

- Shared seeds: 50
- Both success: 42
- Candidate recoveries: 2
- Candidate regressions: 6
- Both failure: 0
- Mean step delta: -23.06
- Mean final distance delta: -0.0144 m

## Interpretation

- BC-v2-curriculum reduced overall success rate by 0.080.
- BC-v2-curriculum finished successful/failed rollouts faster on average by 23.1 steps.
- Seed-level comparison shows 2 recoveries and 6 regressions.

## Candidate Regressions

- seed=13 baseline_dist=0.0597 candidate_dist=0.0973 candidate_grasped=1
- seed=20 baseline_dist=0.0595 candidate_dist=0.0612 candidate_grasped=1
- seed=27 baseline_dist=0.0597 candidate_dist=0.0788 candidate_grasped=1
- seed=28 baseline_dist=0.0597 candidate_dist=0.0974 candidate_grasped=1
- seed=37 baseline_dist=0.0596 candidate_dist=0.0762 candidate_grasped=1
- seed=52 baseline_dist=0.0598 candidate_dist=0.3417 candidate_grasped=1

## Candidate Recoveries

- seed=38 baseline_dist=0.5803 candidate_dist=0.0595 candidate_grasped=1
- seed=56 baseline_dist=0.6658 candidate_dist=0.0599 candidate_grasped=1
