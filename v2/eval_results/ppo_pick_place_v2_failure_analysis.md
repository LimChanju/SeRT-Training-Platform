# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/ppo_pick_place_v2_rollout_eval.json`

## Summary

- Episodes: 50
- Successes: 44
- Failures: 6
- Success rate: 0.880
- Success distance: 0.060 m

## Failure Categories

- `released_outside_success_radius`: 4
- `grasp_never_established`: 2

## Success vs Failure Stats

| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |
|---|---:|---:|---:|---:|---:|
| success | 44 | 636.4 | 0.0595 | 0.0595 | 1.000 |
| failure | 6 | 1200.0 | 0.4423 | 0.2298 | 0.667 |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 4 | 0.235 | 0.1942 |
| cube_1 | 17 | 1 | 0.059 | 0.0598 |
| cube_2 | 16 | 1 | 0.062 | 0.0597 |

## Failed Episodes

| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 4 | 15 | cube_1 | `released_outside_success_radius` | 1200 | 0.0672 | 0.0623 | 1 | 10 |
| 20 | 31 | cube_2 | `released_outside_success_radius` | 1200 | 0.0621 | 0.0620 | 1 | 10 |
| 24 | 35 | cube_0 | `released_outside_success_radius` | 1200 | 1.2039 | 0.0610 | 1 | 10 |
| 27 | 38 | cube_0 | `grasp_never_established` | 1200 | 0.5803 | 0.5625 | 0 | 6 |
| 45 | 56 | cube_0 | `grasp_never_established` | 1200 | 0.6658 | 0.5571 | 0 | 6 |
| 48 | 59 | cube_0 | `released_outside_success_radius` | 1200 | 0.0743 | 0.0742 | 1 | 10 |

## Recommendations

- Prioritize placement refinement: add reward/termination pressure for final cube-target distance after release.
- Consider a short post-release settle check or a tighter event-6/event-7 release gate before opening the gripper.
- The policy usually grasps but releases outside the success radius; grasp reward is less urgent than placement accuracy.
- Some failures never grasped the cube; tune close timing or grasp-phase rewards.

## Source Rollout Summary

```json
{
  "episodes": 50,
  "successes": 44,
  "success_rate": 0.8799999952316284,
  "truncated_rate": 0.11999999731779099,
  "grasp_rate": 0.9599999785423279,
  "mean_steps": 704.0,
  "std_steps": 189.5059814453125,
  "mean_total_reward": -216.17633056640625,
  "std_total_reward": 413.9540710449219,
  "mean_final_cube_target_dist": 0.10546623915433884,
  "std_final_cube_target_dist": 0.19199849665164948,
  "mean_min_cube_target_dist": 0.07997285574674606,
  "std_min_cube_target_dist": 0.09797151386737823
}
```
