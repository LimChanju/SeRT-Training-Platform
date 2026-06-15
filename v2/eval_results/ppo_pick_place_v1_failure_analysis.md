# Rollout Failure Analysis

Input: `/home/railabchan/isaac_vr_project/v2/eval_results/ppo_pick_place_v1_rollout_eval.json`

## Summary

- Episodes: 50
- Successes: 45
- Failures: 5
- Success rate: 0.900
- Success distance: 0.060 m

## Failure Categories

- `released_outside_success_radius`: 1
- `grasp_never_established`: 4

## Success vs Failure Stats

| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |
|---|---:|---:|---:|---:|---:|
| success | 45 | 750.6 | 0.0597 | 0.0597 | 1.000 |
| failure | 5 | 1200.0 | 0.6480 | 0.1697 | 0.200 |

## Failure By Cube

| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |
|---|---:|---:|---:|---:|
| cube_0 | 17 | 0 | 0.000 | 0.0597 |
| cube_1 | 17 | 3 | 0.176 | 0.0781 |
| cube_2 | 16 | 2 | 0.125 | 0.2239 |

## Failed Episodes

| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 2 | 13 | cube_2 | `released_outside_success_radius` | 1200 | 1.2128 | 0.0655 | 1 | 10 |
| 10 | 21 | cube_1 | `grasp_never_established` | 1200 | 0.1776 | 0.1771 | 0 | 10 |
| 25 | 36 | cube_1 | `grasp_never_established` | 1200 | 0.1424 | 0.1423 | 0 | 10 |
| 34 | 45 | cube_1 | `grasp_never_established` | 1200 | 0.1723 | 0.1448 | 0 | 10 |
| 41 | 52 | cube_2 | `grasp_never_established` | 1200 | 1.5349 | 0.3189 | 0 | 10 |

## Recommendations

- Prioritize placement refinement: add reward/termination pressure for final cube-target distance after release.
- Consider a short post-release settle check or a tighter event-6/event-7 release gate before opening the gripper.
- The policy usually grasps but releases outside the success radius; grasp reward is less urgent than placement accuracy.
- Some failures never grasped the cube; tune close timing or grasp-phase rewards.

## Source Rollout Summary

```json
{
  "episodes": 50,
  "successes": 45,
  "success_rate": 0.8999999761581421,
  "truncated_rate": 0.10000000149011612,
  "grasp_rate": 0.9200000166893005,
  "mean_steps": 795.52001953125,
  "std_steps": 148.98297119140625,
  "mean_total_reward": -56.571075439453125,
  "std_total_reward": 159.79498291015625,
  "mean_final_cube_target_dist": 0.1185164526104927,
  "std_final_cube_target_dist": 0.25947296619415283,
  "mean_min_cube_target_dist": 0.0706886500120163,
  "std_min_cube_target_dist": 0.042197275906801224
}
```
