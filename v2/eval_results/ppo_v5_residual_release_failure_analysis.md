# PPO v5 Residual Release Failure Analysis

## Context

Checkpoint:

```text
v2/policies/ppo_pick_place_v5_residual_best.pt
```

Evaluation setting:

```text
episodes=50
seed=11
release_gate_dist=0.06
release_gate_max_hold=360
success_dist=0.06
```

50-episode result:

```text
successes=47/50
grasp_rate=1.0
mean_final_cube_target_dist=0.0605 m
```

The residual policy removed the large no-grasp failures seen in PPO-v2, but introduced small release/placement misses.

## Failed Episodes

From the 50-episode rollout:

```text
episode=13 seed=24 active_cube=cube_1 final_dist=0.0629 min_dist=0.0624 grasped=True final_event=8
episode=36 seed=47 active_cube=cube_0 final_dist=0.0857 min_dist=0.0671 grasped=True final_event=8
episode=49 seed=60 active_cube=cube_1 final_dist=0.0693 min_dist=0.0614 grasped=True final_event=8
```

All failures established a grasp. The failure mode is therefore not grasping; it is final placement/release precision.

## Step-Level Debug Reproduction

Debug rollout command reproduced the same failure pattern for episodes 13 and 49. Episode 36 was non-deterministic and succeeded in the debug run, but still shows the same target-approach behavior.

```text
episode=13 seed=24:
  min_cube_target_dist=0.0626 at event 6
  release at event 7/8 after hold expires
  final_cube_target_dist=0.0794

episode=36 seed=47:
  min_cube_target_dist=0.0600 at event 6
  debug success=True

episode=49 seed=60:
  min_cube_target_dist=0.0614 at event 6
  release at event 7/8 after hold expires
  final_cube_target_dist=0.0812
```

## BC Comparison

BC-v1 succeeds on the same episode/seed pairs:

```text
episode=13 seed=24 final_dist=0.0596
episode=36 seed=47 final_dist=0.0595
episode=49 seed=60 final_dist=0.0595
```

BC reaches the success radius during event 6 before release. Residual PPO reaches close to the target but often stalls just outside the success radius, then drifts away while the release gate is holding.

## Diagnosis

The release gate is working as designed: it prevents release until the cube-target distance is below `0.06 m`, or until the max hold expires.

The problem is that residual PPO sometimes approaches the target but does not stabilize inside the success radius:

```text
target radius: 0.0600 m
residual near misses: 0.0614-0.0626 m
```

After the near miss, the policy continues applying event-6 actions that move the cube away from the target instead of correcting the small residual error. Increasing `release_gate_max_hold` to 720 did not help and lowered the 50-episode result to 46/50.

## Next Fix

Do not tune grasp further. Grasp is already stable for residual PPO.

Next target:

```text
event 5-7 placement/release precision
```

Recommended changes:

- add a stronger near-target shaping term during event 6,
- penalize increasing cube-target distance after the cube has entered a near-target band,
- add a target-zone hold bonus when `cube_target_dist < 0.065`,
- optionally make release success require staying close for a short window rather than only checking one instant,
- keep residual PPO but tune residual scale/reward around placement rather than grasp.

