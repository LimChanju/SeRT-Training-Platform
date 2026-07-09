# Artifact Manifest

This project keeps large training artifacts out of normal Git history. Code,
configuration, and documentation are committed to the repository; validated
model/data artifacts should be uploaded to a GitHub Release or another artifact
store.

## robot-only-baseline-v1

Recommended release tag:

```text
robot-only-baseline-v1
```

Purpose:

- Robot-only Franka Panda pick-and-place baseline.
- Expert trajectories collected from the Isaac/Franka PickPlaceController.
- BC policy trained from expert trajectories.
- Residual PPO policy fine-tuned from the BC policy.

Validated headline result:

```text
checkpoint: v2/policies/ppo_pick_place_v7_residual_rewardv4_strict_best.pt
eval: 50 episodes, seed=11, release_gate_dist=0.06, require_release_for_success=True
success_rate: 50/50 = 1.00
grasp_rate: 1.00
mean_final_cube_target_dist: approximately 0.0299 m
```

Files to attach to the release:

| File | Purpose | Size | SHA256 |
| --- | --- | ---: | --- |
| `v2/trajectories/expert_pick_place_v1.hdf5` | Robot-only expert trajectory dataset, 100 episodes | 64,725,781 bytes | `0554362f6c81bc13bcacccc54b82d6a51ba15b05e2162ab47e0a435303fdc7df` |
| `v2/policies/bc_pick_place_v1_100eps.pt` | BC policy checkpoint trained from `expert_pick_place_v1.hdf5` | 366,976 bytes | `fc4801dfdd7ae4e3edf8760277a1d87fd540c12bf27aeb0a06a3eefdbd5ba8c3` |
| `v2/policies/bc_pick_place_v1_100eps_history.json` | BC training history | 21,207 bytes | `dd47763a0fed177a5f42cf4d99ccabfa19131f6e726c1ae36f6c34c274b252c5` |
| `v2/policies/ppo_pick_place_v7_residual_rewardv4_strict_best.pt` | Best strict robot-only residual PPO checkpoint | 1,078,564 bytes | `39c4f2ff1290e3afed72459f29d54045fa545027185dfafb1379f4cc40bef702` |
| `v2/policies/ppo_pick_place_v7_residual_rewardv4_strict_history.json` | PPO training history | 20,578 bytes | `65fdd1c363944e8f796eb262f9b75762e8242309e78c88f8177aaef6a9964ab1` |

Related source/documentation:

- `docs/rl_progress.md`
- `docs/rl_trajectory_schema.md`
- `v2/train_rl.py`
- `v2/evaluate_rollout_policy.py`
- `v2/rl/`

## Rebuild Notes

Use Isaac Sim's Python/runtime for data collection, rollout, and PPO training.
The BC/PPO policies use the same 84D observation schema and normalized 5D action
schema documented in `docs/rl_progress.md`.

The important schema versions for this baseline are:

```text
observation_version: obs_v1_state_controller_phase
action_version: action_v1_controller_target_delta
reward_version: reward_v4_post_release_stability_hri_errp
```
