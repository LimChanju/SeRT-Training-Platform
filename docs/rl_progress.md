# RL Platform Progress

Updated: 2026-06-15

## Current Goal

Build a learning platform for HRI-aware robot pick-and-place in Isaac Sim. The current milestone is a working baseline pipeline:

1. collect expert pick-and-place trajectories,
2. train a behavior cloning policy,
3. roll out the trained policy in Isaac Sim,
4. prepare the code surface for RL with pseudo-ErrP and later EEG replay feedback.

## Runtime Setup

- Simulator: Isaac Sim 4.5
- Robot: Franka Panda
- Task: tabletop pick-and-place with randomized cubes and one placement target
- Expert: Isaac/Franka PickPlaceController backed by RMPFlow
- Policy action: 5D task-space command
- Observation: 84D state vector with controller phase information
- Reward: v1 placement-focused HRI/ErrP-compatible shaped reward

## Implemented Components

### Trajectory Schema v1

The trajectory format now records transition data suitable for offline learning.

- Observation version: `obs_v1_state_controller_phase`
- Action version: `action_v1_controller_target_delta`
- Reward version in the first saved expert dataset: `reward_v0_hri_errp`
- Schema version: `trajectory_v0_transitions`

Each transition stores:

- `obs` and `next_obs`
- flattened policy observations
- 5D task-space actions
- expert target action and target position
- controller event index and event progress `t`
- 9D raw expert joint action for diagnostic or later joint-action experiments
- reward total and reward components
- ErrP labels, feedback, source code, and EEG replay placeholders

### Observation v1

The policy observation is currently 84D. It includes:

- Panda arm joint position and velocity
- gripper width
- end-effector pose
- active cube pose and velocity
- placement target position
- relative vectors among EE, cube, and target
- optional human head/hand positions
- hand-gripper distance
- human collision and near-human flags
- task event flags such as protected-cube collision, pick miss, drop/throw
- grasp estimate
- high-level task phase one-hot
- PickPlaceController event one-hot
- PickPlaceController event progress

### Action v1

The policy action is 5D:

```text
[dx, dy, dz, dyaw, gripper_cmd]
```

For v1, `dx/dy/dz` encode the expert controller target delta, not only the next observed EE displacement. This fixed the earlier issue where the BC policy learned actions that looked low-error offline but did not reproduce the controller's intended target during rollout.

### BC Training

Expert trajectories were collected with the Isaac PickPlaceController and used to train a behavior cloning policy.

Confirmed dataset metadata:

```text
action_dim: 5
observation_dim: 84
action_version: action_v1_controller_target_delta
observation_version: obs_v1_state_controller_phase
reward_version: reward_v0_hri_errp
schema_version: trajectory_v0_transitions
```

The 100-episode BC model was trained on GPU:

```text
torch: 2.5.1+cu118
cuda: NVIDIA GeForce RTX 4090
checkpoint: v2/policies/bc_pick_place_v1_100eps.pt
```

### Reward v1

Failure analysis showed that failed BC rollout episodes usually grasped the cube but released it outside the success radius. The default reward was therefore moved from `reward_v0_hri_errp` to `reward_v1_placement_hri_errp`.

Main changes:

- reduced the per-step grasp bonus, because holding the cube for a long time was over-rewarded,
- added extra cube-to-target progress while carrying or after the grasp phase,
- added a normalized placement error penalty after the grasp phase,
- added a target-zone bonus,
- added a one-shot penalty for releasing outside the target radius,
- kept human proximity, collision, and ErrP penalties in the reward.

`IsaacPickPlaceEnv` also exposes an optional release gate. It is disabled by default to preserve baseline comparability, but RL runs can set `release_gate_dist` to delay opening the gripper until the cube is inside the target radius. RL runs can also set `require_release_for_success=True` so an episode only succeeds after the cube has been released inside the target radius.

### Rollout Fixes

The first rollout failure was not mainly a BC learning failure. Two rollout/execution mismatches were found and fixed.

1. Gripper action mismatch

   RMPFlow returns a 7D arm action, while `gripper.forward()` returns a gripper-scoped action. The previous code tried to splice finger-joint commands into the arm action and crashed around the close-gripper phase with an index error. The rollout now mirrors PickPlaceController behavior and sends gripper-only actions during close/open events.

2. Orientation mismatch

   The expert PickPlaceController uses a fixed top-down gripper orientation. The rollout was leaving orientation unconstrained, so the BC policy could approach the cube but fail to grasp consistently. The rollout now uses fixed top-down orientation by default.

Additional phase gating was added so event-mode rollout does not advance from lowering to closing before the gripper is close enough to the cube.

Verified rollout:

```bash
ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/rollout_bc_policy.py" \
  --checkpoint v2/policies/bc_pick_place_v1_100eps.pt \
  --episodes 3 \
  --max-steps 1200 \
  --device cuda \
  --log-every 180
```

Result:

```text
success_rate=3/3
```

## RL Environment Wrapper

Added `IsaacPickPlaceEnv` in `v2/rl/pick_place_env.py`.

It follows a Gymnasium-style interface without requiring Gymnasium as a dependency:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

The wrapper owns:

- Isaac world setup
- cube randomization
- Panda robot setup
- RMPFlow control
- policy observation construction
- v1 reward computation
- event-mode gripper control
- phase gating
- pseudo-ErrP feedback placeholder

Basic use inside an Isaac Sim script:

```python
from rl import IsaacPickPlaceEnv, PickPlaceEnvConfig

env = IsaacPickPlaceEnv(PickPlaceEnvConfig(render=False))
obs, info = env.reset()

for _ in range(1200):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

By default, observations are returned as flattened policy vectors. Set `observation_mode="dict"` in `PickPlaceEnvConfig` to receive the structured observation dictionary.

## PPO Fine-Tuning

Added `v2/train_rl.py`, a minimal PPO trainer that runs directly on `IsaacPickPlaceEnv` without adding Stable-Baselines or Gymnasium as hard dependencies.

The PPO actor uses the same `MLPPolicy` class as BC, so the saved PPO actor checkpoint is compatible with `v2/evaluate_rollout_policy.py`.

Default RL behavior:

- initializes the actor from `v2/policies/bc_pick_place_v1_100eps.pt`,
- reuses BC observation normalization,
- uses reward v1,
- uses near-deterministic Gaussian exploration noise for BC warm-start stability,
- scales rewards before PPO value/advantage updates,
- regularizes the actor toward the loaded BC actor,
- rejects actor updates that drift too far from the BC actor,
- keeps the BC baseline success condition unless `--require-release-for-success` is passed.

Starter command for the first PPO fine-tuning pass:

```bash
ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/train_rl.py" \
  --bc-checkpoint v2/policies/bc_pick_place_v1_100eps.pt \
  --output v2/policies/ppo_pick_place_v1.pt \
  --total-steps 20000 \
  --rollout-steps 1024 \
  --log-std-init -8.0 \
  --reward-scale 0.05 \
  --bc-action-coef 10.0 \
  --max-bc-action-mse 0.002 \
  --device cuda
```

Controller-target actions are very sensitive to per-step noise and unconstrained PPO updates. A 1-episode parity check with `--log-std-init -8.0` reproduced BC success, while larger noise or actor drift broke the grasp/place sequence.

After the stable pass is confirmed, run a stricter placement/release experiment by adding `--release-gate-dist 0.06 --require-release-for-success`.

Smoke verification passed with an 8-step CPU run, and the resulting PPO actor checkpoint loaded successfully in `v2/evaluate_rollout_policy.py`.

### BC vs PPO Rollout Comparison

Added `v2/compare_rollout_analyses.py` to compare rollout JSON files and failure-analysis JSON files.

Current 50-episode comparison:

```text
BC success_rate:  0.900
PPO success_rate: 0.900
PPO mean_steps delta: -37.3
PPO mean_final_cube_target_dist delta: +0.0511 m
```

Failure-mode shift:

- BC failures: `released_outside_success_radius=5`
- PPO failures: `released_outside_success_radius=1`, `grasp_never_established=4`

Interpretation: PPO preserved overall success rate and reduced release-outside-target failures, but introduced more grasp-establishment failures. Seed-level comparison showed 4 PPO recoveries and 4 PPO regressions.

### Grasp Failure Debugging

Added `v2/debug_rollout_seeds.py` for step-level rollout diagnostics on selected `episode:seed` pairs. This matters because `evaluate_rollout_policy.py` changes the active cube by episode index, so seed-only reproduction can inspect the wrong cube.

Current PPO grasp-regression episodes:

```text
10:21, 25:36, 34:45, 41:52
```

The debug comparison shows that BC succeeds on these same episode/seed pairs with `min_ee_cube_event_1_3` around `0.0575 m`, while PPO fails with `min_ee_cube_event_1_3` around `0.069-0.088 m`. PPO then advances through the close/pick phase without establishing a grasp.

Working diagnosis: PPO grasp regressions are caused by insufficient close-phase approach before event progression. The next fix should target event 1-3 grasp stability, for example by tightening/holding the phase gate or adding a grasp-phase penalty when EE-cube distance stays too large.

## Current Status

Working:

- expert trajectory collection
- v1 HDF5 trajectory schema
- BC training on GPU
- BC offline evaluation
- BC rollout in Isaac Sim
- rollout fixed-orientation/gripper event fixes
- initial RL environment wrapper
- PPO fine-tuning script
- BC vs PPO rollout comparison report
- seed-level grasp failure debugger

Still experimental:

- pseudo-ErrP mapping weights
- final release-gate setting for RL training
- grasp stability during PPO fine-tuning
- EEG replay integration
- human/avatar asset linkage
- gripper camera occlusion metric

## Recommended Next Steps

1. Add grasp-stability pressure for event-mode PPO, especially around event 1-3 close timing.
2. Re-run PPO with the grasp-stability change and compare against the current BC/PPO report.
3. Tune reward v1 weights and release-gate settings only after grasp stability no longer regresses.
4. Add pseudo-ErrP to the wrapper reward path using collision, near-human distance, and occlusion flags.
5. Replace pseudo-ErrP with EEG replay feedback once the replay dataset and time-event mapping are ready.
