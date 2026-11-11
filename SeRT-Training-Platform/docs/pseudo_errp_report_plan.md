# Pseudo-ErrP Report Plan

## What To Claim

Use pseudo-ErrP as an implemented HRI feedback abstraction, not as a completed
EEG classifier.

Recommended wording:

> This work implements a pseudo-ErrP feedback pathway for HRI-aware robot
> learning. Instead of using a live EEG classifier, the current system maps
> task and human-safety events into soft error feedback. Human-robot collision,
> unsafe hand-gripper proximity, protected-object collision, pick misses, drops,
> and gripper-camera occlusion can produce a pseudo-ErrP score in `[0, 1]`.
> The same interface stores `errp_label`, `errp_feedback`,
> `errp_uncertainty`, and `errp_source_code`, so a future EEG/EDL classifier can
> replace the pseudo source without changing the RL observation schema.

Do not claim:

- live EEG classification is complete,
- a human EEG model was trained,
- the final policy was optimized directly from real ErrP.

Safe claim:

- pseudo-ErrP is implemented and connected to reward/logging,
- it is compatible with later EEG replay or online EEG classifier output,
- it lets the report discuss HRI feedback before the real EEG pipeline is ready.

## Where It Exists In Code

- `v2/rl/pseudo_errp.py`: converts HRI/task events into soft feedback.
- `v2/rl/pick_place_env.py`: injects pseudo-ErrP into env reward/info.
- `v2/rl/rewards.py`: applies `errp_penalty`.
- `v2/train_rl.py`: exposes `--pseudo-errp`, `--no-pseudo-errp`,
  and `--pseudo-errp-sources`.
- `v2/evaluate_rollout_policy.py`: logs per-episode `errp_count`,
  `errp_feedback_sum`, `mean_errp_feedback`, and `max_errp_feedback`.
- `v2/rl/trajectory_recorder.py`: stores ErrP fields in trajectories.

## Minimal Experiment For Tomorrow

The fastest defensible experiment is not a new long RL run. Use the current best
checkpoint and run a short evaluation with a random synthetic human hand. The
hand sweeps near the gripper during some episodes. If it enters the near-human
radius, the observation raises `near_human`; if it enters the collision radius,
the observation raises `human_robot_collision`. Pseudo-ErrP then converts those
events into `errp_feedback`, `errp_label`, and `errp_uncertainty`.

Run two short evaluations:

1. synthetic hand + pseudo-ErrP enabled
2. synthetic hand + pseudo-ErrP disabled

This comparison shows whether the random hand disturbance is being detected and
logged as pseudo-ErrP feedback.

### Synthetic Hand + Pseudo-ErrP Enabled

```bash
ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/evaluate_rollout_policy.py" \
  --checkpoint v2/policies/ppo_pick_place_v7_residual_rewardv4_strict_best.pt \
  --episodes 10 \
  --max-steps 1200 \
  --device cuda \
  --gripper-mode event \
  --release-gate-dist 0.06 \
  --release-gate-max-hold 360 \
  --require-release-for-success \
  --pseudo-errp \
  --pseudo-errp-sources all \
  --synthetic-human \
  --synthetic-human-episode-prob 1.0 \
  --synthetic-human-start-min-step 180 \
  --synthetic-human-start-max-step 620 \
  --synthetic-human-duration-steps 120 \
  --output-json v2/eval_results/ppo_v7_pseudo_errp_smoke_eval.json \
  --output-csv v2/eval_results/ppo_v7_pseudo_errp_smoke_eval.csv
```

### Synthetic Hand + Pseudo-ErrP Disabled

```bash
ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/evaluate_rollout_policy.py" \
  --checkpoint v2/policies/ppo_pick_place_v7_residual_rewardv4_strict_best.pt \
  --episodes 10 \
  --max-steps 1200 \
  --device cuda \
  --gripper-mode event \
  --release-gate-dist 0.06 \
  --release-gate-max-hold 360 \
  --require-release-for-success \
  --no-pseudo-errp \
  --synthetic-human \
  --synthetic-human-episode-prob 1.0 \
  --synthetic-human-start-min-step 180 \
  --synthetic-human-start-max-step 620 \
  --synthetic-human-duration-steps 120 \
  --output-json v2/eval_results/ppo_v7_no_pseudo_errp_smoke_eval.json \
  --output-csv v2/eval_results/ppo_v7_no_pseudo_errp_smoke_eval.csv
```

## Optional Short Training Run

If there is enough time, run a short PPO fine-tuning smoke with synthetic human
disturbances. This is not the main final result; it demonstrates that RL can be
trained while pseudo-ErrP feedback is active.

```bash
ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/train_rl.py" \
  --policy-mode residual \
  --residual-scale 0.1 \
  --bc-checkpoint v2/policies/bc_pick_place_v1_100eps.pt \
  --total-steps 4096 \
  --rollout-steps 512 \
  --max-episode-steps 900 \
  --output v2/policies/ppo_v7_synthetic_errp_smoke.pt \
  --best-output v2/policies/ppo_v7_synthetic_errp_smoke_best.pt \
  --device cuda \
  --release-gate-dist 0.06 \
  --release-gate-max-hold 360 \
  --require-release-for-success \
  --pseudo-errp \
  --pseudo-errp-sources human_robot_collision,near_human \
  --synthetic-human \
  --synthetic-human-episode-prob 1.0 \
  --synthetic-human-start-min-step 180 \
  --synthetic-human-start-max-step 620 \
  --synthetic-human-duration-steps 120
```

## If You Have Time For One More Result

Run with recorded human replay if a suitable HDF5 trajectory contains human
state fields:

```bash
ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/evaluate_rollout_policy.py" \
  --checkpoint v2/policies/ppo_pick_place_v7_residual_rewardv4_strict_best.pt \
  --episodes 10 \
  --max-steps 1200 \
  --device cuda \
  --gripper-mode event \
  --release-gate-dist 0.06 \
  --release-gate-max-hold 360 \
  --require-release-for-success \
  --pseudo-errp \
  --pseudo-errp-sources human_robot_collision,near_human,collision_green \
  --human-replay-data v2/trajectories/expert_pick_place_v1.hdf5 \
  --output-json v2/eval_results/ppo_v7_human_replay_pseudo_errp_eval.json \
  --output-csv v2/eval_results/ppo_v7_human_replay_pseudo_errp_eval.csv
```

Only use this result if the replay file actually has human-state data and the
run produces nonzero feedback. Otherwise, keep it as future work.

## Report Result Template

Fill this after the short evaluation:

| Condition | Episodes | Success Rate | Mean Final Distance | ErrP Count | Mean ErrP Feedback | Max ErrP Feedback |
|---|---:|---:|---:|---:|---:|---:|
| PPO v7 + synthetic hand + pseudo-ErrP | 10 | TODO | TODO | TODO | TODO | TODO |
| PPO v7 + synthetic hand without pseudo-ErrP | 10 | TODO | TODO | TODO | TODO | TODO |

If ErrP stays zero even with synthetic hand enabled:

> In this short synthetic-hand rollout, the random hand trajectory did not cross
> the configured near-human or collision threshold. The pseudo-ErrP pathway was
> enabled, but no feedback event was triggered. A denser disturbance schedule or
> larger collision radius is required for feedback-rich stress testing.

If ErrP is nonzero:

> With pseudo-ErrP enabled, the environment reported nonzero soft error feedback
> during episodes containing HRI/task-risk events. These signals entered the
> reward through the ErrP penalty term and were logged per episode as feedback
> sums, mean feedback, maximum feedback, labels, and source codes.

## Suggested Report Section

### Pseudo-ErrP Feedback Design

The current system does not yet use a real-time EEG classifier. Instead, it
implements a pseudo-ErrP layer that approximates human error perception from
observable HRI risk events. Human-robot collision, unsafe hand-gripper
proximity, protected-object collision, pick misses, drops, and occlusion events
are converted into source-specific risk scores. These scores are combined into a
soft feedback value `errp_feedback` in `[0, 1]`. A binary `errp_label` is derived
by thresholding the soft value, and `errp_uncertainty` is highest near ambiguous
feedback values.

This design keeps ErrP feedback separate from the policy observation vector. The
policy still uses the fixed 84-dimensional observation schema, while feedback
affects the scalar reward and is recorded in rollout logs. This makes the
current pseudo-ErrP implementation compatible with later EEG replay or online
EEG classifier output.

### Current Limitation

The present result should be interpreted as a pseudo-feedback implementation and
logging/reward integration result, not as a completed EEG-based human-subject
experiment. Real EEG/EDL feedback can be added later by replacing the pseudo
source with classifier probabilities while preserving the same `errp_feedback`
and `errp_uncertainty` interface.
