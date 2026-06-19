# Takehome Report Files

Bring these files if you are writing the report on another computer.

## Must Bring

### `docs/rl_progress.md`

Main progress summary.

Use this for:

- overall project goal,
- BC -> PPO pipeline,
- reward version changes,
- pseudo-ErrP explanation,
- final limitations and future work.

Interpretation:

> The project built an Isaac Sim robot learning pipeline. It starts from expert
> demonstrations and behavior cloning, then adds PPO/residual RL and an
> HRI-aware pseudo-ErrP feedback pathway.

### `docs/pipeline.md`

System architecture diagram and runtime data flow.

Use this for:

- VR/Isaac/bHaptics/logging system overview,
- pipeline figure,
- explanation of `errp_markers.csv` and `session_samples.csv`.

Interpretation:

> The system connects VR input, Isaac Sim robot control, human/avatar collision
> checking, pseudo-ErrP/event logging, and optional haptic feedback.

### `docs/pseudo_errp_report_plan.md`

Pseudo-ErrP report wording and experiment plan.

Use this for:

- safe wording about pseudo-ErrP,
- what not to claim,
- experiment commands,
- report section draft.

Interpretation:

> Pseudo-ErrP is not real EEG. It is a simulated error-feedback signal generated
> when the robot gets too close to a synthetic/virtual human hand or collides
> with it.

### `v2/eval_results/ppo_v6_vs_v7_postrelease_comparison.md`

Best normal task comparison.

Key result:

- PPO v6 strict success rate: `0.98`
- PPO v7 strict success rate: `1.00`
- mean final cube-target distance improved from `0.0434 m` to `0.0299 m`
- one post-release drift failure was removed.

Interpretation:

> Under normal pick-and-place conditions, the final PPO v7 policy performs very
> well. It succeeds on all 50 evaluation seeds and places the cube closer to the
> target than the previous version.

### `v2/eval_results/ppo_pick_place_v7_residual_rewardv4_strict_best_require_release_rollout_eval.json`

Full JSON for the best normal-task PPO v7 evaluation.

Key result:

- `episodes`: 50
- `success_rate`: 1.0
- `grasp_rate`: 1.0
- `mean_final_cube_target_dist`: about `0.0299 m`

Interpretation:

> This is the strongest baseline result. Without synthetic human disturbance,
> the robot policy reliably completes the task.

### `v2/eval_results/ppo_v7_synthetic_hand_pseudo_errp_eval.csv`

Mild synthetic hand disturbance result.

Key result:

- `success_rate`: 0.70
- `grasp_rate`: 0.70
- `mean_final_cube_target_dist`: about `0.1621 m`
- pseudo-ErrP count stayed `0`

Interpretation:

> Mild synthetic hand disturbance reduced performance, but it did not cross the
> configured pseudo-ErrP threshold. Use this as evidence that human-hand
> disturbance can degrade the existing task policy even before strong collision
> feedback is triggered.

### `v2/eval_results/ppo_v7_synthetic_hand_pseudo_errp_eval_v2.csv`

Strong synthetic hand disturbance result.

Key result:

- `success_rate`: 0.10
- `grasp_rate`: 0.70
- `mean_final_cube_target_dist`: about `0.4742 m`
- `near_human`: 868 source events
- `human_robot_collision`: 314 source events
- `mean_errp_count`: 31.4 per episode
- `max_errp_feedback`: 1.0

Interpretation:

> Strong synthetic hand disturbance created many near-human and collision-level
> events. These events were successfully recorded as pseudo-ErrP feedback. The
> task success rate dropped sharply, showing that the current task policy is not
> robust to strong human-hand interference.

### `v2/eval_results/ppo_v7_synthetic_hand_pseudo_errp_eval_v2.json`

Full JSON for the strong synthetic hand pseudo-ErrP result.

Use this for:

- exact numeric table,
- per-episode results,
- source counts and feedback statistics.

Interpretation:

> This is the main pseudo-ErrP evidence file. It shows that the system detected
> hand proximity and collision-like events and converted them into pseudo-ErrP
> feedback logs.

## Optional But Useful

### `docs/rl_trajectory_schema.md`

Use this if the report needs implementation detail about observations, actions,
rewards, and ErrP storage.

Interpretation:

> The data schema stores robot state, human state, task phase, action, reward,
> and ErrP fields in a reusable format.

### `v2/evaluate_rollout_policy.py`

Use this only if you need to show how the evaluation logs were produced.

Interpretation:

> This script loads a trained policy, runs evaluation episodes, and saves success
> metrics plus pseudo-ErrP feedback metrics to JSON/CSV.

### `v2/rl/pick_place_env.py`

Use this only if you need to show implementation evidence.

Interpretation:

> This environment wrapper builds observations, computes reward, tracks task
> success, and injects the synthetic human hand disturbance used for the
> pseudo-ErrP stress test.

### `v2/rl/pseudo_errp.py`

Use this only if you need to explain pseudo-ErrP logic.

Interpretation:

> This module converts HRI risk sources such as `near_human` and
> `human_robot_collision` into soft pseudo-ErrP feedback.

## Simple Story For The Report

1. Without human disturbance, PPO v7 succeeds reliably.
2. When a synthetic human hand is added, performance drops.
3. With stronger hand disturbance, the system records near-human and collision
   events as pseudo-ErrP feedback.
4. Therefore, the current system demonstrates a working HRI-aware feedback
   pathway, but more RL fine-tuning is needed for robust behavior around human
   hands.

## Recommended One-Sentence Conclusion

> The robot policy can solve the normal pick-and-place task, but synthetic human
> hand interference greatly reduces performance; importantly, the system can
> detect these risky moments and log them as pseudo-ErrP feedback for future
> HRI-aware RL fine-tuning.
