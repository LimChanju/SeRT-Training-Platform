# Physical Safety Baselines

## Scope

The physical-safety experiment freezes the robot-only task policy and compares
controllers on the same held-out encounter, scene layout, and random seed.
Startle-response and ErrP evaluation are outside this baseline and will be
added as later cognitive-safety experiments.

## Controllers

| Mode | Nominal task command | Physical-safety intervention | Status |
| --- | --- | --- | --- |
| `none` | Frozen task PPO through RMPflow | None | Runtime verified |
| `rmpflow` | Frozen task PPO through RMPflow | Two tracked hands are dynamic inflated sphere obstacles | Runtime verified |
| `cbf` | Frozen task PPO through RMPflow | Velocity-level CBF projection from exact distal-link surface gap | Runtime verified |
| `rmpflow_cbf` | Frozen task PPO through RMPflow | RMPflow obstacle avoidance followed by the CBF filter | Runtime verified |
| `curobo` | cuRobo MPPI/MPC tracks the frozen task-policy EE target | Hand cuboids and the work table are collision obstacles | Optional; runtime verified with cuRobo 0.7.8 |
| `curobo_cbf` | cuRobo MPPI/MPC tracks the frozen task-policy EE target | cuRobo collision cost followed by the CBF filter | Optional; runtime verified with cuRobo 0.7.8 |

The CBF solves a minimum-change projection of the nominal joint velocity:

```text
minimize    0.5 * ||qdot - qdot_nom||^2
subject to  n^T (J_point qdot - v_hand) + gamma * h_pred >= 0
            -qdot_max <= qdot <= qdot_max
```

`h_pred` is the exact signed surface gap plus a bounded closing-speed
prediction. A positive common slack is reported when the constraints cannot be
satisfied within the configured joint-speed limits; it is never silently
treated as a safe solution.

## Paired Evaluation

```bash
python v3_chan/run_physical_safety_benchmark.py \
  --eval-manifest v3_chan/trajectories/manifests/p01_v8_cv4/fold_01_eval.json \
  --controllers none,rmpflow,cbf,rmpflow_cbf \
  --eval-seeds 11,1011,2011 \
  --experiment-tag physical_safety_cv4_fold01_v1
```

Primary safety outcomes are collision episode/rate, near and near-miss rates,
minimum surface gap, and predicted-gap violation. Task success and completion
steps measure the safety-performance trade-off. Motion-quality outcomes include
end-effector path length and raw finite-difference acceleration and jerk using
simulation timestamps. The gate-active 95th-percentile jerk is the primary
abruptness screen; step trajectories remain available for a consistently
filtered paper analysis. Controller diagnostics include activation/intervention
rate, intervention norm, CBF slack and feasibility, and solve time.

An Isaac Sim smoke test establishes API and runtime correctness only. A
baseline is considered experimentally complete after the paired held-out
evaluation has been run and the output pairing checks pass.

## Visual Check

The source-aligned Fold 1 benchmark prepares a three-encounter manifest from
the highest task-only collision rates. Replay the same encounter set in the
Isaac Sim window with the recorded human proxies and physical-safety geometry:

```bash
./v3_chan/run_physical_safety_visual_check.sh none
./v3_chan/run_physical_safety_visual_check.sh rmpflow
./v3_chan/run_physical_safety_visual_check.sh cbf
./v3_chan/run_physical_safety_visual_check.sh rmpflow_cbf
```

Set `VISUAL_DELAY_SEC` to adjust playback speed. For example,
`VISUAL_DELAY_SEC=0.02` slows each rendered simulation step further.

## Optional cuRobo Setup

The adapter targets cuRobo `v0.7.8` for Isaac Sim 4.5. Build it with the same
PyTorch and CUDA environment loaded by `launch_isaac.sh`:

```bash
git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git /tmp/curobo-v0.7.8
CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.9 \
  ISAAC_SKIP_VR_WAIT=1 ISAAC_SKIP_XR_RUNTIME_SEARCH=1 \
  ./launch_isaac.sh -m pip install /tmp/curobo-v0.7.8 \
  --no-build-isolation --no-deps
```

The local runtime check used Isaac Sim's `torch 2.5.1+cu118` on an RTX 4090.
cuRobo is an optional soft-cost MPC comparison, while CBF remains the explicit
constraint-based physical-safety baseline.
