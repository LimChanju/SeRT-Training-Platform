# Grasp Failure Debug Summary

Compared PPO grasp-regression episodes against BC on the same `episode:seed` pairs and active-cube assignments.

## Episode Pairs

| Episode | Seed | Active cube index | BC success | PPO success | BC min EE-cube event 1-3 | PPO min EE-cube event 1-3 | BC first grasp step | PPO first grasp step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 21 | 1 | 1 | 0 | 0.0575 | 0.0728 | 487 | - |
| 25 | 36 | 1 | 1 | 0 | 0.0574 | 0.0689 | 487 | - |
| 34 | 45 | 1 | 1 | 0 | 0.0574 | 0.0814 | 487 | - |
| 41 | 52 | 2 | 1 | 0 | 0.0575 | 0.0883 | 487 | - |

## Diagnosis

BC reaches roughly `0.0575 m` EE-cube distance during the close/grasp phase and establishes a grasp. PPO stays farther away, roughly `0.069-0.088 m`, then event progression continues without a grasp.

The immediate fix should target event 1-3 grasp stability:

- tighten or extend phase gating before close/pick progression,
- add a grasp-phase penalty when EE-cube distance stays above the graspable range,
- keep the existing BC-drift guard so placement improvements do not destabilize grasp timing.
