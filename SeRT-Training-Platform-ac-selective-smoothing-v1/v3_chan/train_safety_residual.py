from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RL_DIR = os.path.join(SCRIPT_DIR, "rl")
PYTHON_PACKAGE_DIR = os.path.join(SCRIPT_DIR, ".python_packages")
ISAAC_TORCH_BUNDLE = os.environ.get(
    "ISAAC_TORCH_BUNDLE",
    os.path.expanduser("~/isaac-sim-4.5.0/exts/omni.isaac.ml_archive/pip_prebundle"),
)

for path in (RL_DIR, PYTHON_PACKAGE_DIR, SCRIPT_DIR, PROJECT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a distance-gated HRI safety residual policy on top of a frozen task policy."
    )
    parser.add_argument(
        "--task-checkpoint",
        default=os.path.join(
            SCRIPT_DIR, "policies", "ppo_pick_place_v7_residual_rewardv4_strict_best.pt"
        ),
        help="Frozen pi_task checkpoint. Its policy input is masked to robot-only HRI defaults.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "policies", "ppo_safety_residual_hri_v1.pt"),
        help="Output pi_safe checkpoint. Compatible with evaluate_rollout_policy.py --safety-residual-checkpoint.",
    )
    parser.add_argument("--best-output", default="")
    parser.add_argument("--best-min-episodes", type=int, default=5)
    parser.add_argument(
        "--human-replay-data",
        default=os.path.join(SCRIPT_DIR, "trajectories", "hri_vr_sphere_obs.hdf5"),
    )
    parser.add_argument("--human-replay-mode", choices=("step", "loop"), default="step")
    parser.add_argument(
        "--human-replay-episode-policy", choices=("cycle", "random"), default="cycle"
    )
    parser.add_argument(
        "--encounter-manifest",
        default="",
        help=(
            "Optional encounter JSON. When set, one phase-matched scenario is "
            "sampled per task episode instead of replaying a full HDF5 episode."
        ),
    )
    parser.add_argument(
        "--encounter-policy",
        choices=("cycle", "random"),
        default="random",
    )
    parser.add_argument(
        "--encounter-severity-mix",
        default="safe=0.40,gate_only=0.25,near=0.20,near_miss=0.10,collision=0.05",
    )
    parser.add_argument(
        "--encounter-anchor-mode",
        choices=("ee", "world"),
        default="ee",
    )
    parser.add_argument(
        "--encounter-timebase",
        choices=("recorded", "step"),
        default="recorded",
        help="recorded preserves monotonic collection time; step preserves legacy one-frame-per-step replay.",
    )
    parser.add_argument("--encounter-playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--allow-legacy-source-configuration",
        action="store_true",
        help="Explicitly allow random-layout fallback for legacy encounter sources.",
    )
    parser.add_argument(
        "--no-encounter-phase-match",
        dest="encounter_phase_match",
        action="store_false",
    )
    parser.add_argument(
        "--no-encounter-event-match",
        dest="encounter_event_match",
        action="store_false",
    )
    parser.set_defaults(
        encounter_phase_match=True,
        encounter_event_match=True,
    )
    parser.add_argument(
        "--visualize-human-replay",
        action="store_true",
        help="Show replayed head/hand positions as spheres when rendering.",
    )
    parser.add_argument(
        "--human-replay-visual-z-offset",
        type=float,
        default=0.0,
        help="Visualization-only z offset in meters for replayed head/hand spheres.",
    )
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.05)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-std-init", type=float, default=-3.0)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument(
        "--distance-progress-weight",
        type=float,
        default=0.0,
        help="Reward per meter of increased hand-gripper clearance. Zero preserves the legacy objective.",
    )
    parser.add_argument("--distance-progress-clip-m", type=float, default=0.03)
    parser.add_argument(
        "--avoidance-direction-weight",
        type=float,
        default=0.0,
        help=(
            "Reward weight for projecting the XYZ residual away from the nearest hand. "
            "Zero disables direction shaping."
        ),
    )
    parser.add_argument(
        "--avoidance-target-residual-norm",
        type=float,
        default=0.01,
        help="Positive away-direction residual magnitude that saturates direction shaping.",
    )
    parser.add_argument(
        "--avoidance-aux-coef",
        type=float,
        default=0.0,
        help="Auxiliary actor loss coefficient for matching an analytic away-from-hand target.",
    )
    parser.add_argument(
        "--avoidance-aux-target-norm",
        type=float,
        default=0.01,
        help="XYZ residual norm of the analytic target used by the auxiliary actor loss.",
    )
    parser.add_argument(
        "--gate-active-only",
        action="store_true",
        help="Update the safety actor/critic only from transitions where the distance gate is active.",
    )
    parser.add_argument(
        "--xyz-only-residual",
        action="store_true",
        help="Train and apply only dx/dy/dz residuals; dyaw and gripper residuals stay zero.",
    )
    parser.add_argument("--residual-alpha", type=float, default=0.1)
    parser.add_argument("--safety-gate-start-dist", type=float, default=0.13)
    parser.add_argument("--safety-gate-full-dist", type=float, default=0.05)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--success-dist", type=float, default=0.06)
    parser.add_argument("--phase-gate-close-dist", type=float, default=0.075)
    parser.add_argument("--phase-gate-max-hold", type=int, default=320)
    parser.add_argument("--release-gate-dist", type=float, default=-1.0)
    parser.add_argument("--release-gate-max-hold", type=int, default=240)
    parser.add_argument(
        "--pseudo-errp", dest="pseudo_errp_enabled", action="store_true"
    )
    parser.add_argument(
        "--no-pseudo-errp", dest="pseudo_errp_enabled", action="store_false"
    )
    parser.set_defaults(pseudo_errp_enabled=True)
    parser.add_argument("--pseudo-errp-sources", default="all")
    parser.add_argument("--save-every-updates", type=int, default=5)
    return parser.parse_args()


args = _parse_args()


def _ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if os.path.isdir(ISAAC_TORCH_BUNDLE) and ISAAC_TORCH_BUNDLE not in sys.path:
            sys.path.insert(0, ISAAC_TORCH_BUNDLE)
        import torch  # noqa: F401


_ensure_torch()

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": not args.render,
        "width": 1280,
        "height": 720,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
    },
    experience=os.environ.get("ISAAC_SIM_EXPERIENCE", ""),
)
print(f"[TrainSafetyResidual] SimulationApp headless={not args.render}", flush=True)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from rl import (  # noqa: E402
    ACTION_DIM,
    DYNAMIC_HRI_OBS_DIM as HRI_OBS_DIM,
    DYNAMIC_HRI_OBS_FIELD_NAMES as HRI_OBS_FIELD_NAMES,
    DYNAMIC_HRI_OBSERVATION_VERSION as HRI_OBSERVATION_VERSION,
    HumanEncounterReplay,
    HumanTrajectoryReplay,
    IsaacPickPlaceEnv,
    PickPlaceEnvConfig,
    flatten_dynamic_hri_observation as flatten_hri_observation,
    parse_pseudo_errp_sources,
)
from rl.actions import clip_action  # noqa: E402
from rl.observations import MISSING_DISTANCE_M, observation_slices  # noqa: E402
from rl.policies import MLPPolicy  # noqa: E402


def _reset_training_episode(
    env: IsaacPickPlaceEnv,
    human_replay: HumanTrajectoryReplay | HumanEncounterReplay | None,
    episode_index: int,
    seed: int,
):
    source_restoration = None
    if human_replay is not None:
        human_replay.reset(episode_index, seed=seed)
    if isinstance(human_replay, HumanEncounterReplay):
        source_restoration = human_replay.source_restoration(
            screening_seed=seed,
            allow_legacy_fallback=args.allow_legacy_source_configuration,
        )
        if not bool(source_restoration.get("source_configuration_available", False)):
            raise ValueError(
                "source_configuration_unavailable: "
                f"{source_restoration.get('restoration_reason', 'unknown')}"
            )
    return env.reset(seed=seed, source_restoration=source_restoration)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        log_std_init: float,
        controlled_action_dims: int,
    ) -> None:
        super().__init__()
        self.actor = MLPPolicy(obs_dim, action_dim, hidden_dims=hidden_dims)
        self.value = _mlp(obs_dim, 1, hidden_dims)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.controlled_action_dims = int(controlled_action_dims)

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action)[..., : self.controlled_action_dims].sum(
            dim=-1
        )
        action = torch.clamp(raw_action, -1.0, 1.0)
        if self.controlled_action_dims < action.shape[-1]:
            action = action.clone()
            action[..., self.controlled_action_dims :] = 0.0
        return action, log_prob, self.value(obs).squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        log_prob = dist.log_prob(actions)[..., : self.controlled_action_dims].sum(
            dim=-1
        )
        entropy = dist.entropy()[..., : self.controlled_action_dims].sum(dim=-1)
        return log_prob, entropy, self.value(obs).squeeze(-1)


class TaskPolicyRunner:
    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        self.path = _resolve_project_path(checkpoint_path)
        checkpoint = _torch_load(self.path, device)
        self.device = device
        self.obs_mean = _tensor_to_numpy(checkpoint["obs_mean"]).reshape(1, -1)
        self.obs_std = np.maximum(
            _tensor_to_numpy(checkpoint["obs_std"]).reshape(1, -1), 1e-6
        )
        self.obs_dim = int(checkpoint.get("obs_dim", 84))
        self.action_dim = int(checkpoint.get("action_dim", ACTION_DIM))
        hidden_dims = tuple(int(v) for v in checkpoint.get("hidden_dims", (256, 256)))
        self.policy_mode = str(checkpoint.get("policy_mode", "direct"))
        self.residual_scale = float(checkpoint.get("residual_scale", 1.0))
        self.model = MLPPolicy(
            self.obs_dim, self.action_dim, hidden_dims=hidden_dims
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.base_model = None
        self.base_obs_mean = None
        self.base_obs_std = None
        self.base_obs_dim = None
        if self.policy_mode == "residual":
            source = str(checkpoint.get("source_bc_checkpoint", ""))
            if not source:
                raise ValueError(
                    "Residual task checkpoint is missing source_bc_checkpoint."
                )
            base_path = _resolve_project_path(source)
            base = _torch_load(base_path, device)
            self.base_obs_dim = int(base.get("obs_dim", 84))
            base_hidden = tuple(int(v) for v in base.get("hidden_dims", hidden_dims))
            self.base_obs_mean = _tensor_to_numpy(base["obs_mean"]).reshape(1, -1)
            self.base_obs_std = np.maximum(
                _tensor_to_numpy(base["obs_std"]).reshape(1, -1), 1e-6
            )
            self.base_model = MLPPolicy(
                self.base_obs_dim, self.action_dim, hidden_dims=base_hidden
            ).to(device)
            self.base_model.load_state_dict(base["model_state_dict"])
            self.base_model.eval()
        for module in (self.model, self.base_model):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        masked = _mask_human_obs_for_policy(obs)
        residual_or_action = self._predict_model(
            self.model, masked, self.obs_mean, self.obs_std, self.obs_dim
        )
        if self.policy_mode != "residual":
            return residual_or_action
        if (
            self.base_model is None
            or self.base_obs_mean is None
            or self.base_obs_std is None
        ):
            raise RuntimeError("Residual task policy has no base policy.")
        base_action = self._predict_model(
            self.base_model,
            masked,
            self.base_obs_mean,
            self.base_obs_std,
            int(self.base_obs_dim),
        )
        return clip_action(
            base_action + float(self.residual_scale) * residual_or_action
        )

    def _predict_model(
        self,
        model: MLPPolicy,
        obs: np.ndarray,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        obs_dim: int,
    ) -> np.ndarray:
        obs_policy = _align_obs_dim(
            np.asarray(obs, dtype=np.float32).reshape(1, -1), obs_dim
        )
        obs_norm = (obs_policy - _align_obs_dim(obs_mean, obs_dim)) / np.maximum(
            _align_obs_dim(obs_std, obs_dim, fill_value=1.0), 1e-6
        )
        with torch.no_grad():
            action = (
                model(torch.from_numpy(obs_norm.astype(np.float32)).to(self.device))
                .detach()
                .cpu()
                .numpy()[0]
            )
        return clip_action(action)


def _run() -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    task_policy = TaskPolicyRunner(args.task_checkpoint, device)
    controlled_action_dims = 3 if args.xyz_only_residual else ACTION_DIM
    model = ActorCritic(
        HRI_OBS_DIM,
        ACTION_DIM,
        hidden_dims,
        args.log_std_init,
        controlled_action_dims,
    ).to(device)
    _zero_last_actor_layer(model.actor)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    obs_mean = np.zeros((1, HRI_OBS_DIM), dtype=np.float32)
    obs_std = np.ones((1, HRI_OBS_DIM), dtype=np.float32)

    human_replay = _load_human_replay()
    release_gate_dist = (
        None if args.release_gate_dist < 0.0 else float(args.release_gate_dist)
    )
    env = IsaacPickPlaceEnv(
        PickPlaceEnvConfig(
            max_episode_steps=args.max_episode_steps,
            success_dist=args.success_dist,
            action_scale=args.action_scale,
            gripper_mode="event",
            phase_gate_close_dist=args.phase_gate_close_dist,
            phase_gate_max_hold=args.phase_gate_max_hold,
            release_gate_dist=release_gate_dist,
            release_gate_max_hold=args.release_gate_max_hold,
            observation_mode="flat",
            seed=args.seed,
            render=args.render,
            pseudo_errp_enabled=args.pseudo_errp_enabled,
            pseudo_errp_sources=parse_pseudo_errp_sources(args.pseudo_errp_sources),
            visualize_human_replay=args.visualize_human_replay,
            human_replay_visual_z_offset=args.human_replay_visual_z_offset,
        ),
        human_state_fn=human_replay,
    )

    output_path = _resolve_output_path(args.output)
    best_path = _resolve_best_output_path(args.best_output, output_path)
    best_metric = {"success_rate": -1.0, "return": -float("inf"), "update": 0}
    history: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    started_at = time.time()

    obs, info = _reset_training_episode(env, human_replay, 0, args.seed)
    episode_return = 0.0
    episode_length = 0
    total_steps = 0
    update_idx = 0
    last_done = False

    print(
        f"[TrainSafetyResidual] task={_resolve_project_path(args.task_checkpoint)} "
        f"human_replay={human_replay.path if human_replay is not None else 'off'} "
        f"alpha={args.residual_alpha} gate=({args.safety_gate_start_dist},{args.safety_gate_full_dist})",
        flush=True,
    )

    try:
        while total_steps < int(args.total_steps):
            update_idx += 1
            rollout = _collect_rollout(
                env,
                model,
                task_policy,
                obs,
                info,
                obs_mean,
                obs_std,
                device,
                remaining_steps=int(args.total_steps) - total_steps,
                episode_return=episode_return,
                episode_length=episode_length,
                episode_stats=episode_stats,
                total_steps=total_steps,
                human_replay=human_replay,
            )
            obs = rollout["last_obs"]
            info = rollout["last_info"]
            episode_return = rollout["episode_return"]
            episode_length = rollout["episode_length"]
            total_steps = rollout["total_steps"]
            last_done = rollout["last_done"]
            next_value = (
                0.0
                if last_done
                else _predict_value(model, info, obs_mean, obs_std, device)
            )
            advantages, returns = _compute_gae(
                rewards=rollout["rewards"],
                dones=rollout["dones"],
                values=rollout["values"],
                next_value=next_value,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            metrics = _ppo_update(
                model,
                optimizer,
                rollout["hri_obs"],
                rollout["actions"],
                rollout["log_probs"],
                advantages,
                returns,
                obs_mean,
                obs_std,
                device,
                rollout["avoidance_targets"],
                sample_mask=(rollout["gates"] > 0.0) if args.gate_active_only else None,
            )
            recent = episode_stats[-20:]
            update_record = {
                "update": update_idx,
                "total_steps": total_steps,
                "episodes": len(episode_stats),
                "recent_success_rate": (
                    float(np.mean([row["success"] for row in recent]))
                    if recent
                    else 0.0
                ),
                "recent_return": (
                    float(np.mean([row["return"] for row in recent])) if recent else 0.0
                ),
                "recent_errp_feedback": (
                    float(np.mean([row["errp_feedback_sum"] for row in recent]))
                    if recent
                    else 0.0
                ),
                "recent_gate_active": (
                    float(np.mean([row["gate_active_count"] for row in recent]))
                    if recent
                    else 0.0
                ),
                "recent_distance_progress_reward": (
                    float(
                        np.mean([row["distance_progress_reward_sum"] for row in recent])
                    )
                    if recent
                    else 0.0
                ),
                "recent_avoidance_direction_reward": (
                    float(
                        np.mean(
                            [
                                row.get("avoidance_direction_reward_sum", 0.0)
                                for row in recent
                            ]
                        )
                    )
                    if recent
                    else 0.0
                ),
                **metrics,
            }
            history.append(update_record)
            if _is_better(update_record, best_metric, args.best_min_episodes):
                best_metric = {
                    "success_rate": update_record["recent_success_rate"],
                    "return": update_record["recent_return"],
                    "update": update_record["update"],
                }
                _save_checkpoint(
                    best_path,
                    model,
                    obs_mean,
                    obs_std,
                    hidden_dims,
                    history,
                    episode_stats,
                    best_metric,
                )
                print(
                    f"[TrainSafetyResidual] saved best checkpoint: {best_path}",
                    flush=True,
                )
            print(
                f"[TrainSafetyResidual] update={update_idx:04d} steps={total_steps} "
                f"episodes={len(episode_stats)} recent_success={update_record['recent_success_rate']:.3f} "
                f"recent_return={update_record['recent_return']:.2f} "
                f"recent_errp={update_record['recent_errp_feedback']:.2f} "
                f"pi={metrics['policy_loss']:.4f} vf={metrics['value_loss']:.4f} "
                f"entropy={metrics['entropy']:.3f}",
                flush=True,
            )
            if (
                args.save_every_updates > 0
                and update_idx % int(args.save_every_updates) == 0
            ):
                _save_checkpoint(
                    output_path,
                    model,
                    obs_mean,
                    obs_std,
                    hidden_dims,
                    history,
                    episode_stats,
                    best_metric,
                )
    finally:
        env.close()
        if human_replay is not None:
            human_replay.close()

    _save_checkpoint(
        output_path,
        model,
        obs_mean,
        obs_std,
        hidden_dims,
        history,
        episode_stats,
        best_metric,
    )
    _save_history(output_path, history, episode_stats, started_at)
    print(f"[TrainSafetyResidual] saved checkpoint: {output_path}", flush=True)


def _collect_rollout(
    env: IsaacPickPlaceEnv,
    model: ActorCritic,
    task_policy: TaskPolicyRunner,
    obs: np.ndarray,
    info: dict[str, Any],
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
    *,
    remaining_steps: int,
    episode_return: float,
    episode_length: int,
    episode_stats: list[dict[str, Any]],
    total_steps: int,
    human_replay: HumanTrajectoryReplay | HumanEncounterReplay | None,
) -> dict[str, Any]:
    hri_buf = []
    action_buf = []
    log_prob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    gate_buf = []
    avoidance_target_buf = []
    last_done = False
    gate_sum = 0.0
    gate_active_count = 0
    errp_feedback_sum = 0.0
    distance_progress_reward_sum = 0.0
    avoidance_direction_reward_sum = 0.0
    steps_to_collect = min(int(args.rollout_steps), int(remaining_steps))

    for _ in range(steps_to_collect):
        obs_dict = info.get("obs_dict", {})
        hri_obs = flatten_hri_observation(obs_dict)
        gate = _safety_distance_gate(obs_dict)
        hand_dist = _obs_scalar(
            obs_dict,
            "min_hand_end_effector_surface_gap",
            default=MISSING_DISTANCE_M,
        )
        task_action = task_policy.predict(np.asarray(obs, dtype=np.float32))
        residual_action, log_prob, value = _sample_safety_action(
            model, hri_obs, obs_mean, obs_std, device
        )
        avoidance_target = _avoidance_target_action(
            obs_dict,
            gate=float(gate),
            target_norm=float(args.avoidance_aux_target_norm),
        )
        env_action = clip_action(
            task_action + float(gate) * float(args.residual_alpha) * residual_action
        )
        next_obs, reward, terminated, truncated, next_info = env.step(env_action)
        done = bool(terminated or truncated)
        next_obs_dict = next_info.get("obs_dict", {})
        next_gate = _safety_distance_gate(next_obs_dict)
        next_hand_dist = _obs_scalar(
            next_obs_dict,
            "min_hand_end_effector_surface_gap",
            default=MISSING_DISTANCE_M,
        )
        distance_progress_reward = _distance_progress_reward(
            hand_dist,
            next_hand_dist,
            gate=max(float(gate), float(next_gate)),
        )
        avoidance_direction_reward = _avoidance_direction_reward(
            obs_dict,
            residual_action,
            gate=float(gate),
        )
        training_reward = (
            float(reward)
            + float(distance_progress_reward)
            + float(avoidance_direction_reward)
        )

        hri_buf.append(hri_obs)
        action_buf.append(residual_action)
        log_prob_buf.append(float(log_prob))
        reward_buf.append(training_reward * float(args.reward_scale))
        done_buf.append(float(done))
        value_buf.append(float(value))
        gate_buf.append(max(float(gate), float(next_gate)))
        avoidance_target_buf.append(avoidance_target)
        gate_sum += float(gate)
        gate_active_count += int(gate > 0.0)
        errp_feedback_sum += float(next_info.get("errp_feedback", 0.0))
        distance_progress_reward_sum += float(distance_progress_reward)
        avoidance_direction_reward_sum += float(avoidance_direction_reward)

        episode_return += training_reward
        episode_length += 1
        total_steps += 1
        last_done = done
        obs = np.asarray(next_obs, dtype=np.float32)
        info = next_info

        if done:
            encounter = (
                human_replay.current_scenario
                if isinstance(human_replay, HumanEncounterReplay)
                else {}
            )
            episode_stats.append(
                {
                    "episode": len(episode_stats),
                    "return": float(episode_return),
                    "length": int(episode_length),
                    "success": bool(terminated),
                    "truncated": bool(truncated),
                    "cube_target_dist": float(info["cube_target_dist"]),
                    "grasped": bool(info["has_grasped_cube"]),
                    "errp_feedback_sum": float(errp_feedback_sum),
                    "gate_sum": float(gate_sum),
                    "gate_active_count": int(gate_active_count),
                    "distance_progress_reward_sum": float(distance_progress_reward_sum),
                    "avoidance_direction_reward_sum": float(
                        avoidance_direction_reward_sum
                    ),
                    "encounter_id": str(encounter.get("id", "")),
                    "encounter_target_severity": str(
                        encounter.get("target_severity", "")
                    ),
                    "encounter_target_phase": str(
                        encounter.get("task_phase", "")
                    ),
                    "encounter_source_session": str(
                        encounter.get("session_id", "")
                    ),
                }
            )
            next_episode_index = len(episode_stats)
            next_seed = int(args.seed + next_episode_index)
            obs, info = _reset_training_episode(
                env,
                human_replay,
                next_episode_index,
                next_seed,
            )
            obs = np.asarray(obs, dtype=np.float32)
            episode_return = 0.0
            episode_length = 0
            gate_sum = 0.0
            gate_active_count = 0
            errp_feedback_sum = 0.0
            distance_progress_reward_sum = 0.0
            avoidance_direction_reward_sum = 0.0

    return {
        "hri_obs": np.asarray(hri_buf, dtype=np.float32),
        "actions": np.asarray(action_buf, dtype=np.float32),
        "log_probs": np.asarray(log_prob_buf, dtype=np.float32),
        "rewards": np.asarray(reward_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
        "values": np.asarray(value_buf, dtype=np.float32),
        "gates": np.asarray(gate_buf, dtype=np.float32),
        "avoidance_targets": np.asarray(avoidance_target_buf, dtype=np.float32),
        "last_obs": obs,
        "last_info": info,
        "last_done": last_done,
        "episode_return": episode_return,
        "episode_length": episode_length,
        "total_steps": total_steps,
    }


def _ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs_np: np.ndarray,
    actions_np: np.ndarray,
    old_log_probs_np: np.ndarray,
    advantages_np: np.ndarray,
    returns_np: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
    avoidance_targets_np: np.ndarray,
    sample_mask: np.ndarray | None = None,
) -> dict[str, float]:
    if sample_mask is not None:
        selected = np.asarray(sample_mask, dtype=bool).reshape(-1)
        obs_np = obs_np[selected]
        actions_np = actions_np[selected]
        old_log_probs_np = old_log_probs_np[selected]
        advantages_np = advantages_np[selected]
        returns_np = returns_np[selected]
        avoidance_targets_np = avoidance_targets_np[selected]
    active_samples = int(len(obs_np))
    if active_samples < 2:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "avoidance_aux_loss": 0.0,
            "active_train_samples": active_samples,
        }
    obs_tensor = torch.from_numpy(_normalize_obs(obs_np, obs_mean, obs_std)).to(device)
    actions = torch.from_numpy(actions_np).to(device)
    old_log_probs = torch.from_numpy(old_log_probs_np).to(device)
    advantages = torch.from_numpy(advantages_np).to(device)
    returns = torch.from_numpy(returns_np).to(device)
    avoidance_targets = torch.from_numpy(avoidance_targets_np).to(device)
    advantages = (advantages - advantages.mean()) / torch.clamp(
        advantages.std(), min=1e-6
    )
    n = int(obs_tensor.shape[0])
    losses = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "avoidance_aux_loss": 0.0,
    }
    updates = 0
    for _ in range(int(args.update_epochs)):
        order = torch.randperm(n, device=device)
        for start in range(0, n, int(args.batch_size)):
            idx = order[start : start + int(args.batch_size)]
            log_prob, entropy, value = model.evaluate_actions(
                obs_tensor[idx], actions[idx]
            )
            ratio = torch.exp(log_prob - old_log_probs[idx])
            unclipped = ratio * advantages[idx]
            clipped = (
                torch.clamp(
                    ratio, 1.0 - float(args.clip_ratio), 1.0 + float(args.clip_ratio)
                )
                * advantages[idx]
            )
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.mean((value - returns[idx]) ** 2)
            entropy_mean = entropy.mean()
            actor_mean = model.actor(obs_tensor[idx])
            avoidance_aux_loss = torch.mean(
                (actor_mean[:, :3] - avoidance_targets[idx, :3]) ** 2
            )
            loss = (
                policy_loss
                + float(args.value_coef) * value_loss
                - float(args.entropy_coef) * entropy_mean
                + float(args.avoidance_aux_coef) * avoidance_aux_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses["policy_loss"] += float(policy_loss.item())
            losses["value_loss"] += float(value_loss.item())
            losses["entropy"] += float(entropy_mean.item())
            losses["avoidance_aux_loss"] += float(avoidance_aux_loss.item())
            updates += 1
    result = {key: value / max(1, updates) for key, value in losses.items()}
    result["active_train_samples"] = active_samples
    return result


def _sample_safety_action(
    model: ActorCritic,
    hri_obs: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    obs_norm = _normalize_obs(
        np.asarray(hri_obs, dtype=np.float32).reshape(1, -1), obs_mean, obs_std
    )
    with torch.no_grad():
        action, log_prob, value = model.act(torch.from_numpy(obs_norm).to(device))
    return (
        action.detach().cpu().numpy()[0].astype(np.float32),
        float(log_prob.detach().cpu().numpy()[0]),
        float(value.detach().cpu().numpy()[0]),
    )


def _predict_value(
    model: ActorCritic,
    info: dict[str, Any],
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
) -> float:
    hri_obs = flatten_hri_observation(info.get("obs_dict", {}))
    obs_norm = _normalize_obs(hri_obs.reshape(1, -1), obs_mean, obs_std)
    with torch.no_grad():
        value = model.value(torch.from_numpy(obs_norm).to(device)).squeeze(-1)
    return float(value.detach().cpu().numpy()[0])


def _compute_gae(
    *,
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = float(next_value) if t == len(rewards) - 1 else float(values[t + 1])
        nonterminal = 1.0 - float(dones[t])
        delta = (
            float(rewards[t]) + float(gamma) * next_v * nonterminal - float(values[t])
        )
        gae = delta + float(gamma) * float(gae_lambda) * nonterminal * gae
        advantages[t] = gae
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


def _save_checkpoint(
    path: str,
    model: ActorCritic,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    hidden_dims: tuple[int, ...],
    history: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    best_metric: dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        "algo": "ppo_safety_residual",
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.actor.state_dict().items()
        },
        "actor_critic_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "obs_mean": torch.from_numpy(obs_mean.squeeze(0).astype(np.float32)),
        "obs_std": torch.from_numpy(obs_std.squeeze(0).astype(np.float32)),
        "obs_dim": HRI_OBS_DIM,
        "observation_fields": list(HRI_OBS_FIELD_NAMES),
        "action_dim": ACTION_DIM,
        "hidden_dims": hidden_dims,
        "observation_version": HRI_OBSERVATION_VERSION,
        "policy_mode": "direct",
        "residual_alpha": float(args.residual_alpha),
        "safety_gate_start_dist": float(args.safety_gate_start_dist),
        "safety_gate_full_dist": float(args.safety_gate_full_dist),
        "task_checkpoint": _resolve_project_path(args.task_checkpoint),
        "human_replay_data": _resolve_project_path(args.human_replay_data),
        "encounter_manifest": (
            _resolve_project_path(args.encounter_manifest)
            if args.encounter_manifest
            else ""
        ),
        "pseudo_errp_enabled": bool(args.pseudo_errp_enabled),
        "pseudo_errp_sources": parse_pseudo_errp_sources(args.pseudo_errp_sources),
        "gate_active_only": bool(args.gate_active_only),
        "xyz_only_residual": bool(args.xyz_only_residual),
        "controlled_action_dims": int(model.controlled_action_dims),
        "distance_progress_weight": float(args.distance_progress_weight),
        "train_args": vars(args),
        "history": history,
        "episode_stats": episode_stats,
        "best_metric": best_metric,
        "log_std": model.log_std.detach().cpu(),
    }
    torch.save(checkpoint, path)


def _save_history(
    output_path: str,
    history: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    started_at: float,
) -> None:
    path = os.path.splitext(output_path)[0] + "_history.json"
    payload = {
        "created_unix": time.time(),
        "duration_sec": time.time() - started_at,
        "updates": history,
        "episodes": episode_stats,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[TrainSafetyResidual] saved history: {path}", flush=True)


def _load_human_replay() -> HumanTrajectoryReplay | HumanEncounterReplay | None:
    if args.encounter_manifest:
        manifest_path = _resolve_project_path(args.encounter_manifest)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"--encounter-manifest not found: {manifest_path}"
            )
        return HumanEncounterReplay(
            manifest_path,
            episode_policy=args.encounter_policy,
            severity_mix=args.encounter_severity_mix,
            anchor_mode=args.encounter_anchor_mode,
            phase_match=args.encounter_phase_match,
            event_match=args.encounter_event_match,
            playback_timebase=args.encounter_timebase,
            playback_speed=args.encounter_playback_speed,
            seed=args.seed,
        )
    if not args.human_replay_data:
        return None
    path = _resolve_project_path(args.human_replay_data)
    if not os.path.exists(path):
        raise FileNotFoundError(f"--human-replay-data not found: {path}")
    return HumanTrajectoryReplay(
        path,
        mode=args.human_replay_mode,
        episode_policy=args.human_replay_episode_policy,
        seed=args.seed,
    )


def _mlp(
    input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(input_dim)
    for hidden in hidden_dims:
        layers.append(nn.Linear(dim, int(hidden)))
        layers.append(nn.ReLU())
        dim = int(hidden)
    layers.append(nn.Linear(dim, int(output_dim)))
    return nn.Sequential(*layers)


def _zero_last_actor_layer(policy: MLPPolicy) -> None:
    for module in reversed(policy.net):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
            return


def _safety_distance_gate(obs: dict[str, np.ndarray]) -> float:
    dist = _obs_scalar(
        obs,
        "min_hand_end_effector_surface_gap",
        default=MISSING_DISTANCE_M,
    )
    denom = float(args.safety_gate_start_dist) - float(args.safety_gate_full_dist)
    if denom <= 1e-6:
        return 0.0
    return float(np.clip((float(args.safety_gate_start_dist) - dist) / denom, 0.0, 1.0))


def _distance_progress_reward(
    previous_dist: float, current_dist: float, *, gate: float
) -> float:
    if float(args.distance_progress_weight) == 0.0 or gate <= 0.0:
        return 0.0
    if not (np.isfinite(previous_dist) and np.isfinite(current_dist)):
        return 0.0
    if previous_dist >= 1.0 or current_dist >= 1.0:
        return 0.0
    clip_m = max(0.0, float(args.distance_progress_clip_m))
    progress_m = float(current_dist) - float(previous_dist)
    if clip_m > 0.0:
        progress_m = float(np.clip(progress_m, -clip_m, clip_m))
    return float(args.distance_progress_weight) * float(gate) * progress_m


def _avoidance_direction_reward(
    obs: dict[str, np.ndarray],
    residual_action: np.ndarray,
    *,
    gate: float,
) -> float:
    weight = float(args.avoidance_direction_weight)
    if weight == 0.0 or gate <= 0.0:
        return 0.0
    target_norm = float(args.avoidance_target_residual_norm)
    if target_norm <= 0.0:
        raise ValueError("--avoidance-target-residual-norm must be positive")
    away_direction = _nearest_hand_away_direction(obs)
    if away_direction is None:
        return 0.0
    residual_xyz = np.asarray(residual_action, dtype=np.float32).reshape(-1)[:3]
    away_projection = float(np.dot(residual_xyz, away_direction))
    normalized_projection = float(np.clip(away_projection / target_norm, -1.0, 1.0))
    return weight * float(gate) * normalized_projection


def _avoidance_target_action(
    obs: dict[str, np.ndarray],
    *,
    gate: float,
    target_norm: float,
) -> np.ndarray:
    target = np.zeros(ACTION_DIM, dtype=np.float32)
    if float(args.avoidance_aux_coef) == 0.0 or gate <= 0.0:
        return target
    if target_norm <= 0.0:
        raise ValueError("--avoidance-aux-target-norm must be positive")
    away_direction = _nearest_hand_away_direction(obs)
    if away_direction is not None:
        target[:3] = away_direction * float(target_norm)
    return target


def _nearest_hand_away_direction(
    obs: dict[str, np.ndarray],
) -> np.ndarray | None:
    relative_hands = []
    for field_name in ("ee_to_left_hand", "ee_to_right_hand"):
        value = np.asarray(obs.get(field_name, np.zeros(3)), dtype=np.float32).reshape(
            -1
        )
        if value.size >= 3 and np.all(np.isfinite(value[:3])):
            distance = float(np.linalg.norm(value[:3]))
            if distance > 1e-6:
                relative_hands.append((distance, value[:3]))
    if not relative_hands:
        return None
    _, nearest_ee_to_hand = min(relative_hands, key=lambda item: item[0])
    return -nearest_ee_to_hand / max(float(np.linalg.norm(nearest_ee_to_hand)), 1e-6)


def _mask_human_obs_for_policy(obs: np.ndarray) -> np.ndarray:
    obs_policy = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
    slices = observation_slices()
    for field_name in (
        "human_head_pos",
        "human_left_hand_pos",
        "human_right_hand_pos",
        "ee_to_left_hand",
        "ee_to_right_hand",
        "human_robot_collision",
        "near_human",
    ):
        field_slice = slices.get(field_name)
        if field_slice is not None and field_slice.stop <= obs_policy.size:
            obs_policy[field_slice] = 0.0
    dist_slice = slices.get("min_hand_gripper_dist")
    if dist_slice is not None and dist_slice.stop <= obs_policy.size:
        obs_policy[dist_slice] = float(MISSING_DISTANCE_M)
    return obs_policy


def _obs_scalar(obs: Any, field_name: str, *, default: float = 0.0) -> float:
    if not isinstance(obs, dict) or field_name not in obs:
        return float(default)
    value = np.asarray(obs[field_name], dtype=float).reshape(-1)
    if value.size == 0 or not np.isfinite(value[0]):
        return float(default)
    return float(value[0])


def _normalize_obs(
    obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray
) -> np.ndarray:
    obs = _align_obs_dim(np.asarray(obs, dtype=np.float32), HRI_OBS_DIM)
    return ((obs - obs_mean) / np.maximum(obs_std, 1e-6)).astype(np.float32)


def _align_obs_dim(
    obs: np.ndarray, expected_dim: int, *, fill_value: float = 0.0
) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[1] == expected_dim:
        return obs
    if obs.shape[1] > expected_dim:
        return obs[:, :expected_dim]
    pad = np.full(
        (obs.shape[0], expected_dim - obs.shape[1]), fill_value, dtype=obs.dtype
    )
    return np.concatenate([obs, pad], axis=1)


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested --device cuda, but torch.cuda.is_available() is False"
        )
    return torch.device(requested)


def _parse_hidden_dims(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    return tuple(int(value) for value in values) if values else (256, 256)


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


def _resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


def _resolve_best_output_path(best_path: str, output_path: str) -> str:
    if best_path:
        return _resolve_output_path(best_path)
    stem, ext = os.path.splitext(output_path)
    return f"{stem}_best{ext or '.pt'}"


def _is_better(
    update_record: dict[str, Any], best_metric: dict[str, float], min_episodes: int
) -> bool:
    if int(update_record["episodes"]) < int(min_episodes):
        return False
    success_rate = float(update_record["recent_success_rate"])
    recent_return = float(update_record["recent_return"])
    if success_rate > float(best_metric["success_rate"]) + 1e-9:
        return True
    return abs(
        success_rate - float(best_metric["success_rate"])
    ) <= 1e-9 and recent_return > float(best_metric["return"])


try:
    _run()
except BaseException as exc:
    print(
        f"[TrainSafetyResidual] terminated by {type(exc).__name__}: {exc}", flush=True
    )
    raise
finally:
    simulation_app.close()
