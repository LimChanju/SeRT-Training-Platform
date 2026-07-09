# train_rl.py -- PPO fine-tuning for the Isaac pick-and-place RL wrapper
#
# Example:
#   ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/train_rl.py" \
#       --bc-checkpoint v2/policies/bc_pick_place_v1_100eps.pt \
#       --total-steps 20000 --device cuda

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
    parser = argparse.ArgumentParser(description="Train a PPO policy in Isaac pick-and-place.")
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "policies", "ppo_pick_place_v2.pt"),
        help="Output actor checkpoint. It is compatible with evaluate_rollout_policy.py.",
    )
    parser.add_argument(
        "--best-output",
        default="",
        help="Optional path for the best actor checkpoint. Empty uses '<output stem>_best.pt'.",
    )
    parser.add_argument(
        "--best-min-episodes",
        type=int,
        default=5,
        help="Do not save a best checkpoint until at least this many episodes have completed.",
    )
    parser.add_argument(
        "--bc-checkpoint",
        default=os.path.join(SCRIPT_DIR, "policies", "bc_pick_place_v1_100eps.pt"),
        help="Optional BC checkpoint used to initialize the actor and observation normalization.",
    )
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--render", action="store_true", help="Render the training window.")
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument(
        "--policy-mode",
        choices=("direct", "residual", "safety_residual"),
        default="direct",
        help=(
            "direct learns the full action; residual learns a correction added to the BC action. "
            "safety_residual feeds full HRI observation to the residual actor but masks human "
            "state before the frozen base actor."
        ),
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.1,
        help="Multiplier for residual policy output when --policy-mode residual is used.",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.05)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--bc-action-coef",
        type=float,
        default=10.0,
        help="Keep the actor close to the loaded BC actor on PPO minibatch observations.",
    )
    parser.add_argument(
        "--max-bc-action-mse",
        type=float,
        default=0.002,
        help="Reject an actor update if it drifts farther than this MSE from the loaded BC actor.",
    )
    parser.add_argument(
        "--bc-anchor-data",
        default="",
        help=(
            "Optional HDF5 expert dataset used as an offline BC anchor during PPO. "
            "This constrains the actor on expert states, not only on the current rollout."
        ),
    )
    parser.add_argument(
        "--bc-anchor-target-dataset",
        default="expert_target_action",
        choices=("expert_task_action", "expert_target_action"),
        help="Action dataset to use for --bc-anchor-data.",
    )
    parser.add_argument(
        "--bc-anchor-coef",
        type=float,
        default=0.0,
        help="MSE coefficient for offline expert anchor batches. Zero disables it.",
    )
    parser.add_argument("--bc-anchor-batch-size", type=int, default=256)
    parser.add_argument(
        "--bc-anchor-grasp-weight",
        type=float,
        default=1.0,
        help="Extra sample weight for controller events 1-3 in the offline anchor loss.",
    )
    parser.add_argument(
        "--bc-anchor-transport-weight",
        type=float,
        default=1.0,
        help="Extra sample weight for controller events 4-6 in the offline anchor loss.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=-8.0,
        help="Initial Gaussian exploration std in log space. Controller-target actions need tiny warm-start noise.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.05,
        help="Scale rewards before PPO advantage/value updates. Raw episode returns are still logged.",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--success-dist", type=float, default=0.06)
    parser.add_argument(
        "--release-gate-dist",
        type=float,
        default=-1.0,
        help="Hold release until this cube-target distance. Negative disables it for BC parity.",
    )
    parser.add_argument("--release-gate-max-hold", type=int, default=240)
    success_group = parser.add_mutually_exclusive_group()
    success_group.add_argument(
        "--require-release-for-success",
        dest="require_release_for_success",
        action="store_true",
        help="Count success only after the cube has been released inside the target radius.",
    )
    success_group.add_argument(
        "--allow-success-before-release",
        dest="require_release_for_success",
        action="store_false",
        help="Use the BC baseline success condition: cube near target is enough even before release.",
    )
    parser.set_defaults(require_release_for_success=False)
    parser.add_argument("--phase-gate-close-dist", type=float, default=0.075)
    parser.add_argument("--phase-gate-max-hold", type=int, default=320)
    parser.add_argument(
        "--human-observation-mode",
        choices=("policy_and_reward", "reward_only"),
        default="policy_and_reward",
        help=(
            "policy_and_reward exposes replayed human state to the policy and reward. "
            "reward_only hides human state from policy input but still uses it for pseudo-ErrP/reward."
        ),
    )
    pseudo_errp_group = parser.add_mutually_exclusive_group()
    pseudo_errp_group.add_argument(
        "--pseudo-errp",
        dest="pseudo_errp_enabled",
        action="store_true",
        help="Enable pseudo-ErrP feedback from configured task/HRI flags.",
    )
    pseudo_errp_group.add_argument(
        "--no-pseudo-errp",
        dest="pseudo_errp_enabled",
        action="store_false",
        help="Disable pseudo-ErrP feedback; reward still logs source flags in info.",
    )
    parser.set_defaults(pseudo_errp_enabled=True)
    parser.add_argument(
        "--pseudo-errp-sources",
        default="all",
        help=(
            "Comma-separated pseudo-ErrP sources, 'all', or 'none'. "
            "Known sources include human_robot_collision, near_human, collision_green, "
            "pick_miss_recent, drop_throw_recent, gripper_camera_occluded."
        ),
    )
    parser.add_argument(
        "--human-replay-data",
        default="",
        help=(
            "Optional HDF5 trajectory file containing recorded human motion. "
            "When set, human head/hand state is replayed during PPO while the robot policy acts."
        ),
    )
    parser.add_argument(
        "--human-replay-mode",
        choices=("step", "loop"),
        default="step",
        help="step holds the last human sample after the replay ends; loop repeats it.",
    )
    parser.add_argument(
        "--human-replay-episode-policy",
        choices=("cycle", "random"),
        default="cycle",
        help="How to choose a recorded human episode for each RL episode.",
    )
    parser.add_argument(
        "--human-replay-offset",
        default="0,0,0",
        help="World-space x,y,z offset applied to replayed human head/hand positions.",
    )
    parser.add_argument(
        "--human-replay-mirror-y",
        action="store_true",
        help="Mirror replayed human head/hand positions across the world Y=0 plane.",
    )
    parser.add_argument(
        "--visualize-human-replay",
        action="store_true",
        help="Show replayed head and hand positions as simple visual markers when rendering.",
    )
    parser.add_argument(
        "--synthetic-human",
        action="store_true",
        help="Inject a random synthetic hand sweep near the gripper for pseudo-ErrP stress tests.",
    )
    parser.add_argument("--synthetic-human-episode-prob", type=float, default=0.35)
    parser.add_argument("--synthetic-human-start-min-step", type=int, default=120)
    parser.add_argument("--synthetic-human-start-max-step", type=int, default=520)
    parser.add_argument("--synthetic-human-duration-steps", type=int, default=90)
    parser.add_argument("--synthetic-human-near-dist", type=float, default=0.12)
    parser.add_argument("--synthetic-human-collision-dist", type=float, default=0.035)
    parser.add_argument(
        "--early-close-on-grasp-gate",
        action="store_true",
        help="Close the gripper as soon as the grasp gate distance is reached during grasp approach.",
    )
    parser.add_argument(
        "--fast-forward-grasp-gate",
        action="store_true",
        help="When early close triggers, jump the event clock to close_gripper to avoid lingering in approach.",
    )
    parser.add_argument(
        "--strict-grasp-phase-gate",
        action="store_true",
        help="Do not advance out of grasp phases until a grasp is actually detected.",
    )
    parser.add_argument(
        "--strict-release-phase-gate",
        action="store_true",
        help="Do not advance out of release approach until the cube is inside --release-gate-dist.",
    )
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
    }
)
print(f"[TrainPPO] SimulationApp headless={not args.render}", flush=True)

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.distributions import Normal  # noqa: E402

from rl import (  # noqa: E402
    ACTION_DIM,
    ACTION_VERSION,
    OBSERVATION_DIM,
    OBSERVATION_VERSION,
    REWARD_VERSION,
    IsaacPickPlaceEnv,
    HumanTrajectoryReplay,
    PickPlaceEnvConfig,
    observation_slices,
    parse_pseudo_errp_sources,
)
from rl.policies import MLPPolicy  # noqa: E402


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dims: tuple[int, ...],
        log_std_init: float,
    ) -> None:
        super().__init__()
        self.actor = MLPPolicy(obs_dim, action_dim, hidden_dims=hidden_dims)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.critic = _make_mlp(obs_dim, 1, hidden_dims)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).clamp(1e-4, 2.0)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        sample = dist.sample()
        action = torch.clamp(sample, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, self.value(obs)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, self.value(obs)


def _make_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, int(output_dim)))
    return nn.Sequential(*layers)


def _maybe_load_human_replay() -> HumanTrajectoryReplay | None:
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
        position_offset=_parse_vec3(args.human_replay_offset),
        mirror_y=args.human_replay_mirror_y,
    )


def _train() -> None:
    started_at = time.time()
    output_path = _resolve_output_path(args.output)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    pseudo_errp_sources = parse_pseudo_errp_sources(args.pseudo_errp_sources)
    human_replay = _maybe_load_human_replay()
    if args.policy_mode == "safety_residual" and args.human_observation_mode != "policy_and_reward":
        print(
            "[TrainPPO] --policy-mode safety_residual requires human state in the safety stream; "
            "overriding human_observation_mode=policy_and_reward.",
            flush=True,
        )
        args.human_observation_mode = "policy_and_reward"

    release_gate_dist = None if args.release_gate_dist < 0.0 else float(args.release_gate_dist)
    env = IsaacPickPlaceEnv(
        PickPlaceEnvConfig(
            max_episode_steps=args.max_episode_steps,
            success_dist=args.success_dist,
            action_scale=args.action_scale,
            fixed_orientation=True,
            gripper_mode="event",
            phase_gate_close_dist=args.phase_gate_close_dist,
            phase_gate_max_hold=args.phase_gate_max_hold,
            early_close_on_grasp_gate=args.early_close_on_grasp_gate,
            fast_forward_grasp_gate=args.fast_forward_grasp_gate,
            strict_grasp_phase_gate=args.strict_grasp_phase_gate,
            release_gate_dist=release_gate_dist,
            release_gate_max_hold=args.release_gate_max_hold,
            strict_release_phase_gate=args.strict_release_phase_gate,
            require_release_for_success=args.require_release_for_success,
            observation_mode="flat",
            human_observation_mode=args.human_observation_mode,
            seed=args.seed,
            render=args.render,
            pseudo_errp_enabled=args.pseudo_errp_enabled,
            pseudo_errp_sources=pseudo_errp_sources,
            visualize_human_replay=args.visualize_human_replay,
            synthetic_human_enabled=args.synthetic_human,
            synthetic_human_episode_prob=args.synthetic_human_episode_prob,
            synthetic_human_start_min_step=args.synthetic_human_start_min_step,
            synthetic_human_start_max_step=args.synthetic_human_start_max_step,
            synthetic_human_duration_steps=args.synthetic_human_duration_steps,
            synthetic_human_near_dist=args.synthetic_human_near_dist,
            synthetic_human_collision_dist=args.synthetic_human_collision_dist,
        ),
        human_state_fn=human_replay,
    )

    model = ActorCritic(
        OBSERVATION_DIM,
        ACTION_DIM,
        hidden_dims=hidden_dims,
        log_std_init=args.log_std_init,
    ).to(device)
    obs_mean, obs_std, bc_meta = _maybe_load_bc_checkpoint(
        model,
        args.bc_checkpoint,
        device,
        load_actor=args.policy_mode == "direct",
    )
    bc_actor = _load_frozen_bc_actor(args.bc_checkpoint, hidden_dims, device)
    if args.policy_mode in ("residual", "safety_residual"):
        if bc_actor is None:
            raise RuntimeError(f"--policy-mode {args.policy_mode} requires a valid --bc-checkpoint")
        _zero_policy_output(model.actor)
    bc_anchor = _load_bc_anchor_dataset(args.bc_anchor_data, obs_mean, obs_std, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"[TrainPPO] total_steps={args.total_steps} rollout_steps={args.rollout_steps} "
        f"device={device} torch={torch.__version__} reward={REWARD_VERSION} "
        f"reward_scale={args.reward_scale} log_std_init={args.log_std_init} "
        f"policy_mode={args.policy_mode} residual_scale={args.residual_scale} "
        f"release_gate_dist={release_gate_dist} bc_action_coef={args.bc_action_coef} "
        f"max_bc_action_mse={args.max_bc_action_mse} "
        f"bc_anchor_coef={args.bc_anchor_coef} "
        f"pseudo_errp={args.pseudo_errp_enabled} "
        f"pseudo_errp_sources={','.join(pseudo_errp_sources) if pseudo_errp_sources else 'none'} "
        f"human_replay={human_replay.path if human_replay is not None else 'off'} "
        f"human_observation_mode={args.human_observation_mode} "
        f"human_replay_offset={args.human_replay_offset} "
        f"human_replay_mirror_y={args.human_replay_mirror_y} "
        f"visualize_human_replay={args.visualize_human_replay} "
        f"synthetic_human={args.synthetic_human}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"[TrainPPO] cuda={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[TrainPPO] output={output_path}", flush=True)
    if bc_meta:
        print(f"[TrainPPO] initialized actor from BC checkpoint: {bc_meta['path']}", flush=True)
    if bc_anchor is not None:
        print(
            f"[TrainPPO] offline BC anchor loaded: {bc_anchor['path']} "
            f"transitions={bc_anchor['obs'].shape[0]} target={args.bc_anchor_target_dataset}",
            flush=True,
        )
    if human_replay is not None:
        replay_info = human_replay.info
        print(
            f"[TrainPPO] human replay loaded: {replay_info.path} "
            f"episodes={replay_info.episode_count} mode={replay_info.mode} "
            f"episode_policy={replay_info.episode_policy}",
            flush=True,
        )

    if human_replay is not None:
        human_replay.reset(0, seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    total_steps = 0
    update_idx = 0
    episode_stats: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    last_done = False
    best_path = _resolve_best_output_path(args.best_output, output_path)
    best_metric = {"success_rate": -1.0, "return": -float("inf"), "update": 0}

    try:
        while total_steps < args.total_steps:
            update_idx += 1
            rollout = _collect_rollout(
                env,
                model,
                obs,
                info,
                obs_mean,
                obs_std,
                bc_actor,
                device,
                remaining_steps=args.total_steps - total_steps,
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

            next_value = 0.0
            if not last_done:
                next_value = _predict_value(model, obs, obs_mean, obs_std, device)
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
                rollout["obs"],
                rollout["actions"],
                rollout["log_probs"],
                advantages,
                returns,
                obs_mean,
                obs_std,
                bc_actor,
                bc_anchor,
                device,
            )
            recent = episode_stats[-20:]
            update_record = {
                "update": update_idx,
                "total_steps": total_steps,
                "episodes": len(episode_stats),
                "recent_success_rate": float(np.mean([row["success"] for row in recent])) if recent else 0.0,
                "recent_return": float(np.mean([row["return"] for row in recent])) if recent else 0.0,
                **metrics,
            }
            history.append(update_record)
            if _is_better_checkpoint(update_record, best_metric, args.best_min_episodes):
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
                    bc_meta,
                    best_metric=best_metric,
                )
                print(
                    f"[TrainPPO] saved best checkpoint: {best_path} "
                    f"update={best_metric['update']} "
                    f"recent_success={best_metric['success_rate']:.3f} "
                    f"recent_return={best_metric['return']:.2f}",
                    flush=True,
                )
            print(
                f"[TrainPPO] update={update_idx:04d} steps={total_steps} "
                f"episodes={len(episode_stats)} recent_success={update_record['recent_success_rate']:.3f} "
                f"recent_return={update_record['recent_return']:.2f} "
                f"pi={metrics['policy_loss']:.4f} vf={metrics['value_loss']:.4f} "
                f"bc={metrics['bc_action_loss']:.6f} "
                f"anchor={metrics['bc_anchor_loss']:.6f} "
                f"reject={int(metrics['update_rejected'])} "
                f"entropy={metrics['entropy']:.3f}",
                flush=True,
            )
            if args.save_every_updates > 0 and update_idx % args.save_every_updates == 0:
                _save_checkpoint(
                    output_path,
                    model,
                    obs_mean,
                    obs_std,
                    hidden_dims,
                    history,
                    episode_stats,
                    bc_meta,
                    best_metric=best_metric,
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
        bc_meta,
        best_metric=best_metric,
    )
    _save_history(output_path, history, episode_stats, started_at)
    print(f"[TrainPPO] saved checkpoint: {output_path}", flush=True)
    if best_metric["update"] > 0:
        print(
            f"[TrainPPO] best checkpoint: {best_path} "
            f"update={best_metric['update']} "
            f"recent_success={best_metric['success_rate']:.3f} "
            f"recent_return={best_metric['return']:.2f}",
            flush=True,
        )


def _collect_rollout(
    env: IsaacPickPlaceEnv,
    model: ActorCritic,
    obs: np.ndarray,
    info: dict[str, Any],
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    bc_actor: MLPPolicy | None,
    device: torch.device,
    *,
    remaining_steps: int,
    episode_return: float,
    episode_length: int,
    episode_stats: list[dict[str, Any]],
    total_steps: int,
    human_replay: HumanTrajectoryReplay | None,
) -> dict[str, Any]:
    obs_buf = []
    action_buf = []
    log_prob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    last_done = False
    steps_to_collect = min(int(args.rollout_steps), int(remaining_steps))

    for _ in range(steps_to_collect):
        train_action, env_action, log_prob, value = _sample_action(
            model,
            obs,
            obs_mean,
            obs_std,
            bc_actor,
            device,
        )
        next_obs, reward, terminated, truncated, next_info = env.step(env_action)
        done = bool(terminated or truncated)

        obs_buf.append(np.asarray(obs, dtype=np.float32))
        action_buf.append(np.asarray(train_action, dtype=np.float32))
        log_prob_buf.append(float(log_prob))
        reward_buf.append(float(reward) * float(args.reward_scale))
        done_buf.append(float(done))
        value_buf.append(float(value))

        episode_return += float(reward)
        episode_length += 1
        total_steps += 1
        last_done = done
        obs = np.asarray(next_obs, dtype=np.float32)
        info = next_info

        if done:
            episode_stats.append(
                {
                    "episode": len(episode_stats),
                    "return": float(episode_return),
                    "length": int(episode_length),
                    "success": bool(terminated),
                    "truncated": bool(truncated),
                    "cube_target_dist": float(info["cube_target_dist"]),
                    "grasped": bool(info["has_grasped_cube"]),
                    "controller_event": int(info["controller_event"]),
                    "errp_feedback": float(info.get("errp_feedback", 0.0)),
                    "errp_uncertainty": float(info.get("errp_uncertainty", 0.0)),
                    "errp_label": int(info.get("errp_label", 0)),
                    "errp_source_code": int(info.get("errp_source_code", 0)),
                }
            )
            next_episode_index = len(episode_stats)
            next_seed = int(args.seed + next_episode_index)
            if human_replay is not None:
                human_replay.reset(next_episode_index, seed=next_seed)
            obs, info = env.reset(seed=next_seed)
            obs = np.asarray(obs, dtype=np.float32)
            episode_return = 0.0
            episode_length = 0

    return {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(action_buf, dtype=np.float32),
        "log_probs": np.asarray(log_prob_buf, dtype=np.float32),
        "rewards": np.asarray(reward_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
        "values": np.asarray(value_buf, dtype=np.float32),
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
    bc_actor: MLPPolicy | None,
    bc_anchor: dict[str, Any] | None,
    device: torch.device,
) -> dict[str, float]:
    obs_norm = _normalize_obs(obs_np, obs_mean, obs_std)
    obs_tensor = torch.from_numpy(obs_norm).to(device)
    actions = torch.from_numpy(actions_np).to(device)
    old_log_probs = torch.from_numpy(old_log_probs_np).to(device)
    advantages = torch.from_numpy(advantages_np).to(device)
    returns = torch.from_numpy(returns_np).to(device)
    advantages = (advantages - advantages.mean()) / torch.clamp(advantages.std(), min=1e-6)

    n = obs_tensor.shape[0]
    actor_before = _clone_actor_state(model) if bc_actor is not None and args.max_bc_action_mse > 0.0 else None
    losses = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "bc_action_loss": 0.0,
        "bc_anchor_loss": 0.0,
    }
    updates = 0
    for _ in range(args.update_epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            log_prob, entropy, value = model.evaluate_actions(obs_tensor[idx], actions[idx])
            ratio = torch.exp(log_prob - old_log_probs[idx])
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * advantages[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.mean((value - returns[idx]) ** 2)
            entropy_mean = entropy.mean()
            bc_action_loss = torch.zeros((), device=device)
            if bc_actor is not None and args.bc_action_coef > 0.0:
                policy_mean = model.actor(obs_tensor[idx])
                if args.policy_mode in ("residual", "safety_residual"):
                    bc_action_loss = torch.mean(policy_mean**2)
                else:
                    with torch.no_grad():
                        bc_action = bc_actor(obs_tensor[idx])
                    bc_action_loss = torch.mean((policy_mean - bc_action) ** 2)
            bc_anchor_loss = torch.zeros((), device=device)
            if bc_anchor is not None and args.bc_anchor_coef > 0.0:
                bc_anchor_loss = _sample_bc_anchor_loss(model, bc_actor, bc_anchor, device)
            loss = (
                policy_loss
                + args.value_coef * value_loss
                + args.bc_action_coef * bc_action_loss
                + args.bc_anchor_coef * bc_anchor_loss
                - args.entropy_coef * entropy_mean
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            losses["policy_loss"] += float(policy_loss.item())
            losses["value_loss"] += float(value_loss.item())
            losses["entropy"] += float(entropy_mean.item())
            losses["bc_action_loss"] += float(bc_action_loss.item())
            losses["bc_anchor_loss"] += float(bc_anchor_loss.item())
            updates += 1

    metrics = {key: value / max(1, updates) for key, value in losses.items()}
    metrics["update_rejected"] = 0.0
    if bc_actor is not None and args.max_bc_action_mse > 0.0:
        final_bc_loss = _actor_constraint_mse(model, bc_actor, obs_tensor)
        metrics["bc_action_loss"] = final_bc_loss
        if final_bc_loss > float(args.max_bc_action_mse):
            if actor_before is not None:
                model.actor.load_state_dict(actor_before)
            metrics["update_rejected"] = 1.0
            metrics["bc_action_loss"] = _actor_constraint_mse(model, bc_actor, obs_tensor)
    return metrics


def _clone_actor_state(model: ActorCritic) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.actor.state_dict().items()}


def _actor_constraint_mse(model: ActorCritic, bc_actor: MLPPolicy, obs_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        policy_action = model.actor(obs_tensor)
        if args.policy_mode in ("residual", "safety_residual"):
            loss = torch.mean(policy_action**2)
        else:
            bc_action = bc_actor(obs_tensor)
            loss = torch.mean((policy_action - bc_action) ** 2)
    return float(loss.detach().cpu().item())


def _load_bc_anchor_dataset(
    path: str,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
) -> dict[str, Any] | None:
    if not path or args.bc_anchor_coef <= 0.0:
        return None
    import h5py

    resolved_path = _resolve_project_path(path)
    obs_chunks = []
    action_chunks = []
    action_path = f"actions/{args.bc_anchor_target_dataset}"
    with h5py.File(resolved_path, "r") as h5:
        episodes = h5["episodes"]
        for episode_name in sorted(episodes.keys()):
            group = episodes[episode_name]
            if action_path not in group:
                raise KeyError(f"{episode_name}: missing dataset '{action_path}'")
            obs = np.asarray(group["obs_policy"], dtype=np.float32)
            actions = np.asarray(group[action_path], dtype=np.float32)
            if obs.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"{episode_name}: obs/action length mismatch "
                    f"{obs.shape[0]} != {actions.shape[0]}"
                )
            obs_chunks.append(obs)
            action_chunks.append(actions)
    if not obs_chunks:
        raise ValueError(f"No episodes found in BC anchor data: {resolved_path}")

    obs_all = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    actions_all = np.clip(np.concatenate(action_chunks, axis=0).astype(np.float32), -1.0, 1.0)
    weights = _bc_anchor_sample_weights(obs_all)
    obs_norm = _normalize_obs(obs_all, obs_mean, obs_std)
    return {
        "path": resolved_path,
        "obs": torch.from_numpy(obs_norm).to(device),
        "actions": torch.from_numpy(actions_all).to(device),
        "weights": torch.from_numpy(weights).to(device),
    }


def _bc_anchor_sample_weights(obs_np: np.ndarray) -> np.ndarray:
    weights = np.ones((obs_np.shape[0],), dtype=np.float32)
    slices = observation_slices()
    event_slice = slices.get("controller_event")
    if event_slice is None:
        return weights
    events = np.asarray(obs_np[:, event_slice], dtype=np.float32)
    if events.shape[1] < 7:
        return weights
    event_idx = np.argmax(events, axis=1)
    grasp_mask = np.isin(event_idx, [1, 2, 3])
    transport_mask = np.isin(event_idx, [4, 5, 6])
    weights[grasp_mask] = float(args.bc_anchor_grasp_weight)
    weights[transport_mask] = np.maximum(
        weights[transport_mask],
        float(args.bc_anchor_transport_weight),
    )
    return np.maximum(weights, 1e-6).astype(np.float32)


def _sample_bc_anchor_loss(
    model: ActorCritic,
    bc_actor: MLPPolicy | None,
    bc_anchor: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    obs = bc_anchor["obs"]
    actions = bc_anchor["actions"]
    weights = bc_anchor["weights"]
    n = int(obs.shape[0])
    batch_size = max(1, min(int(args.bc_anchor_batch_size), n))
    idx = torch.randint(0, n, (batch_size,), device=device)
    obs_batch = obs.index_select(0, idx)
    pred = model.actor(obs_batch)
    if args.policy_mode in ("residual", "safety_residual"):
        if bc_actor is None:
            return torch.zeros((), device=device)
        with torch.no_grad():
            base_action = bc_actor(obs_batch)
        pred = torch.clamp(base_action + float(args.residual_scale) * pred, -1.0, 1.0)
    target = actions.index_select(0, idx)
    sample_weights = weights.index_select(0, idx)
    per_sample_loss = torch.mean((pred - target) ** 2, dim=1)
    weighted_loss = per_sample_loss * sample_weights
    return weighted_loss.sum() / torch.clamp(sample_weights.sum(), min=1e-6)


def _sample_action(
    model: ActorCritic,
    obs: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    bc_actor: MLPPolicy | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    obs_norm = _normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1), obs_mean, obs_std)
    with torch.no_grad():
        tensor = torch.from_numpy(obs_norm).to(device)
        train_action, log_prob, value = model.act(tensor)
        env_action = train_action
        if args.policy_mode in ("residual", "safety_residual"):
            if bc_actor is None:
                raise RuntimeError("Residual policy mode requires a loaded BC actor.")
            base_tensor = tensor
            if args.policy_mode == "safety_residual":
                base_obs = _mask_human_flat_observation(
                    np.asarray(obs, dtype=np.float32).reshape(1, -1)
                )
                base_tensor = torch.from_numpy(_normalize_obs(base_obs, obs_mean, obs_std)).to(device)
            base_action = bc_actor(base_tensor)
            env_action = torch.clamp(
                base_action + float(args.residual_scale) * train_action,
                -1.0,
                1.0,
            )
    return (
        train_action.detach().cpu().numpy()[0].astype(np.float32),
        env_action.detach().cpu().numpy()[0].astype(np.float32),
        float(log_prob.detach().cpu().numpy()[0]),
        float(value.detach().cpu().numpy()[0]),
    )


def _predict_value(
    model: ActorCritic,
    obs: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
) -> float:
    obs_norm = _normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1), obs_mean, obs_std)
    with torch.no_grad():
        value = model.value(torch.from_numpy(obs_norm).to(device))
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
        delta = float(rewards[t]) + float(gamma) * next_v * nonterminal - float(values[t])
        gae = delta + float(gamma) * float(gae_lambda) * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def _maybe_load_bc_checkpoint(
    model: ActorCritic,
    checkpoint_path: str,
    device: torch.device,
    *,
    load_actor: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    obs_mean = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
    obs_std = np.ones((1, OBSERVATION_DIM), dtype=np.float32)
    if not checkpoint_path:
        return obs_mean, obs_std, {}

    path = _resolve_project_path(checkpoint_path)
    if not os.path.exists(path):
        print(f"[TrainPPO] BC checkpoint not found; starting actor from scratch: {path}", flush=True)
        return obs_mean, obs_std, {}

    checkpoint = _torch_load(path, device)
    loaded_actor = False
    if load_actor:
        try:
            model.actor.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as exc:
            print(f"[TrainPPO] BC actor load skipped due to shape mismatch: {exc}", flush=True)
        else:
            loaded_actor = True
            print("[TrainPPO] BC actor weights loaded.", flush=True)
    else:
        print("[TrainPPO] BC actor kept frozen as residual base; PPO actor starts as residual.", flush=True)

    if "obs_mean" in checkpoint and "obs_std" in checkpoint:
        obs_mean = _tensor_to_numpy(checkpoint["obs_mean"]).reshape(1, -1)
        obs_std = np.maximum(_tensor_to_numpy(checkpoint["obs_std"]).reshape(1, -1), 1e-6)
        obs_mean = _align_obs_dim(obs_mean, OBSERVATION_DIM)
        obs_std = _align_obs_dim(obs_std, OBSERVATION_DIM, fill_value=1.0)

    return obs_mean.astype(np.float32), obs_std.astype(np.float32), {"path": path, "loaded_actor": loaded_actor}


def _load_frozen_bc_actor(
    checkpoint_path: str,
    hidden_dims: tuple[int, ...],
    device: torch.device,
) -> MLPPolicy | None:
    if not checkpoint_path:
        return None
    path = _resolve_project_path(checkpoint_path)
    if not os.path.exists(path):
        return None
    checkpoint = _torch_load(path, device)
    obs_dim = int(checkpoint.get("obs_dim", OBSERVATION_DIM))
    action_dim = int(checkpoint.get("action_dim", ACTION_DIM))
    if action_dim != ACTION_DIM:
        return None
    checkpoint_policy_mode = str(checkpoint.get("policy_mode", "direct"))
    if args.policy_mode == "safety_residual" and checkpoint_policy_mode != "direct":
        raise ValueError(
            "--policy-mode safety_residual currently requires a direct frozen task policy. "
            f"Got policy_mode={checkpoint_policy_mode!r} from {path}. "
            "Use a BC checkpoint such as v2/policies/bc_pick_place_v1_100eps.pt."
        )
    bc_hidden_dims = tuple(int(value) for value in checkpoint.get("hidden_dims", hidden_dims))
    bc_actor = MLPPolicy(obs_dim, action_dim, hidden_dims=bc_hidden_dims).to(device)
    bc_actor.load_state_dict(checkpoint["model_state_dict"])
    bc_actor.eval()
    for param in bc_actor.parameters():
        param.requires_grad_(False)
    return bc_actor


def _zero_policy_output(policy: MLPPolicy) -> None:
    for module in reversed(policy.net):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
            return


def _save_checkpoint(
    path: str,
    model: ActorCritic,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    hidden_dims: tuple[int, ...],
    history: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    bc_meta: dict[str, Any],
    *,
    best_metric: dict[str, float] | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        "algo": "ppo",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.actor.state_dict().items()},
        "actor_critic_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "obs_mean": torch.from_numpy(obs_mean.squeeze(0).astype(np.float32)),
        "obs_std": torch.from_numpy(obs_std.squeeze(0).astype(np.float32)),
        "obs_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dims": hidden_dims,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "reward_version": REWARD_VERSION,
        "policy_mode": args.policy_mode,
        "residual_scale": float(args.residual_scale),
        "pseudo_errp_enabled": bool(args.pseudo_errp_enabled),
        "pseudo_errp_sources": parse_pseudo_errp_sources(args.pseudo_errp_sources),
        "human_replay_data": _resolve_project_path(args.human_replay_data)
        if args.human_replay_data
        else "",
        "human_replay_mode": args.human_replay_mode,
        "human_replay_episode_policy": args.human_replay_episode_policy,
        "source_bc_checkpoint": bc_meta.get("path", ""),
        "train_args": vars(args),
        "history": history,
        "episode_stats": episode_stats,
        "best_metric": best_metric or {},
        "log_std": model.log_std.detach().cpu(),
    }
    torch.save(checkpoint, path)


def _is_better_checkpoint(
    update_record: dict[str, Any],
    best_metric: dict[str, float],
    min_episodes: int,
) -> bool:
    if int(update_record["episodes"]) < int(min_episodes):
        return False
    success_rate = float(update_record["recent_success_rate"])
    recent_return = float(update_record["recent_return"])
    best_success = float(best_metric["success_rate"])
    best_return = float(best_metric["return"])
    if success_rate > best_success + 1e-9:
        return True
    if abs(success_rate - best_success) <= 1e-9 and recent_return > best_return:
        return True
    return False


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
    print(f"[TrainPPO] saved history: {path}", flush=True)


def _normalize_obs(obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray) -> np.ndarray:
    obs = _align_obs_dim(np.asarray(obs, dtype=np.float32), OBSERVATION_DIM)
    return ((obs - obs_mean) / np.maximum(obs_std, 1e-6)).astype(np.float32)


def _mask_human_flat_observation(obs: np.ndarray) -> np.ndarray:
    """Build the robot-only view expected by the frozen base policy."""

    masked = _align_obs_dim(np.asarray(obs, dtype=np.float32), OBSERVATION_DIM).copy()
    slices = observation_slices()
    for key in (
        "human_head_pos",
        "human_left_hand_pos",
        "human_right_hand_pos",
        "ee_to_left_hand",
        "ee_to_right_hand",
        "human_robot_collision",
        "near_human",
    ):
        field_slice = slices.get(key)
        if field_slice is not None:
            masked[:, field_slice] = 0.0
    dist_slice = slices.get("min_hand_gripper_dist")
    if dist_slice is not None:
        masked[:, dist_slice] = 10.0
    return masked


def _align_obs_dim(obs: np.ndarray, expected_dim: int, *, fill_value: float = 0.0) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[1] == expected_dim:
        return obs
    if obs.shape[1] > expected_dim:
        return obs[:, :expected_dim]
    pad = np.full((obs.shape[0], expected_dim - obs.shape[1]), fill_value, dtype=obs.dtype)
    return np.concatenate([obs, pad], axis=1)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False")
    return torch.device(requested)


def _parse_hidden_dims(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    return tuple(int(value) for value in values) if values else (256, 256)


def _parse_vec3(text: str) -> np.ndarray:
    try:
        values = [float(part.strip()) for part in str(text).split(",")]
    except Exception as exc:
        raise ValueError(f"Expected comma-separated x,y,z vector, got {text!r}") from exc
    if len(values) != 3:
        raise ValueError(f"Expected comma-separated x,y,z vector, got {text!r}")
    return np.asarray(values, dtype=np.float32)


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    project_path = os.path.abspath(os.path.join(PROJECT_DIR, path))
    if os.path.exists(project_path):
        return project_path
    return os.path.abspath(path)


def _resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


def _resolve_best_output_path(best_path: str, output_path: str) -> str:
    if best_path:
        return _resolve_output_path(best_path)
    stem, ext = os.path.splitext(output_path)
    return f"{stem}_best{ext or '.pt'}"


def _torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


try:
    _train()
except BaseException as exc:
    print(f"[TrainPPO] terminated by {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    simulation_app.close()
