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
        default=os.path.join(SCRIPT_DIR, "policies", "ppo_pick_place_v1.pt"),
        help="Output actor checkpoint. It is compatible with evaluate_rollout_policy.py.",
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-std-init", type=float, default=-1.2)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--success-dist", type=float, default=0.06)
    parser.add_argument("--release-gate-dist", type=float, default=0.06)
    parser.add_argument("--release-gate-max-hold", type=int, default=240)
    parser.add_argument(
        "--allow-success-before-release",
        action="store_true",
        help="Use the legacy success condition: cube near target is enough even before release.",
    )
    parser.add_argument("--phase-gate-close-dist", type=float, default=0.066)
    parser.add_argument("--phase-gate-max-hold", type=int, default=160)
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
    PickPlaceEnvConfig,
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


def _train() -> None:
    started_at = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)

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
            release_gate_dist=release_gate_dist,
            release_gate_max_hold=args.release_gate_max_hold,
            require_release_for_success=not args.allow_success_before_release,
            observation_mode="flat",
            seed=args.seed,
            render=args.render,
        )
    )

    model = ActorCritic(
        OBSERVATION_DIM,
        ACTION_DIM,
        hidden_dims=hidden_dims,
        log_std_init=args.log_std_init,
    ).to(device)
    obs_mean, obs_std, bc_meta = _maybe_load_bc_checkpoint(model, args.bc_checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"[TrainPPO] total_steps={args.total_steps} rollout_steps={args.rollout_steps} "
        f"device={device} torch={torch.__version__} reward={REWARD_VERSION}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"[TrainPPO] cuda={torch.cuda.get_device_name(0)}", flush=True)
    if bc_meta:
        print(f"[TrainPPO] initialized actor from BC checkpoint: {bc_meta['path']}", flush=True)

    obs, info = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    total_steps = 0
    update_idx = 0
    episode_stats: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    last_done = False

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
                device,
                remaining_steps=args.total_steps - total_steps,
                episode_return=episode_return,
                episode_length=episode_length,
                episode_stats=episode_stats,
                total_steps=total_steps,
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
            print(
                f"[TrainPPO] update={update_idx:04d} steps={total_steps} "
                f"episodes={len(episode_stats)} recent_success={update_record['recent_success_rate']:.3f} "
                f"recent_return={update_record['recent_return']:.2f} "
                f"pi={metrics['policy_loss']:.4f} vf={metrics['value_loss']:.4f} "
                f"entropy={metrics['entropy']:.3f}",
                flush=True,
            )
            if args.save_every_updates > 0 and update_idx % args.save_every_updates == 0:
                _save_checkpoint(args.output, model, obs_mean, obs_std, hidden_dims, history, episode_stats, bc_meta)
    finally:
        env.close()

    _save_checkpoint(args.output, model, obs_mean, obs_std, hidden_dims, history, episode_stats, bc_meta)
    _save_history(args.output, history, episode_stats, started_at)
    print(f"[TrainPPO] saved checkpoint: {args.output}", flush=True)


def _collect_rollout(
    env: IsaacPickPlaceEnv,
    model: ActorCritic,
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
        action, log_prob, value = _sample_action(model, obs, obs_mean, obs_std, device)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = bool(terminated or truncated)

        obs_buf.append(np.asarray(obs, dtype=np.float32))
        action_buf.append(np.asarray(action, dtype=np.float32))
        log_prob_buf.append(float(log_prob))
        reward_buf.append(float(reward))
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
                }
            )
            obs, info = env.reset(seed=int(args.seed + len(episode_stats)))
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
    losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
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
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            losses["policy_loss"] += float(policy_loss.item())
            losses["value_loss"] += float(value_loss.item())
            losses["entropy"] += float(entropy_mean.item())
            updates += 1

    return {key: value / max(1, updates) for key, value in losses.items()}


def _sample_action(
    model: ActorCritic,
    obs: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    obs_norm = _normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1), obs_mean, obs_std)
    with torch.no_grad():
        tensor = torch.from_numpy(obs_norm).to(device)
        action, log_prob, value = model.act(tensor)
    return (
        action.detach().cpu().numpy()[0].astype(np.float32),
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
    try:
        model.actor.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        print(f"[TrainPPO] BC actor load skipped due to shape mismatch: {exc}", flush=True)
    else:
        print("[TrainPPO] BC actor weights loaded.", flush=True)

    if "obs_mean" in checkpoint and "obs_std" in checkpoint:
        obs_mean = _tensor_to_numpy(checkpoint["obs_mean"]).reshape(1, -1)
        obs_std = np.maximum(_tensor_to_numpy(checkpoint["obs_std"]).reshape(1, -1), 1e-6)
        obs_mean = _align_obs_dim(obs_mean, OBSERVATION_DIM)
        obs_std = _align_obs_dim(obs_std, OBSERVATION_DIM, fill_value=1.0)

    return obs_mean.astype(np.float32), obs_std.astype(np.float32), {"path": path}


def _save_checkpoint(
    path: str,
    model: ActorCritic,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    hidden_dims: tuple[int, ...],
    history: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    bc_meta: dict[str, Any],
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
        "source_bc_checkpoint": bc_meta.get("path", ""),
        "train_args": vars(args),
        "history": history,
        "episode_stats": episode_stats,
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
    print(f"[TrainPPO] saved history: {path}", flush=True)


def _normalize_obs(obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray) -> np.ndarray:
    obs = _align_obs_dim(np.asarray(obs, dtype=np.float32), OBSERVATION_DIM)
    return ((obs - obs_mean) / np.maximum(obs_std, 1e-6)).astype(np.float32)


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


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


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
