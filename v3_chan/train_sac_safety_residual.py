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
for path in (os.path.join(SCRIPT_DIR, "rl"), os.path.join(SCRIPT_DIR, ".python_packages"), SCRIPT_DIR, PROJECT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SAC HRI safety residual over a frozen task policy.")
    parser.add_argument("--task-checkpoint", default=os.path.join(SCRIPT_DIR, "policies", "ppo_pick_place_v7_residual_rewardv4_strict_best.pt"))
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "policies", "sac_safety_residual_hri_v1.pt"))
    parser.add_argument("--best-output", default="")
    parser.add_argument("--best-min-episodes", type=int, default=5)
    parser.add_argument("--human-replay-data", default=os.path.join(SCRIPT_DIR, "trajectories", "prepared", "hri_train_all_new_sessions.hdf5"))
    parser.add_argument("--human-replay-mode", choices=("step", "loop"), default="step")
    parser.add_argument("--human-replay-episode-policy", choices=("cycle", "random"), default="cycle")
    parser.add_argument("--encounter-manifest", default="")
    parser.add_argument("--encounter-policy", choices=("cycle", "random"), default="random")
    parser.add_argument("--encounter-severity-mix", default="safe=0.40,gate_only=0.25,near=0.20,near_miss=0.10,collision=0.05")
    parser.add_argument("--encounter-anchor-mode", choices=("ee", "world"), default="ee")
    parser.add_argument("--encounter-timebase", choices=("recorded", "step"), default="recorded")
    parser.add_argument("--encounter-playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--allow-legacy-source-configuration",
        action="store_true",
        help="Explicitly allow random-layout fallback for legacy encounter sources.",
    )
    parser.add_argument("--no-encounter-phase-match", dest="encounter_phase_match", action="store_false")
    parser.add_argument("--no-encounter-event-match", dest="encounter_event_match", action="store_false")
    parser.set_defaults(encounter_phase_match=True, encounter_event_match=True)
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=1024)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--target-entropy", type=float, default=-5.0)
    parser.add_argument("--init-temperature", type=float, default=0.01)
    parser.add_argument(
        "--fixed-temperature",
        action="store_true",
        help="Keep entropy temperature fixed at --init-temperature.",
    )
    parser.add_argument(
        "--target-q-clip",
        type=float,
        default=0.0,
        help="Symmetric critic-target clip; non-positive disables clipping.",
    )
    parser.add_argument(
        "--critic-loss",
        choices=("mse", "huber"),
        default="mse",
    )
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=-1.5)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--distance-progress-weight", type=float, default=0.0)
    parser.add_argument("--distance-progress-clip-m", type=float, default=0.03)
    parser.add_argument("--residual-reward-penalty", type=float, default=1.0)
    parser.add_argument("--residual-actor-penalty", type=float, default=0.5)
    parser.add_argument("--random-residual-std", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--residual-alpha", type=float, default=0.1)
    parser.add_argument("--xyz-only-residual", action="store_true")
    parser.add_argument("--safety-gate-start-dist", type=float, default=0.13)
    parser.add_argument("--safety-gate-full-dist", type=float, default=0.05)
    parser.add_argument("--gate-sample-ratio", type=float, default=1.0, help="Desired fraction of gate-active transitions per SAC batch.")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--success-dist", type=float, default=0.06)
    parser.add_argument("--phase-gate-close-dist", type=float, default=0.075)
    parser.add_argument("--phase-gate-max-hold", type=int, default=320)
    parser.add_argument("--release-gate-dist", type=float, default=-1.0)
    parser.add_argument("--release-gate-max-hold", type=int, default=240)
    parser.add_argument("--pseudo-errp", dest="pseudo_errp_enabled", action="store_true")
    parser.add_argument("--no-pseudo-errp", dest="pseudo_errp_enabled", action="store_false")
    parser.set_defaults(pseudo_errp_enabled=True)
    parser.add_argument("--pseudo-errp-sources", default="all")
    parser.add_argument("--log-every-steps", type=int, default=1000)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    return parser.parse_args()


args = _parse_args()


def _ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        bundle = os.environ.get("ISAAC_TORCH_BUNDLE", os.path.expanduser("~/isaac-sim-4.5.0/exts/omni.isaac.ml_archive/pip_prebundle"))
        if os.path.isdir(bundle) and bundle not in sys.path:
            sys.path.insert(0, bundle)
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
print(f"[TrainSafetySAC] SimulationApp headless={not args.render}", flush=True)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
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


def _mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(dim, hidden), nn.ReLU()))
        dim = hidden
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class SACActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.mean = MLPPolicy(obs_dim, action_dim, hidden_dims=hidden_dims)
        self.log_std = _mlp(obs_dim, action_dim, hidden_dims)
        _zero_last_layer(self.mean.net)
        _zero_last_layer(self.log_std)
        with torch.no_grad():
            self.log_std[-1].bias.fill_(-3.0)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(obs)
        log_std = torch.clamp(self.log_std(obs), args.log_std_min, args.log_std_max)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
        controlled_dims = 3 if args.xyz_only_residual else ACTION_DIM
        if controlled_dims < ACTION_DIM:
            action = action.clone()
            action[..., controlled_dims:] = 0.0
        return action, log_prob[..., :controlled_dims].sum(dim=-1, keepdim=True)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(self.mean(obs))
        if args.xyz_only_residual:
            action = action.clone()
            action[..., 3:] = 0.0
        return action


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, HRI_OBS_DIM), dtype=np.float32)
        self.actions = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, HRI_OBS_DIM), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.gates = np.zeros((capacity, 1), dtype=np.float32)
        self.next_gates = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    @property
    def active_size(self) -> int:
        return int(np.count_nonzero(self.gates[: self.size, 0] > 0.0))

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        gate: float,
        next_gate: float,
    ) -> None:
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action if gate > 0.0 else 0.0
        self.rewards[i, 0] = reward
        self.next_obs[i], self.dones[i, 0] = next_obs, float(done)
        self.gates[i, 0], self.next_gates[i, 0] = gate, next_gate
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, gate_ratio: float, device: torch.device) -> tuple[torch.Tensor, ...]:
        valid = np.arange(self.size)
        active = valid[self.gates[: self.size, 0] > 0.0]
        inactive = valid[self.gates[: self.size, 0] <= 0.0]
        requested_active = int(round(batch_size * np.clip(gate_ratio, 0.0, 1.0)))
        want_active = requested_active if len(active) else 0
        want_inactive = batch_size - want_active if len(inactive) else 0
        count = want_active + want_inactive
        pools = []
        if want_active:
            pools.append(np.random.choice(active, want_active, replace=len(active) < want_active))
        if want_inactive:
            pools.append(np.random.choice(inactive, want_inactive, replace=len(inactive) < want_inactive))
        idx = np.concatenate(pools) if pools else np.empty(0, dtype=np.int64)
        if count < batch_size:
            idx = np.concatenate((idx, np.random.choice(valid, batch_size - count, replace=self.size < batch_size)))
        np.random.shuffle(idx)
        arrays = (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            self.gates[idx],
            self.next_gates[idx],
        )
        return tuple(torch.from_numpy(value).to(device) for value in arrays)


class TaskPolicyRunner:
    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        self.path = _resolve_project_path(checkpoint_path)
        checkpoint = _torch_load(self.path, device)
        self.device = device
        self.obs_mean = _to_numpy(checkpoint["obs_mean"]).reshape(1, -1)
        self.obs_std = np.maximum(_to_numpy(checkpoint["obs_std"]).reshape(1, -1), 1e-6)
        self.obs_dim = int(checkpoint.get("obs_dim", 84))
        self.action_dim = int(checkpoint.get("action_dim", ACTION_DIM))
        hidden = tuple(int(v) for v in checkpoint.get("hidden_dims", (256, 256)))
        self.policy_mode = str(checkpoint.get("policy_mode", "direct"))
        self.residual_scale = float(checkpoint.get("residual_scale", 1.0))
        self.model = MLPPolicy(self.obs_dim, self.action_dim, hidden_dims=hidden).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.base_model = self.base_mean = self.base_std = self.base_dim = None
        if self.policy_mode == "residual":
            base = _torch_load(_resolve_project_path(str(checkpoint["source_bc_checkpoint"])), device)
            self.base_dim = int(base.get("obs_dim", 84))
            self.base_mean = _to_numpy(base["obs_mean"]).reshape(1, -1)
            self.base_std = np.maximum(_to_numpy(base["obs_std"]).reshape(1, -1), 1e-6)
            self.base_model = MLPPolicy(self.base_dim, self.action_dim, hidden_dims=tuple(int(v) for v in base.get("hidden_dims", hidden))).to(device)
            self.base_model.load_state_dict(base["model_state_dict"])
            self.base_model.eval()

    def _predict(self, model: nn.Module, obs: np.ndarray, mean: np.ndarray, std: np.ndarray, dim: int) -> np.ndarray:
        x = _align(np.asarray(obs, dtype=np.float32).reshape(1, -1), dim)
        with torch.no_grad():
            return model(torch.from_numpy(((x - _align(mean, dim)) / np.maximum(_align(std, dim, 1.0), 1e-6)).astype(np.float32)).to(self.device)).cpu().numpy()[0]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = _mask_human_obs(obs)
        action = self._predict(self.model, obs, self.obs_mean, self.obs_std, self.obs_dim)
        if self.policy_mode == "residual":
            base = self._predict(self.base_model, obs, self.base_mean, self.base_std, int(self.base_dim))
            action = base + self.residual_scale * action
        return clip_action(action)


def _run() -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    hidden = _hidden_dims(args.hidden_dims)
    task_policy = TaskPolicyRunner(args.task_checkpoint, device)
    actor = SACActor(HRI_OBS_DIM, ACTION_DIM, hidden).to(device)
    q1, q2 = _mlp(HRI_OBS_DIM + ACTION_DIM, 1, hidden).to(device), _mlp(HRI_OBS_DIM + ACTION_DIM, 1, hidden).to(device)
    tq1, tq2 = _mlp(HRI_OBS_DIM + ACTION_DIM, 1, hidden).to(device), _mlp(HRI_OBS_DIM + ACTION_DIM, 1, hidden).to(device)
    tq1.load_state_dict(q1.state_dict()); tq2.load_state_dict(q2.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    log_alpha = torch.tensor(
        np.log(args.init_temperature),
        dtype=torch.float32,
        device=device,
        requires_grad=not args.fixed_temperature,
    )
    alpha_opt = (
        None if args.fixed_temperature else torch.optim.Adam([log_alpha], lr=args.lr)
    )
    replay = ReplayBuffer(args.replay_size)
    human_replay = _load_human_replay()
    release_dist = None if args.release_gate_dist < 0 else args.release_gate_dist
    env = IsaacPickPlaceEnv(PickPlaceEnvConfig(max_episode_steps=args.max_episode_steps, success_dist=args.success_dist, action_scale=args.action_scale, gripper_mode="event", phase_gate_close_dist=args.phase_gate_close_dist, phase_gate_max_hold=args.phase_gate_max_hold, release_gate_dist=release_dist, release_gate_max_hold=args.release_gate_max_hold, observation_mode="flat", seed=args.seed, render=args.render, pseudo_errp_enabled=args.pseudo_errp_enabled, pseudo_errp_sources=parse_pseudo_errp_sources(args.pseudo_errp_sources)), human_state_fn=human_replay)
    output = _resolve_project_path(args.output)
    best_output = _resolve_project_path(args.best_output) if args.best_output else os.path.splitext(output)[0] + "_best.pt"
    obs_mean, obs_std = np.zeros(HRI_OBS_DIM, np.float32), np.ones(HRI_OBS_DIM, np.float32)
    history: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    best = {"success_rate": -1.0, "return": -float("inf"), "step": 0}
    metrics: dict[str, float] = {}
    started = time.time()
    obs, info = _reset_training_episode(env, human_replay, 0, args.seed)
    ep_return = ep_len = ep_gate = ep_errp = 0.0
    print(f"[TrainSafetySAC] task={task_policy.path} human_replay={human_replay.path if human_replay else 'off'} alpha={args.residual_alpha} gate=({args.safety_gate_start_dist},{args.safety_gate_full_dist})", flush=True)
    try:
        for step in range(1, args.total_steps + 1):
            hri_obs = flatten_hri_observation(info.get("obs_dict", {}))
            gate = _gate(info.get("obs_dict", {}))
            if step <= args.learning_starts:
                residual = np.clip(
                    np.random.normal(0.0, args.random_residual_std, ACTION_DIM),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            else:
                residual = _actor_action(actor, hri_obs, device, stochastic=True)
            if args.xyz_only_residual:
                residual[3:] = 0.0
            env_action = clip_action(task_policy.predict(obs) + gate * args.residual_alpha * residual)
            next_obs, reward, terminated, truncated, next_info = env.step(env_action)
            done = bool(terminated or truncated)
            next_hri_obs = flatten_hri_observation(next_info.get("obs_dict", {}))
            next_gate = _gate(next_info.get("obs_dict", {}))
            progress_reward = _distance_progress_reward(
                _surface_gap(info.get("obs_dict", {})),
                _surface_gap(next_info.get("obs_dict", {})),
                max(gate, next_gate),
            )
            residual_cost = args.residual_reward_penalty * gate * float(np.square(residual).sum())
            training_reward = (float(reward) + progress_reward - residual_cost) * args.reward_scale
            replay.add(hri_obs, residual, training_reward, next_hri_obs, done, gate, next_gate)
            ep_return += reward; ep_len += 1; ep_gate += int(gate > 0); ep_errp += float(next_info.get("errp_feedback", 0.0))
            obs, info = np.asarray(next_obs, np.float32), next_info
            if (
                replay.size >= max(args.batch_size, args.learning_starts)
                and replay.active_size >= min(32, args.batch_size)
            ):
                for _ in range(args.updates_per_step):
                    metrics = _sac_update(actor, q1, q2, tq1, tq2, actor_opt, critic_opt, log_alpha, alpha_opt, replay, device)
            if done:
                encounter = human_replay.current_scenario if isinstance(human_replay, HumanEncounterReplay) else {}
                episodes.append({"episode": len(episodes), "return": float(ep_return), "length": int(ep_len), "success": bool(terminated), "truncated": bool(truncated), "cube_target_dist": float(info["cube_target_dist"]), "grasped": bool(info["has_grasped_cube"]), "errp_feedback_sum": float(ep_errp), "gate_active_count": int(ep_gate), "encounter_id": str(encounter.get("id", "")), "encounter_target_severity": str(encounter.get("target_severity", "")), "encounter_target_phase": str(encounter.get("task_phase", "")), "encounter_source_session": str(encounter.get("session_id", ""))})
                episode_index = len(episodes); seed = args.seed + episode_index
                obs, info = _reset_training_episode(
                    env,
                    human_replay,
                    episode_index,
                    seed,
                )
                ep_return = ep_len = ep_gate = ep_errp = 0.0
            if step % args.log_every_steps == 0 or step == args.total_steps:
                recent = episodes[-20:]
                record = {"step": step, "episodes": len(episodes), "recent_success_rate": float(np.mean([e["success"] for e in recent])) if recent else 0.0, "recent_return": float(np.mean([e["return"] for e in recent])) if recent else 0.0, "recent_errp_feedback": float(np.mean([e["errp_feedback_sum"] for e in recent])) if recent else 0.0, "replay_size": replay.size, "gate_fraction": float(np.mean(replay.gates[:replay.size, 0] > 0)) if replay.size else 0.0, **metrics}
                history.append(record)
                print(f"[TrainSafetySAC] step={step:06d} episodes={len(episodes)} success={record['recent_success_rate']:.3f} return={record['recent_return']:.2f} errp={record['recent_errp_feedback']:.2f} actor={record.get('actor_loss', 0):.4f} critic={record.get('critic_loss', 0):.4f} alpha={record.get('alpha', args.init_temperature):.4f} q={record.get('mean_q', 0):.3f} residual={record.get('mean_residual_norm', 0):.4f} gate={record.get('gate_fraction', 0):.3f}", flush=True)
                if len(episodes) >= args.best_min_episodes and (record["recent_success_rate"], record["recent_return"]) > (best["success_rate"], best["return"]):
                    best = {"success_rate": record["recent_success_rate"], "return": record["recent_return"], "step": step}
                    _save(best_output, actor, q1, q2, tq1, tq2, log_alpha, obs_mean, obs_std, hidden, history, episodes, best)
                    print(f"[TrainSafetySAC] saved best checkpoint: {best_output}", flush=True)
            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                _save(output, actor, q1, q2, tq1, tq2, log_alpha, obs_mean, obs_std, hidden, history, episodes, best)
    finally:
        env.close()
        if human_replay is not None: human_replay.close()
    _save(output, actor, q1, q2, tq1, tq2, log_alpha, obs_mean, obs_std, hidden, history, episodes, best)
    history_path = os.path.splitext(output)[0] + "_history.json"
    with open(history_path, "w", encoding="utf-8") as f: json.dump({"created_unix": time.time(), "duration_sec": time.time() - started, "updates": history, "episodes": episodes}, f, indent=2)
    print(f"[TrainSafetySAC] saved history: {history_path}\n[TrainSafetySAC] saved checkpoint: {output}", flush=True)


def _sac_update(actor, q1, q2, tq1, tq2, actor_opt, critic_opt, log_alpha, alpha_opt, replay, device) -> dict[str, float]:
    obs, action, reward, next_obs, done, gate, next_gate = replay.sample(
        args.batch_size,
        args.gate_sample_ratio,
        device,
    )
    with torch.no_grad():
        next_action, next_logp = actor.sample(next_obs)
        next_action = next_action * (next_gate > 0.0)
        target_q = torch.minimum(tq1(torch.cat((next_obs, next_action), 1)), tq2(torch.cat((next_obs, next_action), 1)))
        if args.target_q_clip > 0.0:
            target_q = torch.clamp(
                target_q,
                -args.target_q_clip,
                args.target_q_clip,
            )
        target = reward + args.gamma * (1.0 - done) * (
            target_q - (next_gate > 0.0) * log_alpha.exp() * next_logp
        )
        if args.target_q_clip > 0.0:
            target = torch.clamp(
                target,
                -args.target_q_clip,
                args.target_q_clip,
            )
    q1_value, q2_value = q1(torch.cat((obs, action), 1)), q2(torch.cat((obs, action), 1))
    loss_fn = F.smooth_l1_loss if args.critic_loss == "huber" else F.mse_loss
    critic_loss = loss_fn(q1_value, target) + loss_fn(q2_value, target)
    if not torch.isfinite(critic_loss):
        raise FloatingPointError("SAC critic loss became non-finite")
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), args.max_grad_norm)
    critic_opt.step()
    sampled_action, logp = actor.sample(obs)
    sampled_q = torch.minimum(q1(torch.cat((obs, sampled_action), 1)), q2(torch.cat((obs, sampled_action), 1)))
    active_weight = (gate > 0.0).float()
    weight_sum = torch.clamp(active_weight.sum(), min=1.0)
    residual_penalty = sampled_action.square().sum(dim=1, keepdim=True)
    actor_terms = (
        log_alpha.exp().detach() * logp
        - sampled_q
        + args.residual_actor_penalty * residual_penalty
    )
    actor_loss = (active_weight * actor_terms).sum() / weight_sum
    if not torch.isfinite(actor_loss):
        raise FloatingPointError("SAC actor loss became non-finite")
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
    actor_opt.step()
    if alpha_opt is None:
        alpha_loss = torch.zeros((), dtype=torch.float32, device=device)
    else:
        alpha_terms = -(log_alpha * (logp.detach() + args.target_entropy))
        alpha_loss = (active_weight * alpha_terms).sum() / weight_sum
        if not torch.isfinite(alpha_loss):
            raise FloatingPointError("SAC temperature loss became non-finite")
        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_opt.step()
    with torch.no_grad():
        for target_param, param in zip(tq1.parameters(), q1.parameters()): target_param.mul_(1 - args.tau).add_(param, alpha=args.tau)
        for target_param, param in zip(tq2.parameters(), q2.parameters()): target_param.mul_(1 - args.tau).add_(param, alpha=args.tau)
    return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item()), "alpha_loss": float(alpha_loss.item()), "alpha": float(log_alpha.exp().item()), "mean_q": float(sampled_q.mean().item()), "mean_residual_norm": float(sampled_action.norm(dim=1).mean().item())}


def _save(path, actor, q1, q2, tq1, tq2, log_alpha, obs_mean, obs_std, hidden, history, episodes, best) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"algo": "sac_safety_residual_v2", "model_state_dict": {k: v.detach().cpu() for k, v in actor.mean.state_dict().items()}, "sac_actor_state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()}, "q1_state_dict": q1.state_dict(), "q2_state_dict": q2.state_dict(), "target_q1_state_dict": tq1.state_dict(), "target_q2_state_dict": tq2.state_dict(), "log_alpha": log_alpha.detach().cpu(), "obs_mean": torch.from_numpy(obs_mean), "obs_std": torch.from_numpy(obs_std), "obs_dim": HRI_OBS_DIM, "observation_fields": list(HRI_OBS_FIELD_NAMES), "action_dim": ACTION_DIM, "hidden_dims": hidden, "observation_version": HRI_OBSERVATION_VERSION, "policy_mode": "direct", "output_activation": "tanh", "residual_alpha": args.residual_alpha, "xyz_only_residual": args.xyz_only_residual, "safety_gate_start_dist": args.safety_gate_start_dist, "safety_gate_full_dist": args.safety_gate_full_dist, "task_checkpoint": _resolve_project_path(args.task_checkpoint), "human_replay_data": _resolve_project_path(args.human_replay_data), "encounter_manifest": _resolve_project_path(args.encounter_manifest) if args.encounter_manifest else "", "pseudo_errp_enabled": args.pseudo_errp_enabled, "pseudo_errp_sources": parse_pseudo_errp_sources(args.pseudo_errp_sources), "train_args": vars(args), "history": history, "episode_stats": episodes, "best_metric": best}, path)


def _actor_action(actor, obs, device, stochastic: bool) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(obs, np.float32).reshape(1, -1)).to(device)
        action = actor.sample(x)[0] if stochastic else actor.deterministic(x)
    return action.cpu().numpy()[0].astype(np.float32)


def _gate(obs: dict[str, np.ndarray]) -> float:
    dist = _surface_gap(obs)
    denom = args.safety_gate_start_dist - args.safety_gate_full_dist
    return float(np.clip((args.safety_gate_start_dist - dist) / denom, 0, 1)) if denom > 1e-6 else 0.0


def _surface_gap(obs: dict[str, np.ndarray]) -> float:
    value = np.asarray(
        obs.get(
            "min_hand_end_effector_surface_gap",
            [MISSING_DISTANCE_M],
        ),
        dtype=float,
    ).reshape(-1)
    return (
        float(value[0])
        if value.size and np.isfinite(value[0])
        else float(MISSING_DISTANCE_M)
    )


def _distance_progress_reward(
    current_gap: float,
    next_gap: float,
    gate: float,
) -> float:
    if args.distance_progress_weight == 0.0:
        return 0.0
    if (
        current_gap >= MISSING_DISTANCE_M * 0.5
        or next_gap >= MISSING_DISTANCE_M * 0.5
    ):
        return 0.0
    delta = float(
        np.clip(
            next_gap - current_gap,
            -args.distance_progress_clip_m,
            args.distance_progress_clip_m,
        )
    )
    return float(args.distance_progress_weight) * float(gate) * delta


def _mask_human_obs(obs: np.ndarray) -> np.ndarray:
    result = np.asarray(obs, np.float32).reshape(-1).copy(); slices = observation_slices()
    for name in ("human_head_pos", "human_left_hand_pos", "human_right_hand_pos", "ee_to_left_hand", "ee_to_right_hand", "human_robot_collision", "near_human"):
        sl = slices.get(name)
        if sl is not None and sl.stop <= result.size: result[sl] = 0.0
    sl = slices.get("min_hand_gripper_dist")
    if sl is not None and sl.stop <= result.size: result[sl] = MISSING_DISTANCE_M
    return result


def _load_human_replay() -> HumanTrajectoryReplay | HumanEncounterReplay | None:
    if args.encounter_manifest:
        path = _resolve_project_path(args.encounter_manifest)
        if not os.path.exists(path): raise FileNotFoundError(f"--encounter-manifest not found: {path}")
        return HumanEncounterReplay(path, episode_policy=args.encounter_policy, severity_mix=args.encounter_severity_mix, anchor_mode=args.encounter_anchor_mode, phase_match=args.encounter_phase_match, event_match=args.encounter_event_match, playback_timebase=args.encounter_timebase, playback_speed=args.encounter_playback_speed, seed=args.seed)
    if not args.human_replay_data: return None
    path = _resolve_project_path(args.human_replay_data)
    if not os.path.exists(path): raise FileNotFoundError(f"--human-replay-data not found: {path}")
    return HumanTrajectoryReplay(path, mode=args.human_replay_mode, episode_policy=args.human_replay_episode_policy, seed=args.seed)


def _zero_last_layer(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear): nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias); return


def _align(value: np.ndarray, dim: int, fill: float = 0.0) -> np.ndarray:
    value = np.asarray(value, np.float32)
    if value.ndim == 1: value = value.reshape(1, -1)
    if value.shape[1] >= dim: return value[:, :dim]
    return np.concatenate((value, np.full((value.shape[0], dim - value.shape[1]), fill, np.float32)), 1)


def _to_numpy(value) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, np.float32)


def _torch_load(path: str, device):
    try: return torch.load(path, map_location=device, weights_only=False)
    except TypeError: return torch.load(path, map_location=device)


def _resolve_project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_DIR, path))


def _hidden_dims(text: str) -> tuple[int, ...]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    return values or (256, 256)


def _device(requested: str) -> torch.device:
    if requested == "auto": requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("Requested CUDA but it is unavailable")
    return torch.device(requested)


try:
    _run()
except BaseException as exc:
    print(f"[TrainSafetySAC] terminated by {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    simulation_app.close()
