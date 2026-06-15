from __future__ import annotations

import argparse
import csv
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
    default_json = os.path.join(SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval.json")
    default_csv = os.path.join(SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval.csv")
    parser = argparse.ArgumentParser(description="Evaluate a trained BC policy by rolling it out in Isaac Sim.")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(SCRIPT_DIR, "policies", "bc_pick_place_v1_100eps.pt"),
        help="PyTorch checkpoint produced by v2/rl/train_bc.py.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--render", action="store_true", help="Render the rollout window.")
    parser.add_argument("--action-scale", type=float, default=1.0)
    orientation_group = parser.add_mutually_exclusive_group()
    orientation_group.add_argument(
        "--fixed-orientation",
        dest="fixed_orientation",
        action="store_true",
        help="Use the expert's fixed top-down gripper orientation.",
    )
    orientation_group.add_argument(
        "--free-orientation",
        dest="fixed_orientation",
        action="store_false",
        help="Leave the gripper orientation unconstrained.",
    )
    parser.set_defaults(fixed_orientation=True)
    parser.add_argument("--gripper-mode", choices=("event", "rule", "policy"), default="event")
    parser.add_argument("--success-dist", type=float, default=0.06)
    parser.add_argument("--phase-gate-close-dist", type=float, default=0.066)
    parser.add_argument("--phase-gate-max-hold", type=int, default=160)
    parser.add_argument("--log-every", type=int, default=1, help="Episode logging interval. 0 disables logs.")
    parser.add_argument("--output-json", default=default_json, help="Path to save summary and per-episode JSON.")
    parser.add_argument("--output-csv", default=default_csv, help="Path to save per-episode CSV.")
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
print(f"[EvalRollout] SimulationApp headless={not args.render}", flush=True)

import torch  # noqa: E402

from rl import IsaacPickPlaceEnv, PickPlaceEnvConfig  # noqa: E402
from rl.actions import ACTION_DIM, clip_action  # noqa: E402
from rl.observations import OBSERVATION_DIM  # noqa: E402
from rl.policies import MLPPolicy  # noqa: E402


class PolicyRunner:
    def __init__(self, checkpoint_path: str, device_name: str) -> None:
        self.checkpoint_path = _resolve_project_path(checkpoint_path)
        self.device = _select_device(device_name)
        checkpoint = _torch_load(self.checkpoint_path, self.device)
        self.target_version = str(checkpoint.get("target_version", "task_space_action_v0"))
        self.action_version = str(checkpoint.get("action_version", "action_v0_task_space"))
        if self.target_version == "expert_arm_joint_action_v0":
            raise ValueError("evaluate_rollout_policy.py currently supports 5D task-space policies only.")

        self.obs_mean = _tensor_to_numpy(checkpoint["obs_mean"]).reshape(1, -1)
        self.obs_std = _tensor_to_numpy(checkpoint["obs_std"]).reshape(1, -1)
        self.target_mean = None
        self.target_std = None
        self.target_min = None
        self.target_max = None
        if "target_mean" in checkpoint and "target_std" in checkpoint:
            self.target_mean = _tensor_to_numpy(checkpoint["target_mean"]).reshape(1, -1)
            self.target_std = _tensor_to_numpy(checkpoint["target_std"]).reshape(1, -1)
        if "target_min" in checkpoint and "target_max" in checkpoint:
            self.target_min = _tensor_to_numpy(checkpoint["target_min"]).reshape(1, -1)
            self.target_max = _tensor_to_numpy(checkpoint["target_max"]).reshape(1, -1)

        hidden_dims = tuple(int(value) for value in checkpoint.get("hidden_dims", (256, 256)))
        self.obs_dim = int(checkpoint.get("obs_dim", OBSERVATION_DIM))
        action_dim = int(checkpoint.get("action_dim", ACTION_DIM))
        if action_dim != ACTION_DIM:
            raise ValueError(f"Checkpoint action dim {action_dim} != runtime action dim {ACTION_DIM}")

        self.model = MLPPolicy(self.obs_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.metadata = {
            "checkpoint": self.checkpoint_path,
            "target_version": self.target_version,
            "action_version": self.action_version,
            "observation_version": str(checkpoint.get("observation_version", "")),
            "reward_version": str(checkpoint.get("reward_version", "")),
            "obs_dim": self.obs_dim,
            "action_dim": action_dim,
            "hidden_dims": list(hidden_dims),
            "torch_version": torch.__version__,
            "device": str(self.device),
        }
        print(
            f"[EvalRollout] loaded checkpoint={self.checkpoint_path} device={self.device} "
            f"torch={torch.__version__} action={self.action_version} obs_dim={self.obs_dim}",
            flush=True,
        )
        if self.device.type == "cuda":
            print(f"[EvalRollout] cuda={torch.cuda.get_device_name(0)}", flush=True)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_policy = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs_policy = _align_obs_dim(obs_policy, self.obs_dim)
        obs_norm = (obs_policy - self.obs_mean) / np.maximum(self.obs_std, 1e-6)
        with torch.no_grad():
            tensor = torch.from_numpy(obs_norm.astype(np.float32)).to(self.device)
            action = self.model(tensor).detach().cpu().numpy()[0]
        if self.target_mean is not None and self.target_std is not None:
            action = (action.reshape(1, -1) * self.target_std + self.target_mean)[0]
            if self.target_min is not None and self.target_max is not None:
                action = np.clip(action.reshape(1, -1), self.target_min, self.target_max)[0]
            return np.asarray(action, dtype=np.float32)
        return clip_action(action)


def _select_device(requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False")
    return torch.device(requested)


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    project_path = os.path.abspath(os.path.join(PROJECT_DIR, path))
    if os.path.exists(project_path):
        return project_path
    return os.path.abspath(path)


def _resolve_output_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


def _torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _align_obs_dim(obs_policy: np.ndarray, expected_dim: int) -> np.ndarray:
    if obs_policy.shape[1] == expected_dim:
        return obs_policy
    if obs_policy.shape[1] > expected_dim:
        return obs_policy[:, :expected_dim]
    pad = np.zeros((obs_policy.shape[0], expected_dim - obs_policy.shape[1]), dtype=obs_policy.dtype)
    return np.concatenate([obs_policy, pad], axis=1)


def _run() -> None:
    np.random.seed(args.seed)
    started_at = time.time()
    runner = PolicyRunner(args.checkpoint, args.device)
    env = IsaacPickPlaceEnv(
        PickPlaceEnvConfig(
            max_episode_steps=args.max_steps,
            success_dist=args.success_dist,
            action_scale=args.action_scale,
            action_version=runner.action_version,
            fixed_orientation=args.fixed_orientation,
            gripper_mode=args.gripper_mode,
            phase_gate_close_dist=args.phase_gate_close_dist,
            phase_gate_max_hold=args.phase_gate_max_hold,
            observation_mode="flat",
            seed=args.seed,
            render=args.render,
        )
    )

    rows = []
    print(
        f"[EvalRollout] checkpoint={args.checkpoint} episodes={args.episodes} "
        f"max_steps={args.max_steps} render={args.render} gripper_mode={args.gripper_mode}",
        flush=True,
    )
    try:
        for episode_idx in range(args.episodes):
            episode_seed = int(args.seed + episode_idx)
            obs, info = env.reset(seed=episode_seed)
            total_reward = 0.0
            min_cube_target_dist = float(info["cube_target_dist"])
            min_ee_cube_dist = float(info["ee_cube_dist"])
            grasped_any = bool(info["has_grasped_cube"])
            errp_count = 0
            reward_components_total: dict[str, float] = {}
            terminated = False
            truncated = False

            for _ in range(args.max_steps):
                action = runner.predict(np.asarray(obs, dtype=np.float32))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                min_cube_target_dist = min(min_cube_target_dist, float(info["cube_target_dist"]))
                min_ee_cube_dist = min(min_ee_cube_dist, float(info["ee_cube_dist"]))
                grasped_any = grasped_any or bool(info["has_grasped_cube"])
                errp_count += int(float(info["errp_feedback"]) > 0.5)
                for name, value in info["reward_components"].items():
                    reward_components_total[name] = reward_components_total.get(name, 0.0) + float(value)
                if terminated or truncated:
                    break

            row = {
                "episode": episode_idx,
                "seed": episode_seed,
                "active_cube": info["active_cube"],
                "success": bool(terminated),
                "truncated": bool(truncated),
                "steps": int(info["step"]),
                "total_reward": float(total_reward),
                "final_cube_target_dist": float(info["cube_target_dist"]),
                "min_cube_target_dist": float(min_cube_target_dist),
                "final_ee_cube_dist": float(info["ee_cube_dist"]),
                "min_ee_cube_dist": float(min_ee_cube_dist),
                "grasped_any": bool(grasped_any),
                "final_has_grasped": bool(info["has_grasped_cube"]),
                "errp_count": int(errp_count),
                "final_controller_event": int(info["controller_event"]),
                "final_controller_t": float(info["controller_t"]),
                "phase_hold_steps": int(info["phase_hold_steps"]),
                "reward_components_total": reward_components_total,
            }
            rows.append(row)

            if args.log_every > 0 and (
                episode_idx == 0
                or (episode_idx + 1) % args.log_every == 0
                or episode_idx + 1 == args.episodes
            ):
                print(
                    f"[EvalRollout] episode={episode_idx:04d} success={row['success']} "
                    f"steps={row['steps']} reward={row['total_reward']:.3f} "
                    f"cube_target={row['final_cube_target_dist']:.3f} "
                    f"grasped_any={int(row['grasped_any'])}",
                    flush=True,
                )
    finally:
        env.close()

    summary = _summarize(rows)
    result = {
        "created_unix": time.time(),
        "duration_sec": time.time() - started_at,
        "config": {
            "checkpoint": _resolve_project_path(args.checkpoint),
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "render": args.render,
            "action_scale": args.action_scale,
            "fixed_orientation": args.fixed_orientation,
            "gripper_mode": args.gripper_mode,
            "success_dist": args.success_dist,
            "phase_gate_close_dist": args.phase_gate_close_dist,
            "phase_gate_max_hold": args.phase_gate_max_hold,
        },
        "policy": runner.metadata,
        "summary": summary,
        "episodes": rows,
    }
    output_json = _resolve_output_path(args.output_json)
    output_csv = _resolve_output_path(args.output_csv)
    _write_json(output_json, result)
    _write_csv(output_csv, rows)
    print(
        f"[EvalRollout] success_rate={summary['success_rate']:.3f} "
        f"successes={summary['successes']}/{summary['episodes']} "
        f"mean_steps={summary['mean_steps']:.1f} "
        f"mean_final_cube_target_dist={summary['mean_final_cube_target_dist']:.4f}",
        flush=True,
    )
    print(f"[EvalRollout] saved json={output_json}", flush=True)
    print(f"[EvalRollout] saved csv={output_csv}", flush=True)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
        }
    success = np.asarray([row["success"] for row in rows], dtype=np.float32)
    steps = np.asarray([row["steps"] for row in rows], dtype=np.float32)
    rewards = np.asarray([row["total_reward"] for row in rows], dtype=np.float32)
    final_dist = np.asarray([row["final_cube_target_dist"] for row in rows], dtype=np.float32)
    min_dist = np.asarray([row["min_cube_target_dist"] for row in rows], dtype=np.float32)
    grasped = np.asarray([row["grasped_any"] for row in rows], dtype=np.float32)
    truncated = np.asarray([row["truncated"] for row in rows], dtype=np.float32)
    return {
        "episodes": int(len(rows)),
        "successes": int(np.sum(success)),
        "success_rate": float(np.mean(success)),
        "truncated_rate": float(np.mean(truncated)),
        "grasp_rate": float(np.mean(grasped)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "mean_total_reward": float(np.mean(rewards)),
        "std_total_reward": float(np.std(rewards)),
        "mean_final_cube_target_dist": float(np.mean(final_dist)),
        "std_final_cube_target_dist": float(np.std(final_dist)),
        "mean_min_cube_target_dist": float(np.mean(min_dist)),
        "std_min_cube_target_dist": float(np.std(min_dist)),
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "episode",
        "seed",
        "active_cube",
        "success",
        "truncated",
        "steps",
        "total_reward",
        "final_cube_target_dist",
        "min_cube_target_dist",
        "final_ee_cube_dist",
        "min_ee_cube_dist",
        "grasped_any",
        "final_has_grasped",
        "errp_count",
        "final_controller_event",
        "final_controller_t",
        "phase_hold_steps",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


try:
    _run()
except BaseException as exc:
    print(f"[EvalRollout] terminated by {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    simulation_app.close()
