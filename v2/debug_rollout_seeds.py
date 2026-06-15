from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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
    parser = argparse.ArgumentParser(description="Debug selected rollout seeds with step-level diagnostics.")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(SCRIPT_DIR, "policies", "ppo_pick_place_v1.pt"),
        help="Policy checkpoint to inspect.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seeds to roll out. Active cube follows local debug episode index.",
    )
    parser.add_argument(
        "--episode-seeds",
        default="10:21,25:36,34:45,41:52",
        help=(
            "Comma-separated episode:seed pairs to reproduce evaluate_rollout_policy.py "
            "active-cube assignment. Defaults to current PPO grasp-failure episodes."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--gripper-mode", choices=("event", "rule", "policy"), default="event")
    parser.add_argument("--success-dist", type=float, default=0.06)
    parser.add_argument("--phase-gate-close-dist", type=float, default=0.066)
    parser.add_argument("--phase-gate-max-hold", type=int, default=160)
    parser.add_argument(
        "--output-csv",
        default=os.path.join(SCRIPT_DIR, "eval_results", "debug_rollout_seeds_steps.csv"),
        help="Step-level CSV output.",
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(SCRIPT_DIR, "eval_results", "debug_rollout_seeds_summary.json"),
        help="Per-seed summary JSON output.",
    )
    parser.add_argument("--print-every", type=int, default=120, help="Progress interval per seed. 0 disables.")
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
print(f"[DebugRollout] SimulationApp headless={not args.render}", flush=True)

import torch  # noqa: E402

from rl import IsaacPickPlaceEnv, PickPlaceEnvConfig  # noqa: E402
from rl.actions import ACTION_DIM, ACTION_NAMES, clip_action  # noqa: E402
from rl.observations import OBSERVATION_DIM  # noqa: E402
from rl.policies import MLPPolicy  # noqa: E402


class PolicyRunner:
    def __init__(self, checkpoint_path: str, device_name: str) -> None:
        self.checkpoint_path = _resolve_project_path(checkpoint_path)
        self.device = _select_device(device_name)
        checkpoint = _torch_load(self.checkpoint_path, self.device)
        self.action_version = str(checkpoint.get("action_version", "action_v1_controller_target_delta"))
        self.obs_mean = _tensor_to_numpy(checkpoint["obs_mean"]).reshape(1, -1)
        self.obs_std = np.maximum(_tensor_to_numpy(checkpoint["obs_std"]).reshape(1, -1), 1e-6)

        hidden_dims = tuple(int(value) for value in checkpoint.get("hidden_dims", (256, 256)))
        self.obs_dim = int(checkpoint.get("obs_dim", OBSERVATION_DIM))
        action_dim = int(checkpoint.get("action_dim", ACTION_DIM))
        if action_dim != ACTION_DIM:
            raise ValueError(f"Checkpoint action dim {action_dim} != runtime action dim {ACTION_DIM}")

        self.model = MLPPolicy(self.obs_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(
            f"[DebugRollout] loaded checkpoint={self.checkpoint_path} device={self.device} "
            f"action={self.action_version} obs_dim={self.obs_dim}",
            flush=True,
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_policy = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs_policy = _align_obs_dim(obs_policy, self.obs_dim)
        obs_norm = (obs_policy - self.obs_mean) / self.obs_std
        with torch.no_grad():
            tensor = torch.from_numpy(obs_norm.astype(np.float32)).to(self.device)
            action = self.model(tensor).detach().cpu().numpy()[0]
        return clip_action(action)


def _run() -> None:
    episode_specs = _parse_episode_specs(args.episode_seeds, args.seeds)
    runner = PolicyRunner(args.checkpoint, args.device)
    env = IsaacPickPlaceEnv(
        PickPlaceEnvConfig(
            max_episode_steps=args.max_steps,
            success_dist=args.success_dist,
            action_scale=args.action_scale,
            action_version=runner.action_version,
            fixed_orientation=True,
            gripper_mode=args.gripper_mode,
            phase_gate_close_dist=args.phase_gate_close_dist,
            phase_gate_max_hold=args.phase_gate_max_hold,
            observation_mode="flat",
            seed=min(spec["seed"] for spec in episode_specs) if episode_specs else 0,
            render=args.render,
        )
    )

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        for spec in episode_specs:
            summary = _rollout_seed(
                env,
                runner,
                episode=int(spec["episode"]),
                seed=int(spec["seed"]),
                active_cube_index=int(spec["active_cube_index"]),
                rows=rows,
            )
            summaries.append(summary)
            print(
                f"[DebugRollout] episode={summary['episode']} seed={summary['seed']} "
                f"active_cube_index={summary['active_cube_index']} success={summary['success']} steps={summary['steps']} "
                f"final_dist={summary['final_cube_target_dist']:.4f} grasped_any={int(summary['grasped_any'])} "
                f"first_grasp_step={summary['first_grasp_step']} "
                f"min_ee_cube_event_1_3={summary['min_ee_cube_event_1_3']:.4f}",
                flush=True,
            )
    finally:
        env.close()

    output_csv = _resolve_output_path(args.output_csv)
    output_json = _resolve_output_path(args.output_json)
    _write_csv(output_csv, rows)
    _write_json(
        output_json,
        {
            "checkpoint": runner.checkpoint_path,
            "config": {
                "episode_specs": episode_specs,
                "max_steps": args.max_steps,
                "success_dist": args.success_dist,
                "phase_gate_close_dist": args.phase_gate_close_dist,
                "phase_gate_max_hold": args.phase_gate_max_hold,
                "gripper_mode": args.gripper_mode,
            },
            "summaries": summaries,
        },
    )
    print(f"[DebugRollout] saved csv={output_csv}", flush=True)
    print(f"[DebugRollout] saved json={output_json}", flush=True)


def _rollout_seed(
    env: IsaacPickPlaceEnv,
    runner: PolicyRunner,
    *,
    episode: int,
    seed: int,
    active_cube_index: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    obs, info = env.reset(seed=int(seed), active_cube_index=int(active_cube_index))
    grasped_any = bool(info["has_grasped_cube"])
    first_grasp_step: int | None = 0 if grasped_any else None
    first_close_event_step: int | None = None
    ee_cube_at_first_close_event = float("nan")
    min_ee_cube_dist = float(info["ee_cube_dist"])
    min_cube_target_dist = float(info["cube_target_dist"])
    min_ee_cube_event_1_3 = float("inf")
    min_ee_cube_pre_close = float("inf")
    last_row: dict[str, Any] | None = None
    terminated = False
    truncated = False

    for _ in range(args.max_steps):
        action = runner.predict(np.asarray(obs, dtype=np.float32))
        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = info["obs_dict"]
        step = int(info["step"])
        event = int(info["controller_event"])
        ee_cube_dist = float(info["ee_cube_dist"])
        cube_target_dist = float(info["cube_target_dist"])
        min_ee_cube_dist = min(min_ee_cube_dist, ee_cube_dist)
        min_cube_target_dist = min(min_cube_target_dist, cube_target_dist)

        if 1 <= event <= 3:
            min_ee_cube_event_1_3 = min(min_ee_cube_event_1_3, ee_cube_dist)
        if event <= 3:
            min_ee_cube_pre_close = min(min_ee_cube_pre_close, ee_cube_dist)
        if event == 3 and first_close_event_step is None:
            first_close_event_step = step
            ee_cube_at_first_close_event = ee_cube_dist

        has_grasped = bool(info["has_grasped_cube"])
        if has_grasped and first_grasp_step is None:
            first_grasp_step = step
        grasped_any = grasped_any or has_grasped

        row = _step_row(
            episode=episode,
            seed=seed,
            active_cube_index=active_cube_index,
            step=step,
            info=info,
            obs_dict=obs_dict,
            action=action,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
        )
        rows.append(row)
        last_row = row

        if args.print_every > 0 and (step == 1 or step % args.print_every == 0):
            print(
                f"[DebugRollout] episode={episode} seed={seed} active_cube_index={active_cube_index} "
                f"step={step:04d} event={event} "
                f"ee_cube={ee_cube_dist:.4f} cube_target={cube_target_dist:.4f} "
                f"grasp={int(has_grasped)} action={np.array2string(action, precision=3)}",
                flush=True,
            )
        if terminated or truncated:
            break

    if min_ee_cube_event_1_3 == float("inf"):
        min_ee_cube_event_1_3 = float("nan")
    if min_ee_cube_pre_close == float("inf"):
        min_ee_cube_pre_close = float("nan")
    final = last_row or {}
    return {
        "seed": int(seed),
        "episode": int(episode),
        "active_cube_index": int(active_cube_index),
        "active_cube": str(final.get("active_cube", "")),
        "success": bool(terminated),
        "truncated": bool(truncated),
        "steps": int(final.get("step", 0)),
        "final_event": int(final.get("controller_event", -1)),
        "final_cube_target_dist": float(final.get("cube_target_dist", float("nan"))),
        "final_ee_cube_dist": float(final.get("ee_cube_dist", float("nan"))),
        "min_cube_target_dist": float(min_cube_target_dist),
        "min_ee_cube_dist": float(min_ee_cube_dist),
        "min_ee_cube_event_1_3": float(min_ee_cube_event_1_3),
        "min_ee_cube_pre_close": float(min_ee_cube_pre_close),
        "first_close_event_step": first_close_event_step,
        "ee_cube_at_first_close_event": float(ee_cube_at_first_close_event),
        "grasped_any": bool(grasped_any),
        "first_grasp_step": first_grasp_step,
    }


def _step_row(
    *,
    episode: int,
    seed: int,
    active_cube_index: int,
    step: int,
    info: dict[str, Any],
    obs_dict: dict[str, np.ndarray],
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    ee_pos = np.asarray(obs_dict["ee_pos"], dtype=float).reshape(3)
    cube_pos = np.asarray(obs_dict["cube_pos"], dtype=float).reshape(3)
    target_pos = np.asarray(obs_dict["place_target_pos"], dtype=float).reshape(3)
    components = info.get("reward_components", {})
    row: dict[str, Any] = {
        "episode": int(episode),
        "seed": int(seed),
        "active_cube_index": int(active_cube_index),
        "active_cube": str(info.get("active_cube", "")),
        "step": int(step),
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
        "success": int(bool(info["success"])),
        "controller_event": int(info["controller_event"]),
        "controller_t": float(info["controller_t"]),
        "phase_hold_steps": int(info["phase_hold_steps"]),
        "gripper_closed": int(bool(info["gripper_closed"])),
        "has_grasped_cube": int(bool(info["has_grasped_cube"])),
        "ee_cube_dist": float(info["ee_cube_dist"]),
        "cube_target_dist": float(info["cube_target_dist"]),
        "reward": float(reward),
        "ee_x": float(ee_pos[0]),
        "ee_y": float(ee_pos[1]),
        "ee_z": float(ee_pos[2]),
        "cube_x": float(cube_pos[0]),
        "cube_y": float(cube_pos[1]),
        "cube_z": float(cube_pos[2]),
        "target_x": float(target_pos[0]),
        "target_y": float(target_pos[1]),
        "target_z": float(target_pos[2]),
    }
    for idx, name in enumerate(ACTION_NAMES):
        row[f"action_{name}"] = float(action[idx])
    for name in (
        "ee_to_cube_progress",
        "cube_to_target_progress",
        "carrying_cube_to_target_progress",
        "grasp_bonus",
        "placement_error_penalty",
        "release_outside_target_penalty",
        "action_penalty",
    ):
        row[f"reward_{name}"] = float(components.get(name, 0.0))
    return row


def _parse_episode_specs(episode_seed_text: str, seed_text: str) -> list[dict[str, int]]:
    specs = []
    for part in episode_seed_text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("--episode-seeds entries must use episode:seed format")
        episode_text, seed_value = part.split(":", 1)
        episode = int(episode_text.strip())
        seed = int(seed_value.strip())
        specs.append(
            {
                "episode": episode,
                "seed": seed,
                "active_cube_index": episode % 3,
            }
        )
    if specs:
        return specs

    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("Provide --episode-seeds or --seeds")
    return [
        {
            "episode": local_episode,
            "seed": seed,
            "active_cube_index": local_episode % 3,
        }
        for local_episode, seed in enumerate(seeds)
    ]


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


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


try:
    _run()
except BaseException as exc:
    print(f"[DebugRollout] terminated by {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    simulation_app.close()
