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

from end_effector_safety_geometry import (  # noqa: E402
    SafetyThresholds,
    distance_gate,
)


SAFETY_THRESHOLDS = SafetyThresholds.from_env()


def _parse_args() -> argparse.Namespace:
    default_json = os.path.join(SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval.json")
    default_csv = os.path.join(SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval.csv")
    default_step_csv = os.path.join(
        SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval_steps.csv"
    )
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
    parser.add_argument(
        "--residual-gate-mode",
        choices=("checkpoint", "none", "distance"),
        default="checkpoint",
        help=(
            "Override residual gating. checkpoint uses checkpoint metadata; "
            "legacy checkpoints default to none."
        ),
    )
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
    parser.add_argument("--phase-gate-close-dist", type=float, default=0.075)
    parser.add_argument("--phase-gate-max-hold", type=int, default=320)
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
        help="Disable pseudo-ErrP feedback while still reporting source flags.",
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
        help="Optional HDF5 trajectory file containing recorded human head/hand motion.",
    )
    parser.add_argument(
        "--human-replay-mode",
        choices=("step", "loop"),
        default="step",
        help="step holds the last human sample after replay ends; loop repeats it.",
    )
    parser.add_argument(
        "--human-replay-episode-policy",
        choices=("cycle", "random"),
        default="cycle",
        help="How to choose a recorded human episode for each rollout episode.",
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
        "--release-gate-dist",
        type=float,
        default=-1.0,
        help="Hold release until this cube-target distance. Negative disables release gating.",
    )
    parser.add_argument("--release-gate-max-hold", type=int, default=240)
    parser.add_argument(
        "--blend-bc-checkpoint",
        default="",
        help="Optional BC checkpoint to blend into the evaluated policy for selected controller events.",
    )
    parser.add_argument(
        "--blend-bc-events",
        default="",
        help="Comma-separated controller events that should use the BC blend, e.g. '1,2,3'.",
    )
    parser.add_argument(
        "--blend-bc-alpha",
        type=float,
        default=1.0,
        help="Blend weight for --blend-bc-checkpoint. 1.0 means replace policy action with BC action.",
    )
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
        help="Count success when the cube reaches the target radius, even before release.",
    )
    parser.set_defaults(require_release_for_success=False)
    parser.add_argument("--log-every", type=int, default=1, help="Episode logging interval. 0 disables logs.")
    parser.add_argument("--output-json", default=default_json, help="Path to save summary and per-episode JSON.")
    parser.add_argument("--output-csv", default=default_csv, help="Path to save per-episode CSV.")
    parser.add_argument(
        "--output-step-csv",
        default=default_step_csv,
        help="Path to save geometry and safety metrics for every rollout step.",
    )
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

from rl import (  # noqa: E402
    HumanTrajectoryReplay,
    IsaacPickPlaceEnv,
    PickPlaceEnvConfig,
    parse_pseudo_errp_sources,
)
from rl.actions import ACTION_DIM, clip_action  # noqa: E402
from rl.observations import OBSERVATION_DIM, observation_slices  # noqa: E402
from rl.policies import MLPPolicy  # noqa: E402


class PolicyRunner:
    def __init__(
        self,
        checkpoint_path: str,
        device_name: str,
        residual_gate_override: str = "checkpoint",
    ) -> None:
        self.checkpoint_path = _resolve_project_path(checkpoint_path)
        self.device = _select_device(device_name)
        checkpoint = _torch_load(self.checkpoint_path, self.device)
        self.target_version = str(checkpoint.get("target_version", "task_space_action_v0"))
        self.action_version = str(checkpoint.get("action_version", "action_v0_task_space"))
        self.policy_mode = str(checkpoint.get("policy_mode", "direct"))
        self.residual_scale = float(checkpoint.get("residual_scale", 1.0))
        checkpoint_gate_mode = str(checkpoint.get("residual_gate_mode", "none"))
        self.residual_gate_mode = (
            checkpoint_gate_mode
            if residual_gate_override == "checkpoint"
            else residual_gate_override
        )
        if self.residual_gate_mode not in ("none", "distance"):
            raise ValueError(
                f"Unknown residual gate mode: {self.residual_gate_mode}"
            )
        self.last_residual_norm = 0.0
        self.last_residual_gate = 0.0
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
        self.base_model = None
        self.base_obs_mean = None
        self.base_obs_std = None
        self.base_obs_dim = None
        self.base_checkpoint_path = ""
        if self.policy_mode == "residual":
            self._load_residual_base(checkpoint)
        self.metadata = {
            "checkpoint": self.checkpoint_path,
            "target_version": self.target_version,
            "action_version": self.action_version,
            "policy_mode": self.policy_mode,
            "residual_scale": self.residual_scale,
            "residual_gate_mode": self.residual_gate_mode,
            "source_bc_checkpoint": self.base_checkpoint_path,
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
            f"torch={torch.__version__} action={self.action_version} obs_dim={self.obs_dim} "
            f"policy_mode={self.policy_mode} residual_gate={self.residual_gate_mode}",
            flush=True,
        )
        if self.device.type == "cuda":
            print(f"[EvalRollout] cuda={torch.cuda.get_device_name(0)}", flush=True)

    def predict(
        self,
        obs: np.ndarray,
        *,
        safety_gate: float | None = None,
    ) -> np.ndarray:
        residual_or_action = self._predict_model(
            self.model,
            obs,
            self.obs_mean,
            self.obs_std,
            self.obs_dim,
        )
        if self.policy_mode == "residual":
            if self.base_model is None or self.base_obs_mean is None or self.base_obs_std is None:
                raise RuntimeError("Residual policy checkpoint is missing a loadable source BC checkpoint.")
            base_action = self._predict_model(
                self.base_model,
                obs,
                self.base_obs_mean,
                self.base_obs_std,
                int(self.base_obs_dim),
            )
            gate = 1.0
            if self.residual_gate_mode == "distance":
                gate = (
                    float(np.clip(safety_gate, 0.0, 1.0))
                    if safety_gate is not None
                    else _distance_gate_from_flat_obs(obs)
                )
            scaled_residual = gate * float(self.residual_scale) * residual_or_action
            self.last_residual_gate = float(gate)
            self.last_residual_norm = float(np.linalg.norm(scaled_residual))
            return clip_action(base_action + scaled_residual)
        self.last_residual_norm = 0.0
        self.last_residual_gate = 0.0
        return residual_or_action

    def _predict_model(
        self,
        model: MLPPolicy,
        obs: np.ndarray,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        obs_dim: int,
    ) -> np.ndarray:
        obs_policy = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs_policy = _align_obs_dim(obs_policy, obs_dim)
        obs_norm = (obs_policy - obs_mean) / np.maximum(obs_std, 1e-6)
        with torch.no_grad():
            tensor = torch.from_numpy(obs_norm.astype(np.float32)).to(self.device)
            action = model(tensor).detach().cpu().numpy()[0]
        if self.target_mean is not None and self.target_std is not None:
            action = (action.reshape(1, -1) * self.target_std + self.target_mean)[0]
            if self.target_min is not None and self.target_max is not None:
                action = np.clip(action.reshape(1, -1), self.target_min, self.target_max)[0]
            return np.asarray(action, dtype=np.float32)
        return clip_action(action)

    def _load_residual_base(self, checkpoint: dict[str, Any]) -> None:
        source_path = str(checkpoint.get("source_bc_checkpoint", ""))
        if not source_path:
            raise ValueError("Residual checkpoint is missing source_bc_checkpoint metadata.")
        self.base_checkpoint_path = _resolve_project_path(source_path)
        base_checkpoint = _torch_load(self.base_checkpoint_path, self.device)
        base_hidden_dims = tuple(int(value) for value in base_checkpoint.get("hidden_dims", (256, 256)))
        self.base_obs_dim = int(base_checkpoint.get("obs_dim", OBSERVATION_DIM))
        base_action_dim = int(base_checkpoint.get("action_dim", ACTION_DIM))
        if base_action_dim != ACTION_DIM:
            raise ValueError(
                f"Residual base checkpoint action dim {base_action_dim} != runtime action dim {ACTION_DIM}"
            )
        self.base_obs_mean = _tensor_to_numpy(base_checkpoint["obs_mean"]).reshape(1, -1)
        self.base_obs_std = _tensor_to_numpy(base_checkpoint["obs_std"]).reshape(1, -1)
        self.base_model = MLPPolicy(
            int(self.base_obs_dim),
            base_action_dim,
            hidden_dims=base_hidden_dims,
        ).to(self.device)
        self.base_model.load_state_dict(base_checkpoint["model_state_dict"])
        self.base_model.eval()


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


def _distance_gate_from_flat_obs(obs: np.ndarray) -> float:
    obs_policy = np.asarray(obs, dtype=np.float32).reshape(-1)
    gap_index = observation_slices()["min_hand_gripper_dist"].start
    if gap_index >= obs_policy.size:
        return 0.0
    return distance_gate(float(obs_policy[gap_index]), SAFETY_THRESHOLDS)


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
    )


def _run() -> None:
    np.random.seed(args.seed)
    started_at = time.time()
    runner = PolicyRunner(
        args.checkpoint,
        args.device,
        residual_gate_override=args.residual_gate_mode,
    )
    blend_events = _parse_event_set(args.blend_bc_events)
    blend_runner = (
        PolicyRunner(
            args.blend_bc_checkpoint,
            args.device,
            residual_gate_override=args.residual_gate_mode,
        )
        if args.blend_bc_checkpoint and blend_events
        else None
    )
    blend_alpha = float(np.clip(args.blend_bc_alpha, 0.0, 1.0))
    release_gate_dist = None if args.release_gate_dist < 0.0 else float(args.release_gate_dist)
    pseudo_errp_sources = parse_pseudo_errp_sources(args.pseudo_errp_sources)
    human_replay = _maybe_load_human_replay()
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
            early_close_on_grasp_gate=args.early_close_on_grasp_gate,
            fast_forward_grasp_gate=args.fast_forward_grasp_gate,
            release_gate_dist=release_gate_dist,
            release_gate_max_hold=args.release_gate_max_hold,
            require_release_for_success=args.require_release_for_success,
            observation_mode="flat",
            seed=args.seed,
            render=args.render,
            pseudo_errp_enabled=args.pseudo_errp_enabled,
            pseudo_errp_sources=pseudo_errp_sources,
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

    rows = []
    step_rows = []
    print(
        f"[EvalRollout] checkpoint={args.checkpoint} episodes={args.episodes} "
        f"max_steps={args.max_steps} render={args.render} gripper_mode={args.gripper_mode} "
        f"human_replay={human_replay.path if human_replay is not None else 'off'} "
        f"synthetic_human={args.synthetic_human}",
        flush=True,
    )
    try:
        for episode_idx in range(args.episodes):
            episode_seed = int(args.seed + episode_idx)
            if human_replay is not None:
                human_replay.reset(episode_idx, seed=episode_seed)
            obs, info = env.reset(seed=episode_seed)
            total_reward = 0.0
            min_cube_target_dist = float(info["cube_target_dist"])
            min_ee_cube_dist = float(info["ee_cube_dist"])
            grasped_any = bool(info["has_grasped_cube"])
            errp_count = 0
            errp_feedback_sum = 0.0
            errp_uncertainty_sum = 0.0
            max_errp_feedback = 0.0
            max_errp_uncertainty = 0.0
            errp_source_code = 0
            errp_source_counts: dict[str, int] = {}
            reward_components_total: dict[str, float] = {}
            terminated = False
            truncated = False
            bc_blend_count = 0
            collision_steps = 0
            near_steps = 0
            near_miss_steps = 0
            gate_active_steps = 0
            geometry_valid_steps = 0
            distance_gate_sum = 0.0
            max_distance_gate = 0.0
            min_surface_gap = 10.0
            safety_query_time_ms_sum = 0.0
            closest_link_counts: dict[str, int] = {}
            collision_link_counts: dict[str, int] = {}
            collision_event_count = 0
            collision_was_active = False
            gate_collision_overlap_steps = 0
            gated_residual_norm_sum = 0.0
            gated_residual_norm_max = 0.0
            gated_residual_count = 0

            for _ in range(args.max_steps):
                action_safety_gate = float(info.get("distance_gate", 0.0))
                action = runner.predict(
                    np.asarray(obs, dtype=np.float32),
                    safety_gate=action_safety_gate,
                )
                residual_norm = float(runner.last_residual_norm)
                residual_gate = float(runner.last_residual_gate)
                if blend_runner is not None and int(info["controller_event"]) in blend_events:
                    bc_action = blend_runner.predict(
                        np.asarray(obs, dtype=np.float32),
                        safety_gate=action_safety_gate,
                    )
                    action = clip_action((1.0 - blend_alpha) * action + blend_alpha * bc_action)
                    bc_blend_count += 1
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                min_cube_target_dist = min(min_cube_target_dist, float(info["cube_target_dist"]))
                min_ee_cube_dist = min(min_ee_cube_dist, float(info["ee_cube_dist"]))
                grasped_any = grasped_any or bool(info["has_grasped_cube"])
                errp_feedback = float(info["errp_feedback"])
                errp_uncertainty = float(info.get("errp_uncertainty", 0.0))
                errp_count += int(info.get("errp_label", errp_feedback >= 0.5))
                errp_feedback_sum += errp_feedback
                errp_uncertainty_sum += errp_uncertainty
                max_errp_feedback = max(max_errp_feedback, errp_feedback)
                max_errp_uncertainty = max(max_errp_uncertainty, errp_uncertainty)
                errp_source_code |= int(info.get("errp_source_code", 0))
                for source_name in info.get("errp_source_names", ()):
                    errp_source_counts[source_name] = errp_source_counts.get(source_name, 0) + 1
                for name, value in info["reward_components"].items():
                    reward_components_total[name] = reward_components_total.get(name, 0.0) + float(value)
                collision_active = bool(info.get("human_robot_collision", False))
                collision_steps += int(collision_active)
                if collision_active and not collision_was_active:
                    collision_event_count += 1
                collision_was_active = collision_active
                near_steps += int(info.get("near_human", False))
                near_miss_steps += int(info.get("near_miss", False))
                gate = float(info.get("distance_gate", 0.0))
                gate_active_steps += int(gate > 0.0)
                gate_collision_overlap_steps += int(gate > 0.0 and collision_active)
                distance_gate_sum += gate
                max_distance_gate = max(max_distance_gate, gate)
                geometry_valid = bool(info.get("geometry_valid", False))
                geometry_valid_steps += int(geometry_valid)
                if geometry_valid:
                    min_surface_gap = min(
                        min_surface_gap,
                        float(info.get("min_hand_end_effector_surface_gap", 10.0)),
                    )
                safety_query_time_ms_sum += float(info.get("safety_query_time_ms", 0.0))
                for key in ("closest_link_left", "closest_link_right"):
                    link_name = str(info.get(key, ""))
                    if link_name:
                        closest_link_counts[link_name] = closest_link_counts.get(link_name, 0) + 1
                collision_links = {
                    str(info.get("closest_link_left", ""))
                    if info.get("contact_left", False)
                    else "",
                    str(info.get("closest_link_right", ""))
                    if info.get("contact_right", False)
                    else "",
                }
                for collision_link in collision_links - {""}:
                    collision_link_counts[collision_link] = (
                        collision_link_counts.get(collision_link, 0) + 1
                    )
                if action_safety_gate > 0.0:
                    gated_residual_norm_sum += residual_norm
                    gated_residual_norm_max = max(gated_residual_norm_max, residual_norm)
                    gated_residual_count += 1
                step_rows.append(
                    {
                        "episode": episode_idx,
                        "seed": episode_seed,
                        "step": int(info["step"]),
                        "sim_time": float(info.get("sim_time", 0.0)),
                        "end_effector_surface_gap_m": float(
                            info.get("min_hand_end_effector_surface_gap", 10.0)
                        ),
                        "left_end_effector_surface_gap_m": float(
                            info.get("left_end_effector_surface_gap_m", 10.0)
                        ),
                        "right_end_effector_surface_gap_m": float(
                            info.get("right_end_effector_surface_gap_m", 10.0)
                        ),
                        "closest_human_hand": str(info.get("closest_human_hand", "")),
                        "closest_robot_link": str(info.get("closest_robot_link", "")),
                        "closest_collider_prim": str(info.get("closest_collider", "")),
                        "contact_active": int(info.get("contact_active", False)),
                        "contact_left": int(info.get("contact_left", False)),
                        "contact_right": int(info.get("contact_right", False)),
                        "penetration_depth_m": float(
                            info.get("penetration_depth_m", 0.0)
                        ),
                        "near_human": int(info.get("near_human", False)),
                        "near_miss": int(info.get("near_miss", False)),
                        "human_robot_collision": int(collision_active),
                        "safety_gate": gate,
                        "residual_norm": residual_norm,
                        "residual_gate": residual_gate,
                        "action_safety_gate": action_safety_gate,
                        "haptic_pulse_left": 0,
                        "haptic_pulse_right": 0,
                        "geometry_valid": int(geometry_valid),
                        "safety_query_time_ms": float(
                            info.get("safety_query_time_ms", 0.0)
                        ),
                        "task_success": int(terminated),
                    }
                )
                if terminated or truncated:
                    break

            episode_steps = max(1, int(info["step"]))
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
                "errp_feedback_sum": float(errp_feedback_sum),
                "mean_errp_feedback": float(errp_feedback_sum / max(1, int(info["step"]))),
                "max_errp_feedback": float(max_errp_feedback),
                "mean_errp_uncertainty": float(errp_uncertainty_sum / max(1, int(info["step"]))),
                "max_errp_uncertainty": float(max_errp_uncertainty),
                "errp_source_code": int(errp_source_code),
                "errp_sources": sorted(errp_source_counts),
                "errp_source_counts": errp_source_counts,
                "final_controller_event": int(info["controller_event"]),
                "final_controller_t": float(info["controller_t"]),
                "phase_hold_steps": int(info["phase_hold_steps"]),
                "bc_blend_count": int(bc_blend_count),
                "collision_steps": int(collision_steps),
                "near_steps": int(near_steps),
                "near_miss_steps": int(near_miss_steps),
                "gate_active_steps": int(gate_active_steps),
                "geometry_valid_steps": int(geometry_valid_steps),
                "collision_event_count": int(collision_event_count),
                "gate_collision_overlap_steps": int(gate_collision_overlap_steps),
                "collision_rate": float(collision_steps / episode_steps),
                "near_rate": float(near_steps / episode_steps),
                "near_miss_rate": float(near_miss_steps / episode_steps),
                "gate_activation_rate": float(gate_active_steps / episode_steps),
                "mean_distance_gate": float(distance_gate_sum / episode_steps),
                "max_distance_gate": float(max_distance_gate),
                "min_surface_gap": float(min_surface_gap),
                "minimum_end_effector_surface_gap_m": float(min_surface_gap),
                "mean_safety_query_time_ms": float(
                    safety_query_time_ms_sum / episode_steps
                ),
                "mean_residual_norm_during_gate": float(
                    gated_residual_norm_sum / max(1, gated_residual_count)
                ),
                "max_residual_norm_during_gate": float(gated_residual_norm_max),
                "closest_link_counts": closest_link_counts,
                "collision_link_counts": collision_link_counts,
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
                    f"grasped_any={int(row['grasped_any'])} "
                    f"collision_rate={row['collision_rate']:.3f} "
                    f"gate_rate={row['gate_activation_rate']:.3f}",
                    flush=True,
                )
    finally:
        env.close()
        if human_replay is not None:
            human_replay.close()

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
            "residual_gate_mode_override": args.residual_gate_mode,
            "fixed_orientation": args.fixed_orientation,
            "gripper_mode": args.gripper_mode,
            "success_dist": args.success_dist,
            "phase_gate_close_dist": args.phase_gate_close_dist,
            "phase_gate_max_hold": args.phase_gate_max_hold,
            "pseudo_errp_enabled": args.pseudo_errp_enabled,
            "pseudo_errp_sources": list(pseudo_errp_sources),
            "human_replay_data": human_replay.path if human_replay is not None else "",
            "human_replay_mode": args.human_replay_mode,
            "human_replay_episode_policy": args.human_replay_episode_policy,
            "early_close_on_grasp_gate": args.early_close_on_grasp_gate,
            "fast_forward_grasp_gate": args.fast_forward_grasp_gate,
            "release_gate_dist": release_gate_dist,
            "release_gate_max_hold": args.release_gate_max_hold,
            "require_release_for_success": args.require_release_for_success,
            "blend_bc_checkpoint": _resolve_project_path(args.blend_bc_checkpoint)
            if args.blend_bc_checkpoint
            else "",
            "blend_bc_events": sorted(blend_events),
            "blend_bc_alpha": blend_alpha,
            "safety_geometry_source": env.safety_geometry.GEOMETRY_SOURCE,
            "safety_geometry_metadata": env.safety_geometry.metadata(),
        },
        "policy": runner.metadata,
        "blend_policy": blend_runner.metadata if blend_runner is not None else {},
        "summary": summary,
        "episodes": rows,
    }
    output_json = _resolve_output_path(args.output_json)
    output_csv = _resolve_output_path(args.output_csv)
    output_step_csv = _resolve_output_path(args.output_step_csv)
    _write_json(output_json, result)
    _write_csv(output_csv, rows)
    _write_step_csv(output_step_csv, step_rows)
    print(
        f"[EvalRollout] success_rate={summary['success_rate']:.3f} "
        f"successes={summary['successes']}/{summary['episodes']} "
        f"mean_steps={summary['mean_steps']:.1f} "
        f"mean_final_cube_target_dist={summary['mean_final_cube_target_dist']:.4f} "
        f"collision_rate={summary['collision_rate']:.4f} "
        f"gate_activation_rate={summary['gate_activation_rate']:.4f}",
        flush=True,
    )
    print(f"[EvalRollout] saved json={output_json}", flush=True)
    print(f"[EvalRollout] saved csv={output_csv}", flush=True)
    print(f"[EvalRollout] saved step csv={output_step_csv}", flush=True)


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
    errp_counts = np.asarray([row.get("errp_count", 0) for row in rows], dtype=np.float32)
    mean_errp_feedback = np.asarray(
        [row.get("mean_errp_feedback", 0.0) for row in rows],
        dtype=np.float32,
    )
    max_errp_feedback = np.asarray(
        [row.get("max_errp_feedback", 0.0) for row in rows],
        dtype=np.float32,
    )
    mean_errp_uncertainty = np.asarray(
        [row.get("mean_errp_uncertainty", 0.0) for row in rows],
        dtype=np.float32,
    )
    max_errp_uncertainty = np.asarray(
        [row.get("max_errp_uncertainty", 0.0) for row in rows],
        dtype=np.float32,
    )
    total_steps = max(1, int(np.sum(steps)))
    collision_steps = int(sum(row.get("collision_steps", 0) for row in rows))
    near_steps = int(sum(row.get("near_steps", 0) for row in rows))
    near_miss_steps = int(sum(row.get("near_miss_steps", 0) for row in rows))
    gate_active_steps = int(sum(row.get("gate_active_steps", 0) for row in rows))
    collision_event_count = int(sum(row.get("collision_event_count", 0) for row in rows))
    gate_collision_overlap_steps = int(
        sum(row.get("gate_collision_overlap_steps", 0) for row in rows)
    )
    geometry_valid_steps = int(sum(row.get("geometry_valid_steps", 0) for row in rows))
    weighted_gate_sum = float(
        sum(row.get("mean_distance_gate", 0.0) * row["steps"] for row in rows)
    )
    closest_link_counts: dict[str, int] = {}
    collision_link_counts: dict[str, int] = {}
    for row in rows:
        for name, count in row.get("closest_link_counts", {}).items():
            closest_link_counts[name] = closest_link_counts.get(name, 0) + int(count)
        for name, count in row.get("collision_link_counts", {}).items():
            collision_link_counts[name] = collision_link_counts.get(name, 0) + int(count)
    gated_residual_weight = max(1, gate_active_steps)
    gated_residual_sum = float(
        sum(
            row.get("mean_residual_norm_during_gate", 0.0)
            * row.get("gate_active_steps", 0)
            for row in rows
        )
    )
    query_time_sum = float(
        sum(row.get("mean_safety_query_time_ms", 0.0) * row["steps"] for row in rows)
    )
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
        "mean_errp_count": float(np.mean(errp_counts)),
        "max_errp_count": int(np.max(errp_counts)),
        "mean_episode_errp_feedback": float(np.mean(mean_errp_feedback)),
        "max_episode_errp_feedback": float(np.max(max_errp_feedback)),
        "mean_episode_errp_uncertainty": float(np.mean(mean_errp_uncertainty)),
        "max_episode_errp_uncertainty": float(np.max(max_errp_uncertainty)),
        "collision_steps": collision_steps,
        "near_steps": near_steps,
        "near_miss_steps": near_miss_steps,
        "gate_active_steps": gate_active_steps,
        "geometry_valid_steps": geometry_valid_steps,
        "collision_event_count": collision_event_count,
        "gate_collision_overlap_steps": gate_collision_overlap_steps,
        "collision_rate": float(collision_steps / total_steps),
        "near_rate": float(near_steps / total_steps),
        "near_miss_rate": float(near_miss_steps / total_steps),
        "gate_activation_rate": float(gate_active_steps / total_steps),
        "geometry_valid_rate": float(geometry_valid_steps / total_steps),
        "mean_distance_gate": float(weighted_gate_sum / total_steps),
        "gate_collision_overlap_rate": float(
            gate_collision_overlap_steps / max(1, gate_active_steps)
        ),
        "mean_residual_norm_during_gate": float(
            gated_residual_sum / gated_residual_weight
        ),
        "max_residual_norm_during_gate": float(
            max(row.get("max_residual_norm_during_gate", 0.0) for row in rows)
        ),
        "mean_safety_query_time_ms": float(query_time_sum / total_steps),
        "min_surface_gap": float(min(row.get("min_surface_gap", 10.0) for row in rows)),
        "minimum_end_effector_surface_gap_m": float(
            min(row.get("min_surface_gap", 10.0) for row in rows)
        ),
        "closest_link_counts": closest_link_counts,
        "collision_link_counts": collision_link_counts,
    }


def _parse_event_set(text: str) -> set[int]:
    events: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        events.add(int(part))
    return events


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
        "errp_feedback_sum",
        "mean_errp_feedback",
        "max_errp_feedback",
        "mean_errp_uncertainty",
        "max_errp_uncertainty",
        "errp_source_code",
        "errp_sources",
        "final_controller_event",
        "final_controller_t",
        "phase_hold_steps",
        "bc_blend_count",
        "collision_steps",
        "near_steps",
        "near_miss_steps",
        "gate_active_steps",
        "geometry_valid_steps",
        "collision_event_count",
        "gate_collision_overlap_steps",
        "collision_rate",
        "near_rate",
        "near_miss_rate",
        "gate_activation_rate",
        "mean_distance_gate",
        "max_distance_gate",
        "min_surface_gap",
        "minimum_end_effector_surface_gap_m",
        "mean_safety_query_time_ms",
        "mean_residual_norm_during_gate",
        "max_residual_norm_during_gate",
        "closest_link_counts",
        "collision_link_counts",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row[field] for field in fields}
            csv_row["errp_sources"] = ",".join(row.get("errp_sources", []))
            csv_row["closest_link_counts"] = json.dumps(
                row.get("closest_link_counts", {}), sort_keys=True
            )
            csv_row["collision_link_counts"] = json.dumps(
                row.get("collision_link_counts", {}), sort_keys=True
            )
            writer.writerow(csv_row)


def _write_step_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "episode",
        "seed",
        "step",
        "sim_time",
        "end_effector_surface_gap_m",
        "left_end_effector_surface_gap_m",
        "right_end_effector_surface_gap_m",
        "closest_human_hand",
        "closest_robot_link",
        "closest_collider_prim",
        "contact_active",
        "contact_left",
        "contact_right",
        "penetration_depth_m",
        "near_human",
        "near_miss",
        "human_robot_collision",
        "safety_gate",
        "residual_norm",
        "residual_gate",
        "action_safety_gate",
        "haptic_pulse_left",
        "haptic_pulse_right",
        "geometry_valid",
        "safety_query_time_ms",
        "task_success",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


try:
    _run()
except BaseException as exc:
    print(f"[EvalRollout] terminated by {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    simulation_app.close()
