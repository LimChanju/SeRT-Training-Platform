# rollout_bc_policy.py -- run a trained BC task-space policy in Isaac Sim
#
# Example:
#   ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/rollout_bc_policy.py" \
#       --episodes 3 --render

from __future__ import annotations

import argparse
import os
import sys

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
    parser = argparse.ArgumentParser(description="Roll out a BC policy in Isaac pick-and-place.")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(SCRIPT_DIR, "policies", "bc_pick_place_v1.pt"),
        help="PyTorch checkpoint produced by v2/rl/train_bc.py.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--render", action="store_true", help="Render the rollout window.")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Multiplier for BC dx/dy/dz/dyaw.")
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
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.035,
        help="Maximum per-step arm joint delta for joint-target BC checkpoints.",
    )
    parser.add_argument(
        "--gripper-mode",
        choices=("event", "rule", "policy"),
        default="event",
        help="Use distance-based gripper rule or the policy gripper output.",
    )
    parser.add_argument("--close-dist", type=float, default=0.08, help="EE/cube distance to close gripper.")
    parser.add_argument("--release-dist", type=float, default=0.07, help="Cube/target distance to open gripper.")
    parser.add_argument("--success-dist", type=float, default=0.06, help="Cube/target success distance.")
    parser.add_argument(
        "--phase-gate-close-dist",
        type=float,
        default=0.066,
        help="Hold the lowering phase until EE/cube distance is close enough for event-mode grasping.",
    )
    parser.add_argument(
        "--phase-gate-max-hold",
        type=int,
        default=160,
        help="Maximum extra control steps to hold a phase gate before advancing anyway.",
    )
    parser.add_argument("--log-every", type=int, default=60)
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
print(f"[BCRollout] SimulationApp headless={not args.render}", flush=True)

import torch  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from omni.isaac.franka.controllers import RMPFlowController  # noqa: E402

from actions import (  # noqa: E402
    ACTION_DIM,
    CONTROLLER_TARGET_ACTION_VERSION,
    MAX_EE_DELTA_M,
    MAX_YAW_DELTA_RAD,
    clip_action,
    controller_target_from_action,
)
from observations import OBSERVATION_DIM, build_observation, flatten_observation, task_phase_onehot  # noqa: E402
from rl.pick_place_phase import advance_pick_place_event, event_gripper_command, task_phase_from_event  # noqa: E402
from policies import MLPPolicy, MLPRegressor  # noqa: E402
from rewards import is_success  # noqa: E402

from panda_robot import add_panda  # noqa: E402
from scene_setup import create_world, randomize_cubes, setup_scene  # noqa: E402


class PolicyRunner:
    def __init__(self, checkpoint_path: str, device_name: str) -> None:
        checkpoint_path = _resolve_project_path(checkpoint_path)
        self.device = _select_device(device_name)
        checkpoint = _torch_load(checkpoint_path, self.device)
        self.target_version = str(checkpoint.get("target_version", "task_space_action_v0"))
        self.action_version = str(checkpoint.get("action_version", "action_v0_task_space"))
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
        if self.is_joint_policy:
            model_cls = MLPRegressor
        else:
            model_cls = MLPPolicy
        if not self.is_joint_policy and action_dim != ACTION_DIM:
            raise ValueError(f"Checkpoint action dim {action_dim} != runtime action dim {ACTION_DIM}")
        self.model = model_cls(self.obs_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(
            f"[BCRollout] loaded checkpoint={checkpoint_path} device={self.device} "
            f"torch={torch.__version__} target={self.target_version} action={self.action_version} "
            f"obs_dim={self.obs_dim} hidden_dims={hidden_dims}",
            flush=True,
        )
        if self.device.type == "cuda":
            print(f"[BCRollout] cuda={torch.cuda.get_device_name(0)}", flush=True)

    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        obs_policy = flatten_observation(obs).reshape(1, -1).astype(np.float32)
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

    @property
    def is_joint_policy(self) -> bool:
        return self.target_version == "expert_arm_joint_action_v0"


def _select_device(requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False")
    return torch.device(requested)


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
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


def _task_phase(obs: dict[str, np.ndarray]) -> str:
    ee_cube_dist = float(np.linalg.norm(obs["ee_to_cube"]))
    cube_target_dist = float(np.linalg.norm(obs["cube_to_place_target"]))
    has_grasped = bool(obs["has_grasped_cube"][0] > 0.5)
    if not has_grasped and ee_cube_dist > 0.08:
        return "approach_cube"
    if not has_grasped:
        return "grasp_cube"
    if cube_target_dist > 0.08:
        return "move_to_target"
    return "release_cube"


def _gripper_center_from_fingers(robot) -> np.ndarray | None:
    try:
        left_pos, _ = robot.gripper._left_finger.get_world_pose()
        right_pos, _ = robot.gripper._right_finger.get_world_pose()
        return (np.asarray(left_pos, dtype=float) + np.asarray(right_pos, dtype=float)) * 0.5
    except Exception:
        return None


def _has_grasped_cube(robot, cube, gripper_center: np.ndarray | None) -> bool:
    try:
        width = float(np.sum(robot.gripper.get_joint_positions()))
    except Exception:
        width = 0.1
    cube_pos, _ = cube.get_world_pose()
    center = gripper_center
    if center is None:
        center, _ = robot.end_effector.get_world_pose()
    dist = float(np.linalg.norm(np.asarray(cube_pos, dtype=float) - np.asarray(center, dtype=float)))
    return width < 0.065 and dist < 0.11


def _build_obs(
    robot,
    cube,
    place_pos: np.ndarray,
    *,
    controller_event: int | None = None,
    controller_t: float = 0.0,
) -> dict[str, np.ndarray]:
    gripper_center = _gripper_center_from_fingers(robot)
    has_grasped = _has_grasped_cube(robot, cube, gripper_center)
    task_phase = (
        task_phase_from_event(controller_event)
        if controller_event is not None
        else "approach_cube"
    )
    obs = build_observation(
        robot=robot,
        cube=cube,
        place_target=place_pos,
        gripper_center_pos=gripper_center,
        has_grasped_cube=has_grasped,
        task_phase=task_phase,
        controller_event=controller_event,
        controller_t=controller_t,
    )
    if controller_event is None:
        obs["task_phase"] = task_phase_onehot(_task_phase(obs))
    return obs


def _rule_gripper_should_close(
    obs: dict[str, np.ndarray],
    was_closed: bool,
    *,
    close_dist: float,
    release_dist: float,
) -> bool:
    ee_cube_dist = float(np.linalg.norm(obs["ee_to_cube"]))
    cube_target_dist = float(np.linalg.norm(obs["cube_to_place_target"]))
    has_grasped = bool(obs["has_grasped_cube"][0] > 0.5)
    if was_closed and cube_target_dist <= release_dist:
        return False
    if was_closed or has_grasped:
        return True
    return ee_cube_dist <= close_dist


def _policy_gripper_should_close(action: np.ndarray, was_closed: bool) -> bool:
    gripper_cmd = float(action[4])
    if gripper_cmd < -0.2:
        return True
    if gripper_cmd > 0.2:
        return False
    return was_closed


def _merge_gripper_action(robot, arm_action, gripper_command: str | None):
    if gripper_command is None:
        return arm_action
    # Franka RMPFlow returns a 7-DOF arm action while gripper.forward() returns
    # an action scoped to the gripper controller. Trying to splice finger joints
    # into the 7-DOF arm action can make the articulation controller index past
    # the available action array. PickPlaceController also sends gripper-only
    # actions during close/open events, so mirror that behavior here.
    return robot.gripper.forward(action=gripper_command)


def _joint_action_from_prediction(
    robot,
    joint_prediction: np.ndarray,
    max_delta: float,
    template_action=None,
):
    current = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)[:7]
    target = np.asarray(joint_prediction, dtype=float).reshape(-1)[:7]
    delta = np.clip(target - current, -float(max_delta), float(max_delta))
    positions = (current + delta).astype(np.float32)
    if template_action is None:
        from isaacsim.core.utils.types import ArticulationAction

        return ArticulationAction(joint_positions=positions)
    template_action.joint_positions = positions
    template_action.joint_velocities = None
    template_action.joint_efforts = None
    return template_action


def _target_from_action(
    obs: dict[str, np.ndarray],
    action: np.ndarray,
    *,
    yaw: float,
    action_scale: float,
    action_version: str,
    table_top_z: float,
    fixed_orientation: bool,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    ee_pos = np.asarray(obs["ee_pos"], dtype=float)
    scale = float(action_scale)
    if action_version == CONTROLLER_TARGET_ACTION_VERSION:
        target_pos = controller_target_from_action(ee_pos, action, action_scale=scale)
    else:
        delta_pos = np.asarray(action[:3], dtype=float) * MAX_EE_DELTA_M * scale
        target_pos = ee_pos + delta_pos
    next_yaw = float(yaw + float(action[3]) * MAX_YAW_DELTA_RAD * scale)
    target_pos = np.array(
        [
            np.clip(target_pos[0], 0.20, 0.75),
            np.clip(target_pos[1], -0.35, 0.35),
            np.clip(target_pos[2], table_top_z + 0.035, table_top_z + 0.50),
        ],
        dtype=float,
    )
    target_quat = euler_angles_to_quat(np.array([0.0, np.pi, next_yaw])) if fixed_orientation else None
    return target_pos, target_quat, next_yaw


def _run() -> None:
    np.random.seed(args.seed)
    runner = PolicyRunner(args.checkpoint, args.device)

    world = create_world()
    (
        cubes,
        place_target,
        table_top_z,
        cube_size,
        table_xy,
        table_size,
        stack_base_xy,
    ) = setup_scene(world, cube_count=6)
    pick_targets = cubes[:3]
    cube_half = cube_size / 2.0
    cube_center_z = table_top_z + cube_half
    place_pos = np.array([stack_base_xy[0], stack_base_xy[1], cube_center_z])
    place_target.set_world_pose(position=place_pos)

    panda = add_panda(world, base_z=table_top_z)
    world.reset()
    world.play()
    rmp_controller = RMPFlowController(name="bc_rmpflow_controller", robot_articulation=panda)

    print(
        f"[BCRollout] checkpoint={args.checkpoint} episodes={args.episodes} "
        f"max_steps={args.max_steps} render={args.render} gripper_mode={args.gripper_mode}",
        flush=True,
    )
    successes = 0
    for episode_idx in range(args.episodes):
        randomize_cubes(
            cubes,
            table_xy,
            table_size,
            cube_center_z,
            cube_size,
            forbidden_xy=stack_base_xy,
        )
        world.reset()
        world.play()
        rmp_controller.reset()
        place_target.set_world_pose(position=place_pos)
        active_cube = pick_targets[episode_idx % len(pick_targets)]
        gripper_closed = False
        phase_event = 0
        phase_t = 0.0
        phase_hold_steps = 0
        yaw = 0.0
        success = False
        final_dist = float("inf")
        final_ee_cube = float("inf")
        final_grasped = False
        print(f"[BCRollout] episode={episode_idx:04d} active_cube={active_cube.name}", flush=True)

        for step_idx in range(args.max_steps):
            obs = _build_obs(
                panda,
                active_cube,
                place_pos,
                controller_event=phase_event,
                controller_t=phase_t,
            )
            policy_action = runner.predict(obs)
            if runner.is_joint_policy:
                target_pos = None
                target_quat = None
            else:
                target_pos, target_quat, yaw = _target_from_action(
                    obs,
                    policy_action,
                    yaw=yaw,
                    action_scale=args.action_scale,
                    action_version=runner.action_version,
                    table_top_z=table_top_z,
                    fixed_orientation=args.fixed_orientation,
                )
            gripper_command = None
            if args.gripper_mode == "event":
                gripper_closed = event_gripper_command(phase_event, gripper_closed)
                if phase_event == 3:
                    gripper_command = "close"
                elif phase_event == 7:
                    gripper_command = "open"
            elif args.gripper_mode == "rule":
                prev_gripper_closed = gripper_closed
                gripper_closed = _rule_gripper_should_close(
                    obs,
                    gripper_closed,
                    close_dist=args.close_dist,
                    release_dist=args.release_dist,
                )
            else:
                prev_gripper_closed = gripper_closed
                gripper_closed = _policy_gripper_should_close(policy_action, gripper_closed)

            if args.gripper_mode != "event":
                if gripper_closed and not prev_gripper_closed:
                    gripper_command = "close"
                elif not gripper_closed and prev_gripper_closed:
                    gripper_command = "open"

            if runner.is_joint_policy:
                template_action = rmp_controller.forward(
                    target_end_effector_position=np.asarray(obs["ee_pos"], dtype=float),
                    target_end_effector_orientation=None,
                )
                arm_action = _joint_action_from_prediction(
                    panda,
                    policy_action,
                    args.max_joint_delta,
                    template_action=template_action,
                )
            else:
                arm_action = rmp_controller.forward(
                    target_end_effector_position=target_pos,
                    target_end_effector_orientation=target_quat,
                )
            control_action = _merge_gripper_action(panda, arm_action, gripper_command)
            panda.apply_action(control_action)
            world.step(render=args.render)

            next_obs = _build_obs(
                panda,
                active_cube,
                place_pos,
                controller_event=phase_event,
                controller_t=phase_t,
            )
            final_dist = float(np.linalg.norm(next_obs["cube_to_place_target"]))
            final_ee_cube = float(np.linalg.norm(next_obs["ee_to_cube"]))
            final_grasped = bool(next_obs["has_grasped_cube"][0] > 0.5)
            success = is_success(next_obs, threshold_m=args.success_dist)
            next_event, next_t = advance_pick_place_event(phase_event, phase_t)
            hold_lowering_for_grasp = (
                args.gripper_mode == "event"
                and phase_event == 1
                and next_event != phase_event
                and final_ee_cube > args.phase_gate_close_dist
                and phase_hold_steps < args.phase_gate_max_hold
            )
            if hold_lowering_for_grasp:
                phase_hold_steps += 1
            else:
                if next_event != phase_event:
                    phase_hold_steps = 0
                phase_event, phase_t = next_event, next_t

            if args.log_every > 0 and (step_idx == 0 or (step_idx + 1) % args.log_every == 0):
                print(
                    f"[BCRollout] ep={episode_idx:04d} step={step_idx + 1:04d} "
                    f"event={phase_event:02d}:{phase_t:.3f} "
                    f"cube_target={final_dist:.3f} ee_cube={final_ee_cube:.3f} "
                    f"grasp={int(final_grasped)} grip_closed={int(gripper_closed)} "
                    f"hold={phase_hold_steps} "
                    f"policy={np.round(policy_action, 3)}",
                    flush=True,
                )
            if success:
                break

        successes += int(success)
        print(
            f"[BCRollout] episode={episode_idx:04d} success={success} "
            f"steps={step_idx + 1} cube_target={final_dist:.3f} "
            f"ee_cube={final_ee_cube:.3f} grasp={int(final_grasped)}",
            flush=True,
        )

    print(f"[BCRollout] success_rate={successes}/{args.episodes}", flush=True)


try:
    _run()
except BaseException as exc:
    print(f"[BCRollout] terminated by {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    simulation_app.close()
