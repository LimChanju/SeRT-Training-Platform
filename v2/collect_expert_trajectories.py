# collect_expert_trajectories.py -- Isaac Sim 4.5 expert demo collector
#
# Example:
#   ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "$PWD/v2/collect_expert_trajectories.py" \
#       --episodes 10 --overwrite

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_PACKAGE_DIR = os.path.join(SCRIPT_DIR, ".python_packages")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PickPlaceController expert trajectories.")
    parser.add_argument(
        "--output",
        default=os.path.join("v2", "trajectories", "expert_pick_place_v0.hdf5"),
        help="Output HDF5 path.",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to collect.")
    parser.add_argument("--max-steps", type=int, default=1800, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=7, help="Numpy random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing HDF5 file.")
    parser.add_argument("--render", action="store_true", help="Render while collecting.")
    parser.add_argument(
        "--install-missing-deps",
        action="store_true",
        help="Install missing Python dependencies into the active Isaac Python environment.",
    )
    return parser.parse_args()


args = _parse_args()


def _ensure_h5py() -> None:
    if PYTHON_PACKAGE_DIR not in sys.path:
        sys.path.insert(0, PYTHON_PACKAGE_DIR)
    try:
        import h5py  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    if not args.install_missing_deps:
        raise SystemExit(
            "[ExpertCollect] Missing dependency: h5py\n"
            "Install it in the Isaac Python environment once, or rerun with:\n"
            "  --install-missing-deps\n"
            "Example:\n"
            "  ISAAC_SKIP_VR_WAIT=1 ./launch_isaac.sh "
            "\"$PWD/v2/collect_expert_trajectories.py\" --episodes 1 "
            "--overwrite --install-missing-deps"
        )

    os.makedirs(PYTHON_PACKAGE_DIR, exist_ok=True)
    print(
        "[ExpertCollect] h5py not found; installing h5py>=3.8 into project package dir: "
        f"{PYTHON_PACKAGE_DIR}"
    )
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        PYTHON_PACKAGE_DIR,
        "--no-deps",
        "h5py>=3.8",
    ])
    import importlib

    importlib.invalidate_caches()
    import h5py  # noqa: F401


_ensure_h5py()

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
print(f"[ExpertCollect] SimulationApp headless={not args.render}")

sys.path.insert(0, SCRIPT_DIR)

from panda_robot import add_panda  # noqa: E402
from pick_controller import create_pick_controller  # noqa: E402
from scene_setup import create_world, randomize_cubes, setup_scene  # noqa: E402
from rl import (  # noqa: E402
    TrajectoryRecorder,
    build_observation,
    compute_reward,
    expert_joint_action_vector,
    is_success,
    task_action_from_transition,
)


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


def _build_obs(robot, cube, place_pos: np.ndarray) -> dict[str, np.ndarray]:
    gripper_center = _gripper_center_from_fingers(robot)
    has_grasped = _has_grasped_cube(robot, cube, gripper_center)
    obs = build_observation(
        robot=robot,
        cube=cube,
        place_target=place_pos,
        gripper_center_pos=gripper_center,
        has_grasped_cube=has_grasped,
        task_phase="approach_cube",
    )
    obs["task_phase"] = obs["task_phase"] * 0.0
    from rl.observations import task_phase_onehot

    obs["task_phase"] = task_phase_onehot(_task_phase(obs))
    return obs


def _collect() -> None:
    np.random.seed(args.seed)
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

    controller = create_pick_controller(panda, end_effector_initial_height=table_top_z + 0.2)

    with TrajectoryRecorder(args.output, overwrite=args.overwrite) as recorder:
        print(
            f"[ExpertCollect] output={recorder.path} episodes={args.episodes} "
            f"max_steps={args.max_steps} render={args.render}"
        )
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
            controller.reset(end_effector_initial_height=table_top_z + 0.2)
            active_cube = pick_targets[episode_idx % len(pick_targets)]
            place_target.set_world_pose(position=place_pos)

            recorder.start_episode(
                {
                    "collector": "PickPlaceController",
                    "episode_index": episode_idx,
                    "active_cube": active_cube.name,
                    "seed": args.seed,
                }
            )
            success = False
            steps = 0
            for step_idx in range(args.max_steps):
                obs = _build_obs(panda, active_cube, place_pos)
                cube_pos, _ = active_cube.get_world_pose()
                current_joint_positions = panda.get_joint_positions()
                expert_action = controller.forward(
                    picking_position=np.asarray(cube_pos, dtype=float),
                    placing_position=place_pos,
                    current_joint_positions=current_joint_positions,
                    end_effector_offset=np.array([0.0, 0.005, 0.0]),
                )
                expert_joint_action = expert_joint_action_vector(expert_action)
                panda.apply_action(expert_action)
                world.step(render=args.render)

                next_obs = _build_obs(panda, active_cube, place_pos)
                expert_task_action = task_action_from_transition(
                    obs["ee_pos"],
                    next_obs["ee_pos"],
                    gripper_opening_now=float(obs["gripper_width"][0]),
                    gripper_opening_next=float(next_obs["gripper_width"][0]),
                )
                success = bool(controller.is_done() or is_success(next_obs))
                reward = compute_reward(
                    obs,
                    next_obs,
                    expert_task_action,
                    errp_feedback=0.0,
                    success=success,
                )
                recorder.add_transition(
                    obs=obs,
                    next_obs=next_obs,
                    policy_action=expert_task_action,
                    expert_task_action=expert_task_action,
                    expert_joint_action=expert_joint_action,
                    reward=reward,
                    done=success,
                    errp_label=0,
                    errp_feedback=0.0,
                    errp_source_code=0,
                    errp_event_step=0,
                    eeg_replay_used=0,
                    eeg_epoch_id="",
                )
                steps = step_idx + 1
                if success:
                    break
            group_name = recorder.end_episode(
                success=success,
                metadata={
                    "steps": steps,
                    "timeout": not success,
                },
            )
            print(
                f"[ExpertCollect] episode={episode_idx:04d} success={success} "
                f"steps={steps} group={group_name}"
            )


try:
    _collect()
finally:
    simulation_app.close()
