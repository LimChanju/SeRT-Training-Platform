"""Optional cuRobo MPPI/MPC arm controller for Isaac Sim 4.5.

The adapter targets the stable cuRobo v0.7.8 API because the current project
runs Isaac Sim 4.5.  cuRobo is an optional dependency: importing this module is
cheap, while selecting the ``curobo`` runtime mode fails with an actionable
message when the package is unavailable.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

try:
    from v3_chan.physical_safety_controllers import PhysicalSafetyDiagnostics
except ImportError:
    from physical_safety_controllers import PhysicalSafetyDiagnostics


CUROBO_INSTALL_HINT = (
    "cuRobo is required for --physical-safety-controller curobo. "
    "For Isaac Sim 4.5, install NVLabs/curobo tag v0.7.8 into the Isaac "
    "Python environment, then verify examples/isaac_sim/mpc_example.py."
)


class CuRoboMpcArmController:
    """Generate a Franka arm action with cuRobo's MPPI MPC solver."""

    def __init__(
        self,
        *,
        robot,
        physics_dt_s: float,
        hand_radius_m: float,
        safety_margin_m: float,
        table_center_world_m: np.ndarray,
        table_size_m: np.ndarray,
        robot_config: str = "franka.yml",
    ) -> None:
        if not math.isfinite(float(physics_dt_s)) or physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be finite and positive")
        if hand_radius_m <= 0.0 or safety_margin_m < 0.0:
            raise ValueError("invalid cuRobo hand radius or safety margin")
        table_center_world_m = _finite_vector3(
            table_center_world_m, name="table_center_world_m"
        )
        table_size_m = _finite_vector3(table_size_m, name="table_size_m")
        if np.any(table_size_m <= 0.0):
            raise ValueError("table_size_m must contain positive dimensions")
        try:
            import torch
            from curobo.geom.sdf.world import CollisionCheckerType
            from curobo.geom.types import Cuboid, WorldConfig
            from curobo.rollout.rollout_base import Goal
            from curobo.types.base import TensorDeviceType
            from curobo.types.math import Pose
            from curobo.types.state import JointState
            from curobo.util_file import (
                get_robot_configs_path,
                join_path,
                load_yaml,
            )
            from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModuleNotFoundError(CUROBO_INSTALL_HINT) from exc

        self._torch = torch
        self._Cuboid = Cuboid
        self._WorldConfig = WorldConfig
        self._Goal = Goal
        self._Pose = Pose
        self._JointState = JointState
        self._TensorDeviceType = TensorDeviceType
        self.robot = robot
        self.physics_dt_s = float(physics_dt_s)
        self.hand_radius_m = float(hand_radius_m)
        self.safety_margin_m = float(safety_margin_m)
        self.tensor_args = TensorDeviceType()
        self._human_positions: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }

        base_position, base_orientation = self.robot.get_world_pose()
        self._base_position = np.asarray(base_position, dtype=float).reshape(3)
        self._base_orientation = _normalized_quaternion(base_orientation)
        self._base_rotation = _quaternion_rotation_matrix(self._base_orientation)
        table_center_base = self._world_point_to_robot_base(table_center_world_m)

        robot_cfg = load_yaml(
            join_path(get_robot_configs_path(), robot_config)
        )["robot_cfg"]
        diameter = 2.0 * self.hand_radius_m
        parked_pose = [0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0]
        world = WorldConfig(
            cuboid=[
                Cuboid(
                    name="work_table",
                    pose=[*table_center_base.tolist(), 1.0, 0.0, 0.0, 0.0],
                    dims=table_size_m.tolist(),
                ),
                Cuboid(
                    name="human_left_hand",
                    pose=list(parked_pose),
                    dims=[diameter, diameter, diameter],
                ),
                Cuboid(
                    name="human_right_hand",
                    pose=list(parked_pose),
                    dims=[diameter, diameter, diameter],
                ),
            ]
        )
        config = MpcSolverConfig.load_from_robot_config(
            robot_cfg,
            world,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            self_collision_check=True,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            collision_cache={"obb": 8, "mesh": 0},
            collision_activation_distance=self.safety_margin_m,
            use_mppi=True,
            use_lbfgs=False,
            store_rollouts=False,
            step_dt=self.physics_dt_s,
        )
        self._mpc = MpcSolver(config)
        # cuRobo JointState.clone() copies joint_names in place and therefore
        # requires the mutable list used by the upstream Isaac Sim examples.
        self._joint_names = [str(name) for name in self._mpc.rollout_fn.joint_names]
        self._joint_indices = np.asarray(
            [self.robot.get_dof_index(name) for name in self._joint_names],
            dtype=int,
        )
        current_state = self._current_joint_state()
        kinematics = self._mpc.rollout_fn.compute_kinematics(current_state)
        initial_pose = Pose(
            position=kinematics.ee_pos_seq,
            quaternion=kinematics.ee_quat_seq,
        )
        goal = Goal(
            current_state=current_state,
            goal_state=current_state.clone(),
            goal_pose=initial_pose,
        )
        self._goal_buffer = self._mpc.setup_solve_single(goal, 1)
        self._mpc.update_goal(self._goal_buffer)
        self.last_diagnostics = PhysicalSafetyDiagnostics(controller="curobo")

    def reset(self) -> None:
        self._mpc.reset()
        self.last_diagnostics = PhysicalSafetyDiagnostics(controller="curobo")

    def update_human_obstacles(self, human_state: dict[str, Any]) -> None:
        for hand in ("left", "right"):
            value = human_state.get(f"human_{hand}_hand_pos")
            self._human_positions[hand] = _valid_position_or_none(value)

    def forward(
        self,
        *,
        target_end_effector_position: np.ndarray,
        target_end_effector_orientation: np.ndarray | None,
        observation: dict[str, np.ndarray],
    ):
        del observation
        started = time.perf_counter()
        self._update_collision_world()
        target_position = self._world_point_to_robot_base(
            target_end_effector_position
        )
        if target_end_effector_orientation is None:
            _, target_end_effector_orientation = self.robot.end_effector.get_world_pose()
        target_orientation = self._world_quaternion_to_robot_base(
            target_end_effector_orientation
        )
        goal_pose = self._Pose(
            position=self.tensor_args.to_device(target_position),
            quaternion=self.tensor_args.to_device(target_orientation),
        )
        self._goal_buffer.goal_pose.copy_(goal_pose)
        self._mpc.update_goal(self._goal_buffer)

        current_state = self._current_joint_state()
        result = self._mpc.step(current_state, max_attempts=2)
        command = getattr(result, "js_action", None)
        if command is None:
            command = getattr(result, "action", None)
        if command is None:
            raise RuntimeError("cuRobo MPC returned no joint action")
        command = command.get_ordered_joint_state(self._joint_names)
        positions = _tensor_numpy(command.position).reshape(-1)
        velocity_value = getattr(command, "velocity", None)
        velocities = (
            _tensor_numpy(velocity_value).reshape(-1)
            if velocity_value is not None
            else None
        )

        from isaacsim.core.utils.types import ArticulationAction

        action = ArticulationAction(
            joint_positions=positions,
            joint_velocities=velocities,
            joint_indices=self._joint_indices,
        )
        feasible = _result_feasible(result)
        valid_hands = sum(
            position is not None for position in self._human_positions.values()
        )
        diagnostics = PhysicalSafetyDiagnostics(
            controller="curobo",
            active=valid_hands > 0,
            intervention_available=False,
            valid_hand_count=int(valid_hands),
            filtered_velocity_norm_radps=(
                float(np.linalg.norm(velocities)) if velocities is not None else 0.0
            ),
            solve_time_ms=(time.perf_counter() - started) * 1000.0,
            feasible=feasible,
            status="solved" if feasible else "mpc_infeasible",
        )
        self.last_diagnostics = diagnostics
        return action, diagnostics

    def _current_joint_state(self):
        sim_state = self.robot.get_joints_state()
        if sim_state is None:
            raise RuntimeError("Isaac robot joint state is unavailable")
        positions = self.tensor_args.to_device(
            np.asarray(sim_state.positions, dtype=float)[None, :]
        )
        velocities_np = np.asarray(sim_state.velocities, dtype=float)
        velocities = self.tensor_args.to_device(velocities_np[None, :])
        zeros = self._torch.zeros_like(velocities)
        state = self._JointState(
            position=positions,
            velocity=velocities,
            acceleration=zeros,
            jerk=zeros,
            joint_names=[str(name) for name in self.robot.dof_names],
        )
        return state.get_ordered_joint_state(self._joint_names)

    def _update_collision_world(self) -> None:
        parked = np.array([0.0, 0.0, -10.0], dtype=float)
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for hand, world_position in self._human_positions.items():
            position = (
                self._world_point_to_robot_base(world_position)
                if world_position is not None
                else parked
            )
            pose = self._Pose(
                position=self.tensor_args.to_device(position),
                quaternion=self.tensor_args.to_device(identity),
            )
            self._mpc.world_coll_checker.update_obstacle_pose(
                name=f"human_{hand}_hand",
                w_obj_pose=pose,
            )

    def _world_point_to_robot_base(self, point) -> np.ndarray:
        point = np.asarray(point, dtype=float).reshape(3)
        return self._base_rotation.T @ (point - self._base_position)

    def _world_quaternion_to_robot_base(self, quaternion) -> np.ndarray:
        world = _normalized_quaternion(quaternion)
        base_inverse = self._base_orientation.copy()
        base_inverse[1:] *= -1.0
        return _normalized_quaternion(_quaternion_multiply(base_inverse, world))


def _result_feasible(result) -> bool:
    metrics = getattr(result, "metrics", None)
    value = getattr(metrics, "feasible", None) if metrics is not None else None
    if value is None:
        return True
    array = _tensor_numpy(value).reshape(-1)
    return bool(array.size > 0 and np.all(array > 0.5))


def _valid_position_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if (
        array.size < 3
        or not np.all(np.isfinite(array[:3]))
        or float(np.linalg.norm(array[:3])) <= 1e-8
    ):
        return None
    return array[:3].copy()


def _finite_vector3(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    return array.copy()


def _tensor_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalized_quaternion(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float).reshape(-1)[:4]
    if quaternion.size != 4 or not np.all(np.isfinite(quaternion)):
        raise ValueError("invalid quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise ValueError("zero quaternion")
    return quaternion / norm


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def _quaternion_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalized_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
