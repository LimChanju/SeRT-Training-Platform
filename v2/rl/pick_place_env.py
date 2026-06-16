from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from .actions import (
    CONTROLLER_TARGET_ACTION_VERSION,
    MAX_EE_DELTA_M,
    MAX_YAW_DELTA_RAD,
    clip_action,
    controller_target_from_action,
)
from .observations import build_observation, flatten_observation, task_phase_onehot
from .pick_place_phase import advance_pick_place_event, event_gripper_command, task_phase_from_event
from .rewards import DEFAULT_REWARD_WEIGHTS, RewardWeights, compute_reward, is_success


GripperMode = Literal["event", "rule", "policy"]
ObservationMode = Literal["flat", "dict"]


@dataclass
class PickPlaceEnvConfig:
    """Runtime knobs for the Isaac pick-and-place RL environment wrapper."""

    cube_count: int = 6
    max_episode_steps: int = 1200
    success_dist: float = 0.06
    action_scale: float = 1.0
    action_version: str = CONTROLLER_TARGET_ACTION_VERSION
    fixed_orientation: bool = True
    gripper_mode: GripperMode = "event"
    close_dist: float = 0.08
    release_dist: float = 0.07
    phase_gate_close_dist: float = 0.075
    phase_gate_max_hold: int = 320
    early_close_on_grasp_gate: bool = False
    fast_forward_grasp_gate: bool = False
    release_gate_dist: float | None = None
    release_gate_max_hold: int = 240
    require_release_for_success: bool = False
    observation_mode: ObservationMode = "flat"
    seed: int = 11
    render: bool = False
    reward_weights: RewardWeights = field(default_factory=lambda: DEFAULT_REWARD_WEIGHTS)


class IsaacPickPlaceEnv:
    """A light Gymnasium-style wrapper around the current Isaac pick-and-place scene.

    This class assumes `SimulationApp` has already been created by the caller.
    It owns the Isaac World, Panda robot, cubes, target marker, RMPFlow controller,
    observation construction, reward computation, and episode phase clock.
    """

    metadata = {
        "observation_modes": ("flat", "dict"),
        "action_version": CONTROLLER_TARGET_ACTION_VERSION,
    }

    def __init__(
        self,
        config: PickPlaceEnvConfig | None = None,
        *,
        human_state_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or PickPlaceEnvConfig()
        self.human_state_fn = human_state_fn
        self.rng = np.random.default_rng(self.config.seed)

        from isaacsim.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.franka.controllers import RMPFlowController

        from panda_robot import add_panda
        from scene_setup import create_world, setup_scene

        self._euler_angles_to_quat = euler_angles_to_quat
        self.world = create_world()
        (
            self.cubes,
            self.place_target,
            self.table_top_z,
            self.cube_size,
            self.table_xy,
            self.table_size,
            self.stack_base_xy,
        ) = setup_scene(self.world, cube_count=self.config.cube_count)
        self.pick_targets = self.cubes[: min(3, len(self.cubes))]
        self.cube_half = self.cube_size / 2.0
        self.cube_center_z = self.table_top_z + self.cube_half
        self.place_pos = np.array([self.stack_base_xy[0], self.stack_base_xy[1], self.cube_center_z])
        self.place_target.set_world_pose(position=self.place_pos)

        self.robot = add_panda(self.world, base_z=self.table_top_z)
        self.world.reset()
        self.world.play()
        self.controller = RMPFlowController(name="rl_env_rmpflow_controller", robot_articulation=self.robot)

        self.episode_index = 0
        self.current_episode_index = 0
        self.active_cube = self.pick_targets[0]
        self.step_count = 0
        self.phase_event = 0
        self.phase_t = 0.0
        self.phase_hold_steps = 0
        self.gripper_closed = False
        self.yaw = 0.0
        self._last_obs: dict[str, np.ndarray] | None = None

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (5,)

    @property
    def observation_shape(self) -> tuple[int, ...] | None:
        if self.config.observation_mode == "dict":
            return None
        from .observations import OBSERVATION_DIM

        return (OBSERVATION_DIM,)

    def reset(
        self,
        *,
        seed: int | None = None,
        active_cube_index: int | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            np.random.seed(seed)
        else:
            np.random.seed(int(self.rng.integers(0, 2**31 - 1)))

        from scene_setup import randomize_cubes

        randomize_cubes(
            self.cubes,
            self.table_xy,
            self.table_size,
            self.cube_center_z,
            self.cube_size,
            forbidden_xy=self.stack_base_xy,
        )
        self.world.reset()
        self.world.play()
        self.controller.reset()
        self.place_target.set_world_pose(position=self.place_pos)

        self.current_episode_index = self.episode_index
        if active_cube_index is None:
            active_cube_index = self.current_episode_index % len(self.pick_targets)
        self.active_cube = self.pick_targets[int(active_cube_index) % len(self.pick_targets)]
        self.step_count = 0
        self.phase_event = 0
        self.phase_t = 0.0
        self.phase_hold_steps = 0
        self.gripper_closed = False
        self.yaw = 0.0

        obs = self._build_obs()
        self._last_obs = obs
        info = self._info(obs, reward_components={}, errp_feedback=0.0)
        self.episode_index += 1
        return self._format_obs(obs), info

    def step(
        self,
        action: np.ndarray,
        *,
        errp_feedback: float | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before step()")

        action = clip_action(action)
        target_pos, target_quat, self.yaw = self._target_from_action(action)
        gripper_command = self._gripper_command(action, self._last_obs)

        arm_action = self.controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=target_quat,
        )
        control_action = self._merge_gripper_action(arm_action, gripper_command)
        self.robot.apply_action(control_action)
        self.world.step(render=self.config.render)
        self.step_count += 1

        next_obs = self._build_obs()
        self._advance_phase(next_obs)
        success = self._is_success(next_obs)
        truncated = self.step_count >= self.config.max_episode_steps and not success
        errp_value = self._pseudo_errp_feedback(next_obs) if errp_feedback is None else float(errp_feedback)
        reward_result = compute_reward(
            self._last_obs,
            next_obs,
            action,
            errp_feedback=errp_value,
            success=success,
            success_dist=self.config.success_dist,
            weights=self.config.reward_weights,
        )
        self._last_obs = next_obs

        info = self._info(
            next_obs,
            reward_components=reward_result.components,
            errp_feedback=errp_value,
        )
        return self._format_obs(next_obs), reward_result.total, success, truncated, info

    def close(self) -> None:
        self.world.stop()

    def _build_obs(self) -> dict[str, np.ndarray]:
        gripper_center = _gripper_center_from_fingers(self.robot)
        has_grasped = _has_grasped_cube(self.robot, self.active_cube, gripper_center)
        human_state = dict(self.human_state_fn() if self.human_state_fn is not None else {})
        task_phase = task_phase_from_event(self.phase_event)
        obs = build_observation(
            robot=self.robot,
            cube=self.active_cube,
            place_target=self.place_pos,
            gripper_center_pos=gripper_center,
            has_grasped_cube=has_grasped,
            task_phase=task_phase,
            controller_event=self.phase_event,
            controller_t=self.phase_t,
            **human_state,
        )
        if self.phase_event is None:
            obs["task_phase"] = task_phase_onehot("approach_cube")
        return obs

    def _format_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
        if self.config.observation_mode == "dict":
            return obs
        return flatten_observation(obs)

    def _target_from_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, float]:
        ee_pos = np.asarray(self._last_obs["ee_pos"], dtype=float)
        if self.config.action_version == CONTROLLER_TARGET_ACTION_VERSION:
            target_pos = controller_target_from_action(
                ee_pos,
                action,
                action_scale=self.config.action_scale,
            )
        else:
            target_pos = ee_pos + np.asarray(action[:3], dtype=float) * MAX_EE_DELTA_M * self.config.action_scale
        next_yaw = float(self.yaw + float(action[3]) * MAX_YAW_DELTA_RAD * self.config.action_scale)
        target_pos = np.array(
            [
                np.clip(target_pos[0], 0.20, 0.75),
                np.clip(target_pos[1], -0.35, 0.35),
                np.clip(target_pos[2], self.table_top_z + 0.035, self.table_top_z + 0.50),
            ],
            dtype=float,
        )
        target_quat = (
            self._euler_angles_to_quat(np.array([0.0, np.pi, next_yaw]))
            if self.config.fixed_orientation
            else None
        )
        return target_pos, target_quat, next_yaw

    def _gripper_command(self, action: np.ndarray, obs: dict[str, np.ndarray]) -> str | None:
        if self.config.gripper_mode == "event":
            if (
                self.config.early_close_on_grasp_gate
                and self.phase_event in (1, 2)
                and not self.gripper_closed
                and _rule_gripper_should_close(
                    obs,
                    self.gripper_closed,
                    close_dist=self.config.phase_gate_close_dist,
                    release_dist=self.config.release_dist,
                )
            ):
                if self.config.fast_forward_grasp_gate:
                    self.phase_event = 3
                    self.phase_t = 0.0
                    self.phase_hold_steps = 0
                self.gripper_closed = True
                return "close"
            self.gripper_closed = event_gripper_command(self.phase_event, self.gripper_closed)
            if self.phase_event == 3:
                return "close"
            if self.phase_event == 7:
                return "open"
            return None

        previous_closed = self.gripper_closed
        if self.config.gripper_mode == "rule":
            self.gripper_closed = _rule_gripper_should_close(
                obs,
                self.gripper_closed,
                close_dist=self.config.close_dist,
                release_dist=self.config.release_dist,
            )
        elif self.config.gripper_mode == "policy":
            self.gripper_closed = _policy_gripper_should_close(action, self.gripper_closed)
        else:
            raise ValueError(f"Unknown gripper_mode: {self.config.gripper_mode}")

        if self.gripper_closed and not previous_closed:
            return "close"
        if not self.gripper_closed and previous_closed:
            return "open"
        return None

    def _merge_gripper_action(self, arm_action, gripper_command: str | None):
        if gripper_command is None:
            return arm_action
        return self.robot.gripper.forward(action=gripper_command)

    def _advance_phase(self, obs: dict[str, np.ndarray]) -> None:
        next_event, next_t = advance_pick_place_event(self.phase_event, self.phase_t)
        ee_cube_dist = float(np.linalg.norm(obs["ee_to_cube"]))
        cube_target_dist = float(np.linalg.norm(obs["cube_to_place_target"]))
        hold_lowering_for_grasp = (
            self.config.gripper_mode == "event"
            and self.phase_event in (1, 2)
            and next_event != self.phase_event
            and ee_cube_dist > self.config.phase_gate_close_dist
            and float(obs["has_grasped_cube"][0]) <= 0.5
            and self.phase_hold_steps < self.config.phase_gate_max_hold
        )
        hold_release_for_target = (
            self.config.gripper_mode == "event"
            and self.config.release_gate_dist is not None
            and self.phase_event == 6
            and next_event != self.phase_event
            and cube_target_dist > float(self.config.release_gate_dist)
            and self.phase_hold_steps < self.config.release_gate_max_hold
        )
        if hold_lowering_for_grasp or hold_release_for_target:
            self.phase_hold_steps += 1
            return
        if next_event != self.phase_event:
            self.phase_hold_steps = 0
        self.phase_event, self.phase_t = next_event, next_t

    def _pseudo_errp_feedback(self, obs: dict[str, np.ndarray]) -> float:
        flags = (
            "human_robot_collision",
            "near_human",
            "collision_green",
            "pick_miss_recent",
            "drop_throw_recent",
        )
        return float(any(float(np.asarray(obs[name]).reshape(-1)[0]) > 0.5 for name in flags))

    def _is_success(self, obs: dict[str, np.ndarray]) -> bool:
        if not is_success(obs, threshold_m=self.config.success_dist):
            return False
        if not self.config.require_release_for_success:
            return True
        has_grasped = bool(float(np.asarray(obs["has_grasped_cube"]).reshape(-1)[0]) > 0.5)
        return self.phase_event >= 7 and not has_grasped

    def _info(
        self,
        obs: dict[str, np.ndarray],
        *,
        reward_components: dict[str, float],
        errp_feedback: float,
    ) -> dict[str, Any]:
        return {
            "episode_index": self.current_episode_index,
            "step": self.step_count,
            "active_cube": getattr(self.active_cube, "name", ""),
            "controller_event": int(self.phase_event),
            "controller_t": float(self.phase_t),
            "phase_hold_steps": int(self.phase_hold_steps),
            "gripper_closed": bool(self.gripper_closed),
            "success": self._is_success(obs),
            "cube_target_dist": float(np.linalg.norm(obs["cube_to_place_target"])),
            "ee_cube_dist": float(np.linalg.norm(obs["ee_to_cube"])),
            "has_grasped_cube": bool(float(obs["has_grasped_cube"][0]) > 0.5),
            "errp_feedback": float(errp_feedback),
            "reward_components": dict(reward_components),
            "obs_dict": obs,
        }


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
