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
from .observations import MISSING_DISTANCE_M, build_observation, flatten_observation, task_phase_onehot
from .pick_place_phase import advance_pick_place_event, event_gripper_command, task_phase_from_event
from .pseudo_errp import (
    DEFAULT_PSEUDO_ERRP_SOURCES,
    PseudoErrPResult,
    extract_pseudo_errp_aux_flags,
    pseudo_errp_from_observation,
)
from .rewards import DEFAULT_REWARD_WEIGHTS, RewardWeights, compute_reward, is_success


GripperMode = Literal["event", "rule", "policy"]
ObservationMode = Literal["flat", "dict"]
HumanObservationMode = Literal["policy_and_reward", "reward_only"]


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
    strict_grasp_phase_gate: bool = False
    release_gate_dist: float | None = None
    release_gate_max_hold: int = 240
    strict_release_phase_gate: bool = False
    require_release_for_success: bool = False
    observation_mode: ObservationMode = "flat"
    human_observation_mode: HumanObservationMode = "policy_and_reward"
    seed: int = 11
    render: bool = False
    reward_weights: RewardWeights = field(default_factory=lambda: DEFAULT_REWARD_WEIGHTS)
    pseudo_errp_enabled: bool = True
    pseudo_errp_sources: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_PSEUDO_ERRP_SOURCES
    )
    visualize_human_replay: bool = False
    synthetic_human_enabled: bool = False
    synthetic_human_episode_prob: float = 0.35
    synthetic_human_start_min_step: int = 120
    synthetic_human_start_max_step: int = 520
    synthetic_human_duration_steps: int = 90
    synthetic_human_near_dist: float = 0.12
    synthetic_human_collision_dist: float = 0.035


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
        self._pseudo_errp_aux_flags: dict[str, float] = {}
        self._human_replay_aux_state: dict[str, Any] = {}
        self._synthetic_human_active = False
        self._synthetic_human_start_step = 0
        self._synthetic_human_duration_steps = 0
        self._synthetic_human_side = 1.0
        self._synthetic_human_height_offset = 0.0
        self._phase_gate_blocked_reason = ""
        self._human_visual_prims: dict[str, Any] = {}
        if self.config.visualize_human_replay:
            self._setup_human_visuals()

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
        self._phase_gate_blocked_reason = ""
        self._reset_synthetic_human()

        obs = self._build_obs()
        self._last_obs = obs
        errp_result = self._pseudo_errp_result(obs, override_feedback=0.0)
        info = self._info(obs, reward_components={}, errp_result=errp_result)
        self.episode_index += 1
        return self._format_obs(obs), info

    def start_next_pick_target(
        self,
        *,
        active_cube_index: int | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        """Continue the current scene with another cube as the active pick target."""

        self.controller.reset()
        if active_cube_index is None:
            try:
                current_idx = self.pick_targets.index(self.active_cube)
            except ValueError:
                current_idx = -1
            active_cube_index = current_idx + 1
        self.active_cube = self.pick_targets[int(active_cube_index) % len(self.pick_targets)]
        self.phase_event = 0
        self.phase_t = 0.0
        self.phase_hold_steps = 0
        self.gripper_closed = False
        self.yaw = 0.0
        self._phase_gate_blocked_reason = ""

        obs = self._build_obs()
        self._last_obs = obs
        errp_result = self._pseudo_errp_result(obs, override_feedback=0.0)
        info = self._info(obs, reward_components={}, errp_result=errp_result)
        return self._format_obs(obs), info

    def step(
        self,
        action: np.ndarray,
        *,
        errp_feedback: float | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before step()")

        action = _finite_action(action)
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
        errp_result = self._pseudo_errp_result(next_obs, override_feedback=errp_feedback)
        reward_result = compute_reward(
            self._last_obs,
            next_obs,
            action,
            errp_feedback=errp_result.feedback,
            success=success,
            success_dist=self.config.success_dist,
            weights=self.config.reward_weights,
        )
        self._last_obs = next_obs

        info = self._info(
            next_obs,
            reward_components=reward_result.components,
            errp_result=errp_result,
        )
        return self._format_obs(next_obs), reward_result.total, success, truncated, info

    def close(self) -> None:
        self.world.stop()

    def _build_obs(self) -> dict[str, np.ndarray]:
        gripper_center = _gripper_center_from_fingers(self.robot)
        if gripper_center is None:
            try:
                gripper_center, _ = self.robot.end_effector.get_world_pose()
                gripper_center = np.asarray(gripper_center, dtype=float)
            except Exception:
                gripper_center = None
        has_grasped = _has_grasped_cube(self.robot, self.active_cube, gripper_center)
        human_state = dict(self.human_state_fn() if self.human_state_fn is not None else {})
        synthetic_state = self._synthetic_human_state(gripper_center)
        human_state = {**synthetic_state, **human_state}
        human_state, self._pseudo_errp_aux_flags = extract_pseudo_errp_aux_flags(human_state)
        human_state, self._human_replay_aux_state = _split_observation_human_state(human_state)
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
        self._update_human_visuals(obs)
        return obs

    def _setup_human_visuals(self) -> None:
        from omni.isaac.core.objects import VisualSphere

        specs = (
            ("head", "/World/HumanReplay/head", "human_replay_head", 0.045, np.array([0.8, 0.8, 0.8])),
            ("left", "/World/HumanReplay/left_hand", "human_replay_left_hand", 0.035, np.array([0.45, 0.65, 1.0])),
            ("right", "/World/HumanReplay/right_hand", "human_replay_right_hand", 0.035, np.array([1.0, 0.55, 0.25])),
        )
        parked = np.array([0.0, 0.0, -10.0], dtype=float)
        for key, prim_path, name, radius, color in specs:
            self._human_visual_prims[key] = self.world.scene.add(
                VisualSphere(
                    prim_path=prim_path,
                    name=name,
                    position=parked,
                    radius=radius,
                    color=color,
                )
            )

    def _update_human_visuals(self, obs: dict[str, np.ndarray]) -> None:
        if not self._human_visual_prims:
            return
        fields = {
            "head": "human_head_pos",
            "left": "human_left_hand_pos",
            "right": "human_right_hand_pos",
        }
        parked = np.array([0.0, 0.0, -10.0], dtype=float)
        for key, field_name in fields.items():
            prim = self._human_visual_prims.get(key)
            if prim is None:
                continue
            pos = np.asarray(obs.get(field_name, parked), dtype=float).reshape(-1)
            if pos.size < 3 or not np.all(np.isfinite(pos[:3])) or np.linalg.norm(pos[:3]) < 1e-6:
                pos = parked
            prim.set_world_pose(position=pos[:3])

    def _reset_synthetic_human(self) -> None:
        cfg = self.config
        self._synthetic_human_active = (
            bool(cfg.synthetic_human_enabled)
            and float(self.rng.random()) < float(np.clip(cfg.synthetic_human_episode_prob, 0.0, 1.0))
        )
        start_min = max(0, int(cfg.synthetic_human_start_min_step))
        start_max = max(start_min, int(cfg.synthetic_human_start_max_step))
        if start_max > start_min:
            self._synthetic_human_start_step = int(self.rng.integers(start_min, start_max + 1))
        else:
            self._synthetic_human_start_step = start_min
        self._synthetic_human_duration_steps = max(1, int(cfg.synthetic_human_duration_steps))
        self._synthetic_human_side = -1.0 if float(self.rng.random()) < 0.5 else 1.0
        self._synthetic_human_height_offset = float(self.rng.uniform(-0.025, 0.055))

    def _synthetic_human_state(self, gripper_center: np.ndarray) -> dict[str, Any]:
        if not self._synthetic_human_active:
            return {}
        if gripper_center is None:
            return {}
        gripper_center = np.asarray(gripper_center, dtype=float).reshape(-1)
        if gripper_center.size < 3 or not np.all(np.isfinite(gripper_center[:3])):
            return {}
        gripper_center = gripper_center[:3]
        local_step = self.step_count - self._synthetic_human_start_step
        if local_step < 0 or local_step > self._synthetic_human_duration_steps:
            return {}

        progress = float(local_step / max(1, self._synthetic_human_duration_steps))
        cfg = self.config
        near_dist = max(float(cfg.synthetic_human_near_dist), 1e-3)
        collision_dist = max(float(cfg.synthetic_human_collision_dist), 1e-3)
        min_dist = max(collision_dist * 0.5, 0.015)

        # Sweep the hand across the gripper. The midpoint is closest, so some
        # episodes produce only proximity feedback while others produce collision
        # feedback depending on the randomized height offset.
        lateral = self._synthetic_human_side * np.interp(progress, [0.0, 1.0], [near_dist * 1.8, -near_dist * 1.8])
        closest = min_dist + abs(self._synthetic_human_height_offset) * 0.35
        vertical = self._synthetic_human_height_offset
        forward = closest * np.sin(np.pi * progress)
        right_hand = gripper_center + np.array([lateral, forward, vertical], dtype=float)
        left_hand = right_hand + np.array([0.22 * self._synthetic_human_side, -0.18, 0.02], dtype=float)
        head = right_hand + np.array([0.0, -0.55, 0.55], dtype=float)

        dist = float(np.linalg.norm(right_hand - gripper_center))
        return {
            "human_head_pos": head,
            "human_left_hand_pos": left_hand,
            "human_right_hand_pos": right_hand,
            "min_hand_gripper_dist_override": dist,
            "near_human": dist <= near_dist,
            "human_robot_collision": dist <= collision_dist,
        }

    def _format_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
        obs = self._policy_observation(obs)
        if self.config.observation_mode == "dict":
            return obs
        return flatten_observation(obs)

    def _policy_observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.config.human_observation_mode == "policy_and_reward":
            return obs
        if self.config.human_observation_mode == "reward_only":
            return _mask_human_observation(obs)
        raise ValueError(f"Unknown human_observation_mode: {self.config.human_observation_mode}")

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
        current_yaw = self.yaw if np.isfinite(self.yaw) else 0.0
        next_yaw = float(current_yaw + float(action[3]) * MAX_YAW_DELTA_RAD * self.config.action_scale)
        if not np.isfinite(next_yaw):
            next_yaw = 0.0
        target_pos = np.array(
            [
                np.clip(target_pos[0], 0.20, 0.75),
                np.clip(target_pos[1], -0.35, 0.35),
                np.clip(target_pos[2], self.table_top_z + 0.035, self.table_top_z + 0.50),
            ],
            dtype=float,
        )
        target_quat = None
        if self.config.fixed_orientation:
            target_quat = _safe_quat(self._euler_angles_to_quat(np.array([0.0, np.pi, next_yaw])))
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
        has_grasped = bool(float(obs["has_grasped_cube"][0]) > 0.5)
        self._phase_gate_blocked_reason = ""
        hold_lowering_for_grasp = (
            self.config.gripper_mode == "event"
            and self.phase_event in (1, 2)
            and next_event != self.phase_event
            and ee_cube_dist > self.config.phase_gate_close_dist
            and not has_grasped
            and (
                self.config.strict_grasp_phase_gate
                or self.phase_hold_steps < self.config.phase_gate_max_hold
            )
        )
        hold_after_close_until_grasped = (
            self.config.gripper_mode == "event"
            and self.config.strict_grasp_phase_gate
            and self.phase_event in (3, 4, 5, 6)
            and next_event != self.phase_event
            and not has_grasped
        )
        hold_release_for_target = (
            self.config.gripper_mode == "event"
            and self.config.release_gate_dist is not None
            and self.phase_event == 6
            and next_event != self.phase_event
            and cube_target_dist > float(self.config.release_gate_dist)
            and (
                self.config.strict_release_phase_gate
                or self.phase_hold_steps < self.config.release_gate_max_hold
            )
        )
        if hold_lowering_for_grasp:
            self._phase_gate_blocked_reason = "approach_until_close_or_grasped"
        elif hold_after_close_until_grasped:
            self._phase_gate_blocked_reason = "wait_until_grasped"
        elif hold_release_for_target:
            self._phase_gate_blocked_reason = "hold_release_until_target"
        if hold_lowering_for_grasp or hold_after_close_until_grasped or hold_release_for_target:
            self.phase_hold_steps += 1
            return
        if next_event != self.phase_event:
            self.phase_hold_steps = 0
        self.phase_event, self.phase_t = next_event, next_t

    def _pseudo_errp_result(
        self,
        obs: dict[str, np.ndarray],
        *,
        override_feedback: float | None = None,
    ) -> PseudoErrPResult:
        return pseudo_errp_from_observation(
            obs,
            aux_flags=self._pseudo_errp_aux_flags,
            enabled=self.config.pseudo_errp_enabled,
            sources=self.config.pseudo_errp_sources,
            override_feedback=override_feedback,
        )

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
        errp_result: PseudoErrPResult,
    ) -> dict[str, Any]:
        return {
            "episode_index": self.current_episode_index,
            "step": self.step_count,
            "active_cube": getattr(self.active_cube, "name", ""),
            "controller_event": int(self.phase_event),
            "controller_t": float(self.phase_t),
            "phase_hold_steps": int(self.phase_hold_steps),
            "phase_gate_blocked_reason": self._phase_gate_blocked_reason,
            "human_observation_mode": self.config.human_observation_mode,
            "policy_human_observation_masked": self.config.human_observation_mode == "reward_only",
            "gripper_closed": bool(self.gripper_closed),
            "success": self._is_success(obs),
            "cube_target_dist": float(np.linalg.norm(obs["cube_to_place_target"])),
            "ee_cube_dist": float(np.linalg.norm(obs["ee_to_cube"])),
            "has_grasped_cube": bool(float(obs["has_grasped_cube"][0]) > 0.5),
            "errp_feedback": float(errp_result.feedback),
            "errp_uncertainty": float(errp_result.uncertainty),
            "errp_label": int(errp_result.label),
            "errp_source_code": int(errp_result.source_code),
            "errp_source_names": tuple(errp_result.source_names),
            "pseudo_errp_flags": dict(errp_result.flags),
            "pseudo_errp_source_scores": dict(errp_result.source_scores),
            "human_replay_aux_state": dict(self._human_replay_aux_state),
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


def _finite_action(action: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    return clip_action(arr)


_HUMAN_POLICY_OBS_ZERO_KEYS = (
    "human_head_pos",
    "human_left_hand_pos",
    "human_right_hand_pos",
    "ee_to_left_hand",
    "ee_to_right_hand",
    "human_robot_collision",
    "near_human",
)


def _mask_human_observation(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Hide replayed human state from policy input while preserving reward observations."""

    masked = {key: np.array(value, copy=True) for key, value in obs.items()}
    for key in _HUMAN_POLICY_OBS_ZERO_KEYS:
        if key in masked:
            masked[key][...] = 0.0
    if "min_hand_gripper_dist" in masked:
        masked["min_hand_gripper_dist"][...] = MISSING_DISTANCE_M
    return masked


_OBSERVATION_HUMAN_STATE_KEYS = {
    "human_head_pos",
    "human_left_hand_pos",
    "human_right_hand_pos",
    "human_robot_collision",
    "near_human",
    "collision_green",
    "pick_miss_recent",
    "drop_throw_recent",
    "min_hand_gripper_dist_override",
}


def _split_observation_human_state(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep replay metadata out of build_observation's fixed keyword surface."""

    obs_payload: dict[str, Any] = {}
    aux_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _OBSERVATION_HUMAN_STATE_KEYS:
            obs_payload[key] = value
        else:
            aux_payload[key] = value
    return obs_payload, aux_payload


def _safe_quat(quat: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(quat, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size < 4:
        result = np.zeros(4, dtype=float)
        result[: arr.size] = arr
        arr = result
    else:
        arr = arr[:4]
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return arr / norm


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
