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
from .observations import (
    apply_dynamic_hri_observation,
    build_observation,
    flatten_observation,
    task_phase_onehot,
)
from .pick_place_phase import advance_pick_place_event, event_gripper_command, task_phase_from_event
from .pseudo_errp import (
    DEFAULT_PSEUDO_ERRP_SOURCES,
    PseudoErrPResult,
    extract_pseudo_errp_aux_flags,
    pseudo_errp_from_observation,
)
from .rewards import DEFAULT_REWARD_WEIGHTS, RewardWeights, compute_reward, is_success

try:
    from v3_chan.dynamic_safety import DynamicSafetyEstimator
except ImportError:
    from dynamic_safety import DynamicSafetyEstimator

try:
    from v3_chan.scene_randomization import (
        resolve_active_cube_index,
        restore_cube_poses,
        restored_pose_errors,
    )
except ImportError:
    from scene_randomization import (
        resolve_active_cube_index,
        restore_cube_poses,
        restored_pose_errors,
    )

try:
    from v3_chan.physical_safety_controllers import (
        CBFConfig,
        PHYSICAL_SAFETY_MODES,
        DistalLinkVelocityCBF,
        PhysicalSafetyDiagnostics,
        mode_uses_cbf,
        mode_uses_curobo,
        mode_uses_rmpflow_obstacles,
    )
except ImportError:
    from physical_safety_controllers import (
        CBFConfig,
        PHYSICAL_SAFETY_MODES,
        DistalLinkVelocityCBF,
        PhysicalSafetyDiagnostics,
        mode_uses_cbf,
        mode_uses_curobo,
        mode_uses_rmpflow_obstacles,
    )


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
    pseudo_errp_enabled: bool = True
    pseudo_errp_sources: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_PSEUDO_ERRP_SOURCES
    )
    visualize_human_replay: bool = False
    human_replay_visual_z_offset: float = 0.0
    synthetic_human_enabled: bool = False
    synthetic_human_episode_prob: float = 0.35
    synthetic_human_start_min_step: int = 120
    synthetic_human_start_max_step: int = 520
    synthetic_human_duration_steps: int = 90
    synthetic_human_near_dist: float = 0.12
    synthetic_human_collision_dist: float = 0.035
    physical_safety_controller: str = "none"
    rmpflow_human_safety_margin_m: float = 0.05
    visualize_physical_safety: bool = False
    cbf_safe_gap_m: float = 0.05
    cbf_activation_gap_m: float = 0.13
    cbf_gamma_per_s: float = 8.0
    cbf_prediction_horizon_s: float = 0.15
    cbf_max_prediction_buffer_m: float = 0.08
    cbf_max_joint_speed_rad_s: float = 2.0


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
        if self.config.physical_safety_controller not in PHYSICAL_SAFETY_MODES:
            raise ValueError(
                "Unknown physical_safety_controller "
                f"{self.config.physical_safety_controller!r}; "
                f"expected one of {PHYSICAL_SAFETY_MODES}"
            )
        if self.config.rmpflow_human_safety_margin_m < 0.0:
            raise ValueError("rmpflow_human_safety_margin_m must be non-negative")
        self.human_state_fn = human_state_fn
        self.rng = np.random.default_rng(self.config.seed)

        from isaacsim.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.franka.controllers import RMPFlowController

        from panda_robot import add_panda
        from end_effector_safety_runtime import PandaEndEffectorSafetyRuntime
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
        ) = setup_scene(
            self.world,
            cube_count=self.config.cube_count,
            rng=self.rng,
        )
        self.pick_targets = self.cubes[: min(3, len(self.cubes))]
        self.cube_half = self.cube_size / 2.0
        self.cube_center_z = self.table_top_z + self.cube_half
        self.place_pos = np.array([self.stack_base_xy[0], self.stack_base_xy[1], self.cube_center_z])
        self.place_target.set_world_pose(position=self.place_pos)

        self.robot = add_panda(self.world, base_z=self.table_top_z)
        self.world.reset()
        self.world.play()
        self.controller = RMPFlowController(name="rl_env_rmpflow_controller", robot_articulation=self.robot)
        self.safety_geometry = PandaEndEffectorSafetyRuntime(
            robot_prim_path="/World/Franka"
        )
        self.dynamic_safety = DynamicSafetyEstimator()
        self.physics_dt_s = float(self.world.get_physics_dt())
        self._physical_safety_mode = str(self.config.physical_safety_controller)
        self._cbf_filter = (
            DistalLinkVelocityCBF(
                CBFConfig(
                    safe_gap_m=self.config.cbf_safe_gap_m,
                    activation_gap_m=self.config.cbf_activation_gap_m,
                    gamma_per_s=self.config.cbf_gamma_per_s,
                    prediction_horizon_s=self.config.cbf_prediction_horizon_s,
                    max_prediction_buffer_m=self.config.cbf_max_prediction_buffer_m,
                    max_joint_speed_rad_s=self.config.cbf_max_joint_speed_rad_s,
                )
            )
            if mode_uses_cbf(self._physical_safety_mode)
            else None
        )
        self._curobo_controller = None
        self._last_physical_safety_diagnostics = PhysicalSafetyDiagnostics(
            controller=self._physical_safety_mode
        )
        self._rmpflow_human_obstacles: dict[str, Any] = {}
        self._rmpflow_obstacles_registered = False
        self._rmpflow_valid_hand_count = 0

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
        self._last_safety_result = None
        self._last_dynamic_safety_sample = None
        self._source_restoration_diagnostics = _empty_source_restoration_diagnostics()
        self._synthetic_human_active = False
        self._synthetic_human_start_step = 0
        self._synthetic_human_duration_steps = 0
        self._synthetic_human_side = 1.0
        self._synthetic_human_height_offset = 0.0
        self._human_visual_prims: dict[str, Any] = {}
        if self.config.visualize_human_replay:
            self._setup_human_visuals()
        if mode_uses_rmpflow_obstacles(self._physical_safety_mode):
            self._setup_rmpflow_human_obstacles()
        if mode_uses_curobo(self._physical_safety_mode):
            self._setup_curobo_controller()

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
        source_restoration: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            np.random.seed(seed)
        else:
            np.random.seed(int(self.rng.integers(0, 2**31 - 1)))

        from scene_setup import randomize_cubes

        restoration = _normalize_source_restoration(source_restoration, seed)
        restoration_mode = str(restoration["restoration_mode"])
        source = dict(restoration.get("source_configuration", {}))
        if source_restoration is not None and not bool(
            restoration.get("source_configuration_available", False)
        ):
            raise ValueError(
                "source_configuration_unavailable: "
                f"{restoration.get('restoration_reason', 'unknown')}"
            )

        if restoration_mode == "exact_pose":
            restore_cube_poses(
                self.cubes,
                source["cube_names"],
                source["cube_positions_world"],
                source["cube_orientations_wxyz"],
                set_default_state=True,
            )
            _set_robot_default_state(
                self.robot,
                source.get("robot_initial_joint_positions"),
                source.get("robot_initial_joint_velocities"),
            )
        else:
            scene_rng = self.rng
            if restoration_mode == "collection_seed":
                scene_rng = np.random.default_rng(int(restoration["layout_seed"]))
            randomize_cubes(
                self.cubes,
                self.table_xy,
                self.table_size,
                self.cube_center_z,
                self.cube_size,
                forbidden_xy=self.stack_base_xy,
                rng=scene_rng,
            )
        self.world.reset()
        self.world.play()
        self.controller.reset()
        self._rmpflow_obstacles_registered = False
        self._register_rmpflow_human_obstacles()
        if self._cbf_filter is not None:
            self._cbf_filter.reset()
        if self._curobo_controller is not None:
            self._curobo_controller.reset()
        self._last_physical_safety_diagnostics = PhysicalSafetyDiagnostics(
            controller=self._physical_safety_mode
        )
        self.safety_geometry.reset_link_origin_pose_cache()
        if restoration_mode == "exact_pose":
            restore_cube_poses(
                self.cubes,
                source["cube_names"],
                source["cube_positions_world"],
                source["cube_orientations_wxyz"],
                set_default_state=False,
            )
            self.place_target.set_world_pose(
                position=np.asarray(source["place_target_position_world"], dtype=float),
                orientation=np.asarray(
                    source["place_target_orientation_wxyz"], dtype=float
                ),
            )
            robot_restored = _set_robot_joint_state(
                self.robot,
                source.get("robot_initial_joint_positions"),
                source.get("robot_initial_joint_velocities"),
            )
        else:
            self.place_target.set_world_pose(position=self.place_pos)
            robot_restored = False

        self.current_episode_index = self.episode_index
        if source_restoration is not None:
            active_cube_index = restoration.get("source_cube_index")
        elif active_cube_index is None:
            active_cube_index = self.current_episode_index % len(self.pick_targets)
        screening_cube_index = resolve_active_cube_index(
            active_cube_index,
            episode_index=self.current_episode_index,
            cube_count=len(self.pick_targets),
        )
        self.active_cube = self.pick_targets[screening_cube_index]
        restoration["screening_cube_index"] = int(screening_cube_index)
        restoration["screening_cube_name"] = str(
            getattr(self.active_cube, "name", "")
        )
        if restoration_mode == "exact_pose":
            verification = _verify_exact_restoration(
                self.cubes,
                self.place_target,
                source,
                robot_restored=robot_restored,
            )
            source_cube_name = restoration.get("source_cube_name")
            if source_cube_name and str(source_cube_name) != restoration[
                "screening_cube_name"
            ]:
                verification["pose_mismatch"] = True
                reasons = [
                    item
                    for item in str(verification["pose_mismatch_reason"]).split(",")
                    if item
                ]
                reasons.append("active_cube_identity_mismatch")
                verification["pose_mismatch_reason"] = ",".join(reasons)
            restoration.update(verification)
            if bool(restoration["pose_mismatch"]):
                raise ValueError(
                    "source_configuration_pose_mismatch: "
                    f"{restoration['pose_mismatch_reason']}"
                )
        self._source_restoration_diagnostics = restoration
        self.step_count = 0
        self.phase_event = 0
        self.phase_t = 0.0
        self.phase_hold_steps = 0
        self.gripper_closed = False
        self.yaw = 0.0
        self._reset_synthetic_human()
        self.dynamic_safety.reset()
        self._last_dynamic_safety_sample = None

        obs = self._build_obs()
        self._last_obs = obs
        errp_result = self._pseudo_errp_result(obs, override_feedback=0.0)
        info = self._info(obs, reward_components={}, errp_result=errp_result)
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

        action = _finite_action(action)
        target_pos, target_quat, self.yaw = self._target_from_action(action)
        gripper_command = self._gripper_command(action, self._last_obs)

        if self._curobo_controller is not None:
            arm_action, controller_diagnostics = self._curobo_controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=target_quat,
                observation=self._last_obs,
            )
            self._last_physical_safety_diagnostics = controller_diagnostics
        else:
            arm_action = self.controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=target_quat,
            )
            self._last_physical_safety_diagnostics = PhysicalSafetyDiagnostics(
                controller=self._physical_safety_mode,
                active=bool(
                    mode_uses_rmpflow_obstacles(self._physical_safety_mode)
                    and self._rmpflow_valid_hand_count > 0
                    and self._last_safety_result is not None
                    and self._last_safety_result.distance_gate > 0.0
                ),
                valid_hand_count=int(self._rmpflow_valid_hand_count),
                status=(
                    "dynamic_obstacles_registered"
                    if mode_uses_rmpflow_obstacles(self._physical_safety_mode)
                    else "inactive"
                ),
            )
        if (
            self._cbf_filter is not None
            and self._last_safety_result is not None
            and self._last_dynamic_safety_sample is not None
        ):
            arm_action, self._last_physical_safety_diagnostics = (
                self._cbf_filter.filter_action(
                    robot=self.robot,
                    arm_action=arm_action,
                    safety_result=self._last_safety_result,
                    dynamic_sample=self._last_dynamic_safety_sample,
                    safety_geometry=self.safety_geometry,
                    observation=self._last_obs,
                    physics_dt_s=self.physics_dt_s,
                )
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
        ee_pos = None
        try:
            ee_pos, _ = self.robot.end_effector.get_world_pose()
            ee_pos = np.asarray(ee_pos, dtype=float).reshape(-1)[:3]
        except Exception:
            ee_pos = None
        if gripper_center is None:
            gripper_center = ee_pos
        has_grasped = _has_grasped_cube(self.robot, self.active_cube, gripper_center)
        task_phase = task_phase_from_event(self.phase_event)
        replay_context_setter = getattr(
            self.human_state_fn,
            "set_runtime_context",
            None,
        )
        if callable(replay_context_setter):
            replay_context_setter(
                step=self.step_count,
                task_phase=task_phase,
                controller_event=(
                    -1 if self.phase_event is None else int(self.phase_event)
                ),
                controller_t=int(self.phase_t),
                ee_pos=ee_pos,
                playback_time_s=float(self.step_count) * self.physics_dt_s,
            )
        human_state = dict(self.human_state_fn() if self.human_state_fn is not None else {})
        synthetic_state = self._synthetic_human_state(gripper_center)
        human_state = {**synthetic_state, **human_state}
        self._update_rmpflow_human_obstacles(human_state)
        if self._curobo_controller is not None:
            self._curobo_controller.update_human_obstacles(human_state)
        human_state, self._pseudo_errp_aux_flags = extract_pseudo_errp_aux_flags(human_state)
        human_state, self._human_replay_aux_state = _split_observation_human_state(human_state)
        safety_result = self.safety_geometry.evaluate(
            human_state.get("human_left_hand_pos"),
            human_state.get("human_right_hand_pos"),
        )
        self._last_safety_result = safety_result
        dynamic_sample = self._update_dynamic_safety(
            safety_result,
            human_state.get("human_left_hand_pos"),
            human_state.get("human_right_hand_pos"),
        )
        self._last_dynamic_safety_sample = dynamic_sample
        # Recorded labels never drive a rollout. Recompute them against the
        # current robot pose and its composed PhysX collision shapes.
        human_state.update(
            {
                "human_robot_collision": safety_result.collision,
                "near_human": safety_result.near,
                "near_miss": safety_result.near_miss,
                "min_hand_gripper_surface_gap_override": safety_result.min_surface_gap_m,
                "left_hand_surface_gap_override": safety_result.left.surface_gap_m,
                "right_hand_surface_gap_override": safety_result.right.surface_gap_m,
                "left_hand_contact": safety_result.left.contact,
                "right_hand_contact": safety_result.right.contact,
                "distance_gate_override": safety_result.distance_gate,
                "geometry_valid_override": safety_result.geometry_valid,
            }
        )
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
        apply_dynamic_hri_observation(
            obs,
            {
                **dynamic_sample.human_payload(),
                **dynamic_sample.safety_payload(),
            },
        )
        if self.phase_event is None:
            obs["task_phase"] = task_phase_onehot("approach_cube")
        self._update_human_visuals(obs)
        return obs

    def _update_dynamic_safety(
        self,
        safety_result,
        left_hand_pos,
        right_hand_pos,
    ):
        left_origin, left_orientation, _ = self.safety_geometry.closest_link_world_pose(
            safety_result.left
        )
        right_origin, right_orientation, _ = self.safety_geometry.closest_link_world_pose(
            safety_result.right
        )
        _, left_angular_velocity, _ = self.safety_geometry.closest_link_world_velocity(
            safety_result.left
        )
        _, right_angular_velocity, _ = self.safety_geometry.closest_link_world_velocity(
            safety_result.right
        )
        left_surface_point, _ = self.safety_geometry.closest_surface_point_world_position(
            safety_result.left,
            left_hand_pos,
        )
        right_surface_point, _ = self.safety_geometry.closest_surface_point_world_position(
            safety_result.right,
            right_hand_pos,
        )
        return self.dynamic_safety.update(
            sim_time_s=float(self.step_count) * self.physics_dt_s,
            left_hand_pos=left_hand_pos,
            right_hand_pos=right_hand_pos,
            left_tracking_valid=_valid_runtime_position(left_hand_pos),
            right_tracking_valid=_valid_runtime_position(right_hand_pos),
            left_surface_gap_m=safety_result.left.surface_gap_m,
            right_surface_gap_m=safety_result.right.surface_gap_m,
            left_geometry_valid=safety_result.left.geometry_valid,
            right_geometry_valid=safety_result.right.geometry_valid,
            left_closest_collider_id=safety_result.left.closest_collider_id,
            right_closest_collider_id=safety_result.right.closest_collider_id,
            left_closest_robot_origin_pos=left_origin,
            right_closest_robot_origin_pos=right_origin,
            left_closest_robot_orientation_wxyz=left_orientation,
            right_closest_robot_orientation_wxyz=right_orientation,
            left_closest_surface_point_world_pos=left_surface_point,
            right_closest_surface_point_world_pos=right_surface_point,
            left_closest_robot_angular_velocity_world_radps=left_angular_velocity,
            right_closest_robot_angular_velocity_world_radps=right_angular_velocity,
        )

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

    def _setup_rmpflow_human_obstacles(self) -> None:
        from omni.isaac.core.objects import VisualSphere

        parked = np.array([0.0, 0.0, -10.0], dtype=float)
        radius = float(
            self.safety_geometry.thresholds.hand_radius_m
            + self.config.rmpflow_human_safety_margin_m
        )
        visible = bool(self.config.visualize_physical_safety)
        for hand, color in (
            ("left", np.array([0.1, 0.9, 0.9])),
            ("right", np.array([0.95, 0.25, 0.25])),
        ):
            self._rmpflow_human_obstacles[hand] = self.world.scene.add(
                VisualSphere(
                    prim_path=f"/World/PhysicalSafety/rmpflow_{hand}_hand",
                    name=f"rmpflow_{hand}_hand_obstacle",
                    position=parked,
                    radius=radius,
                    color=color,
                    visible=visible,
                )
            )
        self._register_rmpflow_human_obstacles()

    def _register_rmpflow_human_obstacles(self) -> None:
        if self._rmpflow_obstacles_registered:
            return
        if not self._rmpflow_human_obstacles:
            return
        for obstacle in self._rmpflow_human_obstacles.values():
            self.controller.add_obstacle(obstacle, static=False)
        self._rmpflow_obstacles_registered = True

    def _update_rmpflow_human_obstacles(self, human_state: dict[str, Any]) -> None:
        if not self._rmpflow_human_obstacles:
            self._rmpflow_valid_hand_count = 0
            return
        parked = np.array([0.0, 0.0, -10.0], dtype=float)
        valid_count = 0
        for hand, obstacle in self._rmpflow_human_obstacles.items():
            value = human_state.get(f"human_{hand}_hand_pos")
            if _valid_runtime_position(value):
                position = np.asarray(value, dtype=float).reshape(-1)[:3]
                valid_count += 1
            else:
                position = parked
            obstacle.set_world_pose(position=position)
        self._rmpflow_valid_hand_count = valid_count

    def _setup_curobo_controller(self) -> None:
        try:
            from v3_chan.curobo_mpc_controller import CuRoboMpcArmController
        except ImportError:
            from curobo_mpc_controller import CuRoboMpcArmController

        self._curobo_controller = CuRoboMpcArmController(
            robot=self.robot,
            physics_dt_s=self.physics_dt_s,
            hand_radius_m=self.safety_geometry.thresholds.hand_radius_m,
            safety_margin_m=self.config.rmpflow_human_safety_margin_m,
            table_center_world_m=np.array(
                [
                    self.table_xy[0],
                    self.table_xy[1],
                    self.table_top_z - (self.table_size[2] / 2.0),
                ],
                dtype=float,
            ),
            table_size_m=np.asarray(self.table_size, dtype=float),
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
            else:
                pos = pos[:3].copy()
                pos[2] += float(self.config.human_replay_visual_z_offset)
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
        }

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
        physical_safety = self._last_physical_safety_diagnostics.as_dict()
        return {
            "episode_index": self.current_episode_index,
            "step": self.step_count,
            "sim_time": float(getattr(self.world, "current_time", 0.0)),
            "active_cube": getattr(self.active_cube, "name", ""),
            "controller_event": int(self.phase_event),
            "controller_t": float(self.phase_t),
            "phase_hold_steps": int(self.phase_hold_steps),
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
            "source_restoration": dict(self._source_restoration_diagnostics),
            "human_replay_aux_state": dict(self._human_replay_aux_state),
            "human_robot_collision": bool(float(obs["human_robot_collision"][0]) > 0.5),
            "near_human": bool(float(obs["near_human"][0]) > 0.5),
            "near_miss": bool(float(obs["near_miss"][0]) > 0.5),
            "min_hand_end_effector_surface_gap": float(
                obs["min_hand_end_effector_surface_gap"][0]
            ),
            "distance_gate": float(obs["distance_gate"][0]),
            "geometry_valid": bool(float(obs["geometry_valid"][0]) > 0.5),
            "left_end_effector_surface_gap_m": (
                self._last_safety_result.left.surface_gap_m
                if self._last_safety_result is not None
                else 10.0
            ),
            "right_end_effector_surface_gap_m": (
                self._last_safety_result.right.surface_gap_m
                if self._last_safety_result is not None
                else 10.0
            ),
            "contact_left": (
                self._last_safety_result.left.contact
                if self._last_safety_result is not None
                else False
            ),
            "contact_right": (
                self._last_safety_result.right.contact
                if self._last_safety_result is not None
                else False
            ),
            "penetration_left_m": (
                self._last_safety_result.left.penetration_m
                if self._last_safety_result is not None
                else 0.0
            ),
            "penetration_right_m": (
                self._last_safety_result.right.penetration_m
                if self._last_safety_result is not None
                else 0.0
            ),
            "distance_gate_left": (
                self._last_safety_result.left.distance_gate
                if self._last_safety_result is not None
                else 0.0
            ),
            "distance_gate_right": (
                self._last_safety_result.right.distance_gate
                if self._last_safety_result is not None
                else 0.0
            ),
            "closest_link_left": (
                self._last_safety_result.left.closest_link
                if self._last_safety_result is not None
                else ""
            ),
            "closest_link_right": (
                self._last_safety_result.right.closest_link
                if self._last_safety_result is not None
                else ""
            ),
            "closest_collider_left": (
                self._last_safety_result.left.closest_collider_path
                if self._last_safety_result is not None
                else ""
            ),
            "closest_collider_right": (
                self._last_safety_result.right.closest_collider_path
                if self._last_safety_result is not None
                else ""
            ),
            "closest_human_hand": (
                self._last_safety_result.closest_human_hand
                if self._last_safety_result is not None
                else ""
            ),
            "closest_robot_link": (
                self._last_safety_result.closest_robot_link
                if self._last_safety_result is not None
                else ""
            ),
            "closest_collider": (
                self._last_safety_result.closest_collider_path
                if self._last_safety_result is not None
                else ""
            ),
            "contact_active": (
                self._last_safety_result.contact
                if self._last_safety_result is not None
                else False
            ),
            "penetration_depth_m": (
                self._last_safety_result.penetration_depth_m
                if self._last_safety_result is not None
                else 0.0
            ),
            "physical_safety_controller": self._physical_safety_mode,
            "physical_safety": physical_safety,
            "physical_safety_active": bool(physical_safety["active"]),
            "physical_safety_intervention_available": bool(
                physical_safety["intervention_available"]
            ),
            "physical_safety_constraint_count": int(
                physical_safety["constraint_count"]
            ),
            "physical_safety_intervention_norm_radps": float(
                physical_safety["intervention_norm_radps"]
            ),
            "physical_safety_slack_radps": float(physical_safety["slack_radps"]),
            "physical_safety_feasible": bool(physical_safety["feasible"]),
            "physical_safety_status": str(physical_safety["status"]),
            "physical_safety_solve_time_ms": float(
                physical_safety["solve_time_ms"]
            ),
            "rmpflow_valid_hand_obstacles": int(self._rmpflow_valid_hand_count),
            "safety_query_time_ms": (
                self._last_safety_result.left.query_time_ms
                + self._last_safety_result.right.query_time_ms
                if self._last_safety_result is not None
                else 0.0
            ),
            "reward_components": dict(reward_components),
            "obs_dict": obs,
        }


def _empty_source_restoration_diagnostics() -> dict[str, Any]:
    return {
        "source_configuration_available": True,
        "restoration_mode": "not_requested",
        "restoration_reason": "",
        "source_cube_index": None,
        "screening_cube_index": None,
        "source_cube_name": None,
        "screening_cube_name": None,
        "collection_seed": None,
        "layout_seed": None,
        "screening_seed": None,
        "cube_pose_restored": False,
        "target_pose_restored": False,
        "robot_initial_state_restored": False,
        "pose_mismatch": False,
        "pose_mismatch_reason": "",
        "max_cube_position_error_m": None,
        "max_cube_orientation_error_rad": None,
        "target_position_error_m": None,
        "target_orientation_error_rad": None,
        "missing_fields": [],
        "source_configuration": {},
    }


def _normalize_source_restoration(
    source_restoration: dict[str, Any] | None,
    screening_seed: int | None,
) -> dict[str, Any]:
    if source_restoration is None:
        result = _empty_source_restoration_diagnostics()
    else:
        result = {
            **_empty_source_restoration_diagnostics(),
            **dict(source_restoration),
        }
        source = result.get("source_configuration")
        result["source_configuration"] = (
            dict(source) if isinstance(source, dict) else {}
        )
    if result.get("screening_seed") is None and screening_seed is not None:
        result["screening_seed"] = int(screening_seed)
    return result


def _finite_joint_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.size <= 0 or not np.all(np.isfinite(result)):
        return None
    return result


def _set_robot_default_state(robot, positions: Any, velocities: Any) -> bool:
    positions_array, velocities_array = _runtime_robot_joint_state(
        robot,
        positions,
        velocities,
    )
    if positions_array is None or not hasattr(robot, "set_joints_default_state"):
        return False
    try:
        robot.set_joints_default_state(
            positions=positions_array,
            velocities=velocities_array,
        )
    except TypeError:
        robot.set_joints_default_state(positions_array, velocities_array)
    return True


def _set_robot_joint_state(robot, positions: Any, velocities: Any) -> bool:
    positions_array, velocities_array = _runtime_robot_joint_state(
        robot,
        positions,
        velocities,
    )
    if positions_array is None or not hasattr(robot, "set_joint_positions"):
        return False
    robot.set_joint_positions(positions_array)
    if velocities_array is not None and hasattr(robot, "set_joint_velocities"):
        robot.set_joint_velocities(velocities_array)
    return True


def _runtime_robot_joint_state(
    robot,
    positions: Any,
    velocities: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    source_positions = _finite_joint_vector(positions)
    if source_positions is None:
        return None, None
    try:
        runtime_positions = np.asarray(
            robot.get_joint_positions(), dtype=float
        ).reshape(-1)
    except Exception:
        runtime_positions = np.empty(0, dtype=float)
    if runtime_positions.size > 0:
        if source_positions.size > runtime_positions.size:
            return None, None
        positions_array = runtime_positions.copy()
        positions_array[: source_positions.size] = source_positions
    else:
        positions_array = source_positions

    source_velocities = _finite_joint_vector(velocities)
    velocities_array = np.zeros_like(positions_array)
    if source_velocities is not None:
        count = min(source_velocities.size, velocities_array.size)
        velocities_array[:count] = source_velocities[:count]
    return positions_array, velocities_array


def _verify_exact_restoration(
    cubes,
    place_target,
    source: dict[str, Any],
    *,
    robot_restored: bool,
    position_tolerance_m: float = 1e-4,
    orientation_tolerance_rad: float = 1e-4,
) -> dict[str, Any]:
    cube_position_error, cube_orientation_error = restored_pose_errors(
        cubes,
        source["cube_names"],
        source["cube_positions_world"],
        source["cube_orientations_wxyz"],
    )
    target_position, target_orientation = place_target.get_world_pose()
    target_position_error = float(
        np.linalg.norm(
            np.asarray(target_position, dtype=float)
            - np.asarray(source["place_target_position_world"], dtype=float)
        )
    )
    target_orientation_error = _quaternion_angle_error_rad(
        target_orientation,
        source["place_target_orientation_wxyz"],
    )
    cube_restored = bool(
        cube_position_error <= position_tolerance_m
        and cube_orientation_error <= orientation_tolerance_rad
    )
    target_restored = bool(
        target_position_error <= position_tolerance_m
        and target_orientation_error <= orientation_tolerance_rad
    )
    mismatch_reasons = []
    if not cube_restored:
        mismatch_reasons.append("cube_pose_mismatch")
    if not target_restored:
        mismatch_reasons.append("target_pose_mismatch")
    return {
        "cube_pose_restored": cube_restored,
        "target_pose_restored": target_restored,
        "robot_initial_state_restored": bool(robot_restored),
        "pose_mismatch": bool(mismatch_reasons),
        "pose_mismatch_reason": ",".join(mismatch_reasons),
        "max_cube_position_error_m": float(cube_position_error),
        "max_cube_orientation_error_rad": float(cube_orientation_error),
        "target_position_error_m": float(target_position_error),
        "target_orientation_error_rad": float(target_orientation_error),
    }


def _quaternion_angle_error_rad(actual: Any, expected: Any) -> float:
    first = np.asarray(actual, dtype=float).reshape(-1)[:4]
    second = np.asarray(expected, dtype=float).reshape(-1)[:4]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return float("inf")
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


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


def _valid_runtime_position(value) -> bool:
    if value is None:
        return False
    position = np.asarray(value, dtype=float).reshape(-1)
    return bool(
        position.size >= 3
        and np.all(np.isfinite(position[:3]))
        and np.linalg.norm(position[:3]) > 1e-6
    )


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
    "min_hand_gripper_surface_gap_override",
    "left_hand_surface_gap_override",
    "right_hand_surface_gap_override",
    "left_hand_contact",
    "right_hand_contact",
    "near_miss",
    "distance_gate_override",
    "geometry_valid_override",
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
