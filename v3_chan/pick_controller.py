# pick_controller.py - Isaac Sim 4.5 only

from dataclasses import dataclass
from typing import Any

import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController

try:
    from v3_chan.pick_place_execution import GraspGateDecision, GraspReadinessGate
    from v3_chan.pick_place_speed import (
        CartesianSpeedConfig,
        MOTION_EVENT_INDICES,
        PoseSpeedEstimator,
        SpeedLimitRamp,
        blended_speed_limit_mps,
        events_dt_for_speed_profile,
        limit_joint_target_by_cartesian_displacement,
        normalize_speed_profile,
    )
except ImportError:
    from pick_place_execution import GraspGateDecision, GraspReadinessGate
    from pick_place_speed import (
        CartesianSpeedConfig,
        MOTION_EVENT_INDICES,
        PoseSpeedEstimator,
        SpeedLimitRamp,
        blended_speed_limit_mps,
        events_dt_for_speed_profile,
        limit_joint_target_by_cartesian_displacement,
        normalize_speed_profile,
    )


@dataclass(frozen=True)
class CartesianSpeedStep:
    """Command limit and measured EE speeds for one simulation state."""

    controller_event: int
    speed_profile: str
    nominal_phase_mode: str
    speed_limit_mode: str
    distance_gate: float
    surface_gap_m: float
    geometry_valid: bool
    phase_speed_limit_mps: float
    near_human_speed_limit_mps: float
    blended_speed_limit_mps: float
    active_speed_limit_mps: float
    ramp_acceleration_mps2: float
    sim_dt_s: float
    control_dt_valid: bool
    ee_velocity_sim_world_mps: np.ndarray
    ee_speed_sim_mps: float
    ee_speed_sim_valid: bool
    ee_velocity_wall_world_mps: np.ndarray
    ee_speed_wall_mps: float
    ee_speed_wall_valid: bool
    wall_dt_s: float
    command_scale: float
    command_predicted_ee_displacement_m: float
    command_full_target_ee_displacement_m: float
    command_cartesian_speed_mps: float
    command_limit_valid: bool
    command_was_limited: bool
    waypoint_distance_m: float
    waypoint_reached: bool

    def payload(self) -> dict[str, object]:
        return {
            "controller_event": int(self.controller_event),
            "controller_speed_profile": self.speed_profile,
            "nominal_phase_mode": self.nominal_phase_mode,
            "speed_limit_mode": self.speed_limit_mode,
            "distance_gate": float(self.distance_gate),
            "surface_gap_m": float(self.surface_gap_m),
            "geometry_valid": float(self.geometry_valid),
            "phase_speed_limit_mps": float(self.phase_speed_limit_mps),
            "near_human_speed_limit_mps": float(
                self.near_human_speed_limit_mps
            ),
            "blended_speed_limit_mps": float(self.blended_speed_limit_mps),
            "active_speed_limit_mps": float(self.active_speed_limit_mps),
            "ramp_acceleration_mps2": float(self.ramp_acceleration_mps2),
            "sim_dt_s": float(self.sim_dt_s),
            "control_dt_valid": float(self.control_dt_valid),
            "ee_velocity_sim_world_mps": self.ee_velocity_sim_world_mps,
            "ee_speed_sim_mps": float(self.ee_speed_sim_mps),
            "ee_speed_sim_valid": float(self.ee_speed_sim_valid),
            "ee_velocity_wall_world_mps": self.ee_velocity_wall_world_mps,
            "ee_speed_wall_mps": float(self.ee_speed_wall_mps),
            "ee_speed_wall_valid": float(self.ee_speed_wall_valid),
            "wall_dt_s": float(self.wall_dt_s),
            "command_scale": float(self.command_scale),
            "command_predicted_ee_displacement_m": float(
                self.command_predicted_ee_displacement_m
            ),
            "command_full_target_ee_displacement_m": float(
                self.command_full_target_ee_displacement_m
            ),
            "command_cartesian_speed_mps": float(
                self.command_cartesian_speed_mps
            ),
            "command_limit_valid": float(self.command_limit_valid),
            "command_was_limited": float(self.command_was_limited),
            "waypoint_distance_m": float(self.waypoint_distance_m),
            "waypoint_reached": float(self.waypoint_reached),
        }


@dataclass(frozen=True)
class PickPlaceStepResult:
    """One controller step, including grasp and Cartesian-speed status."""

    done: bool
    attempt_failed: bool
    failure_reason: str | None
    action: Any | None
    grasp_gate: GraspGateDecision | None
    speed_control: CartesianSpeedStep | None = None


class CartesianSpeedLimitedPickPlaceController:
    """Isaac PickPlaceController adapter with an EE displacement hard cap.

    The adapter uses the exact RMPFlow Lula end-effector frame for both ends of
    the candidate displacement. It does not approximate Cartesian speed from a
    fixed events_dt multiplier.
    """

    def __init__(
        self,
        controller: PickPlaceController,
        *,
        speed_profile: str,
        physics_dt_s: float,
        config: CartesianSpeedConfig | None = None,
    ) -> None:
        self._controller = controller
        self._speed_profile = normalize_speed_profile(speed_profile)
        self._physics_dt_s = float(physics_dt_s)
        self._config = (config or CartesianSpeedConfig.from_env()).validated()
        self._speed_ramp = SpeedLimitRamp(self._config)
        self._sim_speed = PoseSpeedEstimator(
            self._config.min_control_dt_s,
            self._config.max_control_dt_s,
        )
        # Wall-clock observations are diagnostic only. A slow render frame is
        # still a valid wall observation, so its upper dt bound is wider.
        self._wall_speed = PoseSpeedEstimator(
            self._config.min_control_dt_s,
            max(1.0, self._config.max_control_dt_s),
        )
        self.last_speed_step: CartesianSpeedStep | None = None

        cspace_controller = self._controller._cspace_controller
        articulation_policy = cspace_controller.get_articulation_motion_policy()
        articulation_policy.set_default_physics_dt(self._physics_dt_s)
        self._active_joints = articulation_policy.get_active_joints_subset()
        self._motion_policy = cspace_controller.get_motion_policy()
        if not callable(getattr(self._motion_policy, "get_end_effector_pose", None)):
            raise RuntimeError(
                "RMPFlow motion policy does not expose get_end_effector_pose; "
                "Cartesian speed limiting cannot be verified"
            )

    @property
    def speed_profile(self) -> str:
        return self._speed_profile

    @property
    def speed_config(self) -> CartesianSpeedConfig:
        return self._config

    @property
    def end_effector_frame_name(self) -> str:
        return str(
            getattr(self._motion_policy, "end_effector_frame_name", "unknown")
        )

    def set_speed_profile(self, speed_profile: str) -> None:
        self._speed_profile = normalize_speed_profile(speed_profile)

    def get_current_event(self) -> int:
        return int(self._controller.get_current_event())

    def is_done(self) -> bool:
        return bool(self._controller.is_done())

    def pause(self) -> None:
        self._controller.pause()

    def resume(self) -> None:
        self._controller.resume()

    def is_paused(self) -> bool:
        return bool(self._controller.is_paused())

    def reset(
        self,
        end_effector_initial_height: float | None = None,
        events_dt=None,
        speed_profile: str | None = None,
    ) -> None:
        if speed_profile is not None:
            self.set_speed_profile(speed_profile)
        self._controller.reset(
            end_effector_initial_height=end_effector_initial_height,
            events_dt=list(events_dt_for_speed_profile(self._speed_profile)),
        )
        self._speed_ramp.reset()
        self._sim_speed.reset()
        self._wall_speed.reset()
        self.last_speed_step = None

    def _forward_kinematics(self, active_joint_positions: np.ndarray) -> np.ndarray:
        position, _ = self._motion_policy.get_end_effector_pose(
            np.asarray(active_joint_positions, dtype=float)
        )
        return np.asarray(position, dtype=float).reshape(3)

    def _final_waypoint(
        self,
        event: int,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        end_effector_offset: np.ndarray,
    ) -> np.ndarray | None:
        h1 = float(self._controller._h1)
        pick = np.asarray(picking_position, dtype=float).reshape(3)
        place = np.asarray(placing_position, dtype=float).reshape(3)
        offset = np.asarray(end_effector_offset, dtype=float).reshape(3)
        stored_pick_x = float(getattr(self._controller, "_current_target_x", pick[0]))
        stored_pick_y = float(getattr(self._controller, "_current_target_y", pick[1]))
        if event == 0:
            target = np.array([pick[0], pick[1], h1])
        elif event == 1:
            target = pick
        elif event == 4:
            target = np.array([stored_pick_x, stored_pick_y, h1])
        elif event in (5, 8, 9):
            target = np.array([place[0], place[1], h1])
        elif event == 6:
            target = place
        else:
            return None
        return target + offset

    def _hold_motion_event_until_waypoint(
        self,
        event_before: int,
        waypoint_reached: bool,
    ) -> None:
        if (
            event_before in MOTION_EVENT_INDICES
            and self._controller.get_current_event() != event_before
            and not waypoint_reached
        ):
            # Isaac 4.5 exposes event but not progress mutators. The inspected
            # controller stores both as these private fields; pinning t=1 keeps
            # the final target active without changing gripper event timing.
            self._controller._event = int(event_before)
            self._controller._t = 1.0

    def forward_speed_limited(
        self,
        *,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        sim_time_s: float,
        wall_time_s: float,
        end_effector_position_world: np.ndarray,
        distance_gate: float,
        surface_gap_m: float,
        geometry_valid: bool,
        end_effector_offset: np.ndarray | None = None,
        end_effector_orientation: np.ndarray | None = None,
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.zeros(3)
        event_before = self.get_current_event()
        current_active = np.asarray(
            self._active_joints.get_joint_positions(), dtype=float
        ).reshape(-1)
        ee_position = self._forward_kinematics(current_active)
        sim_speed = self._sim_speed.update(ee_position, sim_time_s)
        wall_speed = self._wall_speed.update(ee_position, wall_time_s)

        (
            phase_limit,
            near_limit,
            blended_limit,
            nominal_mode,
            speed_limit_mode,
        ) = blended_speed_limit_mps(
            self._speed_profile,
            event_before,
            distance_gate,
        )
        active_limit, sim_dt, control_dt_valid = self._speed_ramp.update(
            blended_limit,
            sim_time_s,
        )

        waypoint = self._final_waypoint(
            event_before,
            picking_position,
            placing_position,
            end_effector_offset,
        )
        waypoint_distance = (
            float(np.linalg.norm(waypoint - ee_position))
            if waypoint is not None
            else 0.0
        )
        waypoint_reached = bool(
            waypoint is None or waypoint_distance <= self._config.reach_tolerance_m
        )

        candidate_action = self._controller.forward(
            picking_position=picking_position,
            placing_position=placing_position,
            current_joint_positions=current_joint_positions,
            end_effector_offset=end_effector_offset,
            end_effector_orientation=end_effector_orientation,
        )
        self._hold_motion_event_until_waypoint(event_before, waypoint_reached)

        command_scale = 1.0
        predicted_displacement = 0.0
        full_target_displacement = 0.0
        command_valid = event_before not in MOTION_EVENT_INDICES
        command_limited = False
        action = candidate_action
        if event_before in MOTION_EVENT_INDICES:
            candidate_positions = getattr(candidate_action, "joint_positions", None)
            candidate_positions = (
                np.asarray(candidate_positions, dtype=float).reshape(-1)
                if candidate_positions is not None
                else np.asarray([], dtype=float)
            )
            max_displacement = (
                active_limit * sim_dt if control_dt_valid else 0.0
            )
            limit_result = limit_joint_target_by_cartesian_displacement(
                current_active,
                candidate_positions,
                max_displacement,
                self._forward_kinematics,
                iterations=self._config.fk_binary_search_iterations,
                tolerance_m=self._config.fk_displacement_tolerance_m,
            )
            command_scale = limit_result.scale
            predicted_displacement = limit_result.predicted_displacement_m
            full_target_displacement = limit_result.full_target_displacement_m
            command_valid = limit_result.valid
            command_limited = limit_result.limited
            candidate_velocities = getattr(candidate_action, "joint_velocities", None)
            limited_velocities = None
            if candidate_velocities is not None:
                limited_velocities = (
                    np.asarray(candidate_velocities, dtype=float).reshape(-1)
                    * command_scale
                )
            action = ArticulationAction(
                joint_positions=limit_result.joint_positions,
                joint_velocities=limited_velocities,
                joint_efforts=getattr(candidate_action, "joint_efforts", None),
                joint_indices=getattr(candidate_action, "joint_indices", None),
            )

        command_speed = (
            predicted_displacement / sim_dt
            if command_valid and control_dt_valid and sim_dt > 0.0
            else 0.0
        )
        self.last_speed_step = CartesianSpeedStep(
            controller_event=event_before,
            speed_profile=self._speed_profile,
            nominal_phase_mode=nominal_mode,
            speed_limit_mode=speed_limit_mode,
            distance_gate=float(np.clip(distance_gate, 0.0, 1.0)),
            surface_gap_m=float(surface_gap_m),
            geometry_valid=bool(geometry_valid),
            phase_speed_limit_mps=phase_limit,
            near_human_speed_limit_mps=near_limit,
            blended_speed_limit_mps=blended_limit,
            active_speed_limit_mps=active_limit,
            ramp_acceleration_mps2=self._speed_ramp.acceleration_mps2,
            sim_dt_s=sim_dt,
            control_dt_valid=control_dt_valid,
            ee_velocity_sim_world_mps=sim_speed.velocity_world_mps,
            ee_speed_sim_mps=sim_speed.speed_mps,
            ee_speed_sim_valid=sim_speed.valid,
            ee_velocity_wall_world_mps=wall_speed.velocity_world_mps,
            ee_speed_wall_mps=wall_speed.speed_mps,
            ee_speed_wall_valid=wall_speed.valid,
            wall_dt_s=wall_speed.dt_s,
            command_scale=command_scale,
            command_predicted_ee_displacement_m=predicted_displacement,
            command_full_target_ee_displacement_m=full_target_displacement,
            command_cartesian_speed_mps=command_speed,
            command_limit_valid=command_valid,
            command_was_limited=command_limited,
            waypoint_distance_m=waypoint_distance,
            waypoint_reached=waypoint_reached,
        )
        return action


def create_pick_controller(
    robot: Franka,
    end_effector_initial_height: float = None,
    speed_profile: str = "slow",
    physics_dt_s: float = 1.0 / 60.0,
    speed_config: CartesianSpeedConfig | None = None,
) -> CartesianSpeedLimitedPickPlaceController:
    """Create the existing Isaac 10-event controller with an m/s limiter."""

    normalized_profile = normalize_speed_profile(speed_profile)
    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        end_effector_initial_height=end_effector_initial_height,
        events_dt=list(events_dt_for_speed_profile(normalized_profile)),
    )
    return CartesianSpeedLimitedPickPlaceController(
        controller,
        speed_profile=normalized_profile,
        physics_dt_s=physics_dt_s,
        config=speed_config,
    )


def run_pick_place(
    controller: CartesianSpeedLimitedPickPlaceController,
    robot: Franka,
    pick_position: np.ndarray,
    place_position: np.ndarray,
    grasp_gate: GraspReadinessGate | None = None,
    grasp_reference_position: np.ndarray | None = None,
    sim_time_s: float | None = None,
    wall_time_s: float | None = None,
    end_effector_position_world: np.ndarray | None = None,
    distance_gate: float = 0.0,
    surface_gap_m: float = 10.0,
    geometry_valid: bool = False,
    return_action: bool = False,
    return_status: bool = False,
):
    """Advance one physics step and apply a Cartesian-speed-limited action."""

    event_before_step = int(controller.get_current_event())
    gate_decision = None
    if grasp_gate is not None:
        gate_decision = grasp_gate.update(
            controller_event=event_before_step,
            grasp_reference_position=grasp_reference_position,
            cube_position=pick_position,
        )
        if event_before_step == GraspReadinessGate.SETTLE_EVENT:
            if gate_decision.timed_out:
                controller.pause()
                result = PickPlaceStepResult(
                    done=False,
                    attempt_failed=True,
                    failure_reason="grasp_gate_timeout",
                    action=None,
                    grasp_gate=gate_decision,
                    speed_control=controller.last_speed_step,
                )
                if return_status:
                    return result
                if return_action:
                    return result.done, result.action
                return result.done
            if gate_decision.release:
                controller.resume()
            else:
                controller.pause()
        else:
            controller.resume()

    current_joint_positions = robot.get_joint_positions()
    if sim_time_s is None or wall_time_s is None:
        raise ValueError(
            "Cartesian speed control requires explicit simulation and monotonic wall timestamps"
        )
    if end_effector_position_world is None:
        end_effector_position_world, _ = robot.end_effector.get_world_pose()

    actions = controller.forward_speed_limited(
        picking_position=pick_position,
        placing_position=place_position,
        current_joint_positions=current_joint_positions,
        sim_time_s=sim_time_s,
        wall_time_s=wall_time_s,
        end_effector_position_world=end_effector_position_world,
        distance_gate=distance_gate,
        surface_gap_m=surface_gap_m,
        geometry_valid=geometry_valid,
        end_effector_offset=np.array([0.0, 0.005, 0.0]),
    )

    robot.apply_action(actions)
    result = PickPlaceStepResult(
        done=controller.is_done(),
        attempt_failed=False,
        failure_reason=None,
        action=actions,
        grasp_gate=gate_decision,
        speed_control=controller.last_speed_step,
    )
    if return_status:
        return result
    if return_action:
        return result.done, result.action
    return result.done
