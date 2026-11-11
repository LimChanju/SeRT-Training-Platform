from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


ORIENT_STAGE_PREGRASP = "orient_pregrasp"
ORIENT_STAGE_APPROACH = "orient_approach"
ORIENT_STAGE_CLOSE = "orient_close"
ORIENT_STAGE_ATTACH = "orient_attach"
ORIENT_STAGE_VERIFY = "orient_verify"
ORIENT_STAGE_REOPEN = "orient_reopen"
ORIENT_STAGE_LIFT_PLACE = "orient_lift_place"


HANDOVER_PHASES = (
    "waiting_for_hand",
    "spawned_to_hand",
    "human_holding",
    "robot_takeover",
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_vec3(name: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return np.asarray(default, dtype=float)
    try:
        parts = [float(part.strip()) for part in raw.split(",")]
        if len(parts) != 3:
            raise ValueError
        return np.asarray(parts, dtype=float)
    except ValueError:
        return np.asarray(default, dtype=float)


def _env_hands(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    hands = tuple(
        hand.strip()
        for hand in raw.split(",")
        if hand.strip() in ("left", "right")
    )
    return hands or default


def _euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in np.asarray(euler_xyz, dtype=float).reshape(-1)[:3]]
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9 or not np.all(np.isfinite(quat)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


@dataclass(frozen=True)
class HandoverTaskConfig:
    hand_priority: tuple[str, ...] = ("right", "left")
    spawn_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.02], dtype=float)
    )
    respawn_to_hand: bool = True
    adopt_any_grabbed_cube: bool = False
    follow_hand_until_grabbed: bool = True
    human_cube_physics: bool = False
    human_hold_steps: int = 24
    stable_window_steps: int = 12
    stable_motion_threshold_m: float = 0.035
    stable_max_step_m: float = 0.055
    stable_mean_step_m: float = 0.010
    stable_jitter_threshold_m: float = 0.025
    stable_outlier_tolerance_m: float = 0.055
    stable_required_windows: int = 3
    takeover_release_event: int = 3
    physical_mode: bool = False
    robot_grasp_confirm_event: int = 4
    release_human_on_robot_grasp: bool = True
    robot_fixed_joint_grasp: bool = False
    robot_grasp_body_path: str = "/World/Franka/panda_hand"
    robot_grasp_max_dist_m: float = 0.085
    robot_grasp_closed_width_m: float = 0.075
    robot_place_done_event: int = 8
    robot_disable_cube_collision_during_carry: bool = False
    takeover_arm_by_distance: bool = False
    takeover_release_dist_m: float = 0.04
    robot_approach_collision: bool = False
    max_robot_takeover_steps: int = 900
    place_stack_gap_m: float = 0.002
    workspace_x: tuple[float, float] = (0.22, 0.75)
    workspace_y: tuple[float, float] = (-0.35, 0.35)
    workspace_z_above_table: tuple[float, float] = (0.04, 0.42)
    haptic_intensity: int = 80
    respawn_after_completion: bool = True
    handover_oriented_grasp_enabled: bool = True
    handover_pregrasp_offset_m: float = 0.14
    handover_approach_offset_m: float = 0.008
    handover_grasp_hold_steps: int = 18
    handover_grasp_attach_dist_m: float = 0.08
    handover_grasp_attach_width_m: float = 0.06
    handover_grasp_attach_center_dist_m: float = 0.065
    handover_grasp_jaw_axis_max_m: float = 0.04
    handover_grasp_vertical_max_m: float = 0.07
    handover_grasp_hand_to_fingertip_m: float = 0.103
    handover_grasp_fingertip_offset_m: float = 0.045
    handover_grasp_depth_axis_max_m: float = 0.04
    handover_grasp_head_clearance_m: float = 0.075
    handover_grasp_verify_steps: int = 18
    handover_grasp_verify_timeout_steps: int = 90
    handover_grasp_verify_max_drop_m: float = 0.04
    handover_grasp_retry_enabled: bool = True
    handover_grasp_max_retries: int = 6
    handover_grasp_retry_open_steps: int = 24
    handover_grasp_human_release_timeout_steps: int = 12
    handover_grasp_approach_timeout_steps: int = 180
    handover_move_reached_tolerance_m: float = 0.035
    handover_physical_lift_m: float = 0.14
    handover_physical_place_above_m: float = 0.04
    handover_physical_release_hold_steps: int = 30
    handover_physical_carry_lost_steps: int = 18
    handover_physical_carry_lost_center_m: float = 0.12
    handover_physical_stage_timeout_steps: int = 180
    handover_disable_human_hand_collision_on_robot_grasp: bool = True
    handover_fingertip_joint_takeover_enabled: bool = True
    handover_high_friction_enabled: bool = True
    handover_grasp_static_friction: float = 2.5
    handover_grasp_dynamic_friction: float = 2.0
    handover_side_grasp_when_hand_below: bool = False

    @classmethod
    def from_env(cls) -> "HandoverTaskConfig":
        physical_proxy_default = _env_bool(
            "XR_PHYSICAL_HAND_PROXY",
            _env_bool("ENABLE_PHYSICAL_HAND_PROXY", _env_bool("ENABLE_HANDOVER_TASK", False)),
        )
        respawn_to_hand = _env_bool("HANDOVER_RESPAWN_TO_HAND", not physical_proxy_default)
        return cls(
            hand_priority=_env_hands("HANDOVER_HAND_PRIORITY", ("right", "left")),
            spawn_offset=_env_vec3("HANDOVER_RESPAWN_OFFSET", (0.0, 0.0, -0.02)),
            respawn_to_hand=respawn_to_hand,
            adopt_any_grabbed_cube=_env_bool(
                "HANDOVER_ADOPT_GRABBED_CUBE",
                not respawn_to_hand,
            ),
            follow_hand_until_grabbed=_env_bool(
                "HANDOVER_FOLLOW_HAND_UNTIL_GRABBED",
                not physical_proxy_default,
            ),
            human_cube_physics=_env_bool(
                "HANDOVER_HUMAN_CUBE_PHYSICS",
                physical_proxy_default,
            ),
            human_hold_steps=max(1, _env_int("HANDOVER_HUMAN_HOLD_STEPS", 24)),
            stable_window_steps=max(1, _env_int("HANDOVER_STABLE_WINDOW_STEPS", 12)),
            stable_motion_threshold_m=max(
                0.001,
                _env_float("HANDOVER_STABLE_MOTION_THRESHOLD_M", 0.035),
            ),
            stable_max_step_m=max(
                0.001,
                _env_float("HANDOVER_STABLE_MAX_STEP_M", 0.055),
            ),
            stable_mean_step_m=max(
                0.001,
                _env_float(
                    "HANDOVER_STABLE_MEAN_STEP_M",
                    _env_float("HANDOVER_STABLE_VELOCITY_THRESHOLD_M", 0.010),
                ),
            ),
            stable_jitter_threshold_m=max(
                0.001,
                _env_float("HANDOVER_STABLE_JITTER_THRESHOLD_M", 0.025),
            ),
            stable_outlier_tolerance_m=max(
                0.001,
                _env_float("HANDOVER_STABLE_OUTLIER_TOLERANCE_M", 0.055),
            ),
            stable_required_windows=max(1, _env_int("HANDOVER_STABLE_REQUIRED_WINDOWS", 3)),
            takeover_release_event=max(0, _env_int("HANDOVER_TAKEOVER_RELEASE_EVENT", 3)),
            physical_mode=_env_bool(
                "HANDOVER_PHYSICAL_MODE",
                _env_bool(
                    "XR_PHYSICAL_HAND_FIXED_JOINT_GRAB",
                    _env_bool("PHYSICAL_FIXED_JOINT_GRAB", physical_proxy_default),
                ),
            ),
            robot_grasp_confirm_event=max(0, _env_int("HANDOVER_ROBOT_GRASP_CONFIRM_EVENT", 4)),
            release_human_on_robot_grasp=_env_bool("HANDOVER_RELEASE_HUMAN_ON_ROBOT_GRASP", True),
            robot_fixed_joint_grasp=_env_bool("HANDOVER_ROBOT_FIXED_JOINT_GRASP", False),
            robot_grasp_body_path=os.environ.get(
                "HANDOVER_ROBOT_GRASP_BODY_PATH",
                "/World/Franka/panda_hand",
            ),
            robot_grasp_max_dist_m=max(
                0.01,
                _env_float("HANDOVER_ROBOT_GRASP_MAX_DIST_M", 0.085),
            ),
            robot_grasp_closed_width_m=max(
                0.0,
                _env_float("HANDOVER_ROBOT_GRASP_CLOSED_WIDTH_M", 0.075),
            ),
            robot_place_done_event=max(0, _env_int("HANDOVER_ROBOT_PLACE_DONE_EVENT", 8)),
            robot_disable_cube_collision_during_carry=_env_bool(
                "HANDOVER_ROBOT_DISABLE_CUBE_COLLISION_DURING_CARRY",
                False,
            ),
            takeover_arm_by_distance=_env_bool("HANDOVER_TAKEOVER_ARM_BY_DISTANCE", False),
            takeover_release_dist_m=max(
                0.01,
                _env_float("HANDOVER_TAKEOVER_RELEASE_DIST_M", 0.04),
            ),
            robot_approach_collision=_env_bool("HANDOVER_ROBOT_APPROACH_COLLISION", False),
            max_robot_takeover_steps=max(60, _env_int("HANDOVER_MAX_ROBOT_STEPS", 900)),
            place_stack_gap_m=max(0.0, _env_float("HANDOVER_PLACE_STACK_GAP_M", 0.002)),
            workspace_x=(
                _env_float("HANDOVER_WORKSPACE_X_MIN", 0.22),
                _env_float("HANDOVER_WORKSPACE_X_MAX", 0.75),
            ),
            workspace_y=(
                _env_float("HANDOVER_WORKSPACE_Y_MIN", -0.35),
                _env_float("HANDOVER_WORKSPACE_Y_MAX", 0.35),
            ),
            workspace_z_above_table=(
                _env_float("HANDOVER_WORKSPACE_Z_MIN_ABOVE_TABLE", 0.04),
                _env_float("HANDOVER_WORKSPACE_Z_MAX_ABOVE_TABLE", 0.42),
            ),
            haptic_intensity=max(0, min(100, _env_int("HANDOVER_HAPTIC_INTENSITY", 80))),
            respawn_after_completion=_env_bool("HANDOVER_RESPAWN_AFTER_COMPLETION", True),
            handover_oriented_grasp_enabled=_env_bool("HANDOVER_ORIENTED_GRASP", True),
            handover_pregrasp_offset_m=max(
                0.04,
                _env_float("HANDOVER_PREGRASP_OFFSET_M", 0.14),
            ),
            handover_approach_offset_m=max(
                0.0,
                _env_float("HANDOVER_APPROACH_OFFSET_M", 0.008),
            ),
            handover_grasp_hold_steps=max(
                1,
                _env_int("HANDOVER_GRASP_HOLD_STEPS", 18),
            ),
            handover_grasp_attach_dist_m=max(
                0.01,
                _env_float("HANDOVER_GRASP_ATTACH_DIST_M", 0.08),
            ),
            handover_grasp_attach_width_m=max(
                0.0,
                _env_float("HANDOVER_GRASP_ATTACH_WIDTH_M", 0.06),
            ),
            handover_grasp_attach_center_dist_m=max(
                0.01,
                _env_float("HANDOVER_GRASP_ATTACH_CENTER_DIST_M", 0.065),
            ),
            handover_grasp_jaw_axis_max_m=max(
                0.005,
                _env_float("HANDOVER_GRASP_JAW_AXIS_MAX_M", 0.04),
            ),
            handover_grasp_vertical_max_m=max(
                0.005,
                _env_float("HANDOVER_GRASP_VERTICAL_MAX_M", 0.07),
            ),
            handover_grasp_hand_to_fingertip_m=max(
                0.0,
                _env_float("HANDOVER_GRASP_HAND_TO_FINGERTIP_M", 0.103),
            ),
            handover_grasp_fingertip_offset_m=max(
                0.0,
                _env_float("HANDOVER_GRASP_FINGERTIP_OFFSET_M", 0.045),
            ),
            handover_grasp_depth_axis_max_m=max(
                0.005,
                _env_float("HANDOVER_GRASP_DEPTH_AXIS_MAX_M", 0.04),
            ),
            handover_grasp_head_clearance_m=max(
                0.0,
                _env_float("HANDOVER_GRASP_HEAD_CLEARANCE_M", 0.075),
            ),
            handover_grasp_verify_steps=max(
                1,
                _env_int("HANDOVER_GRASP_VERIFY_STEPS", 18),
            ),
            handover_grasp_verify_timeout_steps=max(
                1,
                _env_int("HANDOVER_GRASP_VERIFY_TIMEOUT_STEPS", 90),
            ),
            handover_grasp_verify_max_drop_m=max(
                0.005,
                _env_float("HANDOVER_GRASP_VERIFY_MAX_DROP_M", 0.04),
            ),
            handover_grasp_retry_enabled=_env_bool("HANDOVER_GRASP_RETRY_ENABLED", True),
            handover_grasp_max_retries=max(
                0,
                _env_int("HANDOVER_GRASP_MAX_RETRIES", 6),
            ),
            handover_grasp_retry_open_steps=max(
                1,
                _env_int("HANDOVER_GRASP_RETRY_OPEN_STEPS", 24),
            ),
            handover_grasp_human_release_timeout_steps=max(
                1,
                _env_int("HANDOVER_GRASP_HUMAN_RELEASE_TIMEOUT_STEPS", 12),
            ),
            handover_grasp_approach_timeout_steps=max(
                30,
                _env_int("HANDOVER_GRASP_APPROACH_TIMEOUT_STEPS", 180),
            ),
            handover_move_reached_tolerance_m=max(
                0.005,
                _env_float("HANDOVER_MOVE_REACHED_TOLERANCE_M", 0.035),
            ),
            handover_physical_lift_m=max(
                0.02,
                _env_float("HANDOVER_PHYSICAL_LIFT_M", 0.14),
            ),
            handover_physical_place_above_m=max(
                0.0,
                _env_float("HANDOVER_PHYSICAL_PLACE_ABOVE_M", 0.04),
            ),
            handover_physical_release_hold_steps=max(
                1,
                _env_int("HANDOVER_PHYSICAL_RELEASE_HOLD_STEPS", 30),
            ),
            handover_physical_carry_lost_steps=max(
                1,
                _env_int("HANDOVER_PHYSICAL_CARRY_LOST_STEPS", 18),
            ),
            handover_physical_carry_lost_center_m=max(
                0.02,
                _env_float("HANDOVER_PHYSICAL_CARRY_LOST_CENTER_M", 0.12),
            ),
            handover_physical_stage_timeout_steps=max(
                30,
                _env_int("HANDOVER_PHYSICAL_STAGE_TIMEOUT_STEPS", 180),
            ),
            handover_disable_human_hand_collision_on_robot_grasp=_env_bool(
                "HANDOVER_DISABLE_HUMAN_HAND_COLLISION_ON_ROBOT_GRASP",
                True,
            ),
            handover_fingertip_joint_takeover_enabled=_env_bool(
                "HANDOVER_ROBOT_FINGERTIP_JOINT_TAKEOVER",
                True,
            ),
            handover_high_friction_enabled=_env_bool("HANDOVER_HIGH_FRICTION_GRASP", True),
            handover_grasp_static_friction=max(
                0.0,
                _env_float("HANDOVER_GRASP_STATIC_FRICTION", 2.5),
            ),
            handover_grasp_dynamic_friction=max(
                0.0,
                _env_float("HANDOVER_GRASP_DYNAMIC_FRICTION", 2.0),
            ),
            handover_side_grasp_when_hand_below=_env_bool(
                "HANDOVER_SIDE_GRASP_WHEN_HAND_BELOW",
                False,
            ),
        )


@dataclass(frozen=True)
class HandoverUpdate:
    phase: str
    active_cube: Any | None
    active_hand: str | None
    completed_picks: int
    current_pick_idx: int
    robot_active: bool
    completed_cycle: bool = False
    message: str = ""


class HandoverTaskManager:
    """State machine for human-to-robot cube handover followed by pick-and-place.

    The VR hand/grab layer owns user attachment. This manager owns task intent:
    respawn a cube near the selected hand, wait for a stable human hold, freeze a
    safe robot pick target, hand the object over to the robot, and then place it.
    """

    def __init__(
        self,
        *,
        cubes,
        pick_targets,
        place_target,
        table_top_z: float,
        cube_size: float,
        stack_base_xy: np.ndarray,
        controller,
        robot,
        approach_height: float,
        grab_manager=None,
        haptics=None,
        logger=None,
        config: HandoverTaskConfig | None = None,
    ) -> None:
        if not pick_targets:
            raise ValueError("HandoverTaskManager requires at least one pick target")
        self.cubes = list(cubes)
        self.pick_targets = list(pick_targets)
        self.place_target = place_target
        self.table_top_z = float(table_top_z)
        self.cube_size = float(cube_size)
        self.cube_half = self.cube_size * 0.5
        self.stack_base_xy = np.asarray(stack_base_xy, dtype=float)
        self.controller = controller
        self.robot = robot
        self.approach_height = approach_height
        self.grab_manager = grab_manager
        self.haptics = haptics
        self.logger = logger
        self.config = config or HandoverTaskConfig()
        self.handover_candidates = (
            list(self.cubes)
            if self.config.adopt_any_grabbed_cube
            else list(self.pick_targets)
        )

        self.phase = "waiting_for_hand"
        self.current_pick_idx = 0
        self.completed_picks = 0
        self.active_hand: str | None = None
        self._active_cube_override = None
        self._hold_steps = 0
        self._robot_steps = 0
        self._target_history: list[np.ndarray] = []
        self._stable_hold_windows = 0
        self._handover_pick_pos: np.ndarray | None = None
        self._physics_armed_for_robot = False
        self._robot_has_taken_over = False
        self._robot_grasp_joint_path: str | None = None
        self._oriented_controller = None
        self._oriented_stage: str | None = None
        self._oriented_stage_steps = 0
        self._oriented_plan: dict[str, np.ndarray | float | str] | None = None
        self._oriented_fallback = False
        self._oriented_retry_count = 0
        self._last_handover_hand_pos: np.ndarray | None = None
        self._oriented_close_hold_pos: np.ndarray | None = None
        self._physical_grasp_stable_steps = 0
        self._physical_grasp_release_pos: np.ndarray | None = None
        self._physical_carry_stage: str | None = None
        self._physical_carry_stage_steps = 0
        self._physical_carry_lost_steps = 0
        self._physical_carry_lift_z: float | None = None
        self._physical_carry_release_target: np.ndarray | None = None
        self._human_hand_collision_suppressed = False
        self._suppressed_human_hand: str | None = None
        self._human_regrab_suppressed = False
        self._suppressed_human_regrab_hand: str | None = None
        self._suppressed_human_regrab_cube = None
        self._grasp_physics_material = None
        self._high_friction_material_applied = False
        self._last_message = ""
        self._learning_episode_active = False
        self._learning_episode_hand: str | None = None
        self._learning_episode_cube_name: str | None = None
        self._learning_last_action: dict[str, Any] | None = None
        self._learning_last_grasp_metrics: dict[str, float] = self._empty_grasp_metrics()
        self._learning_success_event: str | None = None
        self._learning_failure_reason: str | None = None

        self._set_place_target_pose()
        print(
            "[Handover] enabled | "
            f"hands={self.config.hand_priority} "
            f"spawn_offset={np.round(self.config.spawn_offset, 3)} "
            f"respawn_to_hand={self.config.respawn_to_hand} "
            f"adopt_grabbed={self.config.adopt_any_grabbed_cube} "
            f"human_cube_physics={self.config.human_cube_physics} "
            f"follow_until_grabbed={self.config.follow_hand_until_grabbed} "
            f"hold_steps={self.config.human_hold_steps} "
            f"stable_window={self.config.stable_window_steps} "
            f"stable_required={self.config.stable_required_windows} "
            f"release_event>={self.config.takeover_release_event} "
            f"physical_mode={self.config.physical_mode} "
            f"robot_grasp_event>={self.config.robot_grasp_confirm_event} "
            f"robot_fixed_joint_grasp={self.config.robot_fixed_joint_grasp} "
            f"fingertip_takeover={self.config.handover_fingertip_joint_takeover_enabled} "
            f"robot_place_done_event>={self.config.robot_place_done_event} "
            f"arm_by_distance={self.config.takeover_arm_by_distance} "
            f"robot_approach_collision={self.config.robot_approach_collision} "
            f"oriented_grasp={self.config.handover_oriented_grasp_enabled}",
            flush=True,
        )

    @property
    def active_cube(self):
        if self._active_cube_override is not None:
            return self._active_cube_override
        return self.pick_targets[self.current_pick_idx]

    @property
    def grab_target_cube(self):
        if (
            self.config.adopt_any_grabbed_cube
            and self.phase == "waiting_for_hand"
            and self._active_cube_override is None
        ):
            return None
        return self.active_cube

    @property
    def active_place_pos(self) -> np.ndarray:
        stack_height = (
            self.table_top_z
            + self.cube_half
            + (self.completed_picks % self._cycle_length())
            * (self.cube_size + self.config.place_stack_gap_m)
        )
        return np.array([self.stack_base_xy[0], self.stack_base_xy[1], stack_height], dtype=float)

    @property
    def cycle_length(self) -> int:
        return self._cycle_length()

    def reset_cycle(self) -> None:
        if self._learning_episode_active and self._learning_success_event is None:
            self._mark_learning_failure("reset_cycle")
        self._restore_active_human_regrab()
        self._restore_active_hand_collision()
        self.phase = "waiting_for_hand"
        self.current_pick_idx = 0
        self.completed_picks = 0
        self.active_hand = None
        self._active_cube_override = None
        self._hold_steps = 0
        self._robot_steps = 0
        self._target_history = []
        self._stable_hold_windows = 0
        self._handover_pick_pos = None
        self._physics_armed_for_robot = False
        self._robot_has_taken_over = False
        self._remove_robot_grasp_joint()
        self._reset_oriented_grasp_state()
        self._last_message = ""
        self._reset_controller()
        self._set_place_target_pose()

    def get_learning_sample(self) -> dict[str, Any]:
        action = self._learning_last_action or {}
        target_pos = action.get("target_ee_pos")
        target_quat = action.get("target_ee_quat")
        action_vec = action.get("action")
        joint_action = action.get("joint_action")
        return {
            "episode_active": self._learning_episode_active,
            "has_action": self._learning_last_action is not None,
            "phase": action.get("phase") or self.phase,
            "oriented_stage": action.get("oriented_stage") or self._oriented_stage or "",
            "physical_carry_stage": action.get("physical_carry_stage")
            or self._physical_carry_stage
            or "",
            "target_ee_pos": target_pos,
            "target_ee_quat": target_quat,
            "place_pos": action.get("place_pos"),
            "action": action_vec,
            "joint_action": joint_action,
            "gripper_command": action.get("gripper_command"),
            "gripper_value": action.get("gripper_value", 0.0),
            "ee_pos_at_command": action.get("ee_pos_at_command"),
            "grasp_metrics": dict(self._learning_last_grasp_metrics),
            "robot_has_taken_over": self._robot_has_taken_over,
            "active_hand": self._learning_episode_hand or self.active_hand or "",
            "active_cube": self._learning_episode_cube_name
            or getattr(self.active_cube, "name", ""),
            "success_event": self._learning_success_event or "",
            "failure_reason": self._learning_failure_reason or "",
            "controller_event": self._controller_event(),
            "controller_t": self._controller_t(),
            "robot_steps": self._robot_steps,
        }

    def clear_learning_terminal_event(self) -> None:
        self._learning_episode_active = False
        self._learning_episode_hand = None
        self._learning_episode_cube_name = None
        self._learning_last_action = None
        self._learning_last_grasp_metrics = self._empty_grasp_metrics()
        self._learning_success_event = None
        self._learning_failure_reason = None

    def _mark_learning_episode_started(self, hand: str | None, cube) -> None:
        self._learning_episode_active = True
        self._learning_episode_hand = hand
        self._learning_episode_cube_name = getattr(cube, "name", None)
        self._learning_last_action = None
        self._learning_last_grasp_metrics = self._safe_grasp_metrics()
        self._learning_success_event = None
        self._learning_failure_reason = None

    def _mark_learning_success(self, event: str) -> None:
        if not self._learning_episode_active:
            return
        self._learning_success_event = str(event)
        self._learning_failure_reason = None
        self._learning_last_grasp_metrics = self._safe_grasp_metrics()

    def _mark_learning_failure(self, reason: str) -> None:
        if not self._learning_episode_active or self._learning_success_event is not None:
            return
        self._learning_failure_reason = str(reason or "unknown_failure")
        self._learning_last_grasp_metrics = self._safe_grasp_metrics()

    def _record_learning_action(
        self,
        *,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        gripper_command: str | None,
        ee_pos_at_command: np.ndarray | None,
        applied_action,
        gripper_action=None,
    ) -> None:
        if not self._learning_episode_active:
            return
        target_pos = np.asarray(target_pos, dtype=float).reshape(-1)[:3]
        target_quat = np.asarray(target_quat, dtype=float).reshape(-1)[:4]
        if target_pos.size != 3 or target_quat.size != 4:
            return
        if ee_pos_at_command is not None and self._valid_vec3(ee_pos_at_command):
            ee_pos = np.asarray(ee_pos_at_command, dtype=float)
            delta = target_pos - ee_pos
        else:
            ee_pos = np.full(3, np.nan, dtype=float)
            delta = np.zeros(3, dtype=float)
        gripper_value = self._learning_gripper_value(gripper_command)
        self._learning_last_action = {
            "target_ee_pos": target_pos.astype(np.float32),
            "target_ee_quat": target_quat.astype(np.float32),
            "place_pos": np.asarray(self.active_place_pos, dtype=np.float32),
            "phase": self.phase,
            "oriented_stage": self._oriented_stage or "",
            "physical_carry_stage": self._physical_carry_stage or "",
            "gripper_command": gripper_command or "",
            "gripper_value": float(gripper_value),
            "ee_pos_at_command": ee_pos.astype(np.float32),
            "action": np.concatenate(
                [
                    delta.astype(np.float32),
                    target_quat.astype(np.float32),
                    np.array([gripper_value], dtype=np.float32),
                ]
            ),
            "joint_action": self._learning_joint_action_vector(
                applied_action,
                gripper_action=gripper_action,
            ),
        }
        self._learning_last_grasp_metrics = self._safe_grasp_metrics()

    def _learning_gripper_value(self, command: str | None) -> float:
        if command == "open":
            return 1.0
        if command == "close":
            return -1.0
        return 0.0

    def _learning_joint_action_vector(self, action, *, gripper_action=None) -> np.ndarray:
        values = np.full(9, np.nan, dtype=np.float32)
        self._fill_joint_action_vector(values, action)
        self._fill_joint_action_vector(
            values,
            gripper_action,
            fallback_indices=self._gripper_joint_indices(),
        )
        return values

    def _fill_joint_action_vector(
        self,
        values: np.ndarray,
        action,
        *,
        fallback_indices: np.ndarray | None = None,
    ) -> None:
        if action is None:
            return
        positions = getattr(action, "joint_positions", None)
        if positions is None:
            return
        try:
            positions = np.asarray(positions, dtype=float).reshape(-1)
        except Exception:
            return
        if positions.size == 0:
            return
        indices = getattr(action, "joint_indices", None)
        if indices is None:
            if positions.size >= values.size:
                indices = np.arange(positions.size, dtype=int)
            elif fallback_indices is not None and len(fallback_indices) >= positions.size:
                indices = np.asarray(fallback_indices, dtype=int)[: positions.size]
            else:
                indices = np.arange(positions.size, dtype=int)
        else:
            try:
                indices = np.asarray(indices, dtype=int).reshape(-1)
            except Exception:
                return
        for idx, value in zip(indices, positions):
            idx = int(idx)
            if 0 <= idx < values.size:
                values[idx] = float(value)

    def _safe_grasp_metrics(self) -> dict[str, float]:
        try:
            metrics = self._gripper_grasp_metrics()
        except Exception:
            metrics = self._empty_grasp_metrics()
        base = self._empty_grasp_metrics()
        base.update({key: float(metrics.get(key, base[key])) for key in base})
        return base

    def _empty_grasp_metrics(self) -> dict[str, float]:
        return {
            "center_dist": float("inf"),
            "min_dist": float("inf"),
            "width": float("inf"),
            "jaw_offset": float("inf"),
            "vertical_offset": float("inf"),
            "depth_offset": float("inf"),
            "head_dist": float("inf"),
        }

    def update(
        self,
        *,
        hand_positions: dict[str, np.ndarray | None],
        step: int,
        sim_time: float,
    ) -> HandoverUpdate:
        completed_cycle = False
        message = ""
        self._maintain_active_hand_collision_suppression()

        if self.phase == "waiting_for_hand":
            message = self._wait_for_hand(hand_positions)
        elif self.phase == "spawned_to_hand":
            message = self._wait_for_human_grab(hand_positions)
        elif self.phase == "human_holding":
            message = self._watch_human_hold(hand_positions)
        elif self.phase == "robot_takeover":
            done, message = self._run_robot_takeover(hand_positions)
            if done:
                completed_cycle = self._finish_robot_place()
        else:
            self._log_event("handover_error", f"unknown_phase={self.phase}")
            self.phase = "waiting_for_hand"
            message = "recovered_unknown_phase"

        if (
            message
            and message != self._last_message
            and not self._suppress_terminal_status_message(message)
        ):
            print(f"[Handover] {message}", flush=True)
            self._last_message = message

        return HandoverUpdate(
            phase=self.phase,
            active_cube=self.active_cube,
            active_hand=self.active_hand,
            completed_picks=self.completed_picks,
            current_pick_idx=self.current_pick_idx,
            robot_active=self.phase == "robot_takeover",
            completed_cycle=completed_cycle,
            message=message,
        )

    def _suppress_terminal_status_message(self, message: str) -> bool:
        return message.startswith(f"{ORIENT_STAGE_PREGRASP} steps=")

    def _wait_for_hand(self, hand_positions: dict[str, np.ndarray | None]) -> str:
        if self.config.adopt_any_grabbed_cube:
            adopted = self._adopt_held_cube()
            if adopted is not None:
                holding_hand, _ = adopted
                return self._start_human_holding(holding_hand)
        else:
            holding_hand = self._holding_hand_for_active_cube()
            if holding_hand is not None:
                return self._start_human_holding(holding_hand)

        hand, hand_pos = self._select_hand(hand_positions)
        if hand_pos is None:
            return (
                "waiting_for_tracked_hand"
                if self.config.respawn_to_hand
                else f"waiting_for_table_grab target={self._waiting_cube_label()}"
            )

        if not self.config.respawn_to_hand:
            self.active_hand = None
            return f"waiting_for_table_grab hand_seen={hand} target={self._waiting_cube_label()}"

        self.active_hand = hand
        spawn_pos = self._spawn_position_for_hand(hand_pos)
        self._place_cube_for_human(spawn_pos)
        self.phase = "spawned_to_hand"
        self._pulse("object_respawned", hand=hand, intensity=self.config.haptic_intensity)
        self._log_event(
            "handover_object_respawned",
            f"hand={hand},cube={self.active_cube.name},pos={self._fmt(spawn_pos)}",
        )
        return f"object_respawned hand={hand} cube={self.active_cube.name}"

    def _wait_for_human_grab(self, hand_positions: dict[str, np.ndarray | None]) -> str:
        hand = self.active_hand
        hand_pos = hand_positions.get(hand) if hand else None
        if hand is None or hand_pos is None:
            self.phase = "waiting_for_hand"
            return "lost_hand_before_grab"

        if self.config.follow_hand_until_grabbed and not self._is_user_holding_active_cube():
            self._place_cube_for_human(self._spawn_position_for_hand(hand_pos))

        holding_hand = self._holding_hand_for_active_cube()
        if holding_hand is None:
            return f"waiting_for_grab hand={hand} cube={self.active_cube.name}"

        return self._start_human_holding(holding_hand)

    def _start_human_holding(self, holding_hand: str) -> str:
        self.active_hand = holding_hand
        self._hold_steps = 0
        self._target_history = []
        self._stable_hold_windows = 0
        self.phase = "human_holding"
        self._pulse("object_grabbed", hand=holding_hand, intensity=self.config.haptic_intensity)
        self._log_event(
            "handover_object_grabbed",
            f"hand={holding_hand},cube={self.active_cube.name}",
        )
        return f"human_holding hand={holding_hand} cube={self.active_cube.name}"

    def _watch_human_hold(self, hand_positions: dict[str, np.ndarray | None]) -> str:
        holding_hand = self._holding_hand_for_active_cube()
        if holding_hand is None:
            self.phase = "spawned_to_hand" if self.config.respawn_to_hand else "waiting_for_hand"
            if not self.config.respawn_to_hand:
                self.active_hand = None
            if self.config.adopt_any_grabbed_cube:
                self._active_cube_override = None
            self._hold_steps = 0
            self._target_history = []
            self._stable_hold_windows = 0
            return "cube_released_before_takeover"

        self.active_hand = holding_hand
        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            self.phase = "waiting_for_hand"
            if self.config.adopt_any_grabbed_cube:
                self._active_cube_override = None
            self._hold_steps = 0
            self._target_history = []
            self._stable_hold_windows = 0
            return "active_cube_pose_missing"

        self._hold_steps += 1
        self._target_history.append(cube_pos)
        max_history = max(1, self.config.stable_window_steps)
        if len(self._target_history) > max_history:
            self._target_history = self._target_history[-max_history:]

        if self._hold_steps < self.config.human_hold_steps:
            return f"holding_for_transfer {self._hold_steps}/{self.config.human_hold_steps}"

        stable, stable_reason = self._update_hold_stability()
        if not stable:
            return (
                "waiting_for_stable_hand "
                f"reason={stable_reason} "
                f"stable={self._stable_hold_windows}/{self.config.stable_required_windows}"
            )

        self._handover_pick_pos = self._robot_pick_position(cube_pos)
        if not self.config.physical_mode:
            self._force_release_active_cube()
            self._place_cube_for_robot(self._handover_pick_pos, physics_enabled=False)
        self._reset_controller()
        self._robot_steps = 0
        self._physics_armed_for_robot = False
        self._robot_has_taken_over = False
        self._reset_oriented_grasp_state()
        self.phase = "robot_takeover"
        self._mark_learning_episode_started(holding_hand, self.active_cube)
        self._pulse("transfer_ready", hand=holding_hand, intensity=100)
        self._log_event(
            "handover_transfer_ready",
            (
                f"hand={holding_hand},cube={self.active_cube.name},"
                f"pick={self._fmt(self._handover_pick_pos)},"
                f"stability={self._fmt_hold_stability_metrics()}"
            ),
        )
        return f"robot_takeover_started cube={self.active_cube.name}"

    def _run_robot_takeover(
        self,
        hand_positions: dict[str, np.ndarray | None] | None = None,
    ) -> tuple[bool, str]:
        if self._handover_pick_pos is None:
            self._mark_learning_failure("missing_handover_pick_pos")
            self.phase = "waiting_for_hand"
            return False, "missing_handover_pick_pos"

        self._robot_steps += 1
        if self._should_use_oriented_grasp():
            return self._run_oriented_robot_takeover(hand_positions or {})

        if self.config.physical_mode:
            cube_pos = self._cube_position(self.active_cube)
            if cube_pos is not None and not self._robot_has_taken_over:
                self._handover_pick_pos = self._robot_pick_position(cube_pos)
        else:
            if not self._physics_armed_for_robot:
                self._place_cube_for_robot(self._handover_pick_pos, physics_enabled=False)

            should_arm = self._should_enable_physics_for_grasp()
            if should_arm and not self._physics_armed_for_robot:
                self._place_cube_for_robot(self._handover_pick_pos, physics_enabled=True)
                self._physics_armed_for_robot = True
                self._pulse("robot_takeover", hand=self.active_hand, intensity=100)
                self._log_event(
                    "handover_robot_takeover",
                    f"cube={self.active_cube.name},event={self._controller_event()}",
                )

        done = self._run_pick_place(
            controller=self.controller,
            robot=self.robot,
            pick_position=self._handover_pick_pos,
            place_position=self.active_place_pos,
        )
        if self.config.physical_mode:
            self._maybe_confirm_physical_robot_grasp()
            if self._physical_robot_place_done():
                self._complete_physical_robot_place()
                done = True
        if not done and self._controller_event() >= 8 and self._cube_is_at_place():
            done = True
        self._set_place_target_pose()

        if done:
            return True, f"robot_place_done cube={self.active_cube.name}"

        if self._robot_steps > self.config.max_robot_takeover_steps:
            self._log_event(
                "handover_timeout",
                f"cube={self.active_cube.name},steps={self._robot_steps}",
            )
            self._reset_after_failed_takeover("robot_takeover_timeout")
            return False, "robot_takeover_timeout_respawn"

        return False, f"robot_takeover event={self._controller_event()} steps={self._robot_steps}"

    def _finish_robot_place(self) -> bool:
        placed_cube = self.active_cube
        self._log_event(
            "handover_place_complete",
            f"cube={placed_cube.name},place={self._fmt(self.active_place_pos)}",
        )
        self._mark_learning_success("handover_place_complete")
        self._pulse("handover_complete", hand=self.active_hand, intensity=100)
        self.completed_picks += 1
        cycle_length = self._cycle_length()
        completed_cycle = self.completed_picks % cycle_length == 0
        self.current_pick_idx = self.completed_picks % len(self.pick_targets)
        self._restore_active_human_regrab()
        self._restore_active_hand_collision()
        self.active_hand = None
        self._active_cube_override = None
        self.phase = "waiting_for_hand" if self.config.respawn_after_completion else "spawned_to_hand"
        self._hold_steps = 0
        self._robot_steps = 0
        self._target_history = []
        self._stable_hold_windows = 0
        self._handover_pick_pos = None
        self._physics_armed_for_robot = False
        self._robot_has_taken_over = False
        self._remove_robot_grasp_joint()
        self._reset_oriented_grasp_state()
        self._reset_controller()
        self._set_place_target_pose()
        if completed_cycle:
            self._log_event(
                "handover_cycle_complete",
                f"picks={cycle_length}",
            )
        return completed_cycle

    def _reset_after_failed_takeover(self, reason: str = "failed_takeover_reset") -> None:
        self._mark_learning_failure(reason)
        self._restore_active_human_regrab()
        self._force_release_active_cube()
        self._remove_robot_grasp_joint()
        self._place_cube_for_robot(self._safe_table_position(), physics_enabled=True)
        self._reset_controller()
        self._restore_active_hand_collision()
        self.phase = "waiting_for_hand"
        self.active_hand = None
        self._active_cube_override = None
        self._hold_steps = 0
        self._robot_steps = 0
        self._target_history = []
        self._stable_hold_windows = 0
        self._handover_pick_pos = None
        self._physics_armed_for_robot = False
        self._robot_has_taken_over = False
        self._reset_oriented_grasp_state()

    def _select_hand(
        self,
        hand_positions: dict[str, np.ndarray | None],
    ) -> tuple[str | None, np.ndarray | None]:
        for hand in self.config.hand_priority:
            pos = hand_positions.get(hand)
            if self._valid_vec3(pos):
                return hand, np.asarray(pos, dtype=float)
        return None, None

    def _spawn_position_for_hand(self, hand_pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(hand_pos, dtype=float) + self.config.spawn_offset
        pos[2] = max(pos[2], self.table_top_z + self.cube_half)
        return pos

    def _robot_pick_position(self, cube_pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(cube_pos, dtype=float).copy()
        z_min = self.table_top_z + self.config.workspace_z_above_table[0]
        z_max = self.table_top_z + self.config.workspace_z_above_table[1]
        pos[0] = np.clip(pos[0], *self.config.workspace_x)
        pos[1] = np.clip(pos[1], *self.config.workspace_y)
        pos[2] = np.clip(pos[2], z_min, z_max)
        return pos

    def _should_use_oriented_grasp(self) -> bool:
        return bool(
            self.config.physical_mode
            and self.config.handover_oriented_grasp_enabled
            and not self._oriented_fallback
        )

    def _reset_oriented_grasp_state(self) -> None:
        if self._oriented_controller is not None:
            try:
                self._oriented_controller.reset()
            except Exception:
                pass
        self._oriented_stage = None
        self._oriented_stage_steps = 0
        self._oriented_plan = None
        self._oriented_fallback = False
        self._oriented_retry_count = 0
        self._last_handover_hand_pos = None
        self._oriented_close_hold_pos = None
        self._physical_grasp_stable_steps = 0
        self._physical_grasp_release_pos = None
        self._physical_carry_stage = None
        self._physical_carry_stage_steps = 0
        self._physical_carry_lost_steps = 0
        self._physical_carry_lift_z = None
        self._physical_carry_release_target = None
        if not self._human_hand_collision_suppressed:
            self._suppressed_human_hand = None

    def _run_oriented_robot_takeover(
        self,
        hand_positions: dict[str, np.ndarray | None],
    ) -> tuple[bool, str]:
        if self._robot_has_taken_over:
            return self._run_oriented_lift_place()

        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return self._fallback_to_pick_controller("missing_cube_pose")
        self._handover_pick_pos = self._robot_pick_position(cube_pos)

        hand_pos = self._handover_hand_position(hand_positions)
        if hand_pos is not None:
            self._last_handover_hand_pos = hand_pos

        if self._oriented_stage is None:
            if not self._start_oriented_grasp_plan(cube_pos, hand_pos):
                return self._fallback_to_pick_controller("plan_unavailable")
        elif self._oriented_stage in (
            ORIENT_STAGE_REOPEN,
            ORIENT_STAGE_PREGRASP,
            ORIENT_STAGE_APPROACH,
        ):
            self._refresh_oriented_grasp_plan(cube_pos, hand_pos)

        self._oriented_stage_steps += 1
        if self._oriented_stage == ORIENT_STAGE_REOPEN:
            target = np.asarray(self._oriented_plan["pregrasp_pos"], dtype=float)
            quat = np.asarray(self._oriented_plan["quat"], dtype=float)
            reached = self._move_oriented_ee(target, quat, gripper_command="open")
            if reached or self._oriented_stage_steps >= self.config.handover_grasp_retry_open_steps:
                self._advance_oriented_stage(ORIENT_STAGE_PREGRASP)
            return False, (
                f"{ORIENT_STAGE_REOPEN} "
                f"retry={self._oriented_retry_count}/{self.config.handover_grasp_max_retries} "
                f"steps={self._oriented_stage_steps}"
            )

        if self._oriented_stage == ORIENT_STAGE_PREGRASP:
            target = np.asarray(self._oriented_plan["pregrasp_pos"], dtype=float)
            quat = np.asarray(self._oriented_plan["quat"], dtype=float)
            reached = self._move_oriented_ee(target, quat, gripper_command="open")
            if reached or self._oriented_stage_steps > 150:
                self._advance_oriented_stage(ORIENT_STAGE_APPROACH)
            return False, f"{ORIENT_STAGE_PREGRASP} steps={self._oriented_stage_steps}"

        if self._oriented_stage == ORIENT_STAGE_APPROACH:
            target = np.asarray(self._oriented_plan["approach_pos"], dtype=float)
            quat = np.asarray(self._oriented_plan["quat"], dtype=float)
            reached = self._move_oriented_ee(target, quat, gripper_command="open")
            ready_to_close = self._gripper_ready_to_start_close()
            if ready_to_close:
                self._advance_oriented_stage(ORIENT_STAGE_CLOSE)
            elif self._oriented_stage_steps > self.config.handover_grasp_approach_timeout_steps:
                return self._retry_oriented_grasp(
                    f"approach_not_ready reached={int(reached)} {self._grasp_metrics_text()}",
                    cube_pos,
                    hand_pos,
                )
            elif self._robot_steps % 30 == 0:
                print(
                    "[Handover] waiting_oriented_approach "
                    f"reached={int(reached)} {self._grasp_metrics_text()}",
                    flush=True,
                )
            return False, f"{ORIENT_STAGE_APPROACH} steps={self._oriented_stage_steps}"

        if self._oriented_stage == ORIENT_STAGE_CLOSE:
            target = self._oriented_close_target()
            quat = np.asarray(self._oriented_plan["quat"], dtype=float)
            self._move_oriented_ee(target, quat, gripper_command="close")
            metrics = self._gripper_grasp_metrics()
            force_ready = (
                self._use_fingertip_takeover_path()
                and self._oriented_stage_steps >= max(3, self.config.handover_grasp_hold_steps // 4)
                and self._gripper_ready_to_force_detach_human(metrics)
            )
            if force_ready:
                self._log_event(
                    "handover_close_capture_ready",
                    f"cube={self.active_cube.name},{self._grasp_metrics_text(metrics)}",
                )
            if (
                force_ready
                and self._try_begin_human_detach_for_physical_grasp(
                    metrics,
                    reason="close_capture",
                )
            ):
                self._advance_oriented_stage(ORIENT_STAGE_VERIFY)
                return False, f"{ORIENT_STAGE_VERIFY} early_release_started"
            head_min = self.config.handover_grasp_head_clearance_m * 0.65
            if (
                self._oriented_stage_steps > 6
                and metrics["head_dist"] < head_min
                and not force_ready
            ):
                return self._retry_oriented_grasp(
                    f"head_clearance_lost {self._grasp_metrics_text(metrics)}",
                    cube_pos,
                    hand_pos,
                )
            if (
                self._oriented_stage_steps >= self.config.handover_grasp_hold_steps
                or self._oriented_attach_ready()
            ):
                self._advance_oriented_stage(ORIENT_STAGE_ATTACH)
            return False, f"{ORIENT_STAGE_CLOSE} steps={self._oriented_stage_steps}"

        if self._oriented_stage == ORIENT_STAGE_ATTACH:
            target = self._oriented_close_target()
            quat = np.asarray(self._oriented_plan["quat"], dtype=float)
            self._move_oriented_ee(target, quat, gripper_command="close")
            fingertip_takeover = self._use_fingertip_takeover_path()
            if (
                self.config.robot_fixed_joint_grasp
                and not fingertip_takeover
                and self._attach_robot_to_active_cube()
            ):
                self._advance_oriented_stage(ORIENT_STAGE_LIFT_PLACE)
                return False, f"{ORIENT_STAGE_ATTACH} ok"
            metrics = self._gripper_grasp_metrics()
            release_ready = self._gripper_contains_cube(metrics)
            loose_release = False
            if (
                not release_ready
                and self._oriented_stage_steps >= max(4, self.config.handover_grasp_hold_steps // 3)
            ):
                loose_release = self._gripper_ready_to_release_human()
            force_release = self._gripper_ready_to_force_detach_human(metrics)
            if (
                fingertip_takeover
                and (release_ready or loose_release or force_release)
                and self._try_begin_human_detach_for_physical_grasp(
                    metrics,
                    reason="attach_capture",
                    allow_loose=loose_release,
                )
            ):
                self._advance_oriented_stage(ORIENT_STAGE_VERIFY)
                return False, f"{ORIENT_STAGE_VERIFY} release_started"
            if self._robot_steps % 15 == 0:
                print(
                    "[Handover] attach_detach_decision "
                    f"fixed_joint={self.config.robot_fixed_joint_grasp} "
                    f"fingertip_path={fingertip_takeover} "
                    f"strict={release_ready} loose={loose_release} force={force_release} "
                    f"{self._grasp_metrics_text(metrics)}",
                    flush=True,
                )
            if self._oriented_stage_steps > max(60, self.config.handover_grasp_hold_steps * 4):
                return self._retry_oriented_grasp("attach_timeout", cube_pos, hand_pos)
            return False, f"{ORIENT_STAGE_ATTACH} waiting steps={self._oriented_stage_steps}"

        if self._oriented_stage == ORIENT_STAGE_VERIFY:
            target = self._oriented_close_target()
            quat = np.asarray(self._oriented_plan["quat"], dtype=float)
            self._move_oriented_ee(target, quat, gripper_command="close")
            verified, failed_reason = self._update_physical_grasp_verify()
            if verified:
                if self._confirm_physical_robot_grasp():
                    self._advance_oriented_stage(ORIENT_STAGE_LIFT_PLACE)
                    return False, f"{ORIENT_STAGE_VERIFY} ok"
                return self._retry_oriented_grasp("confirm_rejected", cube_pos, hand_pos)
            if failed_reason:
                self._log_event(
                    "handover_physical_grasp_failed",
                    f"cube={self.active_cube.name},reason={failed_reason}",
                )
                print(f"[Handover] physical grasp failed: {failed_reason}", flush=True)
                return self._retry_oriented_grasp(failed_reason, cube_pos, hand_pos)
            return False, (
                f"{ORIENT_STAGE_VERIFY} "
                f"stable={self._physical_grasp_stable_steps}/{self.config.handover_grasp_verify_steps}"
            )

        if self._oriented_stage == ORIENT_STAGE_LIFT_PLACE:
            return self._run_oriented_lift_place()

        return self._fallback_to_pick_controller(f"unknown_stage={self._oriented_stage}")

    def _start_oriented_grasp_plan(
        self,
        cube_pos: np.ndarray,
        hand_pos: np.ndarray | None,
    ) -> bool:
        controller = self._get_oriented_controller()
        if controller is None:
            return False
        plan = self._make_oriented_grasp_plan(cube_pos, hand_pos)
        if plan is None:
            return False
        self._oriented_plan = plan
        self._oriented_stage = ORIENT_STAGE_PREGRASP
        self._oriented_stage_steps = 0
        self._set_cube_collision_enabled(self.active_cube, True)
        if hasattr(self.active_cube, "enable_rigid_body_physics"):
            try:
                self.active_cube.enable_rigid_body_physics()
            except Exception:
                pass
        self._apply_high_friction_grasp_materials()
        self._log_event(
            "handover_oriented_grasp_plan",
            (
                f"cube={self.active_cube.name},hand={self.active_hand},"
                f"mode={plan['mode']},approach={self._fmt(plan['approach_dir'])},"
                f"pregrasp={self._fmt(plan['pregrasp_pos'])},target={self._fmt(plan['approach_pos'])}"
            ),
        )
        self._log_event(
            "handover_oriented_pregrasp",
            f"cube={self.active_cube.name},target={self._fmt(plan['pregrasp_pos'])}",
        )
        print(
            "[Handover] oriented grasp plan "
            f"mode={plan['mode']} approach={np.round(plan['approach_dir'], 3)}",
            flush=True,
        )
        return True

    def _refresh_oriented_grasp_plan(
        self,
        cube_pos: np.ndarray,
        hand_pos: np.ndarray | None,
    ) -> bool:
        plan = self._make_oriented_grasp_plan(cube_pos, hand_pos)
        if plan is None:
            return False
        previous_mode = None
        if self._oriented_plan is not None:
            previous_mode = self._oriented_plan.get("mode")
        self._oriented_plan = plan
        if previous_mode != plan.get("mode") or self._robot_steps % 60 == 0:
            self._log_event(
                "handover_oriented_grasp_plan",
                (
                    f"cube={self.active_cube.name},hand={self.active_hand},"
                    f"mode={plan['mode']},refresh=1,"
                    f"pregrasp={self._fmt(plan['pregrasp_pos'])},"
                    f"target={self._fmt(plan['approach_pos'])}"
                ),
            )
        return True

    def _retry_oriented_grasp(
        self,
        reason: str,
        cube_pos: np.ndarray | None = None,
        hand_pos: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        if not self.config.handover_grasp_retry_enabled:
            self._reset_after_failed_takeover(reason)
            return False, f"physical_grasp_failed_respawn reason={reason}"

        self._oriented_retry_count += 1
        self._restore_active_hand_collision()
        self._physical_grasp_stable_steps = 0
        self._physical_grasp_release_pos = None
        self._oriented_close_hold_pos = None
        self._physical_carry_stage = None
        self._physical_carry_stage_steps = 0
        self._physical_carry_lost_steps = 0
        self._physical_carry_lift_z = None
        self._physical_carry_release_target = None
        self._robot_has_taken_over = False
        self._physics_armed_for_robot = False

        if self._oriented_retry_count > self.config.handover_grasp_max_retries:
            self._log_event(
                "handover_oriented_retry_limit",
                f"cube={self.active_cube.name},reason={reason},retries={self._oriented_retry_count}",
            )
            print(
                "[Handover] grasp retry limit reached "
                f"reason={reason} retries={self._oriented_retry_count}",
                flush=True,
            )
            holding_hand = self._holding_hand_for_active_cube()
            if holding_hand is not None:
                self._restore_active_human_regrab()
                self.phase = "human_holding"
                self.active_hand = holding_hand
                self._hold_steps = 0
                self._robot_steps = 0
                self._target_history = []
                self._stable_hold_windows = 0
                self._handover_pick_pos = None
                self._reset_oriented_grasp_state()
                return False, "retry_limit_returned_to_human_holding"
            self._reset_after_failed_takeover(f"retry_limit:{reason}")
            return False, "retry_limit_respawn"

        if cube_pos is None:
            cube_pos = self._cube_position(self.active_cube)
        if cube_pos is not None:
            self._handover_pick_pos = self._robot_pick_position(cube_pos)
            plan = self._make_oriented_grasp_plan(cube_pos, hand_pos)
            if plan is not None:
                self._oriented_plan = plan
        if self._oriented_plan is None:
            self._reset_after_failed_takeover(f"retry_plan_unavailable:{reason}")
            return False, f"retry_plan_unavailable reason={reason}"

        self._oriented_stage = ORIENT_STAGE_REOPEN
        self._oriented_stage_steps = 0
        self._log_event(
            "handover_oriented_retry",
            (
                f"cube={self.active_cube.name},reason={reason},"
                f"retry={self._oriented_retry_count}/{self.config.handover_grasp_max_retries}"
            ),
        )
        print(
            "[Handover] grasp failed, reopening and retrying "
            f"reason={reason} retry={self._oriented_retry_count}/"
            f"{self.config.handover_grasp_max_retries}",
            flush=True,
        )
        return False, (
            f"{ORIENT_STAGE_REOPEN} reason={reason} "
            f"retry={self._oriented_retry_count}/{self.config.handover_grasp_max_retries}"
        )

    def _make_oriented_grasp_plan(
        self,
        cube_pos: np.ndarray,
        hand_pos: np.ndarray | None,
    ) -> dict[str, np.ndarray | float | str] | None:
        cube_pos = np.asarray(cube_pos, dtype=float)
        hand_delta = None if hand_pos is None else np.asarray(hand_pos, dtype=float) - cube_pos
        mode = "top_down"
        hand_is_below = bool(
            hand_delta is not None
            and hand_delta[2] < -max(0.025, self.cube_half * 0.7)
        )
        if hand_is_below and self.config.handover_side_grasp_when_hand_below:
            lateral = -np.array([hand_delta[0], hand_delta[1], 0.0], dtype=float)
            if np.linalg.norm(lateral[:2]) <= 1e-4:
                ee_pos = self._end_effector_pos()
                if ee_pos is not None:
                    lateral = np.array([ee_pos[0] - cube_pos[0], ee_pos[1] - cube_pos[1], 0.0], dtype=float)
            if np.linalg.norm(lateral[:2]) <= 1e-4:
                lateral = np.array([-1.0, 0.0, 0.0], dtype=float)
            lateral = lateral / max(float(np.linalg.norm(lateral[:2])), 1e-9)
            approach_dir = np.array([lateral[0], lateral[1], 0.28], dtype=float)
            mode = "side_when_hand_below"
        else:
            approach_dir = np.array([0.0, 0.0, 1.0], dtype=float)
        approach_dir = approach_dir / max(float(np.linalg.norm(approach_dir)), 1e-9)

        hand_to_tip = self.config.handover_grasp_hand_to_fingertip_m
        pregrasp_pos = self._clamp_robot_target(
            cube_pos + approach_dir * (hand_to_tip + self.config.handover_pregrasp_offset_m)
        )
        approach_pos = self._clamp_robot_target(
            cube_pos + approach_dir * (hand_to_tip + self.config.handover_approach_offset_m)
        )
        if not self._target_is_reachable(pregrasp_pos) or not self._target_is_reachable(approach_pos):
            return None

        yaw = float(np.arctan2(approach_dir[1], approach_dir[0]))
        tilt = 2.28 if mode == "side_when_hand_below" else np.pi
        quat = _euler_xyz_to_quat_wxyz(np.array([0.0, tilt, yaw], dtype=float))
        return {
            "mode": mode,
            "approach_dir": approach_dir,
            "pregrasp_pos": pregrasp_pos,
            "approach_pos": approach_pos,
            "quat": quat,
            "yaw": yaw,
        }

    def _handover_hand_position(
        self,
        hand_positions: dict[str, np.ndarray | None],
    ) -> np.ndarray | None:
        if self.active_hand in ("left", "right"):
            pos = hand_positions.get(self.active_hand)
            if self._valid_vec3(pos):
                return np.asarray(pos, dtype=float)
        hand = self._holding_hand_for_active_cube()
        if hand is not None:
            pos = hand_positions.get(hand)
            if self._valid_vec3(pos):
                return np.asarray(pos, dtype=float)
        return self._last_handover_hand_pos

    def _get_oriented_controller(self):
        if self._oriented_controller is not None:
            return self._oriented_controller
        try:
            from omni.isaac.franka.controllers import RMPFlowController

            self._oriented_controller = RMPFlowController(
                name="handover_oriented_rmpflow_controller",
                robot_articulation=self.robot,
            )
            print("[Handover] oriented RMPFlow controller ready", flush=True)
        except Exception as exc:
            self._log_event("handover_oriented_fallback", f"reason=rmpflow_unavailable,error={exc}")
            print(f"[Handover] oriented RMPFlow unavailable: {exc}", flush=True)
            self._oriented_controller = None
        return self._oriented_controller

    def _move_oriented_ee(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        gripper_command: str | None,
    ) -> bool:
        controller = self._get_oriented_controller()
        if controller is None:
            return False
        ee_pos_before = self._end_effector_pos()
        gripper_action = None
        try:
            arm_action = controller.forward(
                target_end_effector_position=np.asarray(target_pos, dtype=float),
                target_end_effector_orientation=np.asarray(target_quat, dtype=float),
            )
            if gripper_command is not None:
                arm_action, gripper_action, merged = self._merge_gripper_command(
                    arm_action,
                    gripper_command,
                )
                self._record_learning_action(
                    target_pos=np.asarray(target_pos, dtype=float),
                    target_quat=np.asarray(target_quat, dtype=float),
                    gripper_command=gripper_command,
                    ee_pos_at_command=ee_pos_before,
                    applied_action=arm_action,
                    gripper_action=gripper_action,
                )
                self.robot.apply_action(arm_action)
                if not merged and gripper_action is not None:
                    self._apply_gripper_command_direct(gripper_command)
            else:
                self._record_learning_action(
                    target_pos=np.asarray(target_pos, dtype=float),
                    target_quat=np.asarray(target_quat, dtype=float),
                    gripper_command=None,
                    ee_pos_at_command=ee_pos_before,
                    applied_action=arm_action,
                    gripper_action=None,
                )
                self.robot.apply_action(arm_action)
        except Exception as exc:
            self._log_event("handover_oriented_fallback", f"reason=move_error,error={exc}")
            self._oriented_fallback = True
            return False
        ee_pos = self._end_effector_pos()
        if ee_pos is None:
            return False
        tolerance = max(0.005, self.config.handover_move_reached_tolerance_m)
        return bool(float(np.linalg.norm(ee_pos - np.asarray(target_pos, dtype=float))) <= tolerance)

    def _merge_gripper_command(self, arm_action, command: str):
        try:
            gripper_action = self.robot.gripper.forward(action=command)
        except Exception as exc:
            if self._robot_steps % 30 == 0:
                print(f"[Handover] gripper command failed action={command}: {exc}", flush=True)
            return arm_action, None, True
        merged = self._merge_articulation_actions(arm_action, gripper_action)
        if merged is not None:
            return merged, gripper_action, True
        if self._robot_steps % 30 == 0:
            print(
                f"[Handover] gripper action sent separately action={command}",
                flush=True,
            )
        return arm_action, gripper_action, False

    def _apply_gripper_command_direct(self, command: str) -> None:
        try:
            method = getattr(self.robot.gripper, command)
            method()
        except Exception as exc:
            if self._robot_steps % 30 == 0:
                print(
                    f"[Handover] direct gripper command failed action={command}: {exc}",
                    flush=True,
                )

    def _merge_articulation_actions(self, arm_action, gripper_action):
        try:
            arm_positions = getattr(arm_action, "joint_positions", None)
            grip_positions = getattr(gripper_action, "joint_positions", None)
            if arm_positions is None or grip_positions is None:
                return None
            arm_positions = np.asarray(arm_positions, dtype=float).reshape(-1)
            grip_positions = np.asarray(grip_positions, dtype=float).reshape(-1)
            if arm_positions.size == 0 or grip_positions.size == 0:
                return None

            arm_indices = getattr(arm_action, "joint_indices", None)
            grip_indices = getattr(gripper_action, "joint_indices", None)
            if arm_indices is None:
                if hasattr(self.robot, "num_dof") and arm_positions.size == int(self.robot.num_dof):
                    merged_positions = arm_positions.copy()
                    resolved_grip_indices = self._resolve_action_indices(
                        grip_indices,
                        grip_positions.size,
                        fallback=self._gripper_joint_indices(),
                    )
                    if resolved_grip_indices is None:
                        return None
                    for idx, value in zip(resolved_grip_indices, grip_positions):
                        merged_positions[int(idx)] = float(value)
                    return arm_action.__class__(joint_positions=merged_positions)
                arm_indices = np.arange(arm_positions.size, dtype=int)
            else:
                arm_indices = np.asarray(arm_indices, dtype=int).reshape(-1)

            resolved_grip_indices = self._resolve_action_indices(
                grip_indices,
                grip_positions.size,
                fallback=self._gripper_joint_indices(),
            )
            if resolved_grip_indices is None:
                return None

            values_by_index = {
                int(idx): float(value)
                for idx, value in zip(arm_indices, arm_positions)
            }
            for idx, value in zip(resolved_grip_indices, grip_positions):
                values_by_index[int(idx)] = float(value)
            merged_indices = np.array(list(values_by_index.keys()), dtype=int)
            merged_positions = np.array(
                [values_by_index[int(idx)] for idx in merged_indices],
                dtype=float,
            )
            return arm_action.__class__(
                joint_positions=merged_positions,
                joint_indices=merged_indices,
            )
        except Exception:
            return None

    def _resolve_action_indices(
        self,
        indices,
        value_count: int,
        *,
        fallback: np.ndarray | None,
    ) -> np.ndarray | None:
        if indices is not None:
            arr = np.asarray(indices, dtype=int).reshape(-1)
            if arr.size >= value_count:
                return arr[:value_count]
        if fallback is not None and fallback.size >= value_count:
            return fallback[:value_count]
        return None

    def _gripper_joint_indices(self) -> np.ndarray | None:
        names = getattr(self.robot, "dof_names", None)
        if not names:
            return None
        indices = [
            idx
            for idx, name in enumerate(names)
            if "finger" in str(name).lower()
        ]
        if not indices:
            return None
        return np.asarray(indices, dtype=int)

    def _advance_oriented_stage(self, stage: str) -> None:
        self._oriented_stage = stage
        self._oriented_stage_steps = 0
        if stage == ORIENT_STAGE_CLOSE:
            if self._oriented_plan is not None:
                self._oriented_close_hold_pos = np.asarray(
                    self._oriented_plan["approach_pos"],
                    dtype=float,
                )
            else:
                self._oriented_close_hold_pos = self._end_effector_pos()
            self._suppress_active_hand_collision()
        if stage == ORIENT_STAGE_PREGRASP:
            event_name = "handover_oriented_pregrasp"
        elif stage == ORIENT_STAGE_ATTACH:
            event_name = "handover_oriented_attach"
        elif stage == ORIENT_STAGE_VERIFY:
            event_name = "handover_physical_grasp_verify"
        else:
            event_name = ""
        if event_name:
            self._log_event(event_name, f"cube={self.active_cube.name},stage={stage}")

    def _oriented_close_target(self) -> np.ndarray:
        if self._oriented_close_hold_pos is not None:
            return np.asarray(self._oriented_close_hold_pos, dtype=float)
        if self._oriented_plan is not None:
            return np.asarray(self._oriented_plan["approach_pos"], dtype=float)
        pos = self._end_effector_pos()
        if pos is not None:
            return pos
        return np.asarray(self._handover_pick_pos, dtype=float)

    def _attach_robot_to_active_cube(self) -> bool:
        if not self._oriented_attach_ready():
            return False
        grasp_mode = "fixed_joint" if self.config.robot_fixed_joint_grasp else "physical_gripper"
        if self.config.robot_fixed_joint_grasp:
            joint_path = self._create_robot_cube_fixed_joint()
            if not joint_path:
                self._log_event(
                    "handover_oriented_attach",
                    f"cube={self.active_cube.name},status=joint_failed",
                )
                return False
            self._robot_grasp_joint_path = joint_path
        else:
            self._set_cube_collision_enabled(self.active_cube, True)
            if hasattr(self.active_cube, "enable_rigid_body_physics"):
                try:
                    self.active_cube.enable_rigid_body_physics()
                except Exception:
                    pass
            self._zero_cube_velocity(self.active_cube)
        if self.config.release_human_on_robot_grasp:
            self._suppress_active_hand_collision()
            self._suppress_active_human_regrab()
            self._force_release_active_cube()
        holding_hand = self._holding_hand_for_active_cube()
        if holding_hand is not None:
            self._log_event(
                "handover_oriented_attach",
                f"cube={self.active_cube.name},status=rejected,human_still_holding={holding_hand}",
            )
            self._remove_robot_grasp_joint()
            self._restore_active_hand_collision()
            return False
        if self.config.robot_fixed_joint_grasp and self.config.robot_disable_cube_collision_during_carry:
            self._set_cube_collision_enabled(self.active_cube, False)
            self._zero_cube_velocity(self.active_cube)
        self._robot_has_taken_over = True
        self._physics_armed_for_robot = True
        self._pulse("robot_takeover", hand=self.active_hand, intensity=100)
        self._log_event(
            "handover_oriented_attach",
            (
                f"cube={self.active_cube.name},status={grasp_mode},"
                f"dist={self._gripper_cube_min_dist():.3f}"
            ),
        )
        print(
            f"[Handover] oriented grasp takeover cube={self.active_cube.name} mode={grasp_mode}",
            flush=True,
        )
        return True

    def _begin_physical_grasp_verify(self, *, allow_loose: bool = False) -> bool:
        strict_ready = self._oriented_attach_ready()
        loose_ready = allow_loose and self._gripper_ready_to_release_human()
        if not strict_ready and not loose_ready:
            return False
        mode = "loose_detach" if loose_ready else "strict"
        return self._start_physical_grasp_verify(mode=mode)

    def _try_begin_human_detach_for_physical_grasp(
        self,
        metrics: dict[str, float] | None = None,
        *,
        reason: str,
        allow_loose: bool = True,
    ) -> bool:
        if self._begin_physical_grasp_verify(allow_loose=allow_loose):
            return True
        metrics = metrics or self._gripper_grasp_metrics()
        if not self._gripper_ready_to_force_detach_human(metrics):
            if self._robot_steps % 15 == 0:
                print(
                    "[Handover] waiting_force_detach_ready "
                    f"reason={reason} {self._grasp_metrics_text(metrics)}",
                    flush=True,
                )
            return False
        return self._start_physical_grasp_verify(mode=f"forced_{reason}", metrics=metrics)

    def _start_physical_grasp_verify(
        self,
        *,
        mode: str,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return False
        metrics = metrics or self._gripper_grasp_metrics(cube_pos)
        if self.config.release_human_on_robot_grasp:
            self._suppress_active_hand_collision()
            self._suppress_active_human_regrab()
        self._set_cube_collision_enabled(self.active_cube, True)
        if hasattr(self.active_cube, "enable_rigid_body_physics"):
            try:
                self.active_cube.enable_rigid_body_physics()
            except Exception:
                pass
        self._zero_cube_velocity(self.active_cube)
        if (
            self.config.handover_fingertip_joint_takeover_enabled
            and self._robot_grasp_joint_path is None
        ):
            joint_path = self._create_robot_cube_fingertip_custody_joint()
            if joint_path:
                self._robot_grasp_joint_path = joint_path
            else:
                self._log_event(
                    "handover_fingertip_joint_failed",
                    f"cube={self.active_cube.name},mode={mode}",
                )
        if self.config.release_human_on_robot_grasp:
            self._force_release_active_cube()
            self._maintain_active_hand_collision_suppression()
            self._zero_cube_velocity(self.active_cube)
        self._physical_grasp_release_pos = cube_pos.copy()
        self._physical_grasp_stable_steps = 0
        self._log_event(
            "handover_physical_grasp_release",
            (
                f"cube={self.active_cube.name},"
                f"mode={mode},"
                f"pos={self._fmt(cube_pos)},metrics={self._grasp_metrics_text(metrics)}"
            ),
        )
        print(
            "[Handover] physical grasp release started "
            f"cube={self.active_cube.name} mode={mode}",
            flush=True,
        )
        return True

    def _update_physical_grasp_verify(self) -> tuple[bool, str | None]:
        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return False, "cube_pose_missing"
        holding_hand = self._holding_hand_for_active_cube()
        if holding_hand is not None:
            if self.config.release_human_on_robot_grasp:
                self._suppress_active_human_regrab()
                self._force_release_active_cube()
                holding_hand = self._holding_hand_for_active_cube()
            if holding_hand is not None:
                self._physical_grasp_stable_steps = 0
                if self._robot_steps % 15 == 0:
                    print(
                        "[Handover] grasp verify blocked: "
                        f"cube still held by {holding_hand} hand",
                        flush=True,
                    )
                if (
                    self._oriented_stage_steps
                    >= self.config.handover_grasp_human_release_timeout_steps
                ):
                    return False, f"human_still_holding_{holding_hand}"
                return False, None
        if self._robot_grasp_joint_path:
            self._physical_grasp_stable_steps += 1
            required = min(self.config.handover_grasp_verify_steps, 4)
            if self._robot_steps % 15 == 0:
                print(
                    "[Handover] verifying_fingertip_joint_grasp "
                    f"stable={self._physical_grasp_stable_steps}/{required} "
                    f"joint={self._robot_grasp_joint_path}",
                    flush=True,
                )
            if self._physical_grasp_stable_steps >= required:
                return True, None
            return False, None
        metrics = self._gripper_grasp_metrics(cube_pos)
        stable = self._gripper_physical_grasp_stable(metrics)
        if self._physical_grasp_release_pos is not None:
            drop = float(self._physical_grasp_release_pos[2] - cube_pos[2])
            if drop > self.config.handover_grasp_verify_max_drop_m:
                return False, f"cube_dropped_{drop:.3f}m"
        if stable:
            self._physical_grasp_stable_steps += 1
        else:
            self._physical_grasp_stable_steps = 0

        if self._robot_steps % 15 == 0:
            print(
                "[Handover] verifying_physical_grasp "
                f"stable={self._physical_grasp_stable_steps}/{self.config.handover_grasp_verify_steps} "
                f"{self._grasp_metrics_text(metrics)}",
                flush=True,
            )
        if self._physical_grasp_stable_steps >= self.config.handover_grasp_verify_steps:
            return True, None
        if self._oriented_stage_steps > self.config.handover_grasp_verify_timeout_steps:
            return False, f"verify_timeout {self._grasp_metrics_text(metrics)}"
        return False, None

    def _confirm_physical_robot_grasp(self) -> bool:
        holding_hand = self._holding_hand_for_active_cube()
        if holding_hand is not None:
            self._log_event(
                "handover_physical_grasp_rejected",
                f"cube={self.active_cube.name},reason=human_still_holding_{holding_hand}",
            )
            print(
                "[Handover] physical grasp rejected: "
                f"cube still held by {holding_hand} hand",
                flush=True,
            )
            return False
        self._robot_has_taken_over = True
        self._physics_armed_for_robot = True
        self._pulse("robot_takeover", hand=self.active_hand, intensity=100)
        self._log_event(
            "handover_physical_grasp_verified",
            (
                f"cube={self.active_cube.name},"
                f"stable_steps={self._physical_grasp_stable_steps},"
                f"metrics={self._grasp_metrics_text()}"
            ),
        )
        print(
            f"[Handover] physical grasp verified cube={self.active_cube.name}",
            flush=True,
        )
        return True

    def _oriented_attach_ready(self) -> bool:
        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return False
        metrics = self._gripper_grasp_metrics(cube_pos)
        ready = self._gripper_contains_cube(metrics)
        if not ready and self._robot_steps % 30 == 0:
            print(
                "[Handover] waiting_oriented_attach "
                f"stage={self._oriented_stage} "
                f"{self._grasp_metrics_text(metrics)}",
                flush=True,
            )
        return ready

    def _gripper_contains_cube(self, metrics: dict[str, float]) -> bool:
        return bool(
            metrics["center_dist"] <= self.config.handover_grasp_attach_center_dist_m
            and metrics["min_dist"] <= self.config.handover_grasp_attach_dist_m
            and metrics["width"] <= self.config.handover_grasp_attach_width_m
            and metrics["jaw_offset"] <= self.config.handover_grasp_jaw_axis_max_m
            and metrics["vertical_offset"] <= self.config.handover_grasp_vertical_max_m
            and metrics["depth_offset"] <= self.config.handover_grasp_depth_axis_max_m
        )

    def _gripper_ready_to_start_close(self) -> bool:
        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return False
        metrics = self._gripper_grasp_metrics(cube_pos)
        depth_offset = metrics["depth_offset"]
        depth_ok = (
            not np.isfinite(depth_offset)
            or depth_offset <= max(0.090, self.config.handover_grasp_depth_axis_max_m * 2.25)
        )
        return bool(
            metrics["center_dist"] <= max(0.10, self.config.handover_grasp_attach_center_dist_m * 1.8)
            and metrics["min_dist"] <= max(0.09, self.config.handover_grasp_attach_dist_m * 1.4)
            and metrics["jaw_offset"] <= max(0.055, self.config.handover_grasp_jaw_axis_max_m * 1.5)
            and metrics["vertical_offset"] <= max(0.09, self.config.handover_grasp_vertical_max_m * 1.3)
            and depth_ok
        )

    def _gripper_ready_to_release_human(self) -> bool:
        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return False
        metrics = self._gripper_grasp_metrics(cube_pos)
        ready = bool(
            metrics["center_dist"] <= max(0.115, self.config.handover_grasp_attach_center_dist_m * 2.0)
            and metrics["min_dist"] <= max(0.10, self.config.handover_grasp_attach_dist_m * 1.6)
            and metrics["width"] <= max(0.085, self.config.handover_grasp_attach_width_m * 1.6)
            and metrics["jaw_offset"] <= max(0.065, self.config.handover_grasp_jaw_axis_max_m * 1.8)
            and metrics["vertical_offset"] <= max(0.09, self.config.handover_grasp_vertical_max_m * 1.4)
        )
        if not ready and self._robot_steps % 30 == 0:
            print(
                "[Handover] waiting_human_detach_ready "
                f"{self._grasp_metrics_text(metrics)}",
                flush=True,
            )
        return ready

    def _gripper_ready_to_force_detach_human(
        self,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        metrics = metrics or self._gripper_grasp_metrics()
        return bool(
            metrics["center_dist"] <= max(0.090, self.config.handover_grasp_attach_center_dist_m * 1.45)
            and metrics["min_dist"] <= max(0.090, self.config.handover_grasp_attach_dist_m * 1.20)
            and metrics["width"] <= max(0.090, self.config.handover_grasp_attach_width_m * 1.50)
            and metrics["jaw_offset"] <= max(0.055, self.config.handover_grasp_jaw_axis_max_m * 1.45)
            and metrics["vertical_offset"] <= max(0.085, self.config.handover_grasp_vertical_max_m * 1.25)
        )

    def _gripper_physical_grasp_stable(self, metrics: dict[str, float]) -> bool:
        depth_offset = metrics["depth_offset"]
        depth_ok = (
            not np.isfinite(depth_offset)
            or depth_offset <= max(0.085, self.config.handover_grasp_depth_axis_max_m * 2.2)
        )
        return bool(
            metrics["center_dist"] <= max(0.095, self.config.handover_grasp_attach_center_dist_m * 1.50)
            and metrics["min_dist"] <= max(0.095, self.config.handover_grasp_attach_dist_m * 1.25)
            and metrics["width"] <= max(0.090, self.config.handover_grasp_attach_width_m * 1.50)
            and metrics["jaw_offset"] <= max(0.060, self.config.handover_grasp_jaw_axis_max_m * 1.55)
            and metrics["vertical_offset"] <= max(0.090, self.config.handover_grasp_vertical_max_m * 1.30)
            and depth_ok
        )

    def _gripper_grasp_metrics(self, cube_pos: np.ndarray | None = None) -> dict[str, float]:
        if cube_pos is None:
            cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return {
                "center_dist": float("inf"),
                "min_dist": float("inf"),
                "width": float("inf"),
                "jaw_offset": float("inf"),
                "vertical_offset": float("inf"),
                "depth_offset": float("inf"),
                "head_dist": 0.0,
                "jaw_width_world": float("inf"),
            }
        fingertips = self._gripper_fingertip_poses()
        center = fingertips["center"]
        min_dist = self._gripper_cube_min_dist(cube_pos)
        width = self._gripper_width()
        jaw_offset = float("inf")
        vertical_offset = float("inf")
        depth_offset = float("inf")
        head_dist = float("inf")
        jaw_width_world = float("inf")
        if center is not None:
            rel = np.asarray(cube_pos, dtype=float) - np.asarray(center, dtype=float)
            vertical_offset = abs(float(rel[2]))
            left_pos = fingertips["left"]
            right_pos = fingertips["right"]
            if left_pos is not None and right_pos is not None:
                jaw_vec = np.asarray(right_pos, dtype=float) - np.asarray(left_pos, dtype=float)
                jaw_width_world = float(np.linalg.norm(jaw_vec))
                if jaw_width_world > 1e-6:
                    jaw_axis = jaw_vec / jaw_width_world
                    jaw_offset = abs(float(np.dot(rel, jaw_axis)))
                else:
                    jaw_offset = float(np.linalg.norm(rel[:2]))
            else:
                jaw_offset = float(np.linalg.norm(rel[:2]))
            depth_axis = fingertips["depth_axis"]
            if depth_axis is None:
                depth_axis = self._fallback_gripper_depth_axis(fingertips)
            if depth_axis is not None:
                depth_offset = abs(float(np.dot(rel, np.asarray(depth_axis, dtype=float))))
            center_dist = float(np.linalg.norm(rel))
        else:
            center_dist = float("inf")
        hand_pos = fingertips["hand"]
        if hand_pos is not None:
            head_dist = float(np.linalg.norm(np.asarray(cube_pos, dtype=float) - hand_pos))
        return {
            "center_dist": center_dist,
            "min_dist": min_dist,
            "width": width,
            "jaw_offset": jaw_offset,
            "vertical_offset": vertical_offset,
            "depth_offset": depth_offset,
            "head_dist": head_dist,
            "jaw_width_world": jaw_width_world,
        }

    def _grasp_metrics_text(self, metrics: dict[str, float] | None = None) -> str:
        if metrics is None:
            metrics = self._gripper_grasp_metrics()
        return (
            f"center={metrics['center_dist']:.3f}/{self.config.handover_grasp_attach_center_dist_m:.3f} "
            f"dist={metrics['min_dist']:.3f}/{self.config.handover_grasp_attach_dist_m:.3f} "
            f"width={metrics['width']:.3f}/{self.config.handover_grasp_attach_width_m:.3f} "
            f"jaw={metrics['jaw_offset']:.3f}/{self.config.handover_grasp_jaw_axis_max_m:.3f} "
            f"z={metrics['vertical_offset']:.3f}/{self.config.handover_grasp_vertical_max_m:.3f} "
            f"depth={metrics['depth_offset']:.3f}/{self.config.handover_grasp_depth_axis_max_m:.3f} "
            f"head={metrics['head_dist']:.3f}>={self.config.handover_grasp_head_clearance_m:.3f}"
        )

    def _fallback_gripper_depth_axis(
        self,
        fingertips: dict[str, np.ndarray | None],
    ) -> np.ndarray | None:
        if self._oriented_plan is not None:
            approach_dir = self._oriented_plan.get("approach_dir")
            if self._valid_vec3(approach_dir):
                axis = np.asarray(approach_dir, dtype=float)
                norm = float(np.linalg.norm(axis))
                if norm > 1e-6:
                    return axis / norm

        center = fingertips.get("center")
        hand = fingertips.get("hand")
        if self._valid_vec3(center) and self._valid_vec3(hand):
            axis = np.asarray(center, dtype=float) - np.asarray(hand, dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-6:
                return axis / norm
        return None

    def _gripper_cube_min_dist(self, cube_pos: np.ndarray | None = None) -> float:
        if cube_pos is None:
            cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return float("inf")
        points = []
        fingertips = self._gripper_fingertip_poses()
        center = fingertips["center"]
        if center is not None:
            points.append(center)
        for pos in (fingertips["left"], fingertips["right"]):
            if pos is not None:
                points.append(np.asarray(pos, dtype=float))
        if not points:
            ee_pos = self._end_effector_pos()
            if ee_pos is not None:
                points.append(ee_pos)
        if not points:
            return float("inf")
        return float(min(np.linalg.norm(point - cube_pos) for point in points))

    def _use_fingertip_takeover_path(self) -> bool:
        return bool(
            self.config.physical_mode
            and self.config.handover_fingertip_joint_takeover_enabled
        )

    def _gripper_fingertip_poses(self) -> dict[str, np.ndarray | None]:
        hand_pos = self._end_effector_pos()
        left_pos = None
        right_pos = None
        try:
            left_raw, _ = self.robot.gripper._left_finger.get_world_pose()
            if self._valid_vec3(left_raw):
                left_pos = np.asarray(left_raw, dtype=float)
        except Exception:
            pass
        try:
            right_raw, _ = self.robot.gripper._right_finger.get_world_pose()
            if self._valid_vec3(right_raw):
                right_pos = np.asarray(right_raw, dtype=float)
        except Exception:
            pass

        center = None
        depth_axis = None
        if left_pos is not None and right_pos is not None:
            finger_origin_center = (left_pos + right_pos) * 0.5
            center = finger_origin_center
            if hand_pos is not None:
                axis = finger_origin_center - hand_pos
                norm = float(np.linalg.norm(axis))
                if norm > 1e-6:
                    depth_axis = axis / norm
                    offset = self.config.handover_grasp_fingertip_offset_m
                    left_pos = left_pos + depth_axis * offset
                    right_pos = right_pos + depth_axis * offset
                    center = (left_pos + right_pos) * 0.5
        elif hand_pos is not None:
            center = hand_pos

        return {
            "left": left_pos,
            "right": right_pos,
            "center": center,
            "hand": hand_pos,
            "depth_axis": depth_axis,
        }

    def _run_oriented_lift_place(self) -> tuple[bool, str]:
        if self._use_fingertip_takeover_path() or not self.config.robot_fixed_joint_grasp:
            return self._run_physical_carry_place()

        done = self._run_pick_place(
            controller=self.controller,
            robot=self.robot,
            pick_position=self._handover_pick_pos,
            place_position=self.active_place_pos,
        )
        if done or self._physical_robot_place_done() or (
            self._controller_event() >= 8 and self._cube_is_at_place()
        ):
            self._complete_physical_robot_place()
            done = True
        self._set_place_target_pose()
        if done:
            return True, f"robot_place_done cube={self.active_cube.name}"
        if self._robot_steps > self.config.max_robot_takeover_steps:
            self._log_event(
                "handover_timeout",
                f"cube={self.active_cube.name},steps={self._robot_steps},mode=oriented",
            )
            self._reset_after_failed_takeover("oriented_lift_place_timeout")
            return False, "robot_takeover_timeout_respawn"
        return False, f"{ORIENT_STAGE_LIFT_PLACE} event={self._controller_event()} steps={self._robot_steps}"

    def _run_physical_carry_place(self) -> tuple[bool, str]:
        if self._physical_carry_stage is None:
            self._set_physical_carry_stage("lift")

        self._physical_carry_stage_steps += 1
        lost_reason = self._physical_carry_lost_reason()
        if lost_reason is not None and self._physical_carry_stage != "release":
            self._physical_carry_lost_steps += 1
            if self._physical_carry_lost_steps >= self.config.handover_physical_carry_lost_steps:
                self._log_event(
                    "handover_physical_carry_failed",
                    f"cube={self.active_cube.name},reason={lost_reason}",
                )
                print(f"[Handover] physical carry failed: {lost_reason}", flush=True)
                self._reset_after_failed_takeover(f"physical_carry_failed:{lost_reason}")
                return False, "physical_carry_failed_respawn"
        else:
            self._physical_carry_lost_steps = 0

        quat = self._physical_carry_quat()
        stage = self._physical_carry_stage
        if stage == "lift":
            target = self._physical_lift_target()
            reached = self._move_oriented_ee(target, quat, gripper_command="close")
            if reached or self._physical_carry_stage_steps > self.config.handover_physical_stage_timeout_steps:
                self._set_physical_carry_stage("carry")
            return False, self._physical_carry_status(target)

        if stage == "carry":
            target = self._physical_carry_target()
            reached = self._move_oriented_ee(target, quat, gripper_command="close")
            if reached or self._physical_carry_stage_steps > self.config.handover_physical_stage_timeout_steps:
                self._set_physical_carry_stage("descend")
            return False, self._physical_carry_status(target)

        if stage == "descend":
            target = self._physical_release_target()
            reached = self._move_oriented_ee(target, quat, gripper_command="close")
            if reached or self._physical_carry_stage_steps > self.config.handover_physical_stage_timeout_steps:
                self._set_physical_carry_stage("release")
            return False, self._physical_carry_status(target)

        if stage == "release":
            target = self._physical_release_target()
            self._move_oriented_ee(target, quat, gripper_command="open")
            if self._physical_carry_stage_steps >= self.config.handover_physical_release_hold_steps:
                self._log_event(
                    "handover_physical_place_complete",
                    f"cube={self.active_cube.name},place={self._fmt(self.active_place_pos)}",
                )
                self._mark_learning_success("handover_physical_place_complete")
                return True, f"robot_place_done cube={self.active_cube.name}"
            return False, self._physical_carry_status(target)

        self._set_physical_carry_stage("lift")
        return False, "physical_carry_recovered"

    def _set_physical_carry_stage(self, stage: str) -> None:
        self._physical_carry_stage = stage
        self._physical_carry_stage_steps = 0
        if stage == "lift":
            ee_pos = self._end_effector_pos()
            base_z = float(ee_pos[2]) if ee_pos is not None else float(self.active_place_pos[2])
            z_max = self.table_top_z + self.config.workspace_z_above_table[1]
            self._physical_carry_lift_z = min(
                z_max,
                max(base_z + self.config.handover_physical_lift_m, self.active_place_pos[2] + 0.12),
            )
        if stage == "descend":
            self._physical_carry_release_target = None
        self._log_event(
            "handover_physical_carry_stage",
            f"cube={self.active_cube.name},stage={stage}",
        )
        print(f"[Handover] physical carry stage={stage}", flush=True)

    def _physical_carry_quat(self) -> np.ndarray:
        if self._oriented_plan is not None:
            return np.asarray(self._oriented_plan["quat"], dtype=float)
        return _euler_xyz_to_quat_wxyz(np.array([0.0, np.pi, 0.0], dtype=float))

    def _physical_lift_target(self) -> np.ndarray:
        ee_pos = self._end_effector_pos()
        if ee_pos is None:
            ee_pos = self._oriented_close_target()
        target = np.asarray(ee_pos, dtype=float).copy()
        if self._physical_carry_lift_z is None:
            self._set_physical_carry_stage("lift")
        target[2] = float(self._physical_carry_lift_z)
        return self._clamp_robot_target(target)

    def _physical_carry_target(self) -> np.ndarray:
        ee_pos = self._end_effector_pos()
        if ee_pos is None:
            ee_pos = self._physical_lift_target()
        target = np.asarray(ee_pos, dtype=float).copy()
        target[0] = float(self.active_place_pos[0])
        target[1] = float(self.active_place_pos[1])
        target[2] = float(self._physical_carry_lift_z or target[2])
        return self._clamp_robot_target(target)

    def _physical_release_target(self) -> np.ndarray:
        if self._physical_carry_release_target is not None:
            return self._physical_carry_release_target
        target = np.asarray(self.active_place_pos, dtype=float).copy()
        target[2] += self.config.handover_physical_place_above_m
        self._physical_carry_release_target = self._clamp_robot_target(target)
        return self._physical_carry_release_target

    def _physical_carry_lost_reason(self) -> str | None:
        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return "cube_pose_missing"
        if self._robot_grasp_joint_path:
            return None
        metrics = self._gripper_grasp_metrics(cube_pos)
        if metrics["center_dist"] > self.config.handover_physical_carry_lost_center_m:
            return (
                f"center_lost {metrics['center_dist']:.3f}/"
                f"{self.config.handover_physical_carry_lost_center_m:.3f}"
            )
        head_min = max(0.04, self.config.handover_grasp_head_clearance_m * 0.65)
        if metrics["head_dist"] < head_min:
            return (
                f"head_contact {metrics['head_dist']:.3f}/"
                f"{self.config.handover_grasp_head_clearance_m:.3f}"
            )
        depth_max = max(0.08, self.config.handover_grasp_depth_axis_max_m * 2.0)
        if metrics["depth_offset"] > depth_max:
            return (
                f"depth_lost {metrics['depth_offset']:.3f}/"
                f"{self.config.handover_grasp_depth_axis_max_m:.3f}"
            )
        if (
            self._physical_carry_stage in ("lift", "carry")
            and self._physical_grasp_release_pos is not None
        ):
            drop = float(self._physical_grasp_release_pos[2] - cube_pos[2])
            if drop > self.config.handover_grasp_verify_max_drop_m:
                return f"cube_dropped_{drop:.3f}m"
        return None

    def _physical_carry_status(self, target: np.ndarray) -> str:
        if self._robot_steps % 30 == 0:
            print(
                "[Handover] physical_carry "
                f"stage={self._physical_carry_stage} "
                f"target={self._fmt(target)} "
                f"{self._grasp_metrics_text()}",
                flush=True,
            )
        return (
            f"physical_carry stage={self._physical_carry_stage} "
            f"steps={self._physical_carry_stage_steps}"
        )

    def _fallback_to_pick_controller(self, reason: str) -> tuple[bool, str]:
        self._oriented_fallback = True
        self._oriented_stage = None
        self._oriented_stage_steps = 0
        self._oriented_plan = None
        self._reset_controller()
        self._log_event("handover_oriented_fallback", f"cube={self.active_cube.name},reason={reason}")
        print(f"[Handover] oriented grasp fallback: {reason}", flush=True)
        return self._run_robot_takeover({})

    def _clamp_robot_target(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=float).copy()
        z_min = self.table_top_z + self.config.workspace_z_above_table[0]
        z_max = self.table_top_z + self.config.workspace_z_above_table[1]
        pos[0] = np.clip(pos[0], *self.config.workspace_x)
        pos[1] = np.clip(pos[1], *self.config.workspace_y)
        pos[2] = np.clip(pos[2], z_min, z_max)
        return pos

    def _target_is_reachable(self, pos: np.ndarray) -> bool:
        pos = np.asarray(pos, dtype=float)
        z_min = self.table_top_z + self.config.workspace_z_above_table[0]
        z_max = self.table_top_z + self.config.workspace_z_above_table[1]
        return bool(
            self.config.workspace_x[0] <= pos[0] <= self.config.workspace_x[1]
            and self.config.workspace_y[0] <= pos[1] <= self.config.workspace_y[1]
            and z_min <= pos[2] <= z_max
        )

    def _safe_table_position(self) -> np.ndarray:
        return np.array(
            [
                np.clip(0.42, *self.config.workspace_x),
                np.clip(0.0, *self.config.workspace_y),
                self.table_top_z + self.cube_half,
            ],
            dtype=float,
        )

    def _set_place_target_pose(self) -> None:
        try:
            self.place_target.set_world_pose(position=self.active_place_pos)
        except Exception:
            pass

    def _place_cube_for_human(self, pos: np.ndarray) -> None:
        self._set_cube_pose(
            self.active_cube,
            pos,
            physics_enabled=self.config.human_cube_physics,
            collision_enabled=True,
        )

    def _place_cube_for_robot(self, pos: np.ndarray, *, physics_enabled: bool) -> None:
        collision_enabled = physics_enabled or self.config.robot_approach_collision
        self._set_cube_pose(
            self.active_cube,
            pos,
            physics_enabled=physics_enabled,
            collision_enabled=collision_enabled,
        )

    def _set_cube_pose(
        self,
        cube,
        pos: np.ndarray,
        *,
        physics_enabled: bool,
        collision_enabled: bool = True,
    ) -> None:
        pos = np.asarray(pos, dtype=float)
        try:
            if physics_enabled:
                if hasattr(cube, "disable_rigid_body_physics"):
                    cube.disable_rigid_body_physics()
                if hasattr(cube, "set_default_state"):
                    cube.set_default_state(position=pos)
                cube.set_world_pose(position=pos)
                if hasattr(cube, "enable_rigid_body_physics"):
                    cube.enable_rigid_body_physics()
                    self._zero_cube_velocity(cube)
                self._set_cube_collision_enabled(cube, collision_enabled)
            else:
                self._set_cube_collision_enabled(cube, collision_enabled)
                if hasattr(cube, "disable_rigid_body_physics"):
                    cube.disable_rigid_body_physics()
                if hasattr(cube, "set_default_state"):
                    cube.set_default_state(position=pos)
                cube.set_world_pose(position=pos)
        except Exception as exc:
            self._log_event("handover_cube_pose_error", f"cube={cube.name},error={exc}")

    def _set_cube_collision_enabled(self, cube, enabled: bool) -> None:
        method_name = "enable_collision" if enabled else "disable_collision"
        method = getattr(cube, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass

        prim = getattr(cube, "prim", None)
        if prim is None:
            return
        try:
            from pxr import Usd, UsdGeom, UsdPhysics

            geom_types = {
                "Cube",
                "Mesh",
                "Sphere",
                "Capsule",
                "Cylinder",
                "Cone",
            }
            for child in Usd.PrimRange(prim):
                if (
                    not child.HasAPI(UsdPhysics.CollisionAPI)
                    and child.GetTypeName() not in geom_types
                    and not child.IsA(UsdGeom.Gprim)
                ):
                    continue
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    collision_api = UsdPhysics.CollisionAPI(child)
                else:
                    collision_api = UsdPhysics.CollisionAPI.Apply(child)
                attr = collision_api.GetCollisionEnabledAttr()
                if not attr.IsValid():
                    attr = collision_api.CreateCollisionEnabledAttr()
                attr.Set(bool(enabled))
        except Exception:
            pass

    def _apply_high_friction_grasp_materials(self) -> None:
        if not self.config.handover_high_friction_enabled:
            return
        material = self._get_high_friction_material()
        if material is None:
            return
        applied_paths: list[str] = []
        for obj in (
            self.active_cube,
            getattr(self.robot.gripper, "_left_finger", None),
            getattr(self.robot.gripper, "_right_finger", None),
        ):
            if obj is None:
                continue
            if self._apply_physics_material_to_object(obj, material):
                path = self._prim_path_for_object(obj)
                if path:
                    applied_paths.append(path)

        for path in (
            self._cube_prim_path(self.active_cube),
            "/World/Franka/panda_leftfinger",
            "/World/Franka/panda_rightfinger",
        ):
            if path and self._bind_physics_material_to_path(path, material):
                applied_paths.append(path)

        if not self._high_friction_material_applied:
            self._high_friction_material_applied = True
            unique_paths = tuple(dict.fromkeys(applied_paths))
            self._log_event(
                "handover_high_friction_grasp",
                (
                    f"static={self.config.handover_grasp_static_friction:.2f},"
                    f"dynamic={self.config.handover_grasp_dynamic_friction:.2f},"
                    f"paths={unique_paths}"
                ),
            )
            print(
                "[Handover] high friction grasp material applied "
                f"paths={unique_paths}",
                flush=True,
            )

    def _get_high_friction_material(self):
        if self._grasp_physics_material is not None:
            return self._grasp_physics_material
        try:
            import omni.usd
            from isaacsim.core.api.materials.physics_material import PhysicsMaterial
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is not None and not stage.GetPrimAtPath("/World/HandoverMaterials").IsValid():
                UsdGeom.Xform.Define(stage, "/World/HandoverMaterials")
            self._grasp_physics_material = PhysicsMaterial(
                prim_path="/World/HandoverMaterials/high_friction_grasp",
                name="high_friction_grasp",
                static_friction=self.config.handover_grasp_static_friction,
                dynamic_friction=self.config.handover_grasp_dynamic_friction,
                restitution=0.0,
            )
            return self._grasp_physics_material
        except Exception as exc:
            if not self._high_friction_material_applied:
                self._high_friction_material_applied = True
                self._log_event("handover_high_friction_error", f"error={exc}")
                print(f"[Handover] high friction material unavailable: {exc}", flush=True)
            return None

    def _apply_physics_material_to_object(self, obj, material) -> bool:
        apply_material = getattr(obj, "apply_physics_material", None)
        if not callable(apply_material):
            return False
        try:
            apply_material(material)
            return True
        except Exception:
            return False

    def _bind_physics_material_to_path(self, prim_path: str, material) -> bool:
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return False
            prim = stage.GetPrimAtPath(prim_path)
            if prim is None or not prim.IsValid():
                return False
            bound = False
            for child in Usd.PrimRange(prim):
                if (
                    not child.HasAPI(UsdPhysics.CollisionAPI)
                    and not child.IsA(UsdGeom.Gprim)
                ):
                    continue
                UsdShade.MaterialBindingAPI.Apply(child).Bind(material.material)
                bound = True
            return bound
        except Exception:
            return False

    def _prim_path_for_object(self, obj) -> str | None:
        prim = getattr(obj, "prim", None)
        if prim is not None:
            try:
                path = prim.GetPath()
                if path:
                    return str(path)
            except Exception:
                pass
        for attr_name in ("prim_path", "_prim_path"):
            path = getattr(obj, attr_name, None)
            if path:
                return str(path)
        return None

    def _zero_cube_velocity(self, cube) -> None:
        for method_name in ("set_linear_velocity", "set_angular_velocity"):
            if hasattr(cube, method_name):
                try:
                    getattr(cube, method_name)(np.zeros(3))
                except Exception:
                    pass

    def _cube_position(self, cube) -> np.ndarray | None:
        try:
            pos, _ = cube.get_world_pose()
        except Exception:
            return None
        if not self._valid_vec3(pos):
            return None
        return np.asarray(pos, dtype=float)

    def _holding_hand_for_active_cube(self) -> str | None:
        if self.grab_manager is not None:
            getter = getattr(self.grab_manager, "holding_hand_for", None)
            if getter is not None:
                hand = getter(self.active_cube)
                if hand is not None:
                    return hand
            for hand in ("left", "right"):
                grabbed = getattr(self.grab_manager, "grabbed", {}).get(hand)
                if self._same_cube(grabbed, self.active_cube):
                    return hand
        joint_hand = self._human_grab_joint_hand_for_active_cube()
        if joint_hand is not None:
            return joint_hand
        return None

    def _is_user_holding_active_cube(self) -> bool:
        return self._holding_hand_for_active_cube() is not None

    def _adopt_held_cube(self) -> tuple[str, Any] | None:
        if self.grab_manager is None:
            return None
        hands = tuple(
            dict.fromkeys((*self.config.hand_priority, "right", "left"))
        )
        held_getter = getattr(self.grab_manager, "held_cube", None)
        grabbed = getattr(self.grab_manager, "grabbed", {})
        for hand in hands:
            cube = held_getter(hand) if held_getter is not None else grabbed.get(hand)
            if cube is None or not self._is_handover_candidate(cube):
                continue
            previous = self._active_cube_override
            self._active_cube_override = cube
            pick_idx = self._pick_target_index(cube)
            if pick_idx is not None:
                self.current_pick_idx = pick_idx
            if previous is not cube:
                self._log_event(
                    "handover_active_cube_adopted",
                    f"hand={hand},cube={cube.name}",
                )
                print(
                    f"[Handover] adopted grabbed cube '{cube.name}' from {hand} hand",
                    flush=True,
                )
            return hand, cube
        return None

    def _is_handover_candidate(self, cube) -> bool:
        return any(self._same_cube(cube, candidate) for candidate in self.handover_candidates)

    def _pick_target_index(self, cube) -> int | None:
        for idx, target in enumerate(self.pick_targets):
            if self._same_cube(cube, target):
                return idx
        return None

    def _cycle_length(self) -> int:
        return max(1, len(self.handover_candidates))

    def _waiting_cube_label(self) -> str:
        if self.config.adopt_any_grabbed_cube and self._active_cube_override is None:
            return "any_table_cube"
        return getattr(self.active_cube, "name", "active_cube")

    def _same_cube(self, lhs, rhs) -> bool:
        if lhs is None or rhs is None:
            return False
        if lhs is rhs:
            return True
        lhs_path = self._cube_prim_path(lhs)
        rhs_path = self._cube_prim_path(rhs)
        return bool(lhs_path and rhs_path and lhs_path == rhs_path)

    def _human_grab_joint_hand_for_active_cube(self) -> str | None:
        joints = self._human_grab_joint_paths_for_active_cube()
        if not joints:
            return None
        for _, hand in joints:
            if hand in ("left", "right"):
                return hand
        return self.active_hand if self.active_hand in ("left", "right") else "unknown"

    def _human_grab_joint_paths_for_active_cube(self) -> list[tuple[str, str | None]]:
        cube_path = self._cube_prim_path(self.active_cube)
        if not cube_path:
            return []
        try:
            import omni.usd
            from pxr import Usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return []
            root = stage.GetPrimAtPath("/World/HandoverJoints")
            if root is None or not root.IsValid():
                return []

            joint_paths: list[tuple[str, str | None]] = []
            for prim in Usd.PrimRange(root):
                prim_path = str(prim.GetPath())
                if "_fixed_joint" not in prim_path:
                    continue
                body0 = self._joint_targets(prim, "physics:body0")
                body1 = self._joint_targets(prim, "physics:body1")
                all_targets = (*body0, *body1)
                if cube_path not in all_targets:
                    continue
                hand = self._hand_from_human_joint(prim_path, all_targets)
                if hand is not None:
                    joint_paths.append((prim_path, hand))
            return joint_paths
        except Exception as exc:
            self._log_event(
                "handover_human_joint_scan_error",
                f"cube={self.active_cube.name},error={exc}",
            )
            print(f"[Handover] human fixed joint scan failed: {exc}", flush=True)
            return []

    def _hand_from_human_joint(
        self,
        joint_path: str,
        targets: tuple[str, ...],
    ) -> str | None:
        robot_body = self.config.robot_grasp_body_path
        if any(target.startswith("/World/Franka") or target == robot_body for target in targets):
            return None
        for hand in ("left", "right"):
            hand_tokens = (
                f"/World/Avatar/{hand}",
                f"/{hand}_",
                f"{hand}_",
            )
            if any(target.startswith(f"/World/Avatar/{hand}") for target in targets):
                return hand
            basename = joint_path.rsplit("/", 1)[-1]
            if basename.startswith(f"{hand}_") or any(token in joint_path for token in hand_tokens):
                return hand
        if any(target.startswith("/World/Avatar/") for target in targets):
            return self.active_hand if self.active_hand in ("left", "right") else "unknown"
        return None

    def _remove_human_grab_joints_for_active_cube(self) -> int:
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return 0
            remove_paths = [path for path, _ in self._human_grab_joint_paths_for_active_cube()]
            for joint_path in remove_paths:
                stage.RemovePrim(joint_path)
                print(
                    f"[Handover] human fixed joint force-removed: {joint_path}",
                    flush=True,
                )
            return len(remove_paths)
        except Exception as exc:
            self._log_event(
                "handover_human_joint_remove_error",
                f"cube={self.active_cube.name},error={exc}",
            )
            print(f"[Handover] human fixed joint remove failed: {exc}", flush=True)
            return 0

    def _joint_targets(self, prim, relationship_name: str) -> tuple[str, ...]:
        try:
            rel = prim.GetRelationship(relationship_name)
            if rel is None or not rel.IsValid():
                return ()
            return tuple(str(target) for target in rel.GetTargets())
        except Exception:
            return ()

    def _force_release_active_cube(self) -> None:
        removed_count = self._remove_human_grab_joints_for_active_cube()
        if self.grab_manager is None:
            if removed_count:
                self._log_event(
                    "handover_human_joint_force_removed",
                    f"cube={self.active_cube.name},count={removed_count},source=no_grab_manager",
                )
            return
        force_release = getattr(self.grab_manager, "force_release", None)
        if force_release is not None:
            force_release(cube=self.active_cube)
        removed_count += self._remove_human_grab_joints_for_active_cube()
        hand = self._holding_hand_for_active_cube()
        if hand in ("left", "right") and hasattr(self.grab_manager, "_release"):
            self.grab_manager._release(hand)
            prev_grip = getattr(self.grab_manager, "prev_grip", None)
            if isinstance(prev_grip, dict):
                prev_grip[hand] = False
            removed_count += self._remove_human_grab_joints_for_active_cube()
            hand = self._holding_hand_for_active_cube()
        if hand is not None:
            grabbed = getattr(self.grab_manager, "grabbed", {})
            release = getattr(self.grab_manager, "_release", None)
            if callable(release):
                for candidate_hand in ("left", "right"):
                    if self._same_cube(grabbed.get(candidate_hand), self.active_cube):
                        release(candidate_hand)
                        prev_grip = getattr(self.grab_manager, "prev_grip", None)
                        if isinstance(prev_grip, dict):
                            prev_grip[candidate_hand] = False
            removed_count += self._remove_human_grab_joints_for_active_cube()
            hand = self._holding_hand_for_active_cube()
        if removed_count:
            self._log_event(
                "handover_human_joint_force_removed",
                f"cube={self.active_cube.name},count={removed_count}",
            )
        if hand is not None:
            self._log_event(
                "handover_human_release_failed",
                f"cube={self.active_cube.name},hand={hand}",
            )
            print(
                f"[Handover] human release still held by {hand} hand",
                flush=True,
            )

    def _suppress_active_human_regrab(self) -> None:
        if self.grab_manager is None:
            return
        suppress = getattr(self.grab_manager, "set_grab_suppressed", None)
        if not callable(suppress):
            return
        hand = self._holding_hand_for_active_cube()
        if hand not in ("left", "right"):
            hand = self.active_hand if self.active_hand in ("left", "right") else None
        try:
            changed = bool(suppress(hand=hand, cube=self.active_cube, suppressed=True))
        except Exception:
            return
        self._human_regrab_suppressed = True
        self._suppressed_human_regrab_hand = hand
        self._suppressed_human_regrab_cube = self.active_cube
        if changed:
            hand_label = hand if hand in ("left", "right") else "both"
            self._log_event(
                "handover_human_regrab_suppressed",
                f"hand={hand_label},cube={self.active_cube.name}",
            )
            print(
                f"[Handover] {hand_label} hand re-grab suppressed for robot takeover",
                flush=True,
            )

    def _restore_active_human_regrab(self) -> None:
        if not self._human_regrab_suppressed:
            return
        suppress = getattr(self.grab_manager, "set_grab_suppressed", None) if self.grab_manager is not None else None
        hand = self._suppressed_human_regrab_hand
        cube = self._suppressed_human_regrab_cube
        if callable(suppress) and cube is not None:
            try:
                suppress(hand=hand, cube=cube, suppressed=False)
                hand_label = hand if hand in ("left", "right") else "both"
                self._log_event(
                    "handover_human_regrab_restored",
                    f"hand={hand_label},cube={getattr(cube, 'name', 'cube')}",
                )
                print(f"[Handover] {hand_label} hand re-grab restored", flush=True)
            except Exception:
                pass
        self._human_regrab_suppressed = False
        self._suppressed_human_regrab_hand = None
        self._suppressed_human_regrab_cube = None

    def _suppress_active_hand_collision(self) -> None:
        if not self.config.handover_disable_human_hand_collision_on_robot_grasp:
            return
        hand = self.active_hand or self._holding_hand_for_active_cube()
        if hand not in ("left", "right"):
            return
        if self._set_hand_proxy_collision(hand, False):
            if not self._human_hand_collision_suppressed:
                self._log_event(
                    "handover_human_hand_collision_suppressed",
                    f"hand={hand},cube={self.active_cube.name}",
                )
                print(
                    f"[Handover] {hand} hand collision suppressed for robot grasp",
                    flush=True,
                )
            self._human_hand_collision_suppressed = True
            self._suppressed_human_hand = hand

    def _restore_active_hand_collision(self) -> None:
        hand = self._suppressed_human_hand or self.active_hand
        if hand not in ("left", "right"):
            self._human_hand_collision_suppressed = False
            self._suppressed_human_hand = None
            return
        if self._set_hand_proxy_collision(hand, True):
            if self._human_hand_collision_suppressed:
                self._log_event(
                    "handover_human_hand_collision_restored",
                    f"hand={hand}",
                )
                print(f"[Handover] {hand} hand collision restored", flush=True)
        self._human_hand_collision_suppressed = False
        self._suppressed_human_hand = None

    def _maintain_active_hand_collision_suppression(self) -> None:
        if not self._human_hand_collision_suppressed:
            return
        hand = self._suppressed_human_hand
        if hand in ("left", "right"):
            self._set_hand_proxy_collision(hand, False)

    def _set_hand_proxy_collision(self, hand: str, enabled: bool) -> bool:
        avatar = getattr(self.grab_manager, "_avatar", None) if self.grab_manager is not None else None
        if avatar is None:
            return False
        override = getattr(avatar, "set_physical_proxy_collision_forced_disabled", None)
        if callable(override):
            try:
                override(hand, not enabled)
                return True
            except Exception:
                pass
        setter = getattr(avatar, "_set_physical_proxy_collision_enabled", None)
        if callable(setter):
            try:
                setter(hand, enabled)
                return True
            except Exception:
                return False
        return False

    def _maybe_confirm_physical_robot_grasp(self) -> None:
        if self._robot_has_taken_over:
            return
        if self._controller_event() < self.config.robot_grasp_confirm_event:
            return
        if not self._robot_grasp_ready():
            return
        if self.config.robot_fixed_joint_grasp and not self._use_fingertip_takeover_path():
            joint_path = self._create_robot_cube_fixed_joint()
            if not joint_path:
                return
            self._robot_grasp_joint_path = joint_path
        if self.config.release_human_on_robot_grasp:
            self._suppress_active_hand_collision()
            self._suppress_active_human_regrab()
            self._force_release_active_cube()
        holding_hand = self._holding_hand_for_active_cube()
        if holding_hand is not None:
            self._log_event(
                "handover_robot_takeover_rejected",
                f"cube={self.active_cube.name},reason=human_still_holding_{holding_hand}",
            )
            self._remove_robot_grasp_joint()
            self._restore_active_hand_collision()
            return
        if (
            self.config.robot_fixed_joint_grasp
            and not self._use_fingertip_takeover_path()
            and self.config.robot_disable_cube_collision_during_carry
        ):
            self._set_cube_collision_enabled(self.active_cube, False)
            self._zero_cube_velocity(self.active_cube)
        self._robot_has_taken_over = True
        self._physics_armed_for_robot = True
        self._pulse("robot_takeover", hand=self.active_hand, intensity=100)
        self._log_event(
            "handover_robot_takeover",
            f"cube={self.active_cube.name},event={self._controller_event()},physical=1",
        )

    def _robot_grasp_ready(self) -> bool:
        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return False
        metrics = self._gripper_grasp_metrics(cube_pos)
        ready = self._gripper_contains_cube(metrics)
        if not ready and self._robot_steps % 30 == 0:
            print(
                "[Handover] waiting_robot_grasp_contact "
                f"event={self._controller_event()} "
                f"{self._grasp_metrics_text(metrics)}",
                flush=True,
            )
        return ready

    def _gripper_center_pos(self) -> np.ndarray | None:
        center = self._gripper_fingertip_poses().get("center")
        if center is None or not self._valid_vec3(center):
            return self._end_effector_pos()
        return np.asarray(center, dtype=float)

    def _gripper_width(self) -> float:
        try:
            return float(np.sum(np.asarray(self.robot.gripper.get_joint_positions(), dtype=float)))
        except Exception:
            return float("inf")

    def _create_robot_cube_fingertip_custody_joint(self) -> str | None:
        body_path = self._nearest_finger_body_path()
        if not body_path:
            return None
        joint_path = self._create_robot_cube_fixed_joint(
            body_path=body_path,
            joint_name_prefix="robot_finger",
        )
        if joint_path:
            self._log_event(
                "handover_fingertip_joint_takeover",
                f"cube={self.active_cube.name},body={body_path},joint={joint_path}",
            )
            print(
                "[Handover] fingertip custody joint takeover "
                f"cube={self.active_cube.name} body={body_path}",
                flush=True,
            )
        return joint_path

    def _create_robot_cube_fixed_joint(
        self,
        *,
        body_path: str | None = None,
        joint_name_prefix: str = "robot",
    ) -> str | None:
        cube_path = self._cube_prim_path(self.active_cube)
        body_path = body_path or self._robot_grasp_body_path()
        if cube_path is None or not body_path:
            return None
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return None
            if not stage.GetPrimAtPath(body_path).IsValid():
                return None
            if not stage.GetPrimAtPath("/World/HandoverJoints").IsValid():
                UsdGeom.Xform.Define(stage, "/World/HandoverJoints")

            safe_cube_name = getattr(self.active_cube, "name", "cube").replace("/", "_")
            joint_path = f"/World/HandoverJoints/{joint_name_prefix}_{safe_cube_name}_fixed_joint"
            if stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)

            cube_pos = self._cube_position(self.active_cube)
            body_pos, body_quat = self._robot_grasp_body_pose(body_path)
            if cube_pos is None or body_pos is None:
                return None
            local0 = self._world_delta_to_body_local(cube_pos - body_pos, body_quat)

            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(cube_path)])
            joint.CreateLocalPos0Attr().Set(
                Gf.Vec3f(float(local0[0]), float(local0[1]), float(local0[2]))
            )
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateCollisionEnabledAttr().Set(False)
            joint.CreateBreakForceAttr().Set(1.0e20)
            joint.CreateBreakTorqueAttr().Set(1.0e20)
            self._zero_cube_velocity(self.active_cube)
            print(
                f"[Handover] robot fixed joint created: {joint_path} "
                f"body0={body_path} body1={cube_path} offset={np.round(local0, 3)}",
                flush=True,
            )
            return joint_path
        except Exception as exc:
            self._log_event("handover_robot_joint_error", f"cube={self.active_cube.name},error={exc}")
            print(f"[Handover] robot fixed joint failed: {exc}", flush=True)
            return None

    def _robot_grasp_body_path(self) -> str | None:
        configured = (self.config.robot_grasp_body_path or "").strip()
        if configured.lower() not in ("", "auto", "auto_fingers", "fingers", "nearest_finger"):
            return configured
        return self._nearest_finger_body_path() or "/World/Franka/panda_hand"

    def _nearest_finger_body_path(self) -> str | None:
        cube_pos = self._cube_position(self.active_cube)
        candidates: list[tuple[str, np.ndarray]] = []
        for body_path, finger_attr in (
            ("/World/Franka/panda_leftfinger", "_left_finger"),
            ("/World/Franka/panda_rightfinger", "_right_finger"),
        ):
            try:
                finger = getattr(self.robot.gripper, finger_attr)
                pos, _ = finger.get_world_pose()
                if self._valid_vec3(pos):
                    candidates.append((body_path, np.asarray(pos, dtype=float)))
            except Exception:
                pass
        if not candidates:
            return None
        if cube_pos is None:
            return candidates[0][0]
        return min(candidates, key=lambda item: float(np.linalg.norm(item[1] - cube_pos)))[0]

    def _physical_robot_place_done(self) -> bool:
        return bool(
            self.config.physical_mode
            and self._robot_has_taken_over
            and self._controller_event() >= self.config.robot_place_done_event
        )

    def _complete_physical_robot_place(self) -> None:
        self._remove_robot_grasp_joint()
        self._place_cube_for_robot(self.active_place_pos, physics_enabled=True)
        self._zero_cube_velocity(self.active_cube)
        self._log_event(
            "handover_robot_physical_release",
            f"cube={self.active_cube.name},event={self._controller_event()},place={self._fmt(self.active_place_pos)}",
        )

    def _remove_robot_grasp_joint(self) -> None:
        joint_path = self._robot_grasp_joint_path
        self._robot_grasp_joint_path = None
        if not joint_path:
            return
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is not None and stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)
                print(f"[Handover] robot fixed joint removed: {joint_path}", flush=True)
        except Exception:
            pass

    def _cube_prim_path(self, cube) -> str | None:
        prim = getattr(cube, "prim", None)
        if prim is not None:
            try:
                path = prim.GetPath()
                if path:
                    return str(path)
            except Exception:
                pass
        for attr_name in ("prim_path", "_prim_path"):
            path = getattr(cube, attr_name, None)
            if path:
                return str(path)
        return None

    def _robot_grasp_body_pose(
        self,
        body_path: str | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if body_path and body_path.endswith("panda_leftfinger"):
            try:
                pos, quat = self.robot.gripper._left_finger.get_world_pose()
                if self._valid_vec3(pos):
                    return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
            except Exception:
                pass
        if body_path and body_path.endswith("panda_rightfinger"):
            try:
                pos, quat = self.robot.gripper._right_finger.get_world_pose()
                if self._valid_vec3(pos):
                    return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
            except Exception:
                pass
        try:
            pos, quat = self.robot.end_effector.get_world_pose()
            if self._valid_vec3(pos):
                return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
        except Exception:
            pass
        return self._end_effector_pos(), None

    def _world_delta_to_body_local(self, delta: np.ndarray, quat: np.ndarray | None) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if quat is None:
            return delta
        q = np.asarray(quat, dtype=float).reshape(-1)
        if q.size < 4 or not np.all(np.isfinite(q[:4])):
            return delta
        q = q[:4]
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return delta
        q = q / norm
        w, x, y, z = q
        q_vec = np.array([x, y, z], dtype=float)
        # Inverse unit quaternion rotates world delta into the body frame.
        return (
            delta
            - 2.0 * w * np.cross(q_vec, delta)
            + 2.0 * np.cross(q_vec, np.cross(q_vec, delta))
        )

    def _controller_event(self) -> int:
        getter = getattr(self.controller, "get_current_event", None)
        if getter is None:
            return -1
        try:
            return int(getter())
        except Exception:
            return -1

    def _controller_t(self) -> float:
        for attr_name in ("_t", "t", "_event_t", "current_t"):
            value = getattr(self.controller, attr_name, None)
            if value is None:
                continue
            try:
                return float(np.clip(float(value), 0.0, 1.0))
            except Exception:
                continue
        return 0.0

    def _should_enable_physics_for_grasp(self) -> bool:
        event = self._controller_event()
        if event >= self.config.takeover_release_event:
            return True
        if not self.config.takeover_arm_by_distance:
            return False
        ee_pos = self._end_effector_pos()
        if ee_pos is None or self._handover_pick_pos is None:
            return False
        return float(np.linalg.norm(ee_pos - self._handover_pick_pos)) <= self.config.takeover_release_dist_m

    def _end_effector_pos(self) -> np.ndarray | None:
        try:
            pos, _ = self.robot.end_effector.get_world_pose()
        except Exception:
            return None
        if not self._valid_vec3(pos):
            return None
        return np.asarray(pos, dtype=float)

    def _update_hold_stability(self) -> tuple[bool, str]:
        reason = self._hold_stability_failure_reason()
        if reason:
            self._stable_hold_windows = 0
            return False, reason

        self._stable_hold_windows += 1
        required = max(1, self.config.stable_required_windows)
        if self._stable_hold_windows < required:
            return False, "confirming"
        return True, "stable"

    def _hold_stability_failure_reason(self) -> str:
        metrics = self._hold_stability_metrics()
        if metrics is None:
            return "warming_up"

        checks = (
            (
                "window_motion",
                metrics["window_motion"],
                self.config.stable_motion_threshold_m,
            ),
            ("max_step", metrics["max_step"], self.config.stable_max_step_m),
            ("mean_step", metrics["mean_step"], self.config.stable_mean_step_m),
            (
                "jitter",
                metrics["jitter"],
                self.config.stable_jitter_threshold_m,
            ),
            (
                "outlier",
                metrics["outlier"],
                self.config.stable_outlier_tolerance_m,
            ),
        )
        for name, value, limit in checks:
            if value > limit:
                return name
        return ""

    def _hold_stability_metrics(self) -> dict[str, float] | None:
        if len(self._target_history) < max(2, self.config.stable_window_steps):
            return None

        arr = np.asarray(self._target_history, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3 or not np.all(np.isfinite(arr[:, :3])):
            return None
        arr = arr[:, :3]

        step_dist = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        center = np.median(arr, axis=0)
        deviation = np.linalg.norm(arr - center, axis=1)

        return {
            "window_motion": float(np.linalg.norm(arr[-1] - arr[0])),
            "max_step": float(np.max(step_dist)) if step_dist.size else 0.0,
            "mean_step": float(np.mean(step_dist)) if step_dist.size else 0.0,
            "jitter": float(np.percentile(deviation, 90)),
            "outlier": float(np.max(deviation)),
        }

    def _fmt_hold_stability_metrics(self) -> str:
        metrics = self._hold_stability_metrics()
        if metrics is None:
            return "unavailable"
        return (
            f"motion={metrics['window_motion']:.3f};"
            f"max_step={metrics['max_step']:.3f};"
            f"mean_step={metrics['mean_step']:.3f};"
            f"jitter={metrics['jitter']:.3f};"
            f"outlier={metrics['outlier']:.3f}"
        )

    def _cube_is_at_place(self) -> bool:
        cube_pos = self._cube_position(self.active_cube)
        if cube_pos is None:
            return False
        return bool(float(np.linalg.norm(cube_pos - self.active_place_pos)) <= self.cube_size * 1.8)

    def _reset_controller(self) -> None:
        try:
            self.controller.reset(end_effector_initial_height=self.approach_height)
        except TypeError:
            try:
                self.controller.reset()
            except Exception:
                pass
        except Exception:
            pass

    def _run_pick_place(
        self,
        *,
        controller,
        robot,
        pick_position: np.ndarray,
        place_position: np.ndarray,
    ) -> bool:
        from simulation.task.pick_controller import run_pick_place

        return bool(
            run_pick_place(
                controller=controller,
                robot=robot,
                pick_position=pick_position,
                place_position=place_position,
            )
        )

    def _pulse(self, event: str, *, hand: str | None, intensity: int) -> None:
        if self.haptics is None or hand is None:
            return
        try:
            self.haptics.pulse(intensity, hand=hand, event=event)
        except Exception:
            pass

    def _log_event(self, event: str, details: str) -> None:
        if self.logger is None:
            return
        try:
            self.logger.log_event(event, details)
        except Exception:
            pass

    def _fmt(self, pos: np.ndarray | None) -> str:
        if pos is None:
            return "None"
        return ",".join(f"{float(v):.3f}" for v in np.asarray(pos, dtype=float).reshape(-1)[:3])

    def _valid_vec3(self, value) -> bool:
        if value is None:
            return False
        arr = np.asarray(value, dtype=float).reshape(-1)
        return bool(arr.size >= 3 and np.all(np.isfinite(arr[:3])))
