"""Cartesian speed profiles for the HRI pick-and-place controller.

This module is intentionally Isaac-independent.  The Isaac adapter supplies
Lula forward kinematics, while the profile, gate, ramp, and line-search logic
remain testable with NumPy alone.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


SPEED_PROFILE_ORDER = ("slow", "medium", "fast")
COUNTERBALANCED_SPEED_PROFILE_ORDERS = (
    ("slow", "medium", "fast"),
    ("medium", "fast", "slow"),
    ("fast", "slow", "medium"),
)

# Isaac Sim 4.5 PickPlaceController event semantics.
APPROACH_PLACE_EVENT_INDICES = (0, 1, 6)
TRANSPORT_EVENT_INDICES = (4, 5, 8, 9)
SETTLING_GRIPPER_EVENT_INDICES = (2, 3, 7)
MOTION_EVENT_INDICES = APPROACH_PLACE_EVENT_INDICES + TRANSPORT_EVENT_INDICES

# Backward-compatible name for readers that imported this constant. Event 1 is
# now explicitly part of the Cartesian-limited approach/place phase.
GRASP_APPROACH_EVENT_INDICES = (1,)

# Original Isaac timings are retained for the non-motion events. Motion events
# use a final-waypoint target and are held until the measured EE reaches it, so
# their progress increment is deliberately profile-independent. Speed is set by
# the Cartesian limiter below, never by multiplying these values.
BASE_EVENTS_DT = (0.008, 0.005, 1.0, 0.1, 0.05, 0.05, 0.0025, 1.0, 0.008, 0.08)
CARTESIAN_EVENTS_DT = (1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

# Kept only for metadata compatibility with older readers. These values no
# longer control speed and must not be interpreted as velocity multipliers.
SPEED_PROFILE_MOTION_SCALES = {profile: 1.0 for profile in SPEED_PROFILE_ORDER}

SPEED_LIMITS_MPS = {
    "slow": {
        "transport": 0.30,
        "approach_place": 0.10,
        "near_human": 0.05,
    },
    "medium": {
        "transport": 0.40,
        "approach_place": 0.15,
        "near_human": 0.10,
    },
    "fast": {
        "transport": 0.50,
        "approach_place": 0.20,
        "near_human": 0.15,
    },
}

SPEED_SCHEDULE_VERSION = "pick_place_cartesian_speed_v1"
SPEED_STEP_SCHEMA_VERSION = "ee_cartesian_speed_step_v1"
DEFAULT_GATE_START_GAP_M = 0.13
DEFAULT_GATE_FULL_GAP_M = 0.05
DEFAULT_ACCELERATION_LIMIT_MPS2 = 1.0
DEFAULT_JERK_LIMIT_MPS3 = 10.0
DEFAULT_REACH_TOLERANCE_M = 0.005
DEFAULT_MIN_CONTROL_DT_S = 1e-5
DEFAULT_MAX_CONTROL_DT_S = 0.1
DEFAULT_FK_BINARY_SEARCH_ITERATIONS = 24
DEFAULT_FK_DISPLACEMENT_TOLERANCE_M = 1e-6


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def normalize_speed_profile(profile: str) -> str:
    normalized = str(profile).strip().lower()
    if normalized not in SPEED_LIMITS_MPS:
        raise ValueError(
            f"Unknown pick-place speed profile {profile!r}; "
            f"expected one of {SPEED_PROFILE_ORDER}"
        )
    return normalized


@dataclass(frozen=True)
class CartesianSpeedConfig:
    """Configuration shared by the command limiter and HDF5 metadata."""

    gate_start_gap_m: float = DEFAULT_GATE_START_GAP_M
    gate_full_gap_m: float = DEFAULT_GATE_FULL_GAP_M
    acceleration_limit_mps2: float = DEFAULT_ACCELERATION_LIMIT_MPS2
    jerk_limit_mps3: float = DEFAULT_JERK_LIMIT_MPS3
    reach_tolerance_m: float = DEFAULT_REACH_TOLERANCE_M
    min_control_dt_s: float = DEFAULT_MIN_CONTROL_DT_S
    max_control_dt_s: float = DEFAULT_MAX_CONTROL_DT_S
    fk_binary_search_iterations: int = DEFAULT_FK_BINARY_SEARCH_ITERATIONS
    fk_displacement_tolerance_m: float = DEFAULT_FK_DISPLACEMENT_TOLERANCE_M

    @classmethod
    def from_env(cls) -> "CartesianSpeedConfig":
        return cls(
            gate_start_gap_m=_env_float(
                "HRI_DISTANCE_GATE_START_GAP_M", cls.gate_start_gap_m
            ),
            gate_full_gap_m=_env_float(
                "HRI_DISTANCE_GATE_FULL_GAP_M", cls.gate_full_gap_m
            ),
            acceleration_limit_mps2=_env_float(
                "PICK_PLACE_EE_ACCELERATION_LIMIT_MPS2",
                cls.acceleration_limit_mps2,
            ),
            jerk_limit_mps3=_env_float(
                "PICK_PLACE_EE_JERK_LIMIT_MPS3", cls.jerk_limit_mps3
            ),
            reach_tolerance_m=_env_float(
                "PICK_PLACE_EE_REACH_TOLERANCE_M", cls.reach_tolerance_m
            ),
            min_control_dt_s=_env_float(
                "PICK_PLACE_EE_MIN_CONTROL_DT_S", cls.min_control_dt_s
            ),
            max_control_dt_s=_env_float(
                "PICK_PLACE_EE_MAX_CONTROL_DT_S", cls.max_control_dt_s
            ),
            fk_binary_search_iterations=_env_int(
                "PICK_PLACE_EE_FK_BINARY_SEARCH_ITERATIONS",
                cls.fk_binary_search_iterations,
            ),
            fk_displacement_tolerance_m=_env_float(
                "PICK_PLACE_EE_FK_DISPLACEMENT_TOLERANCE_M",
                cls.fk_displacement_tolerance_m,
            ),
        ).validated()

    def validated(self) -> "CartesianSpeedConfig":
        if not self.gate_start_gap_m > self.gate_full_gap_m:
            raise ValueError("gate_start_gap_m must be greater than gate_full_gap_m")
        for name, value in (
            ("acceleration_limit_mps2", self.acceleration_limit_mps2),
            ("jerk_limit_mps3", self.jerk_limit_mps3),
            ("reach_tolerance_m", self.reach_tolerance_m),
            ("min_control_dt_s", self.min_control_dt_s),
            ("max_control_dt_s", self.max_control_dt_s),
            ("fk_displacement_tolerance_m", self.fk_displacement_tolerance_m),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_control_dt_s <= self.min_control_dt_s:
            raise ValueError("max_control_dt_s must be greater than min_control_dt_s")
        if int(self.fk_binary_search_iterations) < 1:
            raise ValueError("fk_binary_search_iterations must be positive")
        return self

    def metadata(self) -> dict[str, object]:
        return {
            "controller_speed_schedule_version": SPEED_SCHEDULE_VERSION,
            "controller_speed_step_schema_version": SPEED_STEP_SCHEMA_VERSION,
            "controller_speed_control_clock": "simulation_time",
            "controller_speed_observation_wall_clock": "monotonic_time",
            "controller_speed_gate_start_gap_m": float(self.gate_start_gap_m),
            "controller_speed_gate_full_gap_m": float(self.gate_full_gap_m),
            "controller_speed_gate_blending": (
                "active=(1-distance_gate)*phase_limit+"
                "distance_gate*near_human_limit"
            ),
            "controller_speed_gate_source": (
                "existing_surface_gap_distance_gate_max_over_hands"
            ),
            "controller_speed_acceleration_limit_mps2": float(
                self.acceleration_limit_mps2
            ),
            "controller_speed_jerk_limit_mps3": float(self.jerk_limit_mps3),
            "controller_speed_ramp_semantics": (
                "jerk_limited_increase_immediate_safety_decrease"
            ),
            "controller_speed_reach_tolerance_m": float(self.reach_tolerance_m),
            "controller_speed_min_control_dt_s": float(self.min_control_dt_s),
            "controller_speed_max_control_dt_s": float(self.max_control_dt_s),
            "controller_speed_action_limit_method": (
                "joint_target_interpolation_lula_fk_binary_search"
            ),
            "controller_speed_fk_binary_search_iterations": int(
                self.fk_binary_search_iterations
            ),
            "controller_speed_fk_displacement_tolerance_m": float(
                self.fk_displacement_tolerance_m
            ),
            "controller_speed_end_effector_reference": (
                "rmpflow_lula_end_effector_frame"
            ),
            "controller_speed_hardware_deployment_approved": 0,
        }


def normalize_speed_profile_order(
    profile_order: str | tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Validate and normalize one three-condition episode order."""

    if profile_order is None:
        values = SPEED_PROFILE_ORDER
    elif isinstance(profile_order, str):
        values = tuple(part.strip().lower() for part in profile_order.split(","))
    else:
        values = tuple(str(part).strip().lower() for part in profile_order)
    if len(values) != len(SPEED_PROFILE_ORDER) or set(values) != set(
        SPEED_PROFILE_ORDER
    ):
        raise ValueError(
            "speed profile order must contain slow, medium, and fast exactly once"
        )
    return values


def counterbalanced_speed_profile_order(order_index: int) -> tuple[str, ...]:
    """Return a cyclic Latin-square order for one session."""

    index = int(order_index)
    if index < 0:
        raise ValueError("order_index must be non-negative")
    return COUNTERBALANCED_SPEED_PROFILE_ORDERS[
        index % len(COUNTERBALANCED_SPEED_PROFILE_ORDERS)
    ]


def speed_profile_for_episode(
    episode_index: int,
    profile_order: str | tuple[str, ...] | list[str] | None = None,
) -> str:
    """Return the scheduled speed profile for a zero-based episode index."""

    index = int(episode_index)
    if index < 0:
        raise ValueError("episode_index must be non-negative")
    order = normalize_speed_profile_order(profile_order)
    return order[index % len(order)]


def controller_phase_mode(event: int) -> str:
    """Map one Isaac PickPlaceController event to an explicit speed phase."""

    event = int(event)
    if event in APPROACH_PLACE_EVENT_INDICES:
        return "approach_place"
    if event in TRANSPORT_EVENT_INDICES:
        return "transport"
    if event in SETTLING_GRIPPER_EVENT_INDICES:
        return "settling_gripper"
    return "done"


def speed_limits_for_profile(profile: str) -> dict[str, float]:
    normalized = normalize_speed_profile(profile)
    return dict(SPEED_LIMITS_MPS[normalized])


def blended_speed_limit_mps(
    profile: str,
    event: int,
    distance_gate: float,
) -> tuple[float, float, float, str, str]:
    """Return phase, near, blended limits and nominal/active mode labels."""

    phase_mode = controller_phase_mode(event)
    if phase_mode not in ("approach_place", "transport"):
        return 0.0, 0.0, 0.0, phase_mode, phase_mode
    limits = speed_limits_for_profile(profile)
    gate = float(np.clip(float(distance_gate), 0.0, 1.0))
    phase_limit = float(limits[phase_mode])
    near_limit = float(limits["near_human"])
    blended = (1.0 - gate) * phase_limit + gate * near_limit
    active_mode = "near_human" if gate > 0.0 else phase_mode
    return phase_limit, near_limit, float(blended), phase_mode, active_mode


def events_dt_for_speed_profile(profile: str) -> tuple[float, ...]:
    """Return profile-independent FSM timing for Cartesian-limited motion."""

    normalize_speed_profile(profile)
    return tuple(float(value) for value in CARTESIAN_EVENTS_DT)


def nominal_profile_steps(profile: str) -> int:
    """Return the minimum FSM calls, excluding Cartesian travel convergence."""

    return sum(
        int(math.ceil(1.0 / value))
        for value in events_dt_for_speed_profile(profile)
    )


@dataclass
class SpeedLimitRamp:
    """Simulation-time ramp for increases, with immediate safety decreases."""

    config: CartesianSpeedConfig = field(default_factory=CartesianSpeedConfig)
    active_limit_mps: float = 0.0
    acceleration_mps2: float = 0.0
    previous_sim_time_s: float | None = None

    def reset(self) -> None:
        self.active_limit_mps = 0.0
        self.acceleration_mps2 = 0.0
        self.previous_sim_time_s = None

    def update(self, target_limit_mps: float, sim_time_s: float) -> tuple[float, float, bool]:
        target = max(0.0, float(target_limit_mps))
        sim_time = float(sim_time_s)
        if not math.isfinite(sim_time):
            self.reset()
            return 0.0, 0.0, False
        if self.previous_sim_time_s is None:
            self.previous_sim_time_s = sim_time
            return self.active_limit_mps, 0.0, False

        dt = sim_time - self.previous_sim_time_s
        self.previous_sim_time_s = sim_time
        if (
            not math.isfinite(dt)
            or dt < self.config.min_control_dt_s
            or dt
            > self.config.max_control_dt_s
            + max(1e-12, self.config.max_control_dt_s * 1e-9)
        ):
            self.active_limit_mps = 0.0
            self.acceleration_mps2 = 0.0
            return 0.0, float(dt) if math.isfinite(dt) else 0.0, False

        # A lower phase/gate cap is a hard upper bound. Upward changes are
        # jerk/acceleration limited so phase transitions cannot command a jump.
        if target <= self.active_limit_mps:
            self.active_limit_mps = target
            self.acceleration_mps2 = 0.0
            return self.active_limit_mps, dt, True

        desired_accel = min(
            self.config.acceleration_limit_mps2,
            (target - self.active_limit_mps) / dt,
        )
        max_accel_change = self.config.jerk_limit_mps3 * dt
        self.acceleration_mps2 = min(
            desired_accel,
            self.acceleration_mps2 + max_accel_change,
        )
        self.active_limit_mps = min(
            target,
            self.active_limit_mps + self.acceleration_mps2 * dt,
        )
        return self.active_limit_mps, dt, True


@dataclass(frozen=True)
class PoseSpeedSample:
    velocity_world_mps: np.ndarray
    speed_mps: float
    dt_s: float
    valid: bool


@dataclass
class PoseSpeedEstimator:
    """Finite-difference speed estimator using an explicitly supplied clock."""

    min_dt_s: float = DEFAULT_MIN_CONTROL_DT_S
    max_dt_s: float = DEFAULT_MAX_CONTROL_DT_S
    previous_time_s: float | None = None
    previous_position: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_time_s = None
        self.previous_position = None

    def update(self, position_world, timestamp_s: float) -> PoseSpeedSample:
        position = np.asarray(position_world, dtype=float).reshape(3)
        timestamp = float(timestamp_s)
        invalid = PoseSpeedSample(np.zeros(3), 0.0, 0.0, False)
        if not np.all(np.isfinite(position)) or not math.isfinite(timestamp):
            self.reset()
            return invalid
        if self.previous_time_s is None or self.previous_position is None:
            self.previous_time_s = timestamp
            self.previous_position = position.copy()
            return invalid
        dt = timestamp - self.previous_time_s
        previous_position = self.previous_position
        self.previous_time_s = timestamp
        self.previous_position = position.copy()
        if (
            not math.isfinite(dt)
            or dt < self.min_dt_s
            or dt > self.max_dt_s + max(1e-12, self.max_dt_s * 1e-9)
        ):
            return PoseSpeedSample(np.zeros(3), 0.0, float(dt) if math.isfinite(dt) else 0.0, False)
        velocity = (position - previous_position) / dt
        if not np.all(np.isfinite(velocity)):
            return PoseSpeedSample(np.zeros(3), 0.0, dt, False)
        return PoseSpeedSample(velocity, float(np.linalg.norm(velocity)), dt, True)


@dataclass(frozen=True)
class JointTargetLimitResult:
    joint_positions: np.ndarray
    scale: float
    predicted_displacement_m: float
    full_target_displacement_m: float
    limited: bool
    valid: bool


def limit_joint_target_by_cartesian_displacement(
    current_joint_positions,
    target_joint_positions,
    max_displacement_m: float,
    forward_kinematics: Callable[[np.ndarray], np.ndarray],
    *,
    iterations: int = DEFAULT_FK_BINARY_SEARCH_ITERATIONS,
    tolerance_m: float = DEFAULT_FK_DISPLACEMENT_TOLERANCE_M,
) -> JointTargetLimitResult:
    """Interpolate a joint target until its FK displacement fits the cap."""

    current = np.asarray(current_joint_positions, dtype=float).reshape(-1)
    target = np.asarray(target_joint_positions, dtype=float).reshape(-1)
    cap = max(0.0, float(max_displacement_m))
    if (
        current.shape != target.shape
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(target))
        or not math.isfinite(cap)
    ):
        return JointTargetLimitResult(current.copy(), 0.0, 0.0, 0.0, True, False)

    try:
        current_position = np.asarray(forward_kinematics(current), dtype=float).reshape(3)
        target_position = np.asarray(forward_kinematics(target), dtype=float).reshape(3)
    except Exception:
        return JointTargetLimitResult(current.copy(), 0.0, 0.0, 0.0, True, False)
    if not np.all(np.isfinite(current_position)) or not np.all(np.isfinite(target_position)):
        return JointTargetLimitResult(current.copy(), 0.0, 0.0, 0.0, True, False)

    full_displacement = float(np.linalg.norm(target_position - current_position))
    if full_displacement <= cap + float(tolerance_m):
        return JointTargetLimitResult(
            target.copy(), 1.0, full_displacement, full_displacement, False, True
        )
    if cap <= 0.0:
        return JointTargetLimitResult(
            current.copy(), 0.0, 0.0, full_displacement, True, True
        )

    delta = target - current
    low = 0.0
    high = 1.0
    low_displacement = 0.0
    for _ in range(max(1, int(iterations))):
        scale = 0.5 * (low + high)
        candidate = current + scale * delta
        candidate_position = np.asarray(
            forward_kinematics(candidate), dtype=float
        ).reshape(3)
        displacement = float(np.linalg.norm(candidate_position - current_position))
        if math.isfinite(displacement) and displacement <= cap:
            low = scale
            low_displacement = displacement
        else:
            high = scale

    limited_target = current + low * delta
    # Verify the selected result independently. A local joint-space segment is
    # normally monotonic in Cartesian displacement, but the final check makes
    # the hard cap explicit even for unusual FK curvature.
    limited_position = np.asarray(
        forward_kinematics(limited_target), dtype=float
    ).reshape(3)
    verified_displacement = float(
        np.linalg.norm(limited_position - current_position)
    )
    valid = math.isfinite(verified_displacement) and (
        verified_displacement <= cap + float(tolerance_m)
    )
    if not valid:
        return JointTargetLimitResult(
            current.copy(), 0.0, 0.0, full_displacement, True, False
        )
    return JointTargetLimitResult(
        limited_target,
        float(low),
        verified_displacement if valid else low_displacement,
        full_displacement,
        True,
        valid,
    )


def speed_profile_metadata(
    profile: str,
    *,
    physics_dt_s: float = 1.0 / 60.0,
    config: CartesianSpeedConfig | None = None,
) -> dict[str, object]:
    """Return HDF5-safe metadata for one episode speed condition."""

    normalized = normalize_speed_profile(profile)
    config = (config or CartesianSpeedConfig.from_env()).validated()
    limits = speed_limits_for_profile(normalized)
    events_dt = events_dt_for_speed_profile(normalized)
    nominal_steps = nominal_profile_steps(normalized)
    return {
        **config.metadata(),
        "controller_speed_profile": normalized,
        "transport_speed_limit_mps": float(limits["transport"]),
        "approach_place_speed_limit_mps": float(limits["approach_place"]),
        "near_human_speed_limit_mps": float(limits["near_human"]),
        "controller_speed_approach_place_event_indices": ",".join(
            str(index) for index in APPROACH_PLACE_EVENT_INDICES
        ),
        "controller_speed_transport_event_indices": ",".join(
            str(index) for index in TRANSPORT_EVENT_INDICES
        ),
        "controller_speed_settling_gripper_event_indices": ",".join(
            str(index) for index in SETTLING_GRIPPER_EVENT_INDICES
        ),
        "controller_motion_phase_scale": 1.0,
        "controller_motion_phase_scale_deprecated": 1,
        "controller_grasp_approach_event_indices": ",".join(
            str(index) for index in GRASP_APPROACH_EVENT_INDICES
        ),
        "controller_events_dt_json": json.dumps(events_dt),
        "controller_nominal_cycle_steps": int(nominal_steps),
        "controller_nominal_cycle_duration_s": float(
            nominal_steps * float(physics_dt_s)
        ),
        "controller_nominal_cycle_duration_semantics": (
            "minimum_fsm_bookkeeping_only_excludes_cartesian_travel"
        ),
        "active_speed_limit_dataset": "controller/active_speed_limit_mps",
        "speed_limit_mode_dataset": "controller/speed_limit_mode",
    }


def speed_schedule_metadata(
    *,
    physics_dt_s: float = 1.0 / 60.0,
    profile_order: str | tuple[str, ...] | list[str] | None = None,
    counterbalance_order_index: int | None = None,
    config: CartesianSpeedConfig | None = None,
) -> dict[str, object]:
    """Return file-level metadata describing the three-speed schedule."""

    order = normalize_speed_profile_order(profile_order)
    config = (config or CartesianSpeedConfig.from_env()).validated()
    profiles = {
        profile: {
            **speed_limits_for_profile(profile),
            "events_dt": events_dt_for_speed_profile(profile),
            "nominal_fsm_min_steps": nominal_profile_steps(profile),
        }
        for profile in SPEED_PROFILE_ORDER
    }
    metadata = {
        **config.metadata(),
        "controller_speed_schedule": ",".join(order),
        "controller_speed_counterbalance_orders_json": json.dumps(
            COUNTERBALANCED_SPEED_PROFILE_ORDERS
        ),
        "controller_speed_motion_event_indices": ",".join(
            str(index) for index in MOTION_EVENT_INDICES
        ),
        "controller_speed_approach_place_event_indices": ",".join(
            str(index) for index in APPROACH_PLACE_EVENT_INDICES
        ),
        "controller_speed_transport_event_indices": ",".join(
            str(index) for index in TRANSPORT_EVENT_INDICES
        ),
        "controller_speed_settling_gripper_event_indices": ",".join(
            str(index) for index in SETTLING_GRIPPER_EVENT_INDICES
        ),
        "controller_speed_unscaled_grasp_approach_event_indices": "",
        "controller_speed_profiles_json": json.dumps(profiles),
    }
    if counterbalance_order_index is not None:
        metadata["controller_speed_counterbalance_order_index"] = int(
            counterbalance_order_index
        )
    return metadata
