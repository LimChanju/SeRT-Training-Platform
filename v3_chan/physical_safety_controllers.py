"""Physical safety controllers shared by rollout and live-VR evaluation.

The learned task policy still selects an end-effector target.  RMPflow turns
that target into a nominal joint command.  The velocity-level CBF implemented
here projects that nominal command onto signed surface-gap constraints for the
tracked hands and the selected distal Panda links.

This module intentionally has no Isaac imports at module import time so the QP
projection and constraint construction can be unit tested without Isaac Sim.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


PHYSICAL_SAFETY_MODES = (
    "none",
    "rmpflow",
    "cbf",
    "rmpflow_cbf",
    "curobo",
    "curobo_cbf",
)


def mode_uses_rmpflow_obstacles(mode: str) -> bool:
    return str(mode) in {"rmpflow", "rmpflow_cbf"}


def mode_uses_cbf(mode: str) -> bool:
    return str(mode) in {"cbf", "rmpflow_cbf", "curobo_cbf"}


def mode_uses_curobo(mode: str) -> bool:
    return str(mode) in {"curobo", "curobo_cbf"}


@dataclass(frozen=True)
class CBFConfig:
    """Configuration for the velocity-level distal-link CBF filter."""

    safe_gap_m: float = 0.05
    activation_gap_m: float = 0.13
    gamma_per_s: float = 8.0
    prediction_horizon_s: float = 0.15
    max_prediction_buffer_m: float = 0.08
    max_joint_speed_rad_s: float = 2.0
    projection_iterations: int = 80
    projection_tolerance: float = 1e-5

    def validated(self) -> "CBFConfig":
        values = (
            self.safe_gap_m,
            self.activation_gap_m,
            self.gamma_per_s,
            self.prediction_horizon_s,
            self.max_prediction_buffer_m,
            self.max_joint_speed_rad_s,
            self.projection_tolerance,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("CBF configuration values must be finite")
        if self.safe_gap_m < 0.0:
            raise ValueError("safe_gap_m must be non-negative")
        if self.activation_gap_m < self.safe_gap_m:
            raise ValueError("activation_gap_m must be >= safe_gap_m")
        if self.gamma_per_s <= 0.0:
            raise ValueError("gamma_per_s must be positive")
        if self.prediction_horizon_s < 0.0:
            raise ValueError("prediction_horizon_s must be non-negative")
        if self.max_prediction_buffer_m < 0.0:
            raise ValueError("max_prediction_buffer_m must be non-negative")
        if self.max_joint_speed_rad_s <= 0.0:
            raise ValueError("max_joint_speed_rad_s must be positive")
        if int(self.projection_iterations) < 1:
            raise ValueError("projection_iterations must be positive")
        if self.projection_tolerance <= 0.0:
            raise ValueError("projection_tolerance must be positive")
        return self


@dataclass(frozen=True)
class PhysicalSafetyDiagnostics:
    controller: str = "none"
    active: bool = False
    intervention_available: bool = False
    constraint_count: int = 0
    valid_hand_count: int = 0
    intervention_norm_radps: float = 0.0
    nominal_velocity_norm_radps: float = 0.0
    filtered_velocity_norm_radps: float = 0.0
    max_constraint_violation_before: float = 0.0
    max_constraint_violation_after: float = 0.0
    slack_radps: float = 0.0
    min_predicted_gap_m: float = 10.0
    solve_time_ms: float = 0.0
    feasible: bool = True
    status: str = "inactive"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _CBFConstraint:
    hand: str
    jacobian_row: np.ndarray
    lower_bound_radps: float
    predicted_gap_m: float


def project_velocity_qp(
    nominal_velocity: np.ndarray,
    constraint_matrix: np.ndarray,
    lower_bounds: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
    *,
    max_iterations: int = 80,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, float, bool, float, float]:
    """Project a velocity onto box and half-space constraints.

    The solved convex problem is ``min 0.5 ||qdot-qdot_nom||^2`` subject to
    ``A qdot >= b`` and joint-velocity bounds.  Dykstra projections give the
    Euclidean projection without an external QP package.  If the constraints
    are infeasible under the velocity limits, a common non-negative slack is
    found by bisection and returned explicitly.
    """

    nominal = np.asarray(nominal_velocity, dtype=float).reshape(-1)
    lower = np.asarray(velocity_lower, dtype=float).reshape(nominal.shape)
    upper = np.asarray(velocity_upper, dtype=float).reshape(nominal.shape)
    matrix = np.asarray(constraint_matrix, dtype=float)
    bounds = np.asarray(lower_bounds, dtype=float).reshape(-1)
    if matrix.size == 0:
        matrix = np.empty((0, nominal.size), dtype=float)
    else:
        matrix = matrix.reshape((-1, nominal.size))
    if bounds.shape != (matrix.shape[0],):
        raise ValueError("constraint bounds do not match constraint matrix")
    if np.any(lower > upper):
        raise ValueError("velocity lower bound exceeds upper bound")
    if not all(
        np.all(np.isfinite(value))
        for value in (nominal, matrix, bounds, lower, upper)
    ):
        raise ValueError("QP inputs must be finite")

    clipped_nominal = np.clip(nominal, lower, upper)
    before = _max_halfspace_violation(matrix, bounds, clipped_nominal)
    if matrix.shape[0] == 0:
        return clipped_nominal, 0.0, True, before, 0.0

    projected, violation = _dykstra_project(
        clipped_nominal,
        matrix,
        bounds,
        lower,
        upper,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if violation <= tolerance:
        return projected, 0.0, True, before, violation

    # A positive common slack keeps the filter deterministic and exposes
    # infeasibility to the evaluator instead of silently returning bad values.
    slack_high = max(float(violation), tolerance)
    for _ in range(20):
        candidate, candidate_violation = _dykstra_project(
            clipped_nominal,
            matrix,
            bounds - slack_high,
            lower,
            upper,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if candidate_violation <= tolerance:
            projected = candidate
            break
        slack_high *= 2.0
    else:
        return clipped_nominal, slack_high, False, before, violation

    slack_low = 0.0
    for _ in range(24):
        slack_mid = 0.5 * (slack_low + slack_high)
        candidate, candidate_violation = _dykstra_project(
            clipped_nominal,
            matrix,
            bounds - slack_mid,
            lower,
            upper,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if candidate_violation <= tolerance:
            slack_high = slack_mid
            projected = candidate
        else:
            slack_low = slack_mid

    original_violation = _max_halfspace_violation(matrix, bounds, projected)
    return projected, slack_high, False, before, original_violation


class DistalLinkVelocityCBF:
    """Filter an Isaac articulation action using distal-link CBF constraints."""

    def __init__(self, config: CBFConfig | None = None) -> None:
        self.config = (config or CBFConfig()).validated()
        self.last_diagnostics = PhysicalSafetyDiagnostics(controller="cbf")

    def reset(self) -> None:
        self.last_diagnostics = PhysicalSafetyDiagnostics(controller="cbf")

    def filter_action(
        self,
        *,
        robot,
        arm_action,
        safety_result,
        dynamic_sample,
        safety_geometry,
        observation: dict[str, np.ndarray],
        physics_dt_s: float,
    ):
        started = time.perf_counter()
        try:
            filtered_action, diagnostics = self._filter_action(
                robot=robot,
                arm_action=arm_action,
                safety_result=safety_result,
                dynamic_sample=dynamic_sample,
                safety_geometry=safety_geometry,
                observation=observation,
                physics_dt_s=physics_dt_s,
            )
        except Exception as exc:
            diagnostics = PhysicalSafetyDiagnostics(
                controller="cbf",
                solve_time_ms=(time.perf_counter() - started) * 1000.0,
                feasible=False,
                status=f"error:{type(exc).__name__}:{exc}",
            )
            self.last_diagnostics = diagnostics
            raise RuntimeError(
                f"CBF safety filter failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.last_diagnostics = diagnostics
        return filtered_action, diagnostics

    def _filter_action(
        self,
        *,
        robot,
        arm_action,
        safety_result,
        dynamic_sample,
        safety_geometry,
        observation: dict[str, np.ndarray],
        physics_dt_s: float,
    ):
        started = time.perf_counter()
        dt_s = float(physics_dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("physics_dt_s must be finite and positive")

        current_all = _to_numpy(robot.get_joint_positions()).reshape(-1)
        joint_indices = getattr(arm_action, "joint_indices", None)
        position_targets = getattr(arm_action, "joint_positions", None)
        velocity_targets = getattr(arm_action, "joint_velocities", None)
        if joint_indices is None:
            target_size = len(position_targets) if position_targets is not None else 0
            joint_indices = np.arange(target_size, dtype=int)
        else:
            joint_indices = _to_numpy(joint_indices).astype(int).reshape(-1)
        if joint_indices.size == 0:
            raise ValueError("arm action has no active joint indices")

        if velocity_targets is not None:
            nominal_velocity = _to_numpy(velocity_targets).astype(float).reshape(-1)
        elif position_targets is not None:
            positions = _to_numpy(position_targets).astype(float).reshape(-1)
            nominal_velocity = (positions - current_all[joint_indices]) / dt_s
        else:
            raise ValueError("arm action has no joint position or velocity targets")
        if nominal_velocity.shape != joint_indices.shape:
            raise ValueError("arm action target shape does not match joint indices")

        jacobians, body_names = _robot_jacobians_and_body_names(robot)
        constraints = list(
            self._constraints(
                safety_result=safety_result,
                dynamic_sample=dynamic_sample,
                safety_geometry=safety_geometry,
                observation=observation,
                jacobians=jacobians,
                body_names=body_names,
                active_joint_indices=joint_indices,
            )
        )
        speed_limit = _joint_speed_limits(robot, joint_indices, self.config)
        velocity_lower = -speed_limit
        velocity_upper = speed_limit
        matrix = np.asarray(
            [constraint.jacobian_row for constraint in constraints], dtype=float
        )
        bounds = np.asarray(
            [constraint.lower_bound_radps for constraint in constraints], dtype=float
        )
        filtered_velocity, slack, feasible, before, after = project_velocity_qp(
            nominal_velocity,
            matrix,
            bounds,
            velocity_lower,
            velocity_upper,
            max_iterations=self.config.projection_iterations,
            tolerance=self.config.projection_tolerance,
        )

        if position_targets is not None:
            safe_positions = current_all[joint_indices] + filtered_velocity * dt_s
            safe_positions = _clip_joint_positions(robot, joint_indices, safe_positions)
            arm_action.joint_positions = safe_positions
        arm_action.joint_velocities = filtered_velocity
        intervention = float(np.linalg.norm(filtered_velocity - nominal_velocity))
        diagnostics = PhysicalSafetyDiagnostics(
            controller="cbf",
            active=bool(constraints),
            intervention_available=True,
            constraint_count=len(constraints),
            valid_hand_count=len(constraints),
            intervention_norm_radps=intervention,
            nominal_velocity_norm_radps=float(np.linalg.norm(nominal_velocity)),
            filtered_velocity_norm_radps=float(np.linalg.norm(filtered_velocity)),
            max_constraint_violation_before=float(before),
            max_constraint_violation_after=float(after),
            slack_radps=float(slack),
            min_predicted_gap_m=min(
                (constraint.predicted_gap_m for constraint in constraints),
                default=10.0,
            ),
            solve_time_ms=(time.perf_counter() - started) * 1000.0,
            feasible=bool(feasible),
            status=(
                "inactive"
                if not constraints
                else ("solved" if feasible else "solved_with_slack")
            ),
        )
        return arm_action, diagnostics

    def _constraints(
        self,
        *,
        safety_result,
        dynamic_sample,
        safety_geometry,
        observation: dict[str, np.ndarray],
        jacobians: np.ndarray,
        body_names: tuple[str, ...],
        active_joint_indices: np.ndarray,
    ) -> Iterable[_CBFConstraint]:
        for hand_name in ("left", "right"):
            hand_result = getattr(safety_result, hand_name)
            if not bool(hand_result.geometry_valid):
                continue
            gap_m = float(hand_result.surface_gap_m)
            if not math.isfinite(gap_m) or gap_m > self.config.activation_gap_m:
                continue
            hand_pos = _observation_vec3(
                observation, f"human_{hand_name}_hand_pos"
            )
            if hand_pos is None:
                continue
            link_origin, _, origin_valid = safety_geometry.closest_link_world_pose(
                hand_result
            )
            if not origin_valid or link_origin is None:
                continue
            link_origin = np.asarray(link_origin, dtype=float).reshape(3)
            surface_point = None
            if bool(hand_result.closest_surface_point_valid):
                candidate = np.asarray(
                    hand_result.closest_surface_point_world_pos, dtype=float
                ).reshape(-1)
                if candidate.size >= 3 and np.all(np.isfinite(candidate[:3])):
                    surface_point = candidate[:3]
            if surface_point is None:
                surface_point = link_origin
            normal = surface_point - hand_pos
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-8:
                continue
            normal /= normal_norm

            body_jacobian = _body_jacobian(
                jacobians, body_names, str(hand_result.closest_link)
            )
            offset = surface_point - link_origin
            point_jacobian = (
                body_jacobian[:3]
                - _skew(offset) @ body_jacobian[3:6]
            )
            point_jacobian = point_jacobian[:, active_joint_indices]
            constraint_row = normal @ point_jacobian
            if float(np.linalg.norm(constraint_row)) <= 1e-9:
                continue

            hand_dynamic = getattr(dynamic_sample, hand_name)
            hand_velocity = np.zeros(3, dtype=float)
            if bool(getattr(hand_dynamic, "hand_velocity_valid", False)):
                candidate = np.asarray(
                    hand_dynamic.hand_velocity_filtered_mps, dtype=float
                ).reshape(-1)
                if candidate.size >= 3 and np.all(np.isfinite(candidate[:3])):
                    hand_velocity = candidate[:3]
            closing_speed = 0.0
            if bool(getattr(hand_dynamic, "closing_speed_valid", False)):
                closing_speed = max(
                    0.0, float(getattr(hand_dynamic, "closing_speed_mps", 0.0))
                )
            prediction_buffer = min(
                self.config.max_prediction_buffer_m,
                self.config.prediction_horizon_s * closing_speed,
            )
            safe_gap = self.config.safe_gap_m + prediction_buffer
            barrier_value = gap_m - safe_gap
            lower_bound = float(
                normal @ hand_velocity - self.config.gamma_per_s * barrier_value
            )
            yield _CBFConstraint(
                hand=hand_name,
                jacobian_row=np.asarray(constraint_row, dtype=float),
                lower_bound_radps=lower_bound,
                predicted_gap_m=float(gap_m - prediction_buffer),
            )


def _dykstra_project(
    initial: np.ndarray,
    matrix: np.ndarray,
    bounds: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    projections = 1 + matrix.shape[0]
    corrections = [np.zeros_like(initial) for _ in range(projections)]
    value = np.asarray(initial, dtype=float).copy()
    for _ in range(int(max_iterations)):
        previous = value.copy()
        shifted = value + corrections[0]
        value = np.clip(shifted, lower, upper)
        corrections[0] = shifted - value
        for index, (row, bound) in enumerate(zip(matrix, bounds), start=1):
            shifted = value + corrections[index]
            row_norm_sq = float(row @ row)
            if row_norm_sq <= 1e-16:
                value = shifted
            else:
                violation = float(bound - row @ shifted)
                value = (
                    shifted + (violation / row_norm_sq) * row
                    if violation > 0.0
                    else shifted
                )
            corrections[index] = shifted - value
        if (
            float(np.linalg.norm(value - previous, ord=np.inf)) <= tolerance
            and _max_halfspace_violation(matrix, bounds, value) <= tolerance
            and np.all(value >= lower - tolerance)
            and np.all(value <= upper + tolerance)
        ):
            break
    value = np.clip(value, lower, upper)
    return value, _max_halfspace_violation(matrix, bounds, value)


def _max_halfspace_violation(
    matrix: np.ndarray, bounds: np.ndarray, value: np.ndarray
) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    return float(max(0.0, np.max(bounds - matrix @ value)))


def _robot_jacobians_and_body_names(robot) -> tuple[np.ndarray, tuple[str, ...]]:
    view = getattr(robot, "_articulation_view", None)
    if view is None:
        raise RuntimeError("robot articulation view is unavailable")
    jacobians = _to_numpy(view.get_jacobians())
    if jacobians.ndim == 4 and jacobians.shape[0] == 1:
        jacobians = jacobians[0]
    if jacobians.ndim != 3 or jacobians.shape[1] != 6:
        raise RuntimeError(f"unexpected articulation Jacobian shape {jacobians.shape}")
    body_names = tuple(str(name) for name in view.body_names)
    return np.asarray(jacobians, dtype=float), body_names


def _body_jacobian(
    jacobians: np.ndarray, body_names: tuple[str, ...], link_name: str
) -> np.ndarray:
    if link_name not in body_names:
        raise RuntimeError(f"link {link_name!r} is absent from articulation metadata")
    body_index = body_names.index(link_name)
    if jacobians.shape[0] == len(body_names) - 1:
        jacobian_index = body_index - 1
    elif jacobians.shape[0] == len(body_names):
        jacobian_index = body_index
    else:
        raise RuntimeError(
            "cannot map body metadata to Jacobian rows: "
            f"bodies={len(body_names)} jacobians={jacobians.shape[0]}"
        )
    if jacobian_index < 0 or jacobian_index >= jacobians.shape[0]:
        raise RuntimeError(f"body {link_name!r} has no movable-link Jacobian")
    return jacobians[jacobian_index]


def _joint_speed_limits(robot, joint_indices: np.ndarray, config: CBFConfig) -> np.ndarray:
    limits = np.full(joint_indices.shape, config.max_joint_speed_rad_s, dtype=float)
    properties = getattr(robot, "dof_properties", None)
    if properties is None:
        return limits
    try:
        candidate = np.asarray(properties["maxVelocity"], dtype=float)[joint_indices]
    except Exception:
        return limits
    valid = np.isfinite(candidate) & (candidate > 0.0)
    limits[valid] = np.minimum(limits[valid], candidate[valid])
    return limits


def _clip_joint_positions(robot, joint_indices: np.ndarray, positions: np.ndarray) -> np.ndarray:
    properties = getattr(robot, "dof_properties", None)
    if properties is None:
        return positions
    try:
        lower = np.asarray(properties["lower"], dtype=float)[joint_indices]
        upper = np.asarray(properties["upper"], dtype=float)[joint_indices]
    except Exception:
        return positions
    valid = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
    result = positions.copy()
    result[valid] = np.clip(result[valid], lower[valid], upper[valid])
    return result


def _observation_vec3(
    observation: dict[str, np.ndarray], name: str
) -> np.ndarray | None:
    value = observation.get(name)
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


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float
    )


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
