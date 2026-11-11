"""Per-control-step rows for ``bc_cbf_online_explicit_feedback_v1``.

The existing action tracer remains authoritative for the BC -> RMPFlow -> CBF
-> articulation boundary.  This module only extends that evidence with the
single-crossing protocol, raw XR orientation/velocity, object state, and clear
runtime-vs-outcome provenance.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from .recorder import DatasetSpec, _row_specs
from .rows import build_transition_row


def online_row_specs(joint_count: int, arm_count: int) -> dict[str, DatasetSpec]:
    specs = dict(_row_specs(joint_count, arm_count))
    additions: dict[str, DatasetSpec] = {
        "control/unix_time_ns": DatasetSpec(np.int64),
        "control/control_mode": DatasetSpec("str"),
        "control/trial_state": DatasetSpec("str"),
        "control/trial_id": DatasetSpec("str"),
        "control/query_id": DatasetSpec("str"),
        "human/left_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "human/right_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "human/left_velocity_raw_world_m_s": DatasetSpec(np.float64, (3,)),
        "human/right_velocity_raw_world_m_s": DatasetSpec(np.float64, (3,)),
        "human/left_velocity_filtered_world_m_s": DatasetSpec(np.float64, (3,)),
        "human/right_velocity_filtered_world_m_s": DatasetSpec(np.float64, (3,)),
        "human/left_velocity_valid": DatasetSpec(np.int8),
        "human/right_velocity_valid": DatasetSpec(np.int8),
        "human_next/left_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "human_next/right_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "robot/ee_position_world_m": DatasetSpec(np.float64, (3,)),
        "robot/ee_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "robot/ee_linear_velocity_world_m_s": DatasetSpec(np.float64, (3,)),
        "object/cube_position_world_m": DatasetSpec(np.float64, (3,)),
        "object/cube_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "object/goal_position_world_m": DatasetSpec(np.float64, (3,)),
        "object/goal_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "task/experimental_phase": DatasetSpec("str"),
        "task/phase_advance_enabled": DatasetSpec(np.int8),
        "task/has_grasped": DatasetSpec(np.int8),
        "task/release_observed": DatasetSpec(np.int8),
        "task/object_drop": DatasetSpec(np.int8),
        "actions/controller_task_action": DatasetSpec(np.float32, (5,)),
        "actions/controller_task_action_reason": DatasetSpec("str"),
        "cbf/active_constraint_identity_available": DatasetSpec(np.int8),
        "cbf/active_constraint_summary_json": DatasetSpec("str"),
        "cbf/active_hand": DatasetSpec("str"),
        "cbf/active_robot_link": DatasetSpec("str"),
        "cbf/active_robot_collider": DatasetSpec("str"),
        "cbf/identity_semantics": DatasetSpec("str"),
        "safety/min_ttc_s": DatasetSpec(np.float64),
        "safety/ttc_valid": DatasetSpec(np.int8),
        "safety/dynamic_measurement_valid": DatasetSpec(np.int8),
        "safety/max_closing_speed_m_s": DatasetSpec(np.float64),
        "safety/closing": DatasetSpec(np.int8),
        "safety/left_proximal_surface_gap_m": DatasetSpec(np.float64),
        "safety/right_proximal_surface_gap_m": DatasetSpec(np.float64),
        "safety/left_proximal_collider_path": DatasetSpec("str"),
        "safety/right_proximal_collider_path": DatasetSpec("str"),
        "crossing/active": DatasetSpec(np.int8),
        "crossing/state": DatasetSpec("str"),
        "crossing/started_now": DatasetSpec(np.int8),
        "crossing/intended_completed_now": DatasetSpec(np.int8),
        "crossing/reverse_completed_now": DatasetSpec(np.int8),
        "crossing/progress_fraction": DatasetSpec(np.float64),
        "crossing/perpendicular_deviation_m": DatasetSpec(np.float64),
        "crossing/intended_hand": DatasetSpec("str"),
        "crossing/observed_hand": DatasetSpec("str"),
        "provenance/runtime_feature_cutoff_available": DatasetSpec(np.int8),
    }
    overlap = sorted(set(specs).intersection(additions))
    if overlap:
        raise RuntimeError(f"online row spec unexpectedly overlaps base paths: {overlap}")
    specs.update(additions)
    return specs


def build_online_transition_row(
    *,
    controller_task_action: Sequence[float],
    controller_task_action_reason: str,
    control_mode: str,
    trial_state: str,
    trial_id: str,
    query_id: str,
    experimental_phase: str,
    phase_advance_enabled: bool,
    object_state: Mapping[str, Any],
    ee_state: Mapping[str, Any],
    crossing_state: Mapping[str, Any],
    proximal_state: Mapping[str, Any] | None = None,
    **base_kwargs: Any,
) -> dict[str, Any]:
    """Build one strict row while retaining the proven existing base checks."""

    row = build_transition_row(**base_kwargs)
    info_before = dict(base_kwargs["info_before"])
    info_after = dict(base_kwargs["info_after"])
    tracking = base_kwargs["tracking"]
    tracking_next = base_kwargs["tracking_next"]
    dynamic = _mapping(info_before.get("dynamic_safety"))
    proximal = _mapping(proximal_state)
    cbf = _mapping(info_after.get("physical_safety"))
    cbf_active = bool(cbf.get("active", False))
    strict_state = _mapping(
        _mapping(info_after.get("strict_task_semantics")).get("state")
    )
    terminal_failure = str(
        strict_state.get("failure_reason", "")
        or info_after.get("task_terminal_reason", "")
    ).lower()
    has_grasped = bool(info_after.get("has_grasped_cube", False))
    event_after = int(info_after.get("controller_event", -1))

    active_constraints = []
    for hand in ("left", "right"):
        gap = _finite_or_default(
            info_before.get(f"{hand}_end_effector_surface_gap_m"), 10.0
        )
        if gap <= 0.13:
            active_constraints.append(
                {
                    "hand": hand,
                    "robot_link": str(
                        info_before.get(f"closest_link_{hand}", "") or ""
                    ),
                    "collider_path": str(
                        info_before.get(f"closest_collider_{hand}", "") or ""
                    ),
                    "surface_gap_m": gap,
                }
            )
    identity_available = bool(
        cbf_active
        and len(active_constraints) == int(cbf.get("constraint_count", -1))
        and all(item["robot_link"] and item["collider_path"] for item in active_constraints)
    )
    identity_semantics = (
        "exact_reconstruction_from_frozen_cbf_filter_input_v1"
        if identity_available
        else "inactive_or_identity_unavailable"
    )
    active_hands = ",".join(item["hand"] for item in active_constraints)
    active_links = ",".join(item["robot_link"] for item in active_constraints)
    active_colliders = ",".join(
        item["collider_path"] for item in active_constraints
    )
    summary = {
        "constraint_count": int(cbf.get("constraint_count", 0)),
        "exact_identity_available": identity_available,
        "constraints": active_constraints,
        "derivation": (
            "same safety_result and observation consumed by "
            "DistalLinkVelocityCBF._constraints; successful runtime row "
            "proves constructed count equals diagnostics"
        ),
    }

    min_ttc = _finite_or_default(dynamic.get("min_ttc_s"), 10.0)
    ttc_valid = bool(dynamic.get("ttc_valid", False))
    closing_speed = max(
        0.0, _finite_or_default(dynamic.get("max_closing_speed_mps"), 0.0)
    )
    row.update(
        {
            "control/unix_time_ns": int(base_kwargs["wall_time_unix_ns"]),
            "control/control_mode": str(control_mode),
            "control/trial_state": str(trial_state),
            "control/trial_id": str(trial_id),
            "control/query_id": str(query_id),
            "human/left_orientation_wxyz": _vec(
                tracking.left.orientation_wxyz, 4, "left orientation"
            ),
            "human/right_orientation_wxyz": _vec(
                tracking.right.orientation_wxyz, 4, "right orientation"
            ),
            "human_next/left_orientation_wxyz": _vec(
                tracking_next.left.orientation_wxyz, 4, "next left orientation"
            ),
            "human_next/right_orientation_wxyz": _vec(
                tracking_next.right.orientation_wxyz, 4, "next right orientation"
            ),
            "human/left_velocity_raw_world_m_s": _vec_or_zero(
                dynamic.get("left_hand_vel_raw_mps"), 3
            ),
            "human/right_velocity_raw_world_m_s": _vec_or_zero(
                dynamic.get("right_hand_vel_raw_mps"), 3
            ),
            "human/left_velocity_filtered_world_m_s": _vec_or_zero(
                dynamic.get("left_hand_vel_filtered_mps"), 3
            ),
            "human/right_velocity_filtered_world_m_s": _vec_or_zero(
                dynamic.get("right_hand_vel_filtered_mps"), 3
            ),
            "human/left_velocity_valid": int(
                bool(dynamic.get("left_hand_velocity_valid", False))
            ),
            "human/right_velocity_valid": int(
                bool(dynamic.get("right_hand_velocity_valid", False))
            ),
            "robot/ee_position_world_m": _vec(
                ee_state.get("position_world_m"), 3, "EE position"
            ),
            "robot/ee_orientation_wxyz": _vec(
                ee_state.get("orientation_wxyz"), 4, "EE orientation"
            ),
            "robot/ee_linear_velocity_world_m_s": _vec(
                ee_state.get("linear_velocity_world_m_s"), 3, "EE velocity"
            ),
            "object/cube_position_world_m": _vec(
                object_state.get("cube_position_world_m"), 3, "cube position"
            ),
            "object/cube_orientation_wxyz": _vec(
                object_state.get("cube_orientation_wxyz"), 4, "cube orientation"
            ),
            "object/goal_position_world_m": _vec(
                object_state.get("goal_position_world_m"), 3, "goal position"
            ),
            "object/goal_orientation_wxyz": _vec(
                object_state.get("goal_orientation_wxyz"), 4, "goal orientation"
            ),
            "task/experimental_phase": str(experimental_phase),
            "task/phase_advance_enabled": int(bool(phase_advance_enabled)),
            "task/has_grasped": int(has_grasped),
            "task/release_observed": int(
                bool(strict_state.get("success_latched", False))
                or (event_after >= 7 and not has_grasped)
            ),
            "task/object_drop": int(
                "drop" in terminal_failure
                or "grasp_lost" in terminal_failure
            ),
            "actions/controller_task_action": _vec(
                controller_task_action, 5, "controller task action"
            ).astype(np.float32),
            "actions/controller_task_action_reason": str(controller_task_action_reason),
            "cbf/active_constraint_identity_available": int(identity_available),
            "cbf/active_constraint_summary_json": json.dumps(
                summary, sort_keys=True, separators=(",", ":")
            ),
            "cbf/active_hand": active_hands if cbf_active else "",
            "cbf/active_robot_link": active_links if cbf_active else "",
            "cbf/active_robot_collider": active_colliders if cbf_active else "",
            "cbf/identity_semantics": identity_semantics,
            "safety/min_ttc_s": min_ttc,
            "safety/ttc_valid": int(ttc_valid),
            "safety/dynamic_measurement_valid": int(
                bool(dynamic.get("dynamic_measurement_valid", False))
            ),
            "safety/max_closing_speed_m_s": closing_speed,
            "safety/closing": int(closing_speed > 0.0),
            "safety/left_proximal_surface_gap_m": _finite_or_default(
                proximal.get("left_gap_m"), 10.0
            ),
            "safety/right_proximal_surface_gap_m": _finite_or_default(
                proximal.get("right_gap_m"), 10.0
            ),
            "safety/left_proximal_collider_path": str(
                proximal.get("left_collider_path", "") or ""
            ),
            "safety/right_proximal_collider_path": str(
                proximal.get("right_collider_path", "") or ""
            ),
            "crossing/active": int(bool(crossing_state.get("active", False))),
            "crossing/state": str(
                crossing_state.get("state", "not_observed")
            ),
            "crossing/started_now": int(
                bool(crossing_state.get("started_now", False))
            ),
            "crossing/intended_completed_now": int(
                bool(crossing_state.get("intended_completed_now", False))
            ),
            "crossing/reverse_completed_now": int(
                bool(crossing_state.get("reverse_completed_now", False))
            ),
            "crossing/progress_fraction": _finite_or_default(
                crossing_state.get("progress_fraction"), 0.0
            ),
            "crossing/perpendicular_deviation_m": _finite_or_default(
                crossing_state.get("perpendicular_deviation_m"), 0.0
            ),
            "crossing/intended_hand": str(crossing_state.get("intended_hand", "")),
            "crossing/observed_hand": str(crossing_state.get("observed_hand", "")),
            "provenance/runtime_feature_cutoff_available": int(
                bool(crossing_state.get("runtime_feature_cutoff_available", False))
            ),
        }
    )
    return row


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _vec(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite ({size},) vector")
    return array.copy()


def _vec_or_zero(value: Any, size: int) -> np.ndarray:
    try:
        return _vec(value, size, "optional vector")
    except (TypeError, ValueError):
        return np.zeros(size, dtype=np.float64)


def _finite_or_default(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)
