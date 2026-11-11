from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from v3_chan.explicit_feedback.action_trace import (
    ActionStepTrace,
    CanonicalJointAction,
)
from v3_chan.explicit_feedback.policy import PolicyOutput, mask_human_observations
from v3_chan.explicit_feedback.recorder import _row_specs
from v3_chan.explicit_feedback.rows import (
    build_transition_row,
    validate_runtime_step,
)
from v3_chan.explicit_feedback.schema import (
    ACTION_TRACE_SCHEMA_VERSION,
    ProtocolViolation,
)
from v3_chan.explicit_feedback.tracking import TrackedPose, TrackingSnapshot


JOINT_NAMES = tuple(
    [f"panda_joint{index}" for index in range(1, 8)]
    + ["panda_finger_joint1", "panda_finger_joint2"]
)
ARM = tuple(range(7))


def _canonical(*, velocity: float, full: bool = False) -> CanonicalJointAction:
    count = len(JOINT_NAMES)
    mask = np.ones(count, dtype=bool) if full else np.asarray([1] * 7 + [0, 0], dtype=bool)
    values = np.zeros(count, dtype=np.float64)
    values[:7] = velocity
    effort_mask = np.zeros(count, dtype=bool)
    return CanonicalJointAction(
        joint_positions=values * 0.01,
        joint_positions_mask=mask.copy(),
        joint_velocities=values,
        joint_velocities_mask=mask.copy(),
        joint_efforts=np.zeros(count, dtype=np.float64),
        joint_efforts_mask=effort_mask,
        raw_joint_indices=None,
    )


def _trace(*, active: bool = True) -> ActionStepTrace:
    nominal = _canonical(velocity=0.2)
    filtered = _canonical(velocity=0.1 if active else 0.2)
    applied = _canonical(velocity=0.1 if active else 0.2, full=True)
    return ActionStepTrace(
        schema_version=ACTION_TRACE_SCHEMA_VERSION,
        step_id=(0, 0),
        joint_names=JOINT_NAMES,
        nominal_rmpflow=nominal,
        cbf_filtered=filtered,
        submitted=filtered,
        pre_applied=_canonical(velocity=0.0, full=True),
        post_applied=applied,
        timestamps_monotonic_ns={
            "step_started": 215,
            "pre_applied": 216,
            "nominal": 220,
            "filtered": 230,
            "submitted": 240,
            "apply_started": 240,
            "post_applied": 245,
            "step_finished": 246,
        },
        rmpflow_output_monotonic_ns=220,
        cbf_output_monotonic_ns=230,
        action_command_monotonic_ns=240,
        arm_action_submitted=True,
        pipeline_complete=True,
    )


def _pose(
    name: str,
    source: str,
    *,
    acquisition_monotonic_ns: int,
) -> TrackedPose:
    return TrackedPose(
        position_world=np.asarray([0.5, 0.0, 1.0]),
        orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        pose_valid=True,
        position_tracked=-1,
        tracking_status_known=False,
        source_name=source,
        source_path=f"/xr/{name}",
        acquisition_monotonic_ns=acquisition_monotonic_ns,
        pose_age_ms=0.00001,
        source_switched=False,
    )


def _tracking(
    *,
    sample_monotonic_ns: int = 195,
    acquisition_monotonic_ns: int = 190,
) -> TrackingSnapshot:
    return TrackingSnapshot(
        sample_monotonic_ns=sample_monotonic_ns,
        head=_pose(
            "head",
            "hmd_xr_physical",
            acquisition_monotonic_ns=acquisition_monotonic_ns,
        ),
        left=_pose(
            "left",
            "xr_physical",
            acquisition_monotonic_ns=acquisition_monotonic_ns,
        ),
        right=_pose(
            "right",
            "xr_raw_physical",
            acquisition_monotonic_ns=acquisition_monotonic_ns,
        ),
        valid_mask=np.ones(3, dtype=np.int8),
        invalid_reasons=(),
    )


def _cbf(*, active: bool = True) -> dict[str, object]:
    return {
        "active": active,
        "intervention_available": True,
        "constraint_count": 1 if active else 0,
        "tracked_hand_count": 2,
        "valid_hand_count": 2,
        "intervention_norm_radps": np.sqrt(7.0) * 0.1 if active else 0.0,
        "nominal_velocity_norm_radps": np.sqrt(7.0) * 0.2,
        "filtered_velocity_norm_radps": np.sqrt(7.0) * (0.1 if active else 0.2),
        "max_constraint_violation_before_mps": 0.01 if active else 0.0,
        "max_constraint_violation_after_mps": 0.0,
        "slack_mps": 0.0,
        "buffered_current_gap_m": 0.10 if active else 10.0,
        "solve_time_ms": 0.2,
        "feasible": True,
        "fallback_applied": False,
        "solver_converged": True,
        "infeasibility_proven": False,
        "relaxed_solution_available": False,
        "relaxed_solution_applied": False,
        "intentional_human_absence": False,
        "failure_reasons": (),
        "status": "solved" if active else "inactive",
        "solver_backend": "test" if active else "none",
        "projection_status": "optimal" if active else "inactive",
    }


def _runtime_info_before() -> dict[str, object]:
    return {
        "geometry_valid": True,
        "left_end_effector_surface_gap_m": 0.10,
        "right_end_effector_surface_gap_m": 0.25,
        "min_hand_end_effector_surface_gap": 0.10,
    }


def test_runtime_row_builder_matches_the_recorder_schema() -> None:
    trace = _trace(active=True)
    tracking = _tracking()
    tracking_next = _tracking(
        sample_monotonic_ns=245,
        acquisition_monotonic_ns=235,
    )
    info_before = {
        "step": 0,
        "controller_event": 2,
        "controller_t": 0.4,
        "left_end_effector_surface_gap_m": 0.10,
        "right_end_effector_surface_gap_m": 0.25,
        "min_hand_end_effector_surface_gap": 0.10,
        "geometry_valid": True,
        "near_human": True,
    }
    info_after = {
        "step": 1,
        "sim_time": 0.01,
        "controller_event": 2,
        "controller_t": 0.41,
        "gripper_command": "none",
        "success": False,
        "task_terminal_reason": "",
        "physical_safety_controller": "cbf",
        "physical_safety": _cbf(active=True),
    }

    validate_runtime_step(
        info_before=info_before,
        info_after=info_after,
        trace=trace,
        tracking=tracking,
        arm_joint_indices=ARM,
    )
    row = build_transition_row(
        source_step=0,
        observation_monotonic_ns=200,
        preflight_monotonic_ns=202,
        wall_time_unix_ns=1_000,
        obs_raw=np.zeros(84, dtype=np.float32),
        obs_next=np.zeros(84, dtype=np.float32),
        policy_output=PolicyOutput(
            policy_input=mask_human_observations(
                np.zeros(84, dtype=np.float32)
            ),
            action=np.zeros(5, dtype=np.float32),
            inference_started_monotonic_ns=205,
            inference_completed_monotonic_ns=210,
        ),
        trace=trace,
        tracking=tracking,
        tracking_next=tracking_next,
        measured_positions_before=np.zeros(9),
        measured_velocities_before=np.zeros(9),
        measured_positions_after=np.zeros(9),
        measured_velocities_after=np.zeros(9),
        info_before=info_before,
        info_after=info_after,
        terminated=False,
        truncated=False,
        post_step_monotonic_ns=250,
        arm_joint_indices=ARM,
        encounter_id="encounter-1",
        encounter_state="encounter_active",
    )

    assert set(row) == set(_row_specs(9, 7))
    np.testing.assert_allclose(
        row["actions/applied_arm_joint_velocities_rad_s"],
        np.full(7, 0.1),
    )
    assert row["cbf/constraint_count"] == 1
    assert row["human/head_tracking_status_known"] == 0
    assert row["control/next_observation_monotonic_ns"] == 245


def test_runtime_validation_rejects_unsafe_cbf_before_row_is_written() -> None:
    info_after = {
        "physical_safety_controller": "cbf",
        "physical_safety": {**_cbf(active=True), "fallback_applied": True},
    }

    with pytest.raises(ProtocolViolation, match="fallback"):
        validate_runtime_step(
            info_before=_runtime_info_before(),
            info_after=info_after,
            trace=_trace(active=True),
            tracking=_tracking(),
            arm_joint_indices=ARM,
        )


def test_runtime_validation_requires_exact_inactive_cbf_passthrough() -> None:
    trace = _trace(active=False)
    changed = replace(
        trace.cbf_filtered,
        joint_velocities=trace.cbf_filtered.joint_velocities.copy(),
    )
    changed.joint_velocities[0] += 0.01
    trace = replace(trace, cbf_filtered=changed)

    with pytest.raises(ProtocolViolation, match="inactive CBF passthrough"):
        validate_runtime_step(
            info_before=_runtime_info_before(),
            info_after={
                "physical_safety_controller": "cbf",
                "physical_safety": _cbf(active=False),
            },
            trace=trace,
            tracking=_tracking(),
            arm_joint_indices=ARM,
        )


def test_runtime_validation_rejects_invalid_safety_geometry() -> None:
    with pytest.raises(ProtocolViolation, match="geometry"):
        validate_runtime_step(
            info_before={
                **_runtime_info_before(),
                "geometry_valid": False,
            },
            info_after={
                "physical_safety_controller": "cbf",
                "physical_safety": _cbf(active=True),
            },
            trace=_trace(active=True),
            tracking=_tracking(),
            arm_joint_indices=ARM,
        )
