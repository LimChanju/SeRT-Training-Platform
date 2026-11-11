from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v3_chan.ac_feedback.online_recorder import (
    OnlineExplicitFeedbackRecorder,
    _validate_free_crossing_summary,
    _validate_free_crossing_transition,
)
from v3_chan.ac_feedback.online_rows import (
    build_online_transition_row,
    online_row_specs,
)
from v3_chan.ac_feedback.online_schema import (
    ONLINE_PROTOCOL_VERSION,
    ONLINE_SCHEMA_VERSION,
)


def _crossing_row(**overrides):
    row = {
        "crossing/intended_hand": "",
        "crossing/non_crossing_hand": "",
        "crossing/monitoring_active": 1,
        "crossing/left_completion_sign": 0,
        "crossing/right_completion_sign": 0,
        "crossing/completed_count": 0,
        "crossing/first_completed_hand": "",
        "crossing/first_completed_direction": "not_observed",
        "crossing/multiple_crossing": 0,
    }
    row.update(overrides)
    return row


def _summary_group(*, multiple: bool = False):
    if multiple:
        right_sign = np.asarray([0, 0, -1], dtype=np.int8)
        counts = np.asarray([0, 1, 2], dtype=np.int64)
        multiple_mask = np.asarray([0, 0, 1], dtype=np.int8)
    else:
        right_sign = np.zeros(3, dtype=np.int8)
        counts = np.asarray([0, 1, 1], dtype=np.int64)
        multiple_mask = np.zeros(3, dtype=np.int8)
    return {
        "crossing/monitoring_active": np.ones(3, dtype=np.int8),
        "crossing/left_completion_sign": np.asarray([0, 1, 0], dtype=np.int8),
        "crossing/right_completion_sign": right_sign,
        "crossing/completed_count": counts,
        "crossing/multiple_crossing": multiple_mask,
        "crossing/first_completed_hand": np.asarray(["", "left", "left"]),
        "crossing/first_completed_direction": np.asarray(
            ["not_observed", "right_to_left", "right_to_left"]
        ),
        "control/source_step": np.asarray([10, 11, 12], dtype=np.int64),
        "control/sim_time": np.asarray([1.0, 1.1, 1.2]),
        "human_next/valid_mask": np.ones((3, 3), dtype=np.int8),
    }


def _summary(*, crossing_count: int = 1):
    return {
        "actual_crossing_count": crossing_count,
        "actual_crossing_hand": "left",
        "actual_crossing_direction": "right_to_left",
        "crossing_start_step": 10,
        "crossing_end_step": 11,
        "crossing_start_simulation_time_s": 1.0,
        "crossing_end_simulation_time_s": 1.1,
        "crossing_detector_semantics": "free_single_crossing_bilateral_gate_v1",
        "crossing_monitoring_valid_rate": 1.0,
    }


class FreeCrossingSchemaTests(unittest.TestCase):
    def test_versions_and_row_specs_are_v3_free_crossing(self) -> None:
        self.assertEqual(
            ONLINE_SCHEMA_VERSION, "bc_cbf_recovery_ac_explicit_feedback_v2"
        )
        self.assertEqual(
            ONLINE_PROTOCOL_VERSION,
            "ac_selective_smoothing_free_single_crossing_pilot_v3",
        )
        specs = online_row_specs(9, 7)
        expected = {
            "crossing/monitoring_active": np.int8,
            "crossing/left_completion_sign": np.int8,
            "crossing/right_completion_sign": np.int8,
            "crossing/completed_count": np.int64,
            "crossing/first_completed_hand": "str",
            "crossing/first_completed_direction": "str",
            "crossing/multiple_crossing": np.int8,
        }
        for path, dtype in expected.items():
            self.assertEqual(specs[path].dtype, dtype)

    @staticmethod
    def _build(crossing_state):
        tracking = SimpleNamespace(
            left=SimpleNamespace(orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0])),
            right=SimpleNamespace(orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0])),
        )
        cbf = {
            "objective_mode": "joint_nominal",
            "correction_smoothness_weight": 0.0,
            "correction_radps": np.zeros(7),
            "previous_correction_radps": np.zeros(7),
            "constraint_count": 0,
            "active": False,
            "feasible": True,
            "solver_converged": True,
            "failure_reasons": (),
        }
        with patch(
            "v3_chan.ac_feedback.online_rows.build_transition_row",
            return_value={"actions/bc_task_action": np.zeros(5)},
        ):
            return build_online_transition_row(
                controller_task_action=np.zeros(5),
                controller_task_action_reason="nominal_bc",
                control_mode="nominal_bc",
                trial_state="RESET",
                trial_id="trial",
                query_id="query",
                condition_id="A_reactive",
                objective_mode="joint_nominal",
                lambda_s=0.0,
                response_phase="PRE_RESPONSE",
                decision_context_available=False,
                experimental_phase="reach_approach",
                phase_advance_enabled=True,
                object_state={
                    "cube_position_world_m": np.zeros(3),
                    "cube_orientation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
                    "cube_linear_velocity_world_m_s": np.zeros(3),
                    "goal_position_world_m": np.zeros(3),
                    "goal_orientation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
                },
                ee_state={
                    "position_world_m": np.zeros(3),
                    "orientation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
                    "linear_velocity_world_m_s": np.zeros(3),
                },
                crossing_state=crossing_state,
                info_before={},
                info_after={"physical_safety": cbf},
                tracking=tracking,
                tracking_next=tracking,
                arm_joint_indices=tuple(range(7)),
                wall_time_unix_ns=1,
            )

    def test_online_row_builder_has_safe_free_crossing_defaults(self) -> None:
        row = self._build({})
        self.assertEqual(row["crossing/intended_hand"], "")
        self.assertEqual(row["crossing/non_crossing_hand"], "")
        self.assertEqual(row["crossing/monitoring_active"], 0)
        self.assertEqual(row["crossing/left_completion_sign"], 0)
        self.assertEqual(row["crossing/right_completion_sign"], 0)
        self.assertEqual(row["crossing/completed_count"], 0)
        self.assertEqual(row["crossing/first_completed_hand"], "")
        self.assertEqual(
            row["crossing/first_completed_direction"], "not_observed"
        )
        self.assertEqual(row["crossing/multiple_crossing"], 0)

    def test_online_row_builder_records_free_crossing_payload(self) -> None:
        row = self._build(
            {
                "monitoring_active": True,
                "left_completion_sign": -1,
                "completed_count": 1,
                "first_completed_hand": "left",
                "first_completed_direction": "left_to_right",
            }
        )
        self.assertEqual(row["crossing/monitoring_active"], 1)
        self.assertEqual(row["crossing/left_completion_sign"], -1)
        self.assertEqual(row["crossing/completed_count"], 1)
        self.assertEqual(row["crossing/first_completed_hand"], "left")
        self.assertEqual(
            row["crossing/first_completed_direction"], "left_to_right"
        )

    def test_transition_validator_rejects_legacy_assignment_and_bad_count(self) -> None:
        valid = _crossing_row(
            **{
                "crossing/left_completion_sign": 1,
                "crossing/completed_count": 1,
                "crossing/first_completed_hand": "left",
                "crossing/first_completed_direction": "right_to_left",
            }
        )
        _validate_free_crossing_transition(valid, previous=None)
        with self.assertRaisesRegex(RuntimeError, "empty legacy"):
            _validate_free_crossing_transition(
                {**valid, "crossing/intended_hand": "left"}, previous=None
            )
        with self.assertRaisesRegex(RuntimeError, "cumulative"):
            _validate_free_crossing_transition(
                {**valid, "crossing/completed_count": 2}, previous=None
            )

    def test_summary_is_bound_to_row_hand_direction_count_and_clocks(self) -> None:
        group = _summary_group()
        summary = _summary()
        _validate_free_crossing_summary(summary, group, protocol_valid=True)
        with self.assertRaisesRegex(RuntimeError, "actual_crossing_count"):
            _validate_free_crossing_summary(
                {**summary, "actual_crossing_count": 2},
                group,
                protocol_valid=False,
            )
        with self.assertRaisesRegex(RuntimeError, "hand/direction"):
            _validate_free_crossing_summary(
                {**summary, "actual_crossing_hand": "right"},
                group,
                protocol_valid=True,
            )

    def test_multiple_crossing_cannot_be_protocol_valid(self) -> None:
        group = _summary_group(multiple=True)
        summary = _summary(crossing_count=2)
        _validate_free_crossing_summary(summary, group, protocol_valid=False)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            _validate_free_crossing_summary(summary, group, protocol_valid=True)

    def test_end_trial_requires_new_free_crossing_summary_fields(self) -> None:
        recorder = object.__new__(OnlineExplicitFeedbackRecorder)
        recorder._closed = False
        recorder._active_trial = object()
        recorder._active_plan = object()
        with self.assertRaisesRegex(RuntimeError, "actual_crossing_hand") as error:
            recorder.end_trial({})
        for field in (
            "crossing_start_step",
            "crossing_end_step",
            "crossing_start_simulation_time_s",
            "crossing_end_simulation_time_s",
            "crossing_detector_semantics",
            "crossing_monitoring_valid_rate",
        ):
            self.assertIn(field, str(error.exception))


if __name__ == "__main__":
    unittest.main()
