from __future__ import annotations

from copy import deepcopy
import json
import unittest

import numpy as np

from v3_chan.ac_feedback.config import (
    DEFAULT_CONFIG,
    canonical_sha256,
    load_config,
)
from v3_chan.ac_feedback.validator import _Validator


class _Dataset:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype

    def __getitem__(self, key: object) -> np.ndarray:
        return self.values[key]

    def asstr(self) -> "_Dataset":
        return self


class _Group(dict[str, _Dataset]):
    def __init__(self, values: dict[str, object], attrs: dict[str, object]) -> None:
        super().__init__((name, _Dataset(value)) for name, value in values.items())
        self.attrs = attrs


def _validator() -> _Validator:
    validator = _Validator(
        config_path=DEFAULT_CONFIG,
        collection_path=None,
        participant_id=None,
        session_id=None,
        seed=None,
        mode=None,
        allow_partial_path=False,
    )
    validator.config = load_config()
    return validator


def _crossing_group(
    *,
    hand: str = "left",
    sign: int = 1,
) -> _Group:
    direction = "right_to_left" if sign > 0 else "left_to_right"
    left_signs = np.zeros(4, dtype=np.int8)
    right_signs = np.zeros(4, dtype=np.int8)
    (left_signs if hand == "left" else right_signs)[2] = sign
    values = {
        "crossing/intended_hand": np.asarray(["", "", "", ""]),
        "crossing/non_crossing_hand": np.asarray(["", "", "", ""]),
        "crossing/monitoring_active": np.asarray([0, 1, 1, 0], dtype=np.int8),
        "crossing/left_completion_sign": left_signs,
        "crossing/right_completion_sign": right_signs,
        "crossing/completed_count": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "crossing/first_completed_hand": np.asarray(["", "", hand, hand]),
        "crossing/first_completed_direction": np.asarray(
            ["not_observed", "not_observed", direction, direction]
        ),
        "crossing/multiple_crossing": np.zeros(4, dtype=np.int8),
        "control/source_step": np.asarray([10, 11, 12, 13], dtype=np.int64),
        "control/sim_time": np.asarray([1.0, 1.1, 1.2, 1.3]),
        "control/trial_state": np.asarray(
            [
                "SHOW_HAND_CROSSING_CUE",
                "TRACK_SAFETY_RESPONSE_EPISODE",
                "TRACK_SAFETY_RESPONSE_EPISODE",
                "PAUSE_NOMINAL_TASK_PROGRESSION",
            ]
        ),
        "human_next/valid_mask": np.asarray(
            [[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1]],
            dtype=np.int8,
        ),
    }
    attrs = {
        # These scheduled provenance fields deliberately disagree with the
        # observed free crossing and must not affect compliance validation.
        "crossing_hand": "right" if hand == "left" else "left",
        "direction": "left_to_right" if direction == "right_to_left" else "right_to_left",
        "speed": "fast",
        "severity": "threat",
        "actual_crossing_count": 1,
        "actual_crossing_hand": hand,
        "actual_crossing_direction": direction,
        "crossing_start_step": 11,
        "crossing_end_step": 12,
        "crossing_start_simulation_time_s": 1.1,
        "crossing_end_simulation_time_s": 1.2,
        "crossing_detector_semantics": "free_single_crossing_bilateral_gate_v1",
        "crossing_monitoring_valid_rate": 0.5,
        "off_protocol": 0,
        "protocol_valid": 1,
    }
    return _Group(values, attrs)


def _corridor_group() -> _Group:
    center = [0.4, 0.1, 1.1]
    frame = {
        "semantics": "free_single_crossing_internal_bilateral_gate_v1",
        "detector_semantics": "free_single_crossing_bilateral_gate_v1",
        "participant_visible": False,
        "participant_prescribed_hand": False,
        "participant_prescribed_direction": False,
        "participant_prescribed_speed": False,
        "participant_prescribed_path": False,
        "center_world_m": center,
        "tangent_world": [0.0, 1.0, 0.0],
        "endpoint_a_world_m": [0.4, -0.1, 1.1],
        "endpoint_b_world_m": [0.4, 0.3, 1.1],
        "gate_half_span_m": 0.2,
        "gate_radius_m": 0.35,
        "interaction_band_radius_m": 0.22,
    }
    audit = {
        "semantics": "free_single_crossing_internal_bilateral_gate_v1",
        "phase": "transport",
        "detector_semantics": "free_single_crossing_bilateral_gate_v1",
        "scheduled_crossing_factors_applied": False,
        "controller": "frozen_bc_plus_assigned_cbf",
        "cbf_enabled": True,
        "trigger_event": 5,
        "trigger_observation_source_step": 4,
        "trigger_observation_side": "after",
        "calibration_control_step": 5,
        "trigger_simulation_time_s": 1.0,
        "cue_lead_clock": "monotonic_wall",
        "trigger_ee_position_world_m": center,
        "geometry_valid": True,
        "internal_detection_frame": frame,
    }
    audit["sha256"] = canonical_sha256({"live_corridor": audit})
    planned = {
        **frame,
        "reference_semantics": "free_single_crossing_internal_bilateral_gate_v1",
        "live_trigger_corridor": audit,
    }
    values = {
        "control/source_step": np.asarray([4], dtype=np.int64),
        "control/result_step": np.asarray([5], dtype=np.int64),
        "control/sim_time": np.asarray([1.0]),
        "robot/ee_position_world_m": np.asarray([center]),
        "task/controller_event_before": np.asarray([4], dtype=np.int64),
        "task/controller_event_after": np.asarray([5], dtype=np.int64),
    }
    return _Group(
        values,
        {
            "task_phase": "transport",
            "planned_corridor_json": json.dumps(planned),
        },
    )


class FreeCrossingValidatorTests(unittest.TestCase):
    def test_signed_crossing_reconstructs_without_scheduled_compliance(self) -> None:
        for hand, sign in (("left", 1), ("right", -1)):
            with self.subTest(hand=hand, sign=sign):
                validator = _validator()
                validator._validate_free_crossing_rows(
                    _crossing_group(hand=hand, sign=sign),
                    "/trials/trial_000000",
                    off_protocol=False,
                    protocol_valid=True,
                )
                self.assertEqual(validator.issues, [])

    def test_direction_sign_convention_is_canonical_plus_y(self) -> None:
        group = _crossing_group(hand="left", sign=1)
        group["crossing/first_completed_direction"] = _Dataset(
            ["not_observed", "not_observed", "left_to_right", "left_to_right"]
        )
        group.attrs["actual_crossing_direction"] = "left_to_right"
        validator = _validator()
        validator._validate_free_crossing_rows(
            group,
            "/trials/trial_000000",
            off_protocol=False,
            protocol_valid=True,
        )
        self.assertIn(
            "crossing.direction_reconstruction",
            {issue.code for issue in validator.issues},
        )

    def test_legacy_prescription_and_monitor_rate_tampering_are_rejected(self) -> None:
        group = _crossing_group()
        group["crossing/intended_hand"] = _Dataset(["", "left", "", ""])
        group.attrs["crossing_monitoring_valid_rate"] = 1.0
        validator = _validator()
        validator._validate_free_crossing_rows(
            group,
            "/trials/trial_000000",
            off_protocol=False,
            protocol_valid=True,
        )
        codes = {issue.code for issue in validator.issues}
        self.assertIn("crossing.legacy_prescription", codes)
        self.assertIn("crossing.monitor_valid_rate", codes)

    def test_multiple_crossing_cumulative_evidence_invalidates_protocol_valid(self) -> None:
        group = _crossing_group()
        group["crossing/right_completion_sign"] = _Dataset([0, 0, 0, -1])
        group["crossing/monitoring_active"] = _Dataset([0, 1, 1, 1])
        group["crossing/completed_count"] = _Dataset([0, 0, 1, 2])
        group["crossing/multiple_crossing"] = _Dataset([0, 0, 0, 1])
        group.attrs["actual_crossing_count"] = 2
        group.attrs["crossing_monitoring_valid_rate"] = 2.0 / 3.0
        validator = _validator()
        validator._validate_free_crossing_rows(
            group,
            "/trials/trial_000000",
            off_protocol=False,
            protocol_valid=True,
        )
        self.assertIn(
            "crossing.protocol_valid", {issue.code for issue in validator.issues}
        )

    def test_monitoring_mask_is_bound_to_the_crossing_fsm_window(self) -> None:
        group = _crossing_group()
        group["crossing/monitoring_active"] = _Dataset([0, 0, 1, 0])
        group.attrs["crossing_monitoring_valid_rate"] = 0.0
        validator = _validator()
        validator._validate_free_crossing_rows(
            group,
            "/trials/trial_000000",
            off_protocol=False,
            protocol_valid=True,
        )
        self.assertIn(
            "crossing.monitor_window", {issue.code for issue in validator.issues}
        )

    def test_start_may_be_any_matching_monitor_row_but_end_is_first_completion(self) -> None:
        group = _crossing_group()
        group.attrs["crossing_start_step"] = 10
        group.attrs["crossing_start_simulation_time_s"] = 1.0
        validator = _validator()
        validator._validate_free_crossing_rows(
            group,
            "/trials/trial_000000",
            off_protocol=False,
            protocol_valid=True,
        )
        self.assertIn("crossing.clock", {issue.code for issue in validator.issues})

        group.attrs["crossing_start_step"] = 11
        group.attrs["crossing_start_simulation_time_s"] = 1.1
        group.attrs["crossing_end_step"] = 13
        group.attrs["crossing_end_simulation_time_s"] = 1.3
        validator = _validator()
        validator._validate_free_crossing_rows(
            group,
            "/trials/trial_000000",
            off_protocol=False,
            protocol_valid=True,
        )
        self.assertIn("crossing.clock", {issue.code for issue in validator.issues})

    def test_hidden_corridor_audit_binds_hash_geometry_and_trigger(self) -> None:
        group = _corridor_group()
        validator = _validator()
        validator._validate_direct_live_corridor(
            group, "/trials/trial_000000"
        )
        self.assertEqual(validator.issues, [])

        planned = json.loads(group.attrs["planned_corridor_json"])
        tampered = deepcopy(planned)
        tampered["participant_visible"] = True
        group.attrs["planned_corridor_json"] = json.dumps(tampered)
        validator = _validator()
        validator._validate_direct_live_corridor(
            group, "/trials/trial_000000"
        )
        codes = {issue.code for issue in validator.issues}
        self.assertIn("corridor.participant_hidden", codes)
        self.assertIn("corridor.geometry_binding", codes)

        tampered = deepcopy(planned)
        tampered["live_trigger_corridor"]["trigger_observation_side"] = "before"
        group.attrs["planned_corridor_json"] = json.dumps(tampered)
        validator = _validator()
        validator._validate_direct_live_corridor(
            group, "/trials/trial_000000"
        )
        codes = {issue.code for issue in validator.issues}
        self.assertIn("corridor.audit_hash", codes)
        self.assertIn("corridor.trigger_binding", codes)

    def test_marker_mapping_is_right_ab_regardless_of_scheduled_hand(self) -> None:
        group = _Group(
            {
        "control/result_step": np.asarray([5], dtype=np.int64),
                "control/sim_time": np.asarray([1.2]),
                "control/post_step_monotonic_ns": np.asarray([120], dtype=np.int64),
                "control/unix_time_ns": np.asarray([220], dtype=np.int64),
                "response/phase": np.asarray(["CBF_ACTIVE"]),
                "response/time_since_response_onset_s": np.asarray([0.2]),
                "response/time_since_recovery_onset_s": np.asarray([-1.0]),
            },
            {
                "condition_id": "A_reactive",
                "encounter_id": "enc-1",
                "crossing_hand": "left",
                "start_monotonic_ns": 100,
                "end_monotonic_ns": 140,
                "start_unix_ns": 200,
                "end_unix_ns": 240,
            },
        )
        common = {
            "participant_id": "P00",
            "session_id": "S00",
            "encounter_id": "enc-1",
            "condition_id": "A_reactive",
            "lambda_s": 0.0,
            "response_phase": "CBF_ACTIVE",
            "control_step": 5,
            "simulation_time_s": 1.2,
            "monotonic_ns": 120,
            "unix_ns": 220,
            "time_since_response_onset_s": 0.2,
            "time_since_recovery_onset_s": -1.0,
            "controller_hand": "right",
        }
        for scheduled_hand in ("left", "right"):
            group.attrs["crossing_hand"] = scheduled_hand
            for marker_type, button in (
                ("realtime_behavior_anomaly", "a"),
                ("realtime_safety_concern", "b"),
            ):
                validator = _validator()
                validator._validate_marker(
                    {
                        **common,
                        "marker_type": marker_type,
                        "button": button,
                        "button_identifier": f"right.{button}",
                    },
                    group,
                    "/trials/trial_000000",
                    "P00",
                    "S00",
                    "enc-1",
                )
                self.assertEqual(validator.issues, [])


if __name__ == "__main__":
    unittest.main()
