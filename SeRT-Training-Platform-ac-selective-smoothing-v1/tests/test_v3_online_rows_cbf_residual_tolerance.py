from __future__ import annotations

import json
import unittest

import numpy as np

from v3_chan.ac_feedback.config import DEFAULT_CONFIG
from v3_chan.ac_feedback.online_rows import _constraint_evidence
from v3_chan.ac_feedback.validator import _Validator
from v3_chan.physical_safety_controllers import CBFConfig


def _evidence(filtered_residual_mps: float) -> dict[str, object]:
    return {
        "hand": "left",
        "closest_link": "panda_hand",
        "closest_collider_path": "/World/Franka/panda_hand/collisions",
        "raw_surface_gap_m": 0.06,
        "prediction_buffer_m": 0.01,
        "effective_safe_gap_m": 0.06,
        "barrier_value_m": 0.0,
        "buffered_current_gap_m": 0.05,
        "closing_speed_mps": 0.0,
        "lower_bound_mps": 0.0,
        "nominal_residual_mps": -0.01,
        "filtered_residual_mps": filtered_residual_mps,
    }


def _validator_group(filtered_residual_mps: float) -> dict[str, np.ndarray]:
    evidence = _evidence(filtered_residual_mps)
    summary = {
        "constraint_count": 1,
        "exact_identity_available": True,
        "constraints": [
            {
                "hand": "left",
                "robot_link": "panda_hand",
                "collider_path": "/World/Franka/panda_hand/collisions",
                "surface_gap_m": 0.06,
            }
        ],
        "derivation": "test fixture",
    }
    return {
        "cbf/constraint_evidence_json": np.asarray([json.dumps([evidence])]),
        "cbf/min_barrier_value_m": np.asarray([0.0]),
        "cbf/max_active_prediction_buffer_m": np.asarray([0.01]),
        "cbf/active_constraint_identity_available": np.asarray([1]),
        "cbf/active_hand": np.asarray(["left"]),
        "cbf/active_robot_link": np.asarray(["panda_hand"]),
        "cbf/active_robot_collider": np.asarray(
            ["/World/Franka/panda_hand/collisions"]
        ),
        "cbf/identity_semantics": np.asarray(
            ["exact_reconstruction_from_frozen_cbf_filter_input_v1"]
        ),
        "cbf/active_constraint_summary_json": np.asarray(
            [json.dumps(summary)]
        ),
    }


def _validator() -> _Validator:
    return _Validator(
        config_path=DEFAULT_CONFIG,
        collection_path=None,
        participant_id=None,
        session_id=None,
        seed=None,
        mode=None,
        allow_partial_path=False,
    )


class OnlineRowsCbfResidualToleranceTests(unittest.TestCase):
    def test_filtered_residual_within_solver_projection_tolerance_is_accepted(
        self,
    ) -> None:
        result = _constraint_evidence(
            {"constraint_evidence": [_evidence(-5e-6)]}
        )

        self.assertAlmostEqual(result[0]["filtered_residual_mps"], -5e-6)

    def test_filtered_residual_beyond_solver_projection_tolerance_is_rejected(
        self,
    ) -> None:
        residual = -1.0001e-5
        tolerance = CBFConfig().projection_tolerance

        with self.assertRaises(ValueError) as caught:
            _constraint_evidence({"constraint_evidence": [_evidence(residual)]})

        message = str(caught.exception)
        self.assertIn(f"residual={residual:.12g}", message)
        self.assertIn(f"projection_tolerance={tolerance:.12g}", message)

    def test_large_filtered_residual_violation_remains_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "filtered residual is unsafe"):
            _constraint_evidence({"constraint_evidence": [_evidence(-0.1)]})

    def test_offline_validator_uses_the_same_solver_tolerance(self) -> None:
        for residual, expected_issue in ((-5e-6, False), (-1.0001e-5, True)):
            with self.subTest(residual=residual):
                validator = _validator()
                validator._validate_constraint_evidence(
                    _validator_group(residual),
                    "/trials/trial_000000",
                    np.asarray([1]),
                    np.asarray([1]),
                )
                codes = {issue.code for issue in validator.issues}
                self.assertEqual(
                    "cbf.constraint_residual" in codes,
                    expected_issue,
                )


if __name__ == "__main__":
    unittest.main()
