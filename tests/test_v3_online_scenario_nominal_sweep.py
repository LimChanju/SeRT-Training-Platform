from types import SimpleNamespace

import numpy as np
import pytest

from v3_chan.explicit_feedback.online_scenario import (
    NOMINAL_BC_SWEEP_SEMANTICS,
    CrossingCorridor,
    NominalBCCorridorSweep,
)


class _DynamicPlaneGeometry:
    def __init__(self) -> None:
        self.surface_z_m = 0.0
        self.invalid_z_m: float | None = None
        self.queries: list[tuple[str, np.ndarray]] = []

    def evaluate_hand(self, hand: str, position_world_m: np.ndarray):
        position = np.asarray(position_world_m, dtype=float).copy()
        self.queries.append((hand, position))
        valid = not (
            self.invalid_z_m is not None
            and np.isclose(position[2], self.invalid_z_m)
        )
        return SimpleNamespace(
            geometry_valid=valid,
            surface_gap_m=(
                float(position[2] - self.surface_z_m) if valid else np.inf
            ),
            closest_link="panda_link7" if valid else "",
            closest_collider_path=(
                "/World/Franka/panda_link7/collider" if valid else ""
            ),
        )


def _base_corridor() -> CrossingCorridor:
    return CrossingCorridor(
        start_world_m=np.asarray([0.0, -0.5, 0.0]),
        end_world_m=np.asarray([0.0, 0.5, 0.0]),
        center_world_m=np.asarray([0.0, 0.0, 0.0]),
        tangent_world=np.asarray([0.0, 1.0, 0.0]),
        outward_normal_world=np.asarray([0.0, 0.0, 1.0]),
        planned_minimum_surface_gap_m=0.0,
        closest_link="panda_link7",
        closest_collider_path="/World/Franka/panda_link7/collider",
        calibration_iterations=1,
    )


def _sweep(**overrides) -> NominalBCCorridorSweep:
    arguments = {
        "phase": "transport",
        "base_corridor": _base_corridor(),
        "crossing_direction": "right_to_left",
        "speed_m_s": 1.0,
        "cue_lead_s": 1.0,
        "corridor_length_m": 1.0,
        "hand": "left",
        "candidate_normal_offsets_m": (0.30, 0.10, 0.20),
        "phase_start_simulation_time_s": 10.0,
        "maximum_sample_interval_s": 0.5,
    }
    arguments.update(overrides)
    return NominalBCCorridorSweep(**arguments)


def test_time_aligned_sweep_selects_nearest_severity_and_audits_cue() -> None:
    geometry = _DynamicPlaneGeometry()
    sweep = _sweep()
    for simulation_time_s, surface_z_m in (
        (10.0, 0.02),
        (10.5, 0.04),
        (11.0, 0.05),
        (11.5, 0.10),
        (12.0, 0.08),
    ):
        geometry.surface_z_m = surface_z_m
        assert sweep.observe(
            simulation_time_s=simulation_time_s,
            safety_geometry=geometry,
        )

    audit = sweep.select_severity(
        severity="shallow",
        target_gap_m=0.10,
        allowed_gap_range_m=(0.08, 0.12),
        minimum_pre_cue_start_gap_m=0.15,
    )

    assert audit.semantics == NOMINAL_BC_SWEEP_SEMANTICS
    assert np.isclose(audit.selected_normal_offset_m, 0.20)
    assert np.isclose(audit.nominal_minimum_surface_gap_m, 0.10)
    assert audit.nominal_minimum_simulation_time_s == 11.5
    assert audit.nominal_minimum_sample_index == 3
    assert np.isclose(audit.nominal_minimum_progress_fraction, 0.5)
    assert audit.closest_link == "panda_link7"
    assert audit.closest_collider_path.endswith("panda_link7/collider")
    assert audit.crossing_start_surface_gap_m == pytest.approx(0.15)
    assert audit.crossing_start_query_simulation_time_s == 11.0
    assert audit.crossing_start_sample_index == 2
    assert audit.crossing_start_closest_link == "panda_link7"
    assert audit.crossing_start_closest_collider_path.endswith(
        "panda_link7/collider"
    )
    assert np.isclose(audit.minimum_pre_cue_start_gap_m, 0.16)
    assert audit.required_minimum_pre_cue_start_gap_m == 0.15
    assert audit.minimum_pre_cue_simulation_time_s == 10.5
    assert audit.minimum_pre_cue_sample_index == 1
    assert audit.pre_cue_sample_count == 2
    assert audit.crossing_sample_count == 3
    assert audit.time_sample_count == 5
    assert audit.geometry_query_count == 15
    assert audit.candidate_count == 3
    assert audit.pre_cue_coverage_fraction == 1.0
    assert audit.crossing_coverage_fraction == 1.0
    assert audit.maximum_observed_sample_interval_s == 0.5
    assert audit.geometry_valid
    assert np.allclose(audit.corridor.center_world_m, [0.0, 0.0, 0.20])
    assert np.allclose(audit.corridor.start_world_m, [0.0, -0.5, 0.20])
    assert np.allclose(audit.corridor.end_world_m, [0.0, 0.5, 0.20])
    assert audit.corridor.planned_minimum_surface_gap_m == pytest.approx(0.10)

    serialized = audit.as_dict()
    assert serialized["coverage"]["geometry_query_count"] == 15
    assert len(serialized["candidate_results"]) == 3
    assert serialized["corridor"]["planned_minimum_surface_gap_m"] == pytest.approx(
        0.10
    )


def test_start_timestamp_belongs_to_crossing_and_direction_reverses() -> None:
    geometry = _DynamicPlaneGeometry()
    sweep = _sweep(
        crossing_direction="left_to_right",
        candidate_normal_offsets_m=(0.10,),
        phase_start_simulation_time_s=0.0,
        maximum_sample_interval_s=1.0,
    )
    for simulation_time_s in (0.0, 1.0, 2.0):
        sweep.observe(
            simulation_time_s=simulation_time_s,
            safety_geometry=geometry,
        )
    audit = sweep.select_severity(
        severity="shallow",
        target_gap_m=0.10,
        allowed_gap_range_m=(0.09, 0.11),
    )

    assert audit.pre_cue_sample_count == 1
    assert audit.crossing_sample_count == 2
    assert audit.first_crossing_sample_simulation_time_s == 1.0
    # Query 0 is pre-cue at start, query 1 is exactly START, query 2 is end.
    assert np.allclose(geometry.queries[1][1], [0.0, 0.5, 0.10])
    assert np.allclose(geometry.queries[2][1], [0.0, -0.5, 0.10])
    assert np.allclose(audit.corridor.tangent_world, [0.0, -1.0, 0.0])


def test_signed_offsets_are_sorted_and_tie_break_toward_inward_candidate() -> None:
    geometry = _DynamicPlaneGeometry()
    sweep = _sweep(
        cue_lead_s=0.0,
        candidate_normal_offsets_m=(0.11, -0.11),
        phase_start_simulation_time_s=0.0,
        maximum_sample_interval_s=1.0,
    )
    # Make the gaps symmetric around the target: -0.11 -> 0.09, +0.11 -> 0.31.
    geometry.surface_z_m = -0.20
    for simulation_time_s in (0.0, 1.0):
        sweep.observe(
            simulation_time_s=simulation_time_s,
            safety_geometry=geometry,
        )
    audit = sweep.select_severity(
        severity="synthetic",
        target_gap_m=0.20,
        allowed_gap_range_m=(0.08, 0.32),
    )

    assert audit.selected_normal_offset_m == -0.11
    assert [item.normal_offset_m for item in audit.candidate_results] == [
        -0.11,
        0.11,
    ]
    assert audit.minimum_pre_cue_start_gap_m is None
    assert audit.pre_cue_coverage_fraction == 1.0


def test_geometry_invalid_candidate_is_excluded_fail_closed() -> None:
    geometry = _DynamicPlaneGeometry()
    geometry.invalid_z_m = 0.20
    sweep = _sweep(candidate_normal_offsets_m=(0.20,))
    for simulation_time_s in (10.0, 10.5, 11.0, 11.5, 12.0):
        sweep.observe(
            simulation_time_s=simulation_time_s,
            safety_geometry=geometry,
        )

    with pytest.raises(RuntimeError, match="no geometry-valid"):
        sweep.select_severity(
            severity="shallow",
            target_gap_m=0.10,
            allowed_gap_range_m=(0.08, 0.12),
        )


def test_pre_cue_clearance_guard_is_applied_to_entire_cue_window() -> None:
    geometry = _DynamicPlaneGeometry()
    sweep = _sweep(candidate_normal_offsets_m=(0.20,))
    for simulation_time_s, surface_z_m in (
        (10.0, 0.02),
        (10.5, 0.04),
        (11.0, 0.10),
        (11.5, 0.10),
        (12.0, 0.10),
    ):
        geometry.surface_z_m = surface_z_m
        sweep.observe(
            simulation_time_s=simulation_time_s,
            safety_geometry=geometry,
        )

    with pytest.raises(RuntimeError, match="no geometry-valid"):
        sweep.select_severity(
            severity="shallow",
            target_gap_m=0.10,
            allowed_gap_range_m=(0.08, 0.12),
            minimum_pre_cue_start_gap_m=0.17,
        )


def test_temporal_hole_fails_closed_before_candidate_selection() -> None:
    geometry = _DynamicPlaneGeometry()
    sweep = _sweep(
        candidate_normal_offsets_m=(0.10,),
        phase_start_simulation_time_s=0.0,
        maximum_sample_interval_s=0.6,
    )
    for simulation_time_s in (0.0, 1.0, 2.0):
        sweep.observe(
            simulation_time_s=simulation_time_s,
            safety_geometry=geometry,
        )

    with pytest.raises(RuntimeError, match="incomplete temporal coverage"):
        sweep.select_severity(
            severity="shallow",
            target_gap_m=0.10,
            allowed_gap_range_m=(0.09, 0.11),
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"candidate_normal_offsets_m": ()}, "must not be empty"),
        ({"candidate_normal_offsets_m": (0.1, 0.1)}, "must be unique"),
        ({"crossing_direction": "diagonal"}, "unsupported"),
        ({"maximum_sample_interval_s": 0.0}, "finite and positive"),
    ],
)
def test_sweep_rejects_ambiguous_or_invalid_inputs(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _sweep(**overrides)
