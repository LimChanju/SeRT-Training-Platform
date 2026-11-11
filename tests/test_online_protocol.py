from __future__ import annotations

from collections import Counter
import math

import pytest

from v3_chan.explicit_feedback.online_protocol import (
    CROSSING_DIRECTIONS,
    CROSSING_SPEEDS,
    SEVERITIES,
    SEVERITY_DEFINITIONS,
    TARGET_PHASES,
    CrossingPathMonitor,
    EncounterDetector,
    build_participant_schedule,
)


def _assert_balanced_pilot_marginals(trials) -> None:
    dimensions = (
        ("task_phase", TARGET_PHASES),
        ("severity", SEVERITIES),
        ("crossing_direction", CROSSING_DIRECTIONS),
        ("crossing_speed", CROSSING_SPEEDS),
    )
    for attribute, categories in dimensions:
        counts = Counter(getattr(trial, attribute) for trial in trials)
        values = [counts[category] for category in categories]
        assert max(values) - min(values) <= 1, (attribute, counts)


def test_full_schedule_is_deterministic_factorial_and_speed_balanced() -> None:
    first = build_participant_schedule(
        "P001", session_id="S01", seed=19, include_anchor_repeats=True
    )
    second = build_participant_schedule(
        "P001", session_id="S01", seed=19, include_anchor_repeats=True
    )

    assert first == second
    assert len(first.practice_trials) == 4
    assert len(first.evaluated_trials) == 28
    assert len(first.anchor_trials) == 4
    assert all(trial.analysis_exclude for trial in first.practice_trials)
    assert not any(trial.analysis_exclude for trial in first.evaluated_trials)

    originals = [trial for trial in first.evaluated_trials if not trial.anchor_repeat]
    assert len(originals) == 24
    assert len({trial.condition_key for trial in originals}) == 24
    assert {
        speed: sum(trial.crossing_speed == speed for trial in originals)
        for speed in CROSSING_SPEEDS
    } == {"slow": 12, "fast": 12}
    assert {trial.task_phase for trial in first.anchor_trials} == set(TARGET_PHASES)
    source_by_id = {trial.trial_id: trial for trial in originals}
    for anchor in first.anchor_trials:
        source = source_by_id[anchor.repeated_trial_id]
        assert anchor.condition_key == source.condition_key
        assert anchor.crossing_speed == source.crossing_speed

    assert len({trial.trial_id for trial in first}) == len(first)
    assert len({trial.query_id for trial in first}) == len(first)
    assert [trial.order_index for trial in first] == list(range(len(first)))
    assert [trial.evaluated_index for trial in first.evaluated_trials] == list(
        range(28)
    )


def test_schedule_changes_by_participant_but_not_frozen_factor_cells() -> None:
    first = build_participant_schedule("P001", include_anchor_repeats=False)
    other = build_participant_schedule("P002", include_anchor_repeats=False)
    first_originals = first.evaluated_trials
    other_originals = other.evaluated_trials

    assert [trial.condition_key for trial in first_originals] != [
        trial.condition_key for trial in other_originals
    ]
    assert {trial.condition_key for trial in first_originals} == {
        trial.condition_key for trial in other_originals
    }
    for trial in first:
        frozen = SEVERITY_DEFINITIONS[trial.severity]
        assert trial.planned_gap_min_m == frozen.minimum_gap_m
        assert trial.planned_gap_max_m == frozen.maximum_gap_m
        assert trial.planned_gap_target_m == frozen.target_gap_m


@pytest.mark.parametrize("evaluated_count", range(12, 17))
def test_pilot_schedule_count_is_total_evaluated_and_balanced(evaluated_count: int) -> None:
    schedule = build_participant_schedule(
        "pilot-1",
        practice_trials=4,
        pilot_evaluated_trials=evaluated_count,
    )

    assert schedule.pilot is True
    assert schedule.anchors_enabled is False
    assert len(schedule.practice_trials) == 4
    assert len(schedule.evaluated_trials) == evaluated_count
    assert len({trial.condition_key for trial in schedule.evaluated_trials}) == evaluated_count
    _assert_balanced_pilot_marginals(schedule.evaluated_trials)
    assert all(trial.pilot for trial in schedule)


@pytest.mark.parametrize("evaluated_count", range(12, 17))
def test_pilot_balances_every_marginal_deterministically_across_seeds(
    evaluated_count: int,
) -> None:
    observed_schedules = set()
    for seed in range(32):
        first = build_participant_schedule(
            "pilot-marginals",
            session_id="S01",
            seed=seed,
            pilot_evaluated_trials=evaluated_count,
        )
        second = build_participant_schedule(
            "pilot-marginals",
            session_id="S01",
            seed=seed,
            pilot_evaluated_trials=evaluated_count,
        )

        assert first == second
        _assert_balanced_pilot_marginals(first.evaluated_trials)
        observed_schedules.add(
            tuple(
                (
                    trial.condition_key,
                    trial.crossing_speed,
                )
                for trial in first.evaluated_trials
            )
        )

    assert len(observed_schedules) > 1


def test_pilot_can_include_four_anchors_inside_requested_total() -> None:
    schedule = build_participant_schedule(
        "pilot-anchor",
        pilot_evaluated_trials=16,
        include_anchor_repeats=True,
    )

    assert len(schedule.evaluated_trials) == 16
    assert len(schedule.anchor_trials) == 4
    assert {trial.task_phase for trial in schedule.anchor_trials} == set(TARGET_PHASES)


@pytest.mark.parametrize("evaluated_count", (0, 11, 17, 100))
def test_invalid_pilot_count_fails(evaluated_count: int) -> None:
    with pytest.raises(ValueError, match="between 12 and 16"):
        build_participant_schedule(
            "pilot-bad", pilot_evaluated_trials=evaluated_count
        )


def _observe(
    detector: EncounterDetector,
    step: int,
    time_s: float,
    *,
    gap: float = 0.30,
    ttc: float = math.inf,
    closing: bool = False,
    norm: float = 0.0,
    active: bool = False,
    dynamic_valid: bool = True,
):
    return detector.observe(
        step=step,
        simulation_time_s=time_s,
        surface_gap_m=gap,
        ttc_s=ttc,
        closing=closing,
        cbf_delta_norm_radps=norm,
        cbf_active=active,
        dynamic_measurement_valid=dynamic_valid,
    )


@pytest.mark.parametrize(
    "trigger",
    (
        {"gap": 0.13},
        {"ttc": 0.75, "closing": True},
        {"norm": 0.05},
    ),
)
def test_encounter_onset_requires_three_consecutive_frames(trigger) -> None:
    detector = EncounterDetector(trial_id="trial-A")

    first = _observe(detector, 0, 0.0, **trigger)
    second = _observe(detector, 1, 0.1, **trigger)
    third = _observe(detector, 2, 0.2, **trigger)

    assert first.state == "onset_candidate"
    assert first.opened is False
    assert second.opened is False
    assert third.opened is True
    assert third.onset_candidate_step == 0
    assert third.onset_confirmed_step == 2
    assert third.row_encounter_id == "trial-A:encounter-01"


def test_interrupted_onset_candidate_restarts_three_frame_count() -> None:
    detector = EncounterDetector(trial_id="trial-A")
    _observe(detector, 0, 0.0, gap=0.12)
    _observe(detector, 1, 0.1, gap=0.30)
    _observe(detector, 2, 0.2, gap=0.12)
    _observe(detector, 3, 0.3, gap=0.12)

    update = _observe(detector, 4, 0.4, gap=0.12)

    assert update.opened is True
    assert update.onset_candidate_step == 2


def test_encounter_offset_requires_all_clear_for_half_second_then_merge_grace() -> None:
    detector = EncounterDetector(trial_id="trial-clear")
    for step in range(3):
        opened = _observe(detector, step, step * 0.1, gap=0.12, active=True)
    assert opened.opened

    # Gap alone is not enough: active CBF and closing TTC each block clearance.
    assert _observe(detector, 3, 0.3, gap=0.20, active=True).state == "active"
    assert (
        _observe(
            detector, 4, 0.4, gap=0.20, closing=True, ttc=0.8, active=False
        ).state
        == "active"
    )
    assert _observe(detector, 5, 0.5, gap=0.20).state == "clearance_confirm"
    assert _observe(detector, 6, 0.9, gap=0.20).state == "clearance_confirm"
    offset = _observe(detector, 7, 1.0, gap=0.20)
    assert offset.state == "merge_wait"
    assert offset.closed is None

    waiting = _observe(detector, 8, 1.9, gap=0.20)
    assert waiting.state == "merge_wait"
    assert waiting.closed is None
    closed = _observe(detector, 9, 2.01, gap=0.20)
    assert closed.closed is not None
    assert closed.closed.offset_step == 7
    assert closed.closed.offset_simulation_time_s == pytest.approx(1.0)
    assert closed.closed.window_end_simulation_time_s == pytest.approx(3.0)


def test_invalid_dynamic_measurement_cannot_start_ttc_risk_or_clear_encounter() -> None:
    detector = EncounterDetector(trial_id="trial-dynamic-validity")

    for step in range(3):
        update = _observe(
            detector,
            step,
            step * 0.1,
            ttc=0.2,
            closing=True,
            dynamic_valid=False,
        )
    assert update.opened is False
    assert detector.active is False

    for step in range(3, 6):
        opened = _observe(detector, step, step * 0.1, gap=0.12)
    assert opened.opened is True

    invalid_clear = _observe(
        detector,
        6,
        0.6,
        gap=0.20,
        dynamic_valid=False,
    )
    assert invalid_clear.state == "active"
    assert detector.active is True


def test_reentry_within_one_second_merges_same_encounter() -> None:
    detector = EncounterDetector(trial_id="trial-merge")
    for step in range(3):
        _observe(detector, step, step * 0.1, gap=0.12)
    _observe(detector, 3, 0.3, gap=0.20)
    _observe(detector, 4, 0.8, gap=0.20)
    before_id = detector.active_encounter_id

    reentry = _observe(detector, 5, 1.7, gap=0.12)

    assert reentry.merged_reentry is True
    assert reentry.row_encounter_id == before_id
    assert detector.state == "active"
    record = detector.flush(step=6, simulation_time_s=1.8)
    assert record is not None
    assert record.encounter_id == before_id
    assert record.reentry_merge_count == 1


def test_cbf_intervention_keeps_candidate_start_and_confirmed_step() -> None:
    detector = EncounterDetector(trial_id="trial-cbf")
    updates = [
        _observe(detector, step, step * 0.1, norm=0.06)
        for step in range(3)
    ]
    assert updates[-1].cbf_intervention_confirmed_now is True

    record = detector.flush(step=3, simulation_time_s=0.3)

    assert record is not None
    assert record.risk_onset_step == 0
    assert record.onset_confirmed_step == 2
    assert record.cbf_intervention_start_step == 0
    assert record.cbf_intervention_confirmed_step == 2
    assert record.maximum_cbf_delta_norm_radps == pytest.approx(0.06)


def test_encounter_timeout_is_flagged_only_after_eight_seconds() -> None:
    detector = EncounterDetector(trial_id="trial-timeout")
    for step in range(3):
        _observe(detector, step, step * 0.1, gap=0.12)
    at_eight = _observe(detector, 3, 8.0, gap=0.12)
    over_eight = _observe(detector, 4, 8.01, gap=0.12)

    assert at_eight.timed_out_now is False
    assert over_eight.timed_out_now is True
    record = detector.flush(step=5, simulation_time_s=8.1)
    assert record is not None and record.timeout is True


def test_crossing_path_monitor_measures_path_speed_and_deviation() -> None:
    monitor = CrossingPathMonitor(
        start_position_world=(0.0, 0.0, 0.0),
        end_position_world=(1.0, 0.0, 0.0),
        path_deviation_threshold_m=0.05,
        expected_speed_mps=1.0,
        relative_speed_tolerance=0.20,
    )

    monitor.observe(position_world=(0.0, 0.0, 0.0), simulation_time_s=0.0)
    monitor.observe(position_world=(0.5, 0.02, 0.0), simulation_time_s=0.5)
    completed = monitor.observe(
        position_world=(1.0, 0.0, 0.0), simulation_time_s=1.0
    )
    metrics = monitor.finalize()

    assert completed.intended_crossing_completed is True
    assert metrics.intended_crossing_count == 1
    assert metrics.reverse_crossing_count == 0
    assert metrics.path_length_m == pytest.approx(1.00079968, rel=1e-6)
    assert metrics.duration_s == pytest.approx(1.0)
    assert metrics.maximum_path_deviation_m == pytest.approx(0.02)
    assert metrics.off_protocol is False


def test_crossing_monitor_marks_large_deviation_and_extra_traversal_off_protocol() -> None:
    monitor = CrossingPathMonitor(
        start_position_world=(0.0, 0.0, 0.0),
        end_position_world=(1.0, 0.0, 0.0),
        path_deviation_threshold_m=0.05,
    )
    monitor.observe(position_world=(0.0, 0.0, 0.0), simulation_time_s=0.0)
    monitor.observe(position_world=(0.5, 0.10, 0.0), simulation_time_s=0.5)
    monitor.observe(position_world=(1.0, 0.0, 0.0), simulation_time_s=1.0)
    monitor.observe(position_world=(0.0, 0.0, 0.0), simulation_time_s=2.0)
    monitor.observe(position_world=(1.0, 0.0, 0.0), simulation_time_s=3.0)

    metrics = monitor.finalize()

    assert metrics.intended_crossing_count == 2
    assert metrics.reverse_crossing_count == 1
    assert metrics.off_protocol is True
    assert "path_deviation_exceeded" in metrics.off_protocol_reasons
    assert "intended_crossing_count_not_one" in metrics.off_protocol_reasons
    assert "reverse_or_extra_crossing" in metrics.off_protocol_reasons


def test_crossing_monitor_enforces_radial_gates_and_reverse_progress_limit() -> None:
    monitor = CrossingPathMonitor(
        start_position_world=(0.0, 0.0, 0.0),
        end_position_world=(1.0, 0.0, 0.0),
        path_deviation_threshold_m=0.2,
        start_radius_m=0.08,
        end_radius_m=0.08,
        minimum_progress_fraction=0.9,
        maximum_reverse_progress_m=0.06,
    )
    # Axially before the gate but far away radially must not arm a crossing.
    update = monitor.observe(
        position_world=(0.0, 0.20, 0.0), simulation_time_s=0.0
    )
    assert update.crossing_started is False
    monitor.observe(position_world=(0.0, 0.0, 0.0), simulation_time_s=0.1)
    monitor.observe(position_world=(0.55, 0.0, 0.0), simulation_time_s=0.5)
    monitor.observe(position_world=(0.45, 0.0, 0.0), simulation_time_s=0.6)
    completed = monitor.observe(
        position_world=(1.0, 0.0, 0.0), simulation_time_s=1.0
    )
    metrics = monitor.finalize()

    assert completed.intended_crossing_completed is True
    assert metrics.maximum_reverse_progress_m == pytest.approx(0.10)
    assert "reverse_progress_exceeded" in metrics.off_protocol_reasons
