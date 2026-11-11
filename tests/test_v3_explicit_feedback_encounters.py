from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from v3_chan.explicit_feedback.encounters import EncounterStateMachine
from v3_chan.explicit_feedback.schema import ProtocolViolation
from v3_chan.explicit_feedback.tracking import (
    TRACKING_NOT_TRACKED,
    TRACKING_TRACKED,
    TrackingDropout,
    TrackingWatchdog,
)


_NOW_NS = 5_000_000_000


def _pose(
    *,
    source_name: str,
    acquisition_monotonic_ns: int = _NOW_NS - 1_000_000,
    pose_age_ms: float = 1.0,
    pose_valid: bool = True,
    tracking_status_known: bool = True,
    position_tracked: int = TRACKING_TRACKED,
    source_switched: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        position_world=np.array([0.5, 0.1, 1.0], dtype=np.float64),
        orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        pose_valid=pose_valid,
        position_tracked=position_tracked,
        tracking_status_known=tracking_status_known,
        source_name=source_name,
        source_path=f"/{source_name}",
        acquisition_monotonic_ns=acquisition_monotonic_ns,
        pose_age_ms=pose_age_ms,
        source_switched=source_switched,
    )


def _valid_samples() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    return (
        _pose(source_name="hmd_xr_physical"),
        _pose(source_name="xr_physical"),
        _pose(source_name="openxr_joint"),
    )


def test_tracking_watchdog_accepts_fresh_allowed_tracked_sources() -> None:
    watchdog = TrackingWatchdog(max_pose_age_ms=100.0)

    snapshot = watchdog.evaluate(_valid_samples(), now_monotonic_ns=_NOW_NS)

    np.testing.assert_array_equal(snapshot.valid_mask, [1, 1, 1])
    assert snapshot.invalid_reasons == ()
    watchdog.require_valid(snapshot)
    payload = snapshot.state_payload()
    np.testing.assert_array_equal(payload["human_valid_mask"], [1.0, 1.0, 1.0])
    assert payload["intentional_human_absence"] is False


def test_tracking_watchdog_rechecks_cached_pose_age_at_control_time() -> None:
    watchdog = TrackingWatchdog(max_pose_age_ms=100.0)
    snapshot = watchdog.evaluate(_valid_samples(), now_monotonic_ns=_NOW_NS)

    watchdog.require_current(snapshot, now_monotonic_ns=_NOW_NS)
    with pytest.raises(TrackingDropout, match="pose_stale"):
        watchdog.require_current(
            snapshot,
            now_monotonic_ns=_NOW_NS + 200_000_000,
        )


def test_tracking_watchdog_combines_source_age_and_control_delay() -> None:
    watchdog = TrackingWatchdog(max_pose_age_ms=100.0)
    head, left, right = _valid_samples()
    left.pose_age_ms = 60.0
    snapshot = watchdog.evaluate((head, left, right), now_monotonic_ns=_NOW_NS)

    with pytest.raises(TrackingDropout, match="effective_pose_stale"):
        watchdog.require_current(
            snapshot,
            now_monotonic_ns=_NOW_NS + 50_000_000,
        )


def test_tracking_watchdog_accepts_valid_physical_pose_without_tracked_bit() -> None:
    watchdog = TrackingWatchdog(max_pose_age_ms=100.0)
    samples = (
        _pose(
            source_name="hmd_xr_physical",
            tracking_status_known=False,
            position_tracked=-1,
        ),
        _pose(
            source_name="xr_physical",
            tracking_status_known=False,
            position_tracked=-1,
        ),
        _pose(
            source_name="xr_raw_physical",
            tracking_status_known=False,
            position_tracked=-1,
        ),
    )

    snapshot = watchdog.evaluate(samples, now_monotonic_ns=_NOW_NS)

    np.testing.assert_array_equal(snapshot.valid_mask, [1, 1, 1])
    watchdog.require_valid(snapshot)


def test_tracking_watchdog_rejects_stale_required_pose() -> None:
    watchdog = TrackingWatchdog(max_pose_age_ms=100.0)
    head, left, right = _valid_samples()
    left.acquisition_monotonic_ns = _NOW_NS - 100_000_001

    snapshot = watchdog.evaluate((head, left, right), now_monotonic_ns=_NOW_NS)

    assert snapshot.valid_mask[1] == 0
    assert "left:pose_stale" in snapshot.invalid_reasons
    with pytest.raises(TrackingDropout, match="left:pose_stale"):
        watchdog.require_valid(snapshot)


def test_tracking_watchdog_rejects_non_allowlisted_source() -> None:
    watchdog = TrackingWatchdog(max_pose_age_ms=100.0)
    head, _, right = _valid_samples()
    cached_visual = _pose(source_name="xr_stage_visual")

    snapshot = watchdog.evaluate(
        (head, cached_visual, right), now_monotonic_ns=_NOW_NS
    )

    assert snapshot.valid_mask[1] == 0
    assert "left:source_not_allowed" in snapshot.invalid_reasons
    with pytest.raises(TrackingDropout, match="left:source_not_allowed"):
        watchdog.require_valid(snapshot)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("source_path", "", "source_path_missing"),
        ("pose_age_ms", -1.0, "source_pose_age_invalid"),
        ("pose_age_ms", float("nan"), "source_pose_age_invalid"),
        ("position_tracked", 7, "tracking_status_invalid"),
    ),
)
def test_tracking_watchdog_rejects_malformed_live_provenance(
    field, value, reason
) -> None:
    watchdog = TrackingWatchdog(max_pose_age_ms=100.0)
    head, left, right = _valid_samples()
    setattr(left, field, value)

    snapshot = watchdog.evaluate((head, left, right), now_monotonic_ns=_NOW_NS)

    with pytest.raises(TrackingDropout, match=reason):
        watchdog.require_valid(snapshot)


def test_tracking_watchdog_rejects_known_not_tracked_pose() -> None:
    watchdog = TrackingWatchdog(max_pose_age_ms=100.0)
    head, left, _ = _valid_samples()
    not_tracked = _pose(
        source_name="openxr_joint",
        tracking_status_known=True,
        position_tracked=TRACKING_NOT_TRACKED,
    )

    snapshot = watchdog.evaluate((head, left, not_tracked), now_monotonic_ns=_NOW_NS)

    assert snapshot.valid_mask[2] == 0
    assert "right:not_position_tracked" in snapshot.invalid_reasons
    with pytest.raises(TrackingDropout, match="right:not_position_tracked"):
        watchdog.require_valid(snapshot)


def _observe(
    machine: EncounterStateMachine,
    *,
    step: int,
    monotonic_ns: int,
    left_gap_m: float,
    right_gap_m: float = 0.30,
    geometry_valid: bool = True,
    constraint_count: int = 0,
    tracking_valid_mask: np.ndarray | None = None,
):
    return machine.observe(
        step=step,
        monotonic_ns=monotonic_ns,
        left_gap_m=left_gap_m,
        right_gap_m=right_gap_m,
        geometry_valid=geometry_valid,
        constraint_count=constraint_count,
        tracking_valid_mask=(
            np.ones(3, dtype=np.int8)
            if tracking_valid_mask is None
            else tracking_valid_mask
        ),
    )


def test_encounter_opens_once_on_rising_edge_and_closes_after_confirmed_clear() -> None:
    machine = EncounterStateMachine(
        activation_gap_m=0.13,
        clear_confirmation_steps=2,
        refractory_s=1.0,
    )
    machine.reset_episode(7)

    idle = _observe(
        machine,
        step=0,
        monotonic_ns=0,
        left_gap_m=0.20,
    )
    opened = _observe(
        machine,
        step=1,
        monotonic_ns=100_000_000,
        left_gap_m=0.12,
        constraint_count=1,
    )
    still_active = _observe(
        machine,
        step=2,
        monotonic_ns=200_000_000,
        left_gap_m=0.08,
        right_gap_m=0.11,
        constraint_count=2,
    )
    clearing = _observe(
        machine,
        step=3,
        monotonic_ns=300_000_000,
        left_gap_m=0.20,
    )
    closed = _observe(
        machine,
        step=4,
        monotonic_ns=400_000_000,
        left_gap_m=0.20,
    )

    assert idle.opened is False
    assert idle.row_encounter_id == ""
    assert opened.opened is True
    assert still_active.opened is False
    assert still_active.row_encounter_id == opened.row_encounter_id
    assert clearing.state == "clearance_confirm"
    assert clearing.closed is None
    assert closed.state == "robot_hold"
    assert closed.closed is not None
    assert closed.closed.encounter_id == opened.row_encounter_id
    assert closed.closed.episode_index == 7
    assert closed.closed.start_step == 1
    assert closed.closed.end_step_exclusive == 5
    assert closed.closed.minimum_surface_gap_m == pytest.approx(0.08)
    assert closed.closed.involved_hand_mask == (1, 1)
    assert closed.closed.eligible is True
    assert closed.closed.terminal_status == "completed"
    assert machine.active is False


def test_refractory_suppresses_entry_and_requires_a_new_rising_edge() -> None:
    machine = EncounterStateMachine(
        activation_gap_m=0.13,
        clear_confirmation_steps=1,
        refractory_s=1.0,
    )

    first = _observe(
        machine,
        step=0,
        monotonic_ns=0,
        left_gap_m=0.10,
        constraint_count=1,
    )
    closed = _observe(
        machine,
        step=1,
        monotonic_ns=100_000_000,
        left_gap_m=0.20,
    )
    suppressed = _observe(
        machine,
        step=2,
        monotonic_ns=200_000_000,
        left_gap_m=0.10,
        constraint_count=1,
    )
    still_inside_after_refractory = _observe(
        machine,
        step=3,
        monotonic_ns=1_200_000_000,
        left_gap_m=0.10,
        constraint_count=1,
    )
    _observe(
        machine,
        step=4,
        monotonic_ns=1_300_000_000,
        left_gap_m=0.20,
    )
    second = _observe(
        machine,
        step=5,
        monotonic_ns=1_400_000_000,
        left_gap_m=0.10,
        constraint_count=1,
    )

    assert first.opened is True
    assert closed.closed is not None
    assert suppressed.state == "refractory"
    assert suppressed.opened is False
    assert still_inside_after_refractory.opened is False
    assert second.opened is True
    assert second.row_encounter_id != first.row_encounter_id


def test_tracking_dropout_during_encounter_fails_closed_and_can_be_aborted() -> None:
    machine = EncounterStateMachine(clear_confirmation_steps=2)
    opened = _observe(
        machine,
        step=0,
        monotonic_ns=100,
        left_gap_m=0.10,
        constraint_count=1,
    )

    with pytest.raises(
        ProtocolViolation,
        match="live tracking invalid during eligible encounter",
    ):
        _observe(
            machine,
            step=1,
            monotonic_ns=200,
            left_gap_m=0.10,
            constraint_count=1,
            tracking_valid_mask=np.array([1, 0, 1], dtype=np.int8),
        )

    assert machine.active is True
    aborted = machine.abort(
        end_step_exclusive=1,
        end_monotonic_ns=200,
        reason="tracking_dropout:left",
    )
    assert aborted is not None
    assert aborted.encounter_id == opened.row_encounter_id
    assert aborted.terminal_status == "aborted"
    assert aborted.abort_reason == "tracking_dropout:left"
    assert machine.active is False


def test_malformed_tracking_mask_during_active_encounter_fails_closed() -> None:
    machine = EncounterStateMachine(clear_confirmation_steps=2)
    _observe(
        machine,
        step=0,
        monotonic_ns=100,
        left_gap_m=0.10,
        constraint_count=1,
    )

    with pytest.raises(ProtocolViolation, match="tracking_valid_mask"):
        _observe(
            machine,
            step=1,
            monotonic_ns=200,
            left_gap_m=0.10,
            constraint_count=1,
            tracking_valid_mask=np.array([1], dtype=np.int8),
        )
