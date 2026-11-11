import json

import pytest

from v3_chan.pick_place_speed import (
    BASE_EVENTS_DT,
    COUNTERBALANCED_SPEED_PROFILE_ORDERS,
    MOTION_EVENT_INDICES,
    SPEED_PROFILE_ORDER,
    counterbalanced_speed_profile_order,
    events_dt_for_speed_profile,
    nominal_profile_steps,
    normalize_speed_profile_order,
    speed_profile_for_episode,
    speed_profile_metadata,
    speed_schedule_metadata,
)


def test_episode_schedule_cycles_slow_medium_fast():
    assert [speed_profile_for_episode(index) for index in range(6)] == [
        "slow",
        "medium",
        "fast",
        "slow",
        "medium",
        "fast",
    ]


def test_session_orders_are_counterbalanced_and_repeat():
    assert [counterbalanced_speed_profile_order(index) for index in range(4)] == [
        ("slow", "medium", "fast"),
        ("medium", "fast", "slow"),
        ("fast", "slow", "medium"),
        ("slow", "medium", "fast"),
    ]
    assert tuple(COUNTERBALANCED_SPEED_PROFILE_ORDERS) == tuple(
        counterbalanced_speed_profile_order(index) for index in range(3)
    )


def test_episode_schedule_uses_session_order():
    order = normalize_speed_profile_order("medium,fast,slow")
    assert [speed_profile_for_episode(index, order) for index in range(3)] == [
        "medium",
        "fast",
        "slow",
    ]


def test_session_order_requires_each_profile_once():
    with pytest.raises(ValueError):
        normalize_speed_profile_order("slow,slow,fast")


def test_only_motion_phase_timing_is_scaled():
    motion_indices = set(MOTION_EVENT_INDICES)
    medium = events_dt_for_speed_profile("medium")
    fast = events_dt_for_speed_profile("fast")

    for index, base_value in enumerate(BASE_EVENTS_DT):
        if index in motion_indices:
            assert medium[index] == pytest.approx(base_value * 1.5)
            assert fast[index] == pytest.approx(base_value * 2.0)
        else:
            assert medium[index] == pytest.approx(base_value)
            assert fast[index] == pytest.approx(base_value)


def test_nominal_cycle_duration_decreases_by_profile():
    assert [nominal_profile_steps(profile) for profile in SPEED_PROFILE_ORDER] == [
        915,
        618,
        465,
    ]


def test_metadata_records_exact_schedule_and_events_dt():
    episode = speed_profile_metadata("medium", physics_dt_s=1.0 / 60.0)
    schedule = speed_schedule_metadata(physics_dt_s=1.0 / 60.0)

    assert episode["controller_speed_profile"] == "medium"
    assert episode["controller_nominal_cycle_steps"] == 618
    assert tuple(json.loads(episode["controller_events_dt_json"])) == pytest.approx(
        events_dt_for_speed_profile("medium")
    )
    assert schedule["controller_speed_schedule"] == "slow,medium,fast"
    assert set(json.loads(schedule["controller_speed_profiles_json"])) == set(
        SPEED_PROFILE_ORDER
    )


def test_metadata_records_selected_counterbalance_order():
    schedule = speed_schedule_metadata(
        profile_order=("fast", "slow", "medium"),
        counterbalance_order_index=2,
    )

    assert schedule["controller_speed_schedule"] == "fast,slow,medium"
    assert schedule["controller_speed_counterbalance_order_index"] == 2
