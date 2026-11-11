import json

import numpy as np
import pytest

from v3_chan.pick_place_speed import (
    APPROACH_PLACE_EVENT_INDICES,
    CARTESIAN_EVENTS_DT,
    COUNTERBALANCED_SPEED_PROFILE_ORDERS,
    SPEED_LIMITS_MPS,
    SPEED_PROFILE_ORDER,
    SPEED_SCHEDULE_VERSION,
    TRANSPORT_EVENT_INDICES,
    CartesianSpeedConfig,
    PoseSpeedEstimator,
    SpeedLimitRamp,
    blended_speed_limit_mps,
    controller_phase_mode,
    counterbalanced_speed_profile_order,
    events_dt_for_speed_profile,
    limit_joint_target_by_cartesian_displacement,
    nominal_profile_steps,
    normalize_speed_profile_order,
    speed_profile_for_episode,
    speed_profile_metadata,
    speed_schedule_metadata,
)
from v3_chan.end_effector_safety_geometry import SafetyThresholds, distance_gate


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


def test_ten_event_phase_mapping_is_explicit():
    assert APPROACH_PLACE_EVENT_INDICES == (0, 1, 6)
    assert TRANSPORT_EVENT_INDICES == (4, 5, 8, 9)
    assert [controller_phase_mode(index) for index in range(10)] == [
        "approach_place",
        "approach_place",
        "settling_gripper",
        "settling_gripper",
        "transport",
        "transport",
        "approach_place",
        "settling_gripper",
        "transport",
        "transport",
    ]


def test_events_dt_is_not_used_as_a_speed_profile_multiplier():
    values = [events_dt_for_speed_profile(profile) for profile in SPEED_PROFILE_ORDER]
    assert values[0] == values[1] == values[2] == CARTESIAN_EVENTS_DT
    assert [nominal_profile_steps(profile) for profile in SPEED_PROFILE_ORDER] == [
        19,
        19,
        19,
    ]


@pytest.mark.parametrize(
    ("profile", "transport", "approach", "near"),
    (
        ("slow", 0.30, 0.10, 0.05),
        ("medium", 0.40, 0.15, 0.10),
        ("fast", 0.50, 0.20, 0.15),
    ),
)
def test_exact_cartesian_speed_limits(profile, transport, approach, near):
    assert SPEED_LIMITS_MPS[profile] == {
        "transport": transport,
        "approach_place": approach,
        "near_human": near,
    }


def test_distance_gate_blends_continuously_to_near_limit():
    normal, near, start, phase, mode = blended_speed_limit_mps("fast", 5, 0.0)
    assert (normal, near, start, phase, mode) == (
        0.50,
        0.15,
        0.50,
        "transport",
        "transport",
    )
    _, _, middle, _, middle_mode = blended_speed_limit_mps("fast", 5, 0.5)
    _, _, full, _, full_mode = blended_speed_limit_mps("fast", 5, 1.0)
    assert middle == pytest.approx(0.325)
    assert middle_mode == "near_human"
    assert full == pytest.approx(0.15)
    assert full_mode == "near_human"

    gates = np.linspace(0.0, 1.0, 101)
    limits = np.array(
        [blended_speed_limit_mps("fast", 5, gate)[2] for gate in gates]
    )
    assert np.all(np.diff(limits) <= 0.0)
    assert np.max(np.abs(np.diff(limits, n=2))) < 1e-12


def test_speed_ramp_uses_sim_time_and_never_exceeds_target():
    config = CartesianSpeedConfig(
        acceleration_limit_mps2=1.0,
        jerk_limit_mps3=10.0,
    ).validated()
    ramp = SpeedLimitRamp(config)
    assert ramp.update(0.5, 10.0) == (0.0, 0.0, False)
    samples = [ramp.update(0.5, 10.0 + index / 60.0)[0] for index in range(1, 61)]
    assert np.all(np.diff(samples) >= -1e-12)
    assert max(samples) <= 0.5
    # A safety decrease is a hard cap and cannot wait for the ramp.
    reduced, _, valid = ramp.update(0.1, 11.0 + 1.0 / 60.0)
    assert valid
    assert reduced == pytest.approx(0.1)


def test_fk_joint_interpolation_obeys_cartesian_displacement_cap():
    def fk(q):
        return np.asarray(q[:3], dtype=float)

    result = limit_joint_target_by_cartesian_displacement(
        np.zeros(3),
        np.array([0.3, 0.4, 0.0]),
        0.01,
        fk,
    )
    assert result.valid
    assert result.limited
    assert result.full_target_displacement_m == pytest.approx(0.5)
    assert result.predicted_displacement_m <= 0.01 + 1e-6
    assert np.linalg.norm(fk(result.joint_positions)) <= 0.01 + 1e-6


@pytest.mark.parametrize(
    ("profile", "near_limit_mps"),
    (("slow", 0.05), ("medium", 0.10), ("fast", 0.15)),
)
def test_full_distance_gate_caps_fk_command_for_each_profile(
    profile, near_limit_mps
):
    physics_dt_s = 1.0 / 60.0
    gate = distance_gate(0.05, SafetyThresholds())
    _, _, blended, _, mode = blended_speed_limit_mps(profile, 5, gate)

    assert gate == 1.0
    assert mode == "near_human"
    assert blended == pytest.approx(near_limit_mps)

    def fk(q):
        return np.asarray(q[:3], dtype=float)

    result = limit_joint_target_by_cartesian_displacement(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        blended * physics_dt_s,
        fk,
    )
    command_speed_mps = result.predicted_displacement_m / physics_dt_s
    assert result.valid
    assert result.limited
    assert command_speed_mps <= near_limit_mps + 1e-6


def test_pose_speed_estimator_excludes_reset_and_bad_timestamp_frames():
    estimator = PoseSpeedEstimator(min_dt_s=1e-5, max_dt_s=0.1)
    first = estimator.update([100.0, 0.0, 0.0], 1.0)
    assert not first.valid
    normal = estimator.update([100.01, 0.0, 0.0], 1.1)
    assert normal.valid
    assert normal.speed_mps == pytest.approx(0.1)
    bad = estimator.update([200.0, 0.0, 0.0], 1.1)
    assert not bad.valid
    estimator.reset()
    after_reset = estimator.update([-500.0, 0.0, 0.0], 2.0)
    assert not after_reset.valid
    assert after_reset.speed_mps == 0.0


def test_metadata_records_cartesian_schedule_and_limiter_settings():
    config = CartesianSpeedConfig().validated()
    episode = speed_profile_metadata(
        "medium", physics_dt_s=1.0 / 60.0, config=config
    )
    schedule = speed_schedule_metadata(
        physics_dt_s=1.0 / 60.0, config=config
    )

    assert episode["controller_speed_profile"] == "medium"
    assert episode["transport_speed_limit_mps"] == pytest.approx(0.40)
    assert episode["approach_place_speed_limit_mps"] == pytest.approx(0.15)
    assert episode["near_human_speed_limit_mps"] == pytest.approx(0.10)
    assert tuple(json.loads(episode["controller_events_dt_json"])) == pytest.approx(
        CARTESIAN_EVENTS_DT
    )
    assert schedule["controller_speed_schedule"] == "slow,medium,fast"
    assert schedule["controller_speed_schedule_version"] == SPEED_SCHEDULE_VERSION
    profiles = json.loads(schedule["controller_speed_profiles_json"])
    assert profiles["fast"]["transport"] == pytest.approx(0.50)
    assert schedule["controller_speed_action_limit_method"] == (
        "joint_target_interpolation_lula_fk_binary_search"
    )


def test_metadata_records_selected_counterbalance_order():
    schedule = speed_schedule_metadata(
        profile_order=("fast", "slow", "medium"),
        counterbalance_order_index=2,
    )
    assert schedule["controller_speed_schedule"] == "fast,slow,medium"
    assert schedule["controller_speed_counterbalance_order_index"] == 2
