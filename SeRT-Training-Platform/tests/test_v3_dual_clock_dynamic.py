import numpy as np

from v3_chan.dynamic_safety import (
    DualClockDynamicSafetyEstimator,
    DynamicSafetyConfig,
)


def _config():
    return DynamicSafetyConfig(
        ema_time_constant_s=0.001,
        ttc_cap_s=10.0,
        min_valid_closing_speed_mps=0.01,
        max_valid_dt_s=0.5,
    )


def _update(estimator, sim_time, wall_time, position, gap, **extra):
    return estimator.update(
        sim_time_s=sim_time,
        wall_time_s=wall_time,
        left_hand_pos=np.asarray(position, dtype=float),
        right_hand_pos=None,
        left_tracking_valid=True,
        right_tracking_valid=False,
        left_surface_gap_m=gap,
        right_surface_gap_m=10.0,
        left_geometry_valid=True,
        right_geometry_valid=False,
        left_closest_collider_id=1,
        right_closest_collider_id=0,
        left_closest_robot_origin_pos=np.zeros(3),
        right_closest_robot_origin_pos=None,
        left_closest_robot_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        right_closest_robot_orientation_wxyz=None,
        left_closest_surface_point_world_pos=np.array([0.5, 0.0, 0.0]),
        right_closest_surface_point_world_pos=None,
        left_closest_robot_angular_velocity_world_radps=np.zeros(3),
        right_closest_robot_angular_velocity_world_radps=None,
        **extra,
    )


def test_dual_clock_separates_wall_and_simulation_velocity():
    estimator = DualClockDynamicSafetyEstimator(_config())
    first = _update(estimator, 0.0, 10.0, [1.0, 0.0, 0.0], 0.5)
    second = _update(estimator, 0.1, 10.2, [0.8, 0.0, 0.0], 0.3)

    assert not first.real_time_factor_valid
    assert second.real_time_factor_valid
    assert np.isclose(second.real_time_factor, 0.5)
    assert np.isclose(second.wall.left.hand_velocity_raw_mps[0], -1.0)
    assert np.isclose(second.simulation.left.hand_velocity_raw_mps[0], -2.0)


def test_validity_distinguishes_measurement_from_ttc():
    estimator = DualClockDynamicSafetyEstimator(_config())
    first = _update(estimator, 0.0, 20.0, [0.5, 0.0, 0.0], 0.5)
    receding = _update(estimator, 0.1, 20.1, [0.6, 0.0, 0.0], 0.6)

    assert first.wall.left.tracking_valid
    assert first.wall.left.gap_measurement_valid
    assert not first.wall.left.gap_rate_valid
    assert receding.wall.left.dynamic_measurement_valid
    assert receding.wall.left.closing_speed_valid
    assert not receding.wall.left.ttc_valid
    assert not receding.wall.left.dynamic_valid


def test_pose_source_switch_resets_both_clock_histories():
    estimator = DualClockDynamicSafetyEstimator(_config())
    _update(estimator, 0.0, 30.0, [0.5, 0.0, 0.0], 0.5)
    _update(estimator, 0.1, 30.1, [0.4, 0.0, 0.0], 0.4)
    switched = _update(
        estimator,
        0.2,
        30.2,
        [4.0, 0.0, 0.0],
        0.3,
        left_pose_source_switched=True,
    )

    assert switched.wall.left.pose_source_switched
    assert switched.simulation.left.pose_source_switched
    assert not switched.wall.left.hand_velocity_valid
    assert not switched.simulation.left.hand_velocity_valid
