import numpy as np

from v3_chan.pick_place_validation import evaluate_place_success


def test_place_success_requires_position_and_stability():
    result = evaluate_place_success(
        [0.602, -0.247, 1.028],
        [0.600, -0.250, 1.029],
        cube_linear_velocity=[0.01, 0.0, 0.0],
        cube_lift_m=0.18,
        grasp_observed=True,
    )

    assert result.success


def test_place_failure_when_cube_stays_at_pick_location():
    result = evaluate_place_success(
        [0.260, 0.098, 0.976],
        [0.600, -0.250, 1.029],
        cube_linear_velocity=np.zeros(3),
        cube_lift_m=0.0,
        grasp_observed=False,
    )

    assert not result.success
    assert result.xy_error_m > 0.4


def test_place_failure_when_cube_is_still_moving():
    result = evaluate_place_success(
        [0.600, -0.250, 1.029],
        [0.600, -0.250, 1.029],
        cube_linear_velocity=[0.08, 0.0, 0.0],
        cube_lift_m=0.18,
        grasp_observed=True,
    )

    assert not result.success


def test_place_failure_without_robot_grasp():
    result = evaluate_place_success(
        [0.600, -0.250, 1.029],
        [0.600, -0.250, 1.029],
        cube_linear_velocity=np.zeros(3),
        cube_lift_m=0.18,
        grasp_observed=False,
    )

    assert not result.success
