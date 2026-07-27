import h5py
import numpy as np

from v3_chan.dynamic_safety import DynamicSafetyConfig, DynamicSafetyEstimator
from v3_chan.hri_obs_recorder import HRIObsRecorder, build_observation


class _PoseObject:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def get_world_pose(self):
        return self.position, np.array([1.0, 0.0, 0.0, 0.0])

    def get_linear_velocity(self):
        return np.zeros(3)

    def get_angular_velocity(self):
        return np.zeros(3)


class _Gripper:
    def get_joint_positions(self):
        return np.array([0.02, 0.02])


class _Robot:
    def __init__(self):
        self.end_effector = _PoseObject([0.0, 0.0, 0.0])
        self.gripper = _Gripper()

    def get_joint_positions(self):
        return np.zeros(9)

    def get_joint_velocities(self):
        return np.zeros(9)


def _observation():
    return build_observation(
        robot=_Robot(),
        cube=_PoseObject([0.1, 0.0, 0.0]),
        place_target=_PoseObject([0.5, 0.0, 0.0]),
        human_left_hand_pos=np.array([0.2, 0.0, 0.0]),
        human_right_hand_pos=np.array([0.3, 0.0, 0.0]),
        gripper_center_pos=np.zeros(3),
        left_hand_surface_gap_override=0.1,
        right_hand_surface_gap_override=0.2,
        geometry_valid_override=True,
    )


def _config():
    return DynamicSafetyConfig(
        ema_time_constant_s=0.1,
        ttc_cap_s=10.0,
        min_valid_closing_speed_mps=0.01,
        max_valid_dt_s=0.25,
    )


def _update(
    estimator,
    time_s,
    left_pos,
    left_gap,
    left_collider=1,
    *,
    left_valid=True,
    robot_origin=None,
    robot_orientation=None,
    surface_point=None,
    robot_angular_velocity=None,
):
    if left_valid:
        robot_origin = np.zeros(3) if robot_origin is None else robot_origin
        robot_orientation = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if robot_orientation is None
            else robot_orientation
        )
        surface_point = np.zeros(3) if surface_point is None else surface_point
        robot_angular_velocity = (
            np.zeros(3) if robot_angular_velocity is None else robot_angular_velocity
        )
    return estimator.update(
        sim_time_s=time_s,
        left_hand_pos=left_pos,
        right_hand_pos=None,
        left_tracking_valid=left_valid,
        right_tracking_valid=False,
        left_surface_gap_m=left_gap,
        right_surface_gap_m=10.0,
        left_geometry_valid=left_valid,
        right_geometry_valid=False,
        left_closest_collider_id=left_collider if left_valid else 0,
        right_closest_collider_id=0,
        left_closest_robot_origin_pos=robot_origin if left_valid else None,
        right_closest_robot_origin_pos=None,
        left_closest_robot_orientation_wxyz=(robot_orientation if left_valid else None),
        left_closest_surface_point_world_pos=(surface_point if left_valid else None),
        left_closest_robot_angular_velocity_world_radps=(
            robot_angular_velocity if left_valid else None
        ),
    )


def test_constant_approach_has_negative_gap_rate_and_finite_ttc():
    estimator = DynamicSafetyEstimator(_config())
    _update(estimator, 0.0, [1.0, 0.0, 0.0], 1.0)
    sample = _update(estimator, 0.1, [0.9, 0.0, 0.0], 0.9)

    assert sample.left.hand_velocity_valid
    assert sample.left.dynamic_valid
    assert np.allclose(sample.left.hand_velocity_raw_mps, [-1.0, 0.0, 0.0])
    assert np.isclose(sample.left.surface_gap_rate_raw_mps, -1.0)
    assert np.isclose(sample.left.surface_gap_rate_filtered_mps, -1.0)
    assert np.isclose(sample.left.closing_speed_mps, 1.0)
    assert np.isclose(sample.left.ttc_s, 0.9)
    assert np.allclose(sample.left.relative_velocity_world_mps, [-1.0, 0.0, 0.0])


def test_receding_hand_has_no_closing_speed_and_invalid_ttc():
    estimator = DynamicSafetyEstimator(_config())
    _update(estimator, 0.0, [0.5, 0.0, 0.0], 0.5)
    sample = _update(estimator, 0.1, [0.6, 0.0, 0.0], 0.6)

    assert sample.left.hand_velocity_valid
    assert not sample.left.dynamic_valid
    assert sample.left.closing_speed_mps == 0.0
    assert sample.left.ttc_s == 10.0


def test_surface_point_velocity_includes_link_rotation():
    estimator = DynamicSafetyEstimator(_config())
    _update(
        estimator,
        0.0,
        [1.0, 0.0, 0.0],
        0.2,
        robot_origin=np.array([-0.05, 0.0, 0.0]),
        surface_point=np.array([0.95, 0.0, 0.0]),
        robot_angular_velocity=np.array([0.0, 0.0, 2.0]),
    )
    sample = _update(
        estimator,
        0.1,
        [0.9, 0.0, 0.0],
        0.1,
        robot_origin=np.zeros(3),
        surface_point=np.array([1.0, 0.0, 0.0]),
        robot_angular_velocity=np.array([0.0, 0.0, 2.0]),
    )

    assert sample.left.robot_surface_velocity_valid
    assert np.allclose(
        sample.left.closest_robot_origin_velocity_world_mps, [0.5, 0.0, 0.0]
    )
    assert np.allclose(
        sample.left.closest_robot_rotational_velocity_world_mps, [0.0, 2.0, 0.0]
    )
    assert np.allclose(sample.left.closest_robot_velocity_world_mps, [0.5, 2.0, 0.0])


def test_penetration_has_zero_ttc():
    estimator = DynamicSafetyEstimator(_config())
    _update(estimator, 0.0, [0.02, 0.0, 0.0], 0.01)
    sample = _update(estimator, 0.1, [0.0, 0.0, 0.0], -0.01)

    assert sample.left.ttc_s == 0.0
    assert sample.min_ttc_s == 0.0


def test_episode_reset_does_not_emit_velocity_spike():
    estimator = DynamicSafetyEstimator(_config())
    _update(estimator, 0.0, [0.0, 0.0, 0.0], 0.2)
    _update(estimator, 0.1, [0.1, 0.0, 0.0], 0.1)
    estimator.reset()
    sample = _update(estimator, 0.2, [8.0, 0.0, 0.0], 0.1)

    assert not sample.left.hand_velocity_valid
    assert not sample.left.dynamic_valid
    assert np.allclose(sample.left.hand_velocity_raw_mps, 0.0)
    assert sample.left.surface_gap_rate_raw_mps == 0.0


def test_tracking_recovery_does_not_emit_velocity_spike():
    estimator = DynamicSafetyEstimator(_config())
    _update(estimator, 0.0, [0.0, 0.0, 0.0], 0.2)
    _update(estimator, 0.1, [0.1, 0.0, 0.0], 0.1)
    invalid = _update(estimator, 0.2, None, 10.0, left_valid=False)
    recovered = _update(estimator, 0.3, [5.0, 0.0, 0.0], 0.2)
    resumed = _update(estimator, 0.4, [5.1, 0.0, 0.0], 0.1)

    assert not invalid.left.hand_velocity_valid
    assert not recovered.left.hand_velocity_valid
    assert np.allclose(recovered.left.hand_velocity_raw_mps, 0.0)
    assert resumed.left.hand_velocity_valid
    assert np.allclose(resumed.left.hand_velocity_raw_mps, [1.0, 0.0, 0.0])


def test_abnormal_simulation_dt_resets_the_estimator():
    estimator = DynamicSafetyEstimator(_config())
    _update(estimator, 0.0, [0.0, 0.0, 0.0], 0.2)
    _update(estimator, 0.1, [0.1, 0.0, 0.0], 0.1)
    reset_sample = _update(estimator, 1.0, [5.0, 0.0, 0.0], 0.1)
    resumed = _update(estimator, 1.1, [5.1, 0.0, 0.0], 0.05)

    assert not reset_sample.left.hand_velocity_valid
    assert np.allclose(reset_sample.left.hand_velocity_raw_mps, 0.0)
    assert resumed.left.hand_velocity_valid
    assert np.allclose(resumed.left.hand_velocity_raw_mps, [1.0, 0.0, 0.0])


def test_closest_collider_change_preserves_hand_velocity_and_resets_geometry():
    estimator = DynamicSafetyEstimator(_config())
    _update(estimator, 0.0, [0.0, 0.0, 0.0], 0.5, left_collider=1)
    before_switch = _update(estimator, 0.1, [0.1, 0.0, 0.0], 0.4, left_collider=1)
    switched = _update(estimator, 0.2, [0.3, 0.0, 0.0], 0.3, left_collider=2)
    resumed = _update(estimator, 0.3, [0.4, 0.0, 0.0], 0.2, left_collider=2)

    assert switched.left.closest_collider_switched
    assert switched.left.hand_velocity_valid
    assert not switched.left.dynamic_valid
    assert switched.left.surface_gap_rate_raw_mps == 0.0
    assert np.allclose(switched.left.hand_velocity_raw_mps, [2.0, 0.0, 0.0])
    assert (
        switched.left.hand_velocity_filtered_mps[0]
        > before_switch.left.hand_velocity_filtered_mps[0]
    )
    assert switched.left.hand_velocity_filtered_mps[0] < 2.0
    assert np.allclose(switched.left.closest_robot_velocity_world_mps, 0.0)
    assert np.allclose(switched.left.relative_velocity_world_mps, 0.0)
    assert resumed.left.dynamic_valid
    assert resumed.left.surface_gap_rate_raw_mps < 0.0


def test_recorder_dynamic_dataset_lengths_match_episode_frames(tmp_path):
    estimator = DynamicSafetyEstimator(_config())
    path = tmp_path / "dynamic_obs.hdf5"
    recorder = HRIObsRecorder(
        str(path),
        overwrite=True,
        compression=None,
        file_metadata=estimator.metadata(),
    )
    recorder.start_episode()
    for step in range(3):
        sample = _update(
            estimator,
            step * 0.1,
            [1.0 - 0.1 * step, 0.0, 0.0],
            1.0 - 0.1 * step,
        )
        recorder.add_sample(
            step=step,
            sim_time=step * 0.1,
            obs=_observation(),
            dynamic={**sample.human_payload(), **sample.safety_payload()},
        )
    recorder.end_episode(success=True)
    recorder.close()

    with h5py.File(path, "r") as data:
        episode = data["episodes/episode_000000"]
        assert data.attrs["schema_version"] == "hri_obs_v6_surface_point_dynamic_safety"
        assert np.isclose(data.attrs["dynamic_ema_time_constant_s"], 0.1)
        assert np.isclose(data.attrs["dynamic_ttc_cap_s"], 10.0)
        assert (
            data.attrs["dynamic_collider_switch_reset_scope"]
            == "gap_rate_and_robot_link_pose_only"
        )
        assert bool(
            data.attrs["dynamic_hand_velocity_preserved_across_collider_switch"]
        )
        assert data.attrs["dynamic_robot_velocity_method"] == (
            "closest_surface_point_rigid_body_twist"
        )
        assert bool(data.attrs["dynamic_angular_velocity_correction_used"])
        expected_human = {
            "left_hand_vel_raw_mps",
            "right_hand_vel_raw_mps",
            "left_hand_vel_filtered_mps",
            "right_hand_vel_filtered_mps",
            "left_hand_velocity_valid",
            "right_hand_velocity_valid",
        }
        expected_safety = {
            "left_closest_surface_point_world_pos",
            "right_closest_surface_point_world_pos",
            "left_closest_robot_origin_velocity_world_mps",
            "right_closest_robot_origin_velocity_world_mps",
            "left_closest_robot_angular_velocity_world_radps",
            "right_closest_robot_angular_velocity_world_radps",
            "left_closest_robot_rotational_velocity_world_mps",
            "right_closest_robot_rotational_velocity_world_mps",
            "left_closest_robot_velocity_world_mps",
            "right_closest_robot_velocity_world_mps",
            "left_robot_surface_velocity_valid",
            "right_robot_surface_velocity_valid",
            "left_relative_velocity_world_mps",
            "right_relative_velocity_world_mps",
            "left_surface_gap_rate_raw_mps",
            "right_surface_gap_rate_raw_mps",
            "left_surface_gap_rate_filtered_mps",
            "right_surface_gap_rate_filtered_mps",
            "left_closing_speed_mps",
            "right_closing_speed_mps",
            "left_ttc_s",
            "right_ttc_s",
            "left_dynamic_valid",
            "right_dynamic_valid",
            "min_ttc_s",
            "max_closing_speed_mps",
            "dynamic_valid",
            "closest_collider_switched_left",
            "closest_collider_switched_right",
        }
        assert expected_human <= set(episode["human"])
        assert expected_safety <= set(episode["safety"])
        assert episode["human/left_hand_vel_raw_mps"].shape == (3, 3)
        assert episode["human/right_hand_velocity_valid"].shape == (3,)
        assert episode["safety/left_relative_velocity_world_mps"].shape == (3, 3)
        assert episode["safety/left_closest_surface_point_world_pos"].shape == (3, 3)
        assert episode["safety/left_ttc_s"].shape == (3,)
        assert episode["safety/closest_collider_switched_left"].shape == (3,)
        for group_name in ("human", "safety"):
            for dataset in episode[group_name].values():
                assert dataset.shape[0] == 3


def test_recorder_persists_aborted_partial_episode(tmp_path):
    path = tmp_path / "interrupted_obs.hdf5"
    recorder = HRIObsRecorder(str(path), overwrite=True, compression=None)
    recorder.start_episode({"mode": "velocity_test"})
    recorder.add_sample(step=1, sim_time=0.01, obs=_observation())
    saved_path = recorder.end_episode(
        success=False,
        metadata={"reason": "interrupted", "aborted": True, "interrupted": True},
    )
    recorder.close()

    assert saved_path == "/episodes/episode_000000"
    with h5py.File(path, "r") as data:
        episode = data["episodes/episode_000000"]
        assert not bool(episode.attrs["success"])
        assert bool(episode.attrs["aborted"])
        assert bool(episode.attrs["interrupted"])
        assert episode.attrs["reason"] == "interrupted"
        assert episode.attrs["episode_length"] == 1
        assert episode["sim_time"].shape == (1,)
