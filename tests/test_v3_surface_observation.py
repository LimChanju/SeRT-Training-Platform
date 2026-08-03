import h5py
import numpy as np

from v3_chan.hri_obs_recorder import (
    HRI_OBS_DIM,
    HRI_OBS_FIELD_NAMES,
    HRIObsRecorder,
    OBSERVATION_DIM,
    build_observation,
    flatten_observation,
)
from v3_chan.rl.observations import (
    DYNAMIC_HRI_OBS_DIM,
    DYNAMIC_HRI_OBS_FIELD_NAMES,
    apply_dynamic_hri_observation,
    flatten_dynamic_hri_observation,
)
from v3_chan.pose_tracking import PoseSample, TRACKING_TRACKED


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


def _observation(surface_gap_override=None):
    return build_observation(
        robot=_Robot(),
        cube=_PoseObject([0.1, 0.0, 0.0]),
        place_target=_PoseObject([0.5, 0.0, 0.0]),
        human_left_hand_pos=np.array([0.2, 0.0, 0.0]),
        gripper_center_pos=np.zeros(3),
        min_hand_gripper_surface_gap_override=surface_gap_override,
        hand_proxy_radius_m=0.035,
        gripper_proxy_radius_m=0.025,
    )


def _scalar(obs, name):
    return float(np.asarray(obs[name]).reshape(-1)[0])


def test_surface_gap_flags_and_policy_compatibility():
    obs = _observation()

    assert OBSERVATION_DIM == 84
    assert flatten_observation(obs).shape == (84,)
    assert np.isclose(_scalar(obs, "min_hand_gripper_center_dist"), 0.2)
    assert np.isclose(_scalar(obs, "min_hand_gripper_surface_gap"), 0.14)
    assert np.isclose(_scalar(obs, "min_hand_gripper_dist"), 0.14)
    assert _scalar(obs, "near_human") == 0.0
    assert _scalar(obs, "near_miss") == 0.0
    assert _scalar(obs, "human_robot_collision") == 0.0

    near_miss_obs = _observation(surface_gap_override=0.01)
    assert _scalar(near_miss_obs, "near_human") == 1.0
    assert _scalar(near_miss_obs, "near_miss") == 1.0
    assert _scalar(near_miss_obs, "human_robot_collision") == 0.0

    collision_obs = _observation(surface_gap_override=-0.01)
    assert _scalar(collision_obs, "near_human") == 1.0
    assert _scalar(collision_obs, "near_miss") == 0.0
    assert _scalar(collision_obs, "human_robot_collision") == 1.0


def test_dynamic_safety_policy_observation_extends_static_schema():
    obs = _observation(surface_gap_override=0.04)
    apply_dynamic_hri_observation(
        obs,
        {
            "left_hand_vel_filtered_mps": [0.2, 0.0, 0.0],
            "left_closest_robot_velocity_world_mps": [0.05, 0.0, 0.0],
            "left_relative_velocity_world_mps": [0.15, 0.0, 0.0],
            "left_closing_speed_mps": 0.15,
            "left_ttc_s": 0.25,
            "left_dynamic_measurement_valid": 1.0,
            "left_ttc_valid": 1.0,
        },
    )
    dynamic = flatten_dynamic_hri_observation(obs)

    assert HRI_OBS_DIM == 83
    assert DYNAMIC_HRI_OBS_DIM == 109
    assert dynamic.shape == (109,)
    assert DYNAMIC_HRI_OBS_FIELD_NAMES[: len(HRI_OBS_FIELD_NAMES)] == (
        HRI_OBS_FIELD_NAMES
    )
    assert np.isclose(_scalar(obs, "left_closing_speed_mps"), 0.15)
    assert np.isclose(_scalar(obs, "left_ttc_s"), 0.25)
    assert _scalar(obs, "left_ttc_valid") == 1.0
    assert np.isclose(_scalar(obs, "right_ttc_s"), 10.0)


def test_hri_recorder_writes_surface_schema(tmp_path):
    class _Action:
        def __init__(self, value):
            self.joint_positions = np.full(9, value, dtype=np.float32)
            self.joint_velocities = None
            self.joint_efforts = None

    path = tmp_path / "surface_obs.hdf5"
    recorder = HRIObsRecorder(
        str(path),
        overwrite=True,
        compression=None,
        file_metadata={
            "session_id": "session_test",
            "participant_id": "participant_test",
            "collection_protocol_version": "protocol_test",
            "time_sync_schema": "sim_monotonic_unix_v1",
        },
    )
    recorder.start_episode(
        initial_scene={
            "layout_id": "layout-001",
            "cube_positions_world": np.zeros((6, 3)),
        }
    )
    tracked_left = PoseSample(
        position_world=[0.2, 0.0, 0.0],
        pose_valid=True,
        position_tracked=TRACKING_TRACKED,
        tracking_status_known=True,
        source_name="openxr_joint",
        source_path="/user/hand/left/palm",
        acquisition_monotonic_ns=120,
    )
    recorder.add_sample(
        step=1,
        sim_time=0.01,
        monotonic_time_ns=123,
        pose_monotonic_time_ns=124,
        wall_time_unix_ns=456,
        real_time_factor=0.75,
        real_time_factor_valid=True,
        action_command_monotonic_ns=125,
        obs=_observation(0.01),
        tracking={"left_hand": tracked_left},
        dynamic_sim={"left_ttc_s": 1.25},
        previous_action=_Action(1.0),
        next_action=_Action(2.0),
    )
    recorder.end_episode(success=True)
    recorder.close()

    assert HRI_OBS_DIM == 83
    assert "min_hand_gripper_center_dist" not in HRI_OBS_FIELD_NAMES
    with h5py.File(path, "r") as data:
        episode = data["episodes/episode_000000"]
        assert data.attrs["schema_version"] == (
            "hri_obs_v8_dual_clock_tracked_action_aligned"
        )
        assert int(data.attrs["observation_dim"]) == 84
        assert int(data.attrs["hri_observation_dim"]) == 83
        assert data.attrs["hri_observation_version"] == (
            "hri_policy_obs_v1_83d_surface_gap"
        )
        assert data.attrs["session_id"] == "session_test"
        assert data.attrs["participant_id"] == "participant_test"
        assert data.attrs["collection_protocol_version"] == "protocol_test"
        assert data.attrs["time_sync_schema"] == "sim_monotonic_unix_v1"
        assert episode["obs_policy"].shape == (1, 84)
        assert episode["hri_obs_policy"].shape == (1, 83)
        assert episode["monotonic_time_ns"].shape == (1,)
        assert int(episode["pose_monotonic_time_ns"][0]) == 124
        assert episode["wall_time_unix_ns"].shape == (1,)
        assert int(episode["monotonic_time_ns"][0]) == 123
        assert int(episode["wall_time_unix_ns"][0]) == 456
        assert np.isclose(episode["real_time_factor"][0], 0.75)
        assert bool(episode["real_time_factor_valid"][0])
        assert int(episode["action_command_monotonic_ns"][0]) == 125
        assert bool(episode["human/left_hand_pose_valid"][0])
        assert int(episode["human/left_hand_position_tracked"][0]) == 1
        assert int(episode["human_valid_mask"][0, 1]) == 1
        assert np.isclose(episode["dynamic_sim/left_ttc_s"][0], 1.25)
        np.testing.assert_allclose(
            episode["actions/previous_applied_joint_positions"][0], 1.0
        )
        np.testing.assert_allclose(
            episode["actions/next_commanded_joint_positions"][0], 2.0
        )
        assert bool(episode["actions/previous_applied_valid"][0])
        assert bool(episode["actions/next_commanded_valid"][0])
        layout_id = episode["initial_scene/layout_id"][()]
        if isinstance(layout_id, bytes):
            layout_id = layout_id.decode("utf-8")
        assert layout_id == "layout-001"
        assert episode["initial_scene/cube_positions_world"].shape == (6, 3)
        assert "episode_start_monotonic_ns" in episode.attrs
        assert "episode_end_monotonic_ns" in episode.attrs
        assert "min_hand_gripper_center_dist" in episode["obs"]
        assert "min_hand_gripper_surface_gap" in episode["obs"]
        assert "near_miss" in episode["obs"]
        assert "distance_gate" in episode["obs"]
        assert "geometry_valid" in episode["obs"]
        assert "min_hand_end_effector_surface_gap" in episode["obs"]
        assert "min_hand_gripper_center_dist_m" in episode["safety"]
        assert "min_hand_gripper_surface_gap_m" in episode["safety"]
        assert "left_hand_surface_gap_m" in episode["safety"]
        assert "closest_collider_left_id" in episode["safety"]
        assert "distance_gate" in episode["safety"]
        assert float(episode["safety/near_miss"][0]) == 1.0


def test_hri_recorder_preserves_nonempty_legacy_schema(tmp_path):
    legacy_path = tmp_path / "legacy.hdf5"
    with h5py.File(legacy_path, "w") as data:
        data.attrs["schema_version"] = "hri_obs_v3_surface_gap"
        episodes = data.create_group("episodes")
        episodes.create_group("episode_000000")

    recorder = HRIObsRecorder(str(legacy_path), overwrite=False, compression=None)
    new_path = recorder.path
    recorder.close()

    assert new_path != str(legacy_path)
    with h5py.File(legacy_path, "r") as data:
        assert data.attrs["schema_version"] == "hri_obs_v3_surface_gap"
        assert "episode_000000" in data["episodes"]
    with h5py.File(new_path, "r") as data:
        assert data.attrs["schema_version"] == (
            "hri_obs_v8_dual_clock_tracked_action_aligned"
        )
