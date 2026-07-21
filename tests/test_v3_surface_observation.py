import h5py
import numpy as np

from v3_chan.hri_obs_recorder import (
    HRI_OBS_DIM,
    HRIObsRecorder,
    OBSERVATION_DIM,
    build_observation,
    flatten_observation,
)


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


def test_hri_recorder_writes_surface_schema(tmp_path):
    path = tmp_path / "surface_obs.hdf5"
    recorder = HRIObsRecorder(str(path), overwrite=True, compression=None)
    recorder.start_episode()
    recorder.add_sample(step=1, sim_time=0.01, obs=_observation(0.01))
    recorder.end_episode(success=True)
    recorder.close()

    assert HRI_OBS_DIM == 84
    with h5py.File(path, "r") as data:
        episode = data["episodes/episode_000000"]
        assert data.attrs["schema_version"] == (
            "hri_obs_v4_builtin_panda_collision_geometry"
        )
        assert int(data.attrs["observation_dim"]) == 84
        assert int(data.attrs["hri_observation_dim"]) == 84
        assert episode["obs_policy"].shape == (1, 84)
        assert episode["hri_obs_policy"].shape == (1, 84)
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
            "hri_obs_v4_builtin_panda_collision_geometry"
        )
