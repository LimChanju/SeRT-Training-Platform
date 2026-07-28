import h5py
import numpy as np
import pytest

from v3_chan.hri_obs_recorder import HRIObsRecorder
from v3_chan.validate_hri_collection import validate_hri_collection


ROW_DATASETS = (
    "sim_time",
    "monotonic_time_ns",
    "pose_monotonic_time_ns",
    "wall_time_unix_ns",
    "real_time_factor",
    "real_time_factor_valid",
    "action_command_monotonic_ns",
    "obs_policy",
    "hri_obs_policy",
    "human_valid_mask",
    "human/left_hand_pose_source_id",
    "human/right_hand_pose_source_id",
    "human/left_hand_position_tracked",
    "human/right_hand_position_tracked",
    "human/left_hand_pose_valid",
    "human/right_hand_pose_valid",
    "human/left_hand_pos",
    "human/right_hand_pos",
    "human/left_hand_vel_filtered_mps",
    "human/right_hand_vel_filtered_mps",
    "actions/previous_applied_joint_positions",
    "actions/next_commanded_joint_positions",
    "dynamic_sim/left_ttc_s",
    "dynamic_sim/right_ttc_s",
    "safety/left_ttc_s",
    "safety/right_ttc_s",
    "safety/haptic_pulse_left",
    "safety/haptic_pulse_right",
)


def _create_valid_file(path):
    with h5py.File(path, "w") as data:
        data.attrs["schema_version"] = HRIObsRecorder.SCHEMA_VERSION
        data.attrs["participant_id"] = "P01"
        data.attrs["participant_id_kind"] = "pseudonym"
        data.attrs["participant_session_index"] = 1
        data.attrs["participant_handedness"] = "right"
        data.attrs["is_practice"] = 0
        data.attrs["experiment_metadata_schema_version"] = "hri_experiment_metadata_v3"
        data.attrs["experiment_condition"] = "pilot"
        data.attrs["experiment_block_id"] = "block_01"
        data.attrs["haptic_experiment_condition"] = "on"
        data.attrs["collection_protocol_version"] = "surface_gap_dynamic_multispeed_dualclock_v4"
        data.attrs["room_calibration_id"] = "vr_room_to_isaac_world_v1"
        data.attrs["collection_label"] = "surface_twist_v8_dualclock_tracked_v1"
        data.attrs["haptics_transport"] = "udp"
        data.attrs["haptics_enabled"] = 1
        data.attrs["haptics_udp_configured"] = 1
        data.attrs["haptics_intensity"] = 100
        data.attrs["haptics_min_interval_s"] = 0.08
        data.attrs["haptics_contact_min_steps"] = 1
        data.attrs["haptics_pulse_flag_semantics"] = "udp_sendto_success_only"
        data.attrs["xr_hand_sphere_enabled"] = 1
        data.attrs["xr_anchor_status"] = "xr_anchor"
        data.attrs["session_seed"] = 42
        data.attrs["code_version"] = "source-sha256:test"
        data.attrs["source_tree_sha256"] = "a" * 64
        episodes = data.create_group("episodes")
        for index, speed in enumerate(("slow", "medium", "fast")):
            episode = episodes.create_group(f"episode_{index:06d}")
            episode.attrs["episode_length"] = 2
            episode.attrs["success"] = True
            episode.attrs["controller_speed_profile"] = speed
            initial = episode.create_group("initial_scene")
            initial.create_dataset("layout_id", data="layout-one")
            initial.create_dataset("cube_positions_world", data=np.zeros((6, 3)))
            initial.create_dataset(
                "cube_orientations_wxyz",
                data=np.tile([1.0, 0.0, 0.0, 0.0], (6, 1)),
            )
            initial.create_dataset("place_target_position_world", data=np.zeros(3))
            for name in ROW_DATASETS:
                parent = episode
                parts = name.split("/")
                for part in parts[:-1]:
                    parent = parent.require_group(part)
                if name.endswith("position_tracked"):
                    payload = np.full(2, -1, dtype=np.int8)
                elif name.endswith("pose_valid"):
                    payload = np.ones(2, dtype=np.float32)
                elif name.endswith(("hand_pos", "filtered_mps")):
                    payload = np.zeros((2, 3), dtype=np.float32)
                elif name in ("real_time_factor", "real_time_factor_valid"):
                    payload = np.ones(2, dtype=np.float32)
                else:
                    payload = (
                        np.zeros((2, 9))
                        if "joint_positions" in name
                        else np.zeros(2)
                    )
                parent.create_dataset(parts[-1], data=payload)
            episode["human"].create_dataset(
                "head_position_tracked", data=np.full(2, -1, dtype=np.int8)
            )


def test_collection_validator_accepts_three_speed_shared_layout(tmp_path):
    path = tmp_path / "valid.hdf5"
    _create_valid_file(path)
    report = validate_hri_collection(str(path))
    assert report.episode_count == 3
    assert report.layout_id == "layout-one"
    assert set(report.speed_profiles) == {"slow", "medium", "fast"}
    assert report.left_hand_pose_valid_fraction == 1.0
    assert report.right_hand_pose_valid_fraction == 1.0
    assert report.real_time_factor_valid_fraction == 1.0


def test_collection_validator_rejects_layout_confound(tmp_path):
    path = tmp_path / "layout_mismatch.hdf5"
    _create_valid_file(path)
    with h5py.File(path, "a") as data:
        del data["episodes/episode_000002/initial_scene/layout_id"]
        data["episodes/episode_000002/initial_scene"].create_dataset(
            "layout_id", data="layout-two"
        )
    with pytest.raises(ValueError, match="share exactly one layout_id"):
        validate_hri_collection(str(path))


def test_collection_validator_rejects_same_id_with_different_cube_pose(tmp_path):
    path = tmp_path / "layout_pose_mismatch.hdf5"
    _create_valid_file(path)
    with h5py.File(path, "a") as data:
        data["episodes/episode_000001/initial_scene/cube_positions_world"][0, 0] = 0.01
    with pytest.raises(ValueError, match="cube positions differ"):
        validate_hri_collection(str(path))


def test_collection_validator_rejects_unknown_code_version(tmp_path):
    path = tmp_path / "unknown_code.hdf5"
    _create_valid_file(path)
    with h5py.File(path, "a") as data:
        data.attrs["code_version"] = "unknown"
    with pytest.raises(ValueError, match="CODE_VERSION"):
        validate_hri_collection(str(path))


def test_collection_validator_rejects_timestamp_inversion(tmp_path):
    path = tmp_path / "time_inversion.hdf5"
    _create_valid_file(path)
    with h5py.File(path, "a") as data:
        data["episodes/episode_000000/monotonic_time_ns"][:] = [2, 1]
    with pytest.raises(ValueError, match="monotonic_time_ns is not monotonic"):
        validate_hri_collection(str(path))


def test_collection_validator_rejects_invalid_hand_tracking_quality(tmp_path):
    path = tmp_path / "invalid_tracking.hdf5"
    _create_valid_file(path)
    with h5py.File(path, "a") as data:
        for episode in data["episodes"].values():
            episode["human/left_hand_pose_valid"][:] = 0
            episode["human/right_hand_pose_valid"][:] = 0
    with pytest.raises(ValueError, match="pose_valid valid fraction"):
        validate_hri_collection(str(path))


def test_collection_validator_rejects_invalid_rtf_quality(tmp_path):
    path = tmp_path / "invalid_rtf.hdf5"
    _create_valid_file(path)
    with h5py.File(path, "a") as data:
        for episode in data["episodes"].values():
            episode["real_time_factor_valid"][:] = 0
    with pytest.raises(ValueError, match="real_time_factor_valid fraction"):
        validate_hri_collection(str(path))


def test_collection_validator_rejects_unapplied_xr_anchor(tmp_path):
    path = tmp_path / "failed_anchor.hdf5"
    _create_valid_file(path)
    with h5py.File(path, "a") as data:
        data.attrs["xr_anchor_status"] = "failed"
    with pytest.raises(ValueError, match="applied XR anchor"):
        validate_hri_collection(str(path))


def test_collection_validator_rejects_haptic_condition_mismatch(tmp_path):
    path = tmp_path / "haptic_mismatch.hdf5"
    _create_valid_file(path)
    with h5py.File(path, "a") as data:
        data.attrs["haptic_experiment_condition"] = "off"
    with pytest.raises(ValueError, match="does not match"):
        validate_hri_collection(str(path))
