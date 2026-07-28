import pytest

from v3_chan.experiment_metadata import (
    build_experiment_metadata,
    validate_experiment_metadata,
)


def _metadata():
    return build_experiment_metadata(
        environment={"XR_HAND_SPHERE_ENABLED": "1"},
        participant_id="P01",
        participant_session_index=2,
        participant_handedness="right",
        is_practice=0,
        experiment_condition="pilot",
        experiment_block_id="block_01",
        haptic_condition="on",
        protocol_version="surface_gap_dynamic_multispeed_dualclock_v4",
        room_calibration_id="vr_room_to_isaac_world_v1",
        xr_mode="vr",
        xr_backend="OpenXR",
        haptics_intensity=100,
        haptics_min_interval_s=0.08,
        haptics_contact_min_steps=1,
        haptics_udp_port=5005,
        haptics_enabled=1,
        haptics_udp_configured=True,
        collection_max_episodes=3,
        sample_interval_steps=1,
        task_success_xy_tolerance_m=0.04,
        task_success_z_tolerance_m=0.03,
        task_success_max_speed_mps=0.05,
        task_success_min_lift_m=0.05,
    )


def test_experiment_metadata_records_participant_and_haptics_fields():
    metadata = _metadata()
    assert metadata["participant_id"] == "P01"
    assert metadata["participant_session_index"] == 2
    assert metadata["participant_handedness"] == "right"
    assert metadata["is_practice"] == 0
    assert metadata["haptics_intensity"] == 100
    assert metadata["haptics_enabled"] == 1
    assert metadata["haptics_min_interval_s"] == 0.08
    assert metadata["haptics_contact_min_steps"] == 1


def _validate_metadata(**overrides):
    values = {
        "production_mode": True,
        "participant_id": "P01",
        "participant_session_index": 1,
        "participant_handedness": "right",
        "is_practice": 0,
        "experiment_condition": "pilot",
        "experiment_block_id": "block_01",
        "haptic_condition": "on",
        "haptics_enabled": 1,
        "haptics_udp_configured": 1,
        "protocol_version": "v1",
        "room_calibration_id": "room_v1",
    }
    values.update(overrides)
    validate_experiment_metadata(**values)


def test_production_metadata_rejects_missing_session_and_handedness():
    with pytest.raises(RuntimeError, match="SESSION_INDEX"):
        _validate_metadata(participant_session_index=0)
    with pytest.raises(RuntimeError, match="HANDEDNESS"):
        _validate_metadata(participant_handedness="unspecified")
    with pytest.raises(RuntimeError, match="must be one of"):
        _validate_metadata(participant_handedness="rgiht")


def test_production_metadata_requires_explicit_practice_and_block():
    with pytest.raises(RuntimeError, match="HRI_IS_PRACTICE"):
        _validate_metadata(is_practice=-1)
    with pytest.raises(RuntimeError, match="EXPERIMENT_BLOCK_ID"):
        _validate_metadata(experiment_block_id="unspecified")


def test_production_metadata_rejects_haptic_condition_mismatch():
    with pytest.raises(RuntimeError, match="does not match"):
        _validate_metadata(haptic_condition="off", haptics_enabled=1)
    with pytest.raises(RuntimeError, match="BHAPTICS_ENABLED"):
        _validate_metadata(haptics_enabled=-1)
    with pytest.raises(RuntimeError, match="NOTEBOOK_IP"):
        _validate_metadata(haptics_udp_configured=0)
