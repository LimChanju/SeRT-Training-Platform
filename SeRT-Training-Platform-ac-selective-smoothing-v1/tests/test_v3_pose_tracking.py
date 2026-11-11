import numpy as np

from v3_chan.pose_tracking import (
    PoseSample,
    PoseSourceLatch,
    TRACKING_TRACKED,
)


def _sample(source: str, x: float) -> PoseSample:
    return PoseSample(
        position_world=np.array([x, 0.0, 1.0]),
        pose_valid=True,
        position_tracked=TRACKING_TRACKED,
        tracking_status_known=True,
        source_name=source,
        source_path=f"/{source}",
        acquisition_monotonic_ns=123,
    )


def test_pose_source_latch_invalidates_transition_until_confirmed():
    latch = PoseSourceLatch(confirmation_frames=3)

    assert latch.update(_sample("xr_physical", 0.0)).pose_valid
    assert not latch.update(_sample("xr_stage_visual", 1.0)).pose_valid
    assert not latch.update(_sample("xr_stage_visual", 1.1)).pose_valid
    switched = latch.update(_sample("xr_stage_visual", 1.2))

    assert switched.pose_valid
    assert switched.source_switched
    assert switched.source_name == "xr_stage_visual"


def test_pose_source_latch_cancels_transient_candidate():
    latch = PoseSourceLatch(confirmation_frames=2)
    latch.update(_sample("xr_physical", 0.0))

    assert not latch.update(_sample("xr_virtual_world", 5.0)).pose_valid
    recovered = latch.update(_sample("xr_physical", 0.1))

    assert recovered.pose_valid
    assert not recovered.source_switched


def test_pose_sample_normalizes_orientation_and_rejects_invalid_position():
    sample = PoseSample(
        position_world=[1.0, 2.0, 3.0],
        orientation_wxyz=[2.0, 0.0, 0.0, 0.0],
        pose_valid=True,
    )
    invalid = PoseSample(position_world=[np.nan, 0.0, 0.0], pose_valid=True)

    np.testing.assert_allclose(sample.orientation_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert sample.pose_valid
    assert not invalid.pose_valid
