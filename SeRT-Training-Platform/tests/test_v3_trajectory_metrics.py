import numpy as np

from v3_chan.trajectory_metrics import CartesianMotionTracker


def test_cartesian_motion_tracker_recovers_constant_velocity():
    tracker = CartesianMotionTracker([0.0, 0.0, 0.0], 0.0)

    samples = [
        tracker.update([0.1, 0.0, 0.0], 0.1),
        tracker.update([0.2, 0.0, 0.0], 0.2),
        tracker.update([0.3, 0.0, 0.0], 0.3),
    ]
    summary = tracker.summary()

    assert all(sample.velocity_valid for sample in samples)
    assert samples[1].acceleration_valid
    assert samples[2].jerk_valid
    np.testing.assert_allclose(samples[-1].velocity_mps, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(samples[-1].acceleration_mps2, 0.0, atol=1e-12)
    np.testing.assert_allclose(samples[-1].jerk_mps3, 0.0, atol=1e-12)
    assert np.isclose(summary["ee_path_length_m"], 0.3)
    assert np.isclose(summary["mean_ee_speed_mps"], 1.0)
    assert np.isclose(summary["rms_ee_jerk_mps3"], 0.0)


def test_cartesian_motion_tracker_integrates_nonzero_jerk():
    tracker = CartesianMotionTracker([0.0, 0.0, 0.0], 0.0)
    positions = [0.001, 0.008, 0.027, 0.064]

    sample = None
    for index, position in enumerate(positions, start=1):
        sample = tracker.update([position, 0.0, 0.0], index * 0.1)

    summary = tracker.summary()
    assert sample is not None and sample.jerk_valid
    assert sample.jerk_norm_mps3 > 0.0
    assert summary["ee_jerk_sample_count"] == 2
    assert summary["p95_ee_jerk_mps3"] > 0.0
    assert summary["integrated_squared_ee_jerk_m2ps5"] > 0.0


def test_cartesian_motion_tracker_resets_after_nonmonotonic_time():
    tracker = CartesianMotionTracker([0.0, 0.0, 0.0], 1.0)
    invalid = tracker.update([0.1, 0.0, 0.0], 1.0)
    recovered = tracker.update([0.2, 0.0, 0.0], 1.1)

    assert not invalid.velocity_valid
    assert recovered.velocity_valid
    assert not recovered.acceleration_valid
