import csv

from v3_chan.event_logger import EventLogger


def _logger(tmp_path):
    return EventLogger(
        log_path=str(tmp_path / "markers.csv"),
        sample_path=str(tmp_path / "samples.csv"),
        cube_size=0.05,
        speed_threshold=0.5,
        collision_dist=0.1,
        stack_drop_threshold=0.03,
        max_human_collisions=100,
        session_id="session_test",
    )


def test_cycle_reset_clears_marker_debounce_and_tracks_episode(tmp_path):
    logger = _logger(tmp_path)
    logger.update_context(
        10,
        1.0,
        monotonic_time_ns=100,
        wall_time_unix_ns=200,
    )
    logger.start_episode()
    logger.check_arm_robot_collision("left", ["/World/Franka/panda_hand"], -0.01)
    logger.end_episode()

    logger.reset_cycle()
    logger.update_context(
        1,
        0.1,
        monotonic_time_ns=300,
        wall_time_unix_ns=400,
    )
    logger.start_episode()
    logger.check_arm_robot_collision("left", ["/World/Franka/panda_hand"], -0.01)

    with open(logger._log_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    collisions = [row for row in rows if row["event"] == "arm_robot_collision"]
    assert [row["episode_index"] for row in collisions] == ["0", "1"]
    assert [row["session_id"] for row in collisions] == [
        "session_test",
        "session_test",
    ]
    assert collisions[1]["monotonic_time_ns"] == "300"
    assert collisions[1]["wall_time_unix_ns"] == "400"


def test_sample_log_contains_session_episode_and_sync_clocks(tmp_path):
    logger = _logger(tmp_path)
    logger.update_context(
        1,
        0.1,
        monotonic_time_ns=123,
        wall_time_unix_ns=456,
    )
    logger.start_episode()
    logger.log_sample(0.2, 0.3, 0.2, False)

    with open(logger._sample_path, newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["session_id"] == "session_test"
    assert row["episode_index"] == "0"
    assert row["monotonic_time_ns"] == "123"
    assert row["wall_time_unix_ns"] == "456"
