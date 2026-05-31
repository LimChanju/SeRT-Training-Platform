"""
TDD 사이클 5 — EventLogger 핵심 기능

[사이클 5] 이벤트 기록 및 컨텍스트 업데이트
  RED   : update_context, log_event 미구현 가정
  GREEN : 구현 후 통과
  REFACTOR: ensure_episode_started 분리
"""

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock
from v2.event_logger import EventLogger


def _mock_cube(name: str, pos: np.ndarray, velocity: np.ndarray = None):
    cube = MagicMock()
    cube.name = name
    cube.get_world_pose.return_value = (pos, None)
    if velocity is not None:
        cube.get_linear_velocity.return_value = velocity
    return cube


@pytest.fixture
def logger(tmp_path):
    log_path = str(tmp_path / "test_events.csv")
    return EventLogger(
        log_path=log_path,
        cube_size=0.05,
        speed_threshold=0.5,
        collision_dist=0.15,
        stack_drop_threshold=0.03,
        max_human_collisions=3,
    )


class TestEventLogger:
    def test_context_update(self, logger):
        # RED: update_context 후 내부 상태 반영
        logger.update_context(step=10, sim_time=0.5)
        assert logger._step == 10
        assert logger._sim_time == pytest.approx(0.5)

    def test_log_event_creates_file(self, logger):
        # GREEN: log_event 호출 시 CSV 파일 생성
        logger.update_context(1, 0.1)
        logger.ensure_episode_started()
        logger.log_event("session_start")
        assert os.path.exists(logger._log_path)

    def test_log_event_writes_row(self, logger):
        # REFACTOR: 로그 파일에 이벤트 행이 기록됨
        logger.update_context(1, 0.1)
        logger.ensure_episode_started()
        logger.log_event("test_event", "detail_value")
        with open(logger._log_path) as f:
            content = f.read()
        assert "test_event" in content

    def test_reset_cycle(self, logger):
        logger.update_context(5, 0.5)
        logger.ensure_episode_started()
        logger._drop_logged_cubes.add("cube_0")
        logger._last_contact_step["cube_0"] = 5
        logger.reset_cycle()
        assert len(logger._drop_logged_cubes) == 0
        assert len(logger._last_contact_step) == 0

    def test_reset_pick_miss(self, logger):
        logger._miss_logged_for_pick = True
        logger.reset_pick_miss()
        assert logger._miss_logged_for_pick is False

    def test_human_collision_count(self, logger):
        logger.update_context(1, 0.1)
        logger.ensure_episode_started()
        ee = np.array([0.5, 0.0, 0.5])
        # 충돌 없는 경우 (프록시 없음)
        result = logger.check_human_collision(ee, [])
        assert result is False

    def test_episode_start_idempotent(self, logger):
        logger.update_context(1, 0.1)
        logger.ensure_episode_started()
        logger.ensure_episode_started()  # 두 번 호출해도 중복 기록 없어야 함
        with open(logger._log_path) as f:
            rows = [l for l in f if "episode_start" in l]
        assert len(rows) <= 1

    def test_update_contact_near_cube(self, logger):
        logger.update_context(5, 0.5)
        ee = np.array([0.5, 0.0, 0.5])
        cube = _mock_cube("cube_0", np.array([0.5, 0.0, 0.5]))
        logger.update_contact(ee, [cube])
        assert logger._last_contact_step.get("cube_0") == 5

    def test_update_contact_far_cube(self, logger):
        logger.update_context(5, 0.5)
        ee = np.array([0.5, 0.0, 0.5])
        cube = _mock_cube("cube_0", np.array([2.0, 2.0, 2.0]))
        logger.update_contact(ee, [cube])
        assert "cube_0" not in logger._last_contact_step

    def test_check_drop_throw_fast(self, logger):
        logger.update_context(10, 1.0)
        logger.ensure_episode_started()
        vel = np.array([1.0, 0.0, 0.0])  # speed=1.0 > threshold 0.5
        cube = _mock_cube("cube_0", np.array([0.5, 0.0, 0.5]), vel)
        logger._last_contact_step["cube_0"] = 5  # 최근 접촉
        logger.check_drop_throw([cube])
        with open(logger._log_path) as f:
            assert "drop_throw" in f.read()

    def test_check_drop_throw_slow(self, logger):
        logger.update_context(10, 1.0)
        logger.ensure_episode_started()
        vel = np.array([0.1, 0.0, 0.0])  # speed=0.1 < threshold 0.5
        cube = _mock_cube("cube_0", np.array([0.5, 0.0, 0.5]), vel)
        logger.check_drop_throw([cube])
        with open(logger._log_path) as f:
            assert "drop_throw" not in f.read()

    def test_check_human_collision_detected(self, logger):
        logger.update_context(10, 1.0)
        logger.ensure_episode_started()
        ee = np.array([0.5, 0.0, 0.5])
        proxy = _mock_cube("hand_proxy", np.array([0.5, 0.0, 0.5]))
        result = logger.check_human_collision(ee, [proxy])
        assert result is False  # max_human_collisions=3 이므로 아직 False
        assert logger._human_collision_count == 1

    def test_check_human_collision_limit(self, logger):
        logger.update_context(100, 10.0)
        logger.ensure_episode_started()
        ee = np.array([0.5, 0.0, 0.5])
        proxies = [_mock_cube(f"proxy_{i}", np.array([0.5, 0.0, 0.5])) for i in range(3)]
        # 3번 충돌 → limit 도달
        for i, proxy in enumerate(proxies):
            logger.update_context(i * 100, i * 10.0)
            logger.check_human_collision(ee, [proxy])
        assert logger._human_collision_count == 3

    def test_check_collision_green(self, logger):
        logger.update_context(10, 1.0)
        logger.ensure_episode_started()
        pick = _mock_cube("pick_0", np.array([0.5, 0.0, 0.5]))
        green = _mock_cube("green_0", np.array([0.5, 0.0, 0.5]))
        logger.check_collision_green([pick], [green])
        with open(logger._log_path) as f:
            assert "collision_green" in f.read()

    def test_record_stack_expected(self, logger):
        logger.record_stack_expected("cube_0", 0.15)
        assert logger._stacked_expected["cube_0"] == pytest.approx(0.15)

    def test_check_stack_failure(self, logger):
        logger.update_context(10, 1.0)
        logger.ensure_episode_started()
        logger.record_stack_expected("cube_0", 0.15)
        cube = _mock_cube("cube_0", np.array([0.5, 0.0, 0.05]))  # z=0.05 < 0.15-0.03
        logger.check_stack_failure([cube])
        with open(logger._log_path) as f:
            assert "stack_failure" in f.read()

    def test_check_arm_robot_collision(self, logger):
        logger.update_context(10, 1.0)
        logger.ensure_episode_started()
        logger.check_arm_robot_collision("left", ["/World/Franka/panda_link3"])
        with open(logger._log_path) as f:
            assert "arm_robot_collision" in f.read()

    def test_check_arm_robot_collision_empty(self, logger):
        logger.update_context(10, 1.0)
        initial_events = len(open(logger._log_path).readlines()) if os.path.exists(logger._log_path) else 0
        logger.ensure_episode_started()
        logger.check_arm_robot_collision("left", [])  # 빈 목록 → 기록 안 됨
        with open(logger._log_path) as f:
            assert "arm_robot_collision" not in f.read()
