"""
TDD 사이클 3 & 4 — A/B 테스트

[사이클 3] variant 할당 일관성
  RED   : 같은 ID가 매번 다른 variant를 받을 수 있다 가정
  GREEN : SHA-256 해시 기반 고정 할당 구현
  REFACTOR: AB_VARIANT_OVERRIDE로 강제 지정 분리

[사이클 4] 이벤트 추적 CSV 기록
  RED   : record() 호출 후 CSV 파일이 없다 가정
  GREEN : _append_to_csv 구현
  REFACTOR: ABEventTracker dataclass로 정리
"""

import csv
import os
import tempfile
import pytest
from v2.ab_test import assign_variant, ABEventTracker, VARIANT_A, VARIANT_B, create_session


# ── 사이클 3: variant 할당 일관성 ────────────────────────────────────────────

class TestVariantAssignment:
    def setup_method(self):
        os.environ.pop("AB_VARIANT_OVERRIDE", None)
        os.environ.pop("AB_VARIANT_B_PCT", None)

    def test_same_id_same_variant(self):
        # RED: 동일 ID는 항상 같은 variant
        v1 = assign_variant("S001")
        v2 = assign_variant("S001")
        assert v1.name == v2.name

    def test_override_a(self):
        # GREEN: 강제 지정 A
        os.environ["AB_VARIANT_OVERRIDE"] = "A"
        assert assign_variant("S999").name == "A"

    def test_override_b(self):
        # GREEN: 강제 지정 B
        os.environ["AB_VARIANT_OVERRIDE"] = "B"
        assert assign_variant("S999").name == "B"

    def test_pct_100_all_b(self):
        # REFACTOR: B 비율 100% → 모두 B
        os.environ["AB_VARIANT_B_PCT"] = "100"
        for i in range(20):
            assert assign_variant(f"S{i:03d}").name == "B"

    def test_pct_0_all_a(self):
        # B 비율 0% → 모두 A
        os.environ["AB_VARIANT_B_PCT"] = "0"
        for i in range(20):
            assert assign_variant(f"S{i:03d}").name == "A"

    def test_distribution_balanced(self):
        # 기본(50%) 설정 시 A/B 어느 정도 분산
        os.environ.pop("AB_VARIANT_B_PCT", None)
        results = [assign_variant(f"X{i:04d}").name for i in range(200)]
        a_count = results.count("A")
        assert 60 <= a_count <= 140  # 30~70% 범위


# ── variant 설정값 검증 ───────────────────────────────────────────────────────

class TestVariantConfig:
    def test_variant_a_cubes(self):
        assert VARIANT_A.num_cubes == 3

    def test_variant_b_cubes(self):
        assert VARIANT_B.num_cubes == 5

    def test_variant_b_stricter_threshold(self):
        assert VARIANT_B.speed_threshold < VARIANT_A.speed_threshold


# ── 사이클 4: 이벤트 추적 CSV ────────────────────────────────────────────────

class TestABEventTracker:
    def _make_tracker(self, tmp_path, subject_id="S001"):
        log_path = str(tmp_path / f"ab_{subject_id}.csv")
        return ABEventTracker(
            subject_id=subject_id,
            variant=VARIANT_A,
            log_path=log_path,
        )

    def test_csv_created_on_init(self, tmp_path):
        # RED: 초기화 시 CSV 파일이 생성돼야 한다
        tracker = self._make_tracker(tmp_path)
        assert os.path.exists(tracker.log_path)

    def test_session_start_logged(self, tmp_path):
        # GREEN: session_start 이벤트가 자동 기록
        tracker = self._make_tracker(tmp_path)
        with open(tracker.log_path) as f:
            content = f.read()
        assert "session_start" in content

    def test_record_appends_row(self, tmp_path):
        # REFACTOR: record() 호출마다 행 추가
        tracker = self._make_tracker(tmp_path)
        before = len(tracker._events)
        tracker.record("test_event", {"value": 42})
        assert len(tracker._events) == before + 1

    def test_task_complete_fields(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record_task_complete(duration_sec=35.5, pick_count=3)
        last = tracker._events[-1]
        assert last["event"] == "task_complete"
        assert last["duration_sec"] == 35.5

    def test_collision_recorded(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record_collision("panda_link3")
        last = tracker._events[-1]
        assert last["event"] == "collision"
        assert last["object"] == "panda_link3"

    def test_erp_marker_recorded(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record_erp_marker("P300", confidence=0.91)
        last = tracker._events[-1]
        assert last["event"] == "erp_marker"
        assert last["marker_type"] == "P300"

    def test_summary_contains_variant(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        s = tracker.summary()
        assert s["variant"] == "A"
        assert s["subject_id"] == "S001"

    def test_create_session_helper(self, tmp_path):
        os.environ.pop("AB_VARIANT_OVERRIDE", None)
        tracker = create_session("S001", log_dir=str(tmp_path))
        assert tracker.subject_id == "S001"
        assert tracker.variant.name in ("A", "B")
