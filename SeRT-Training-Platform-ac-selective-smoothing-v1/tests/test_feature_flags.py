"""
TDD 사이클 1 & 2 — FeatureFlags

[사이클 1] ENABLE_VR 기본 OFF + 환경변수 ON
  RED   : enable_vr가 존재하지 않는다 가정 → AttributeError 예상
  GREEN : enable_vr property 구현 → 통과
  REFACTOR: _env_bool 헬퍼로 중복 제거

[사이클 2] 피험자 ID 해시 bucket 일관성
  RED   : 동일 ID가 다른 bucket을 반환할 수 있다 가정 → 불일치 예상
  GREEN : MD5 해시 고정 → 항상 동일 bucket
  REFACTOR: _subject_bucket 함수로 분리
"""

import os
import pytest
from v2.feature_flags import FeatureFlags, _subject_bucket


# ── 사이클 1: ENABLE_VR 토글 ─────────────────────────────────────────────────

class TestEnableVR:
    def test_default_off(self):
        # RED: 환경변수 없으면 OFF여야 한다
        os.environ.pop("ENABLE_VR", None)
        os.environ.pop("VR_SUBJECT_IDS", None)
        flags = FeatureFlags(subject_id="S001")
        assert flags.enable_vr is False

    def test_env_var_on(self):
        # GREEN: ENABLE_VR=true 설정 시 ON
        os.environ["ENABLE_VR"] = "true"
        flags = FeatureFlags(subject_id="S001")
        assert flags.enable_vr is True
        del os.environ["ENABLE_VR"]

    def test_subject_id_allowlist(self):
        # REFACTOR: VR_SUBJECT_IDS 목록에 있는 피험자만 ON
        os.environ.pop("ENABLE_VR", None)
        os.environ["VR_SUBJECT_IDS"] = "S001,S003"
        assert FeatureFlags(subject_id="S001").enable_vr is True
        assert FeatureFlags(subject_id="S002").enable_vr is False
        del os.environ["VR_SUBJECT_IDS"]

    def test_env_var_false_string(self):
        os.environ["ENABLE_VR"] = "false"
        flags = FeatureFlags(subject_id="S001")
        assert flags.enable_vr is False
        del os.environ["ENABLE_VR"]


# ── 사이클 2: 피험자 bucket 일관성 ───────────────────────────────────────────

class TestSubjectBucket:
    def test_same_id_same_bucket(self):
        # RED: 동일 ID는 항상 같은 bucket이어야 한다
        b1 = _subject_bucket("S042")
        b2 = _subject_bucket("S042")
        assert b1 == b2

    def test_bucket_range(self):
        # GREEN: bucket은 0~99 범위
        for sid in [f"S{i:03d}" for i in range(50)]:
            assert 0 <= _subject_bucket(sid) < 100

    def test_different_ids_distribute(self):
        # REFACTOR: 100명 중 일부는 서로 다른 bucket을 가져야 함
        buckets = {_subject_bucket(f"S{i:03d}") for i in range(100)}
        assert len(buckets) > 10  # 극단적 편향 없음


# ── 나머지 플래그 ─────────────────────────────────────────────────────────────

class TestOtherFlags:
    def setup_method(self):
        for k in ("ENABLE_ERP_LOGGING", "ENABLE_ROBOT_COLLISION",
                  "ENABLE_RAYCAST", "ERP_ROLLOUT_PCT"):
            os.environ.pop(k, None)

    def test_erp_logging_default_on(self):
        flags = FeatureFlags(subject_id="S001")
        assert flags.enable_erp_logging is True

    def test_robot_collision_default_on(self):
        flags = FeatureFlags(subject_id="S001")
        assert flags.enable_robot_collision is True

    def test_raycast_default_off(self):
        flags = FeatureFlags(subject_id="S001")
        assert flags.enable_raycast is False

    def test_erp_rollout_pct_zero(self):
        # ERP_ROLLOUT_PCT=0 이면 아무도 받지 못함
        os.environ["ENABLE_ERP_LOGGING"] = "true"
        os.environ["ERP_ROLLOUT_PCT"] = "0"
        flags = FeatureFlags(subject_id="S001")
        assert flags.enable_erp_logging is False

    def test_summary_keys(self):
        flags = FeatureFlags(subject_id="S001")
        keys = flags.summary().keys()
        for k in ("subject_id", "bucket", "enable_vr",
                  "enable_erp_logging", "enable_robot_collision", "enable_raycast"):
            assert k in keys
