"""
Feature Flag 시스템 — 환경 변수 + 피험자 ID 기반 토글 제어

사용법:
    flags = FeatureFlags(subject_id="S001")
    if flags.enable_vr:
        ...
    if flags.is_enabled_for("ENABLE_RAYCAST"):
        ...
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional


# ── 피험자 ID 기반 토글: subject_id의 해시값 % 100 으로 그룹 결정 ──────────────
def _subject_bucket(subject_id: str) -> int:
    digest = hashlib.md5(subject_id.encode()).hexdigest()
    return int(digest[:8], 16) % 100


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def _env_list(key: str) -> set:
    """쉼표로 구분된 피험자 ID 목록을 환경 변수에서 읽음."""
    raw = os.getenv(key, "")
    return {s.strip() for s in raw.split(",") if s.strip()}


@dataclass
class FeatureFlags:
    subject_id: str = "anonymous"
    _bucket: int = field(init=False)

    def __post_init__(self):
        self._bucket = _subject_bucket(self.subject_id)

    # ── Flag 1: VR 모드 활성화 ───────────────────────────────────────────────
    # 환경 변수 ENABLE_VR=true 또는 허용 피험자 목록(VR_SUBJECT_IDS)에 포함 시 ON
    @property
    def enable_vr(self) -> bool:
        if _env_bool("ENABLE_VR"):
            return True
        allowed = _env_list("VR_SUBJECT_IDS")
        if allowed:
            return self.subject_id in allowed
        return False

    # ── Flag 2: ERP 마커 로깅 활성화 ────────────────────────────────────────
    # 환경 변수 ENABLE_ERP_LOGGING=true 또는 bucket < ERP_ROLLOUT_PCT(기본 100)
    @property
    def enable_erp_logging(self) -> bool:
        if _env_bool("ENABLE_ERP_LOGGING", default=True):
            pct = int(os.getenv("ERP_ROLLOUT_PCT", "100"))
            return self._bucket < pct
        return False

    # ── Flag 3: 로봇 충돌 감지 활성화 ───────────────────────────────────────
    # 환경 변수 ENABLE_ROBOT_COLLISION=true (기본 ON)
    @property
    def enable_robot_collision(self) -> bool:
        return _env_bool("ENABLE_ROBOT_COLLISION", default=True)

    # ── Flag 4: 레이캐스트 시각화 활성화 ────────────────────────────────────
    # 환경 변수 ENABLE_RAYCAST=true (기본 OFF — 실험적 기능)
    @property
    def enable_raycast(self) -> bool:
        return _env_bool("ENABLE_RAYCAST", default=False)

    # ── 범용 조회 ────────────────────────────────────────────────────────────
    def is_enabled_for(self, flag_name: str) -> bool:
        prop = flag_name.lower().replace("enable_", "enable_")
        return getattr(self, prop, False)

    def summary(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "bucket": self._bucket,
            "enable_vr": self.enable_vr,
            "enable_erp_logging": self.enable_erp_logging,
            "enable_robot_collision": self.enable_robot_collision,
            "enable_raycast": self.enable_raycast,
        }
