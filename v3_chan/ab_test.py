"""
A/B 테스트 시스템 — 피험자 할당 일관성 + 이벤트 추적

Variant A (기본): 큐브 3개, 표준 배치, 속도 임계값 0.5 m/s
Variant B (실험): 큐브 5개, 확장 배치, 속도 임계값 0.3 m/s

피험자 ID의 해시를 기반으로 할당 — 동일 피험자는 항상 같은 variant
"""

import csv
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Variant 설정 ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VariantConfig:
    name: str
    num_cubes: int
    cube_positions: List[tuple]
    speed_threshold: float
    collision_dist: float
    description: str


VARIANT_A = VariantConfig(
    name="A",
    num_cubes=3,
    cube_positions=[(0.5, 0.0, 0.5), (0.7, 0.2, 0.5), (0.6, -0.2, 0.5)],
    speed_threshold=0.5,
    collision_dist=0.15,
    description="기본 배치 — 큐브 3개, 표준 간격",
)

VARIANT_B = VariantConfig(
    name="B",
    num_cubes=5,
    cube_positions=[
        (0.5, 0.0, 0.5), (0.7, 0.2, 0.5), (0.6, -0.2, 0.5),
        (0.8, 0.0, 0.5), (0.55, 0.3, 0.5),
    ],
    speed_threshold=0.3,
    collision_dist=0.12,
    description="확장 배치 — 큐브 5개, 좁은 간격",
)


# ── 할당 로직 ─────────────────────────────────────────────────────────────────

def assign_variant(subject_id: str) -> VariantConfig:
    """피험자 ID 해시 기반으로 variant 할당 (동일 ID → 항상 동일 결과)."""
    override = os.getenv("AB_VARIANT_OVERRIDE", "").strip().upper()
    if override == "A":
        return VARIANT_A
    if override == "B":
        return VARIANT_B

    b_pct = int(os.getenv("AB_VARIANT_B_PCT", "50"))
    digest = hashlib.sha256(subject_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return VARIANT_B if bucket < b_pct else VARIANT_A


# ── 이벤트 추적 ───────────────────────────────────────────────────────────────

@dataclass
class ABEventTracker:
    subject_id: str
    variant: VariantConfig
    log_path: str
    _events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        # 세션 시작 이벤트 자동 기록
        self.record("session_start", {
            "variant": self.variant.name,
            "num_cubes": self.variant.num_cubes,
            "speed_threshold": self.variant.speed_threshold,
        })

    def record(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "subject_id": self.subject_id,
            "variant": self.variant.name,
            "event": event_type,
            **(data or {}),
        }
        self._events.append(entry)
        self._append_to_csv(entry)

    def _append_to_csv(self, entry: Dict[str, Any]) -> None:
        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(entry)

    def record_task_complete(self, duration_sec: float, pick_count: int) -> None:
        self.record("task_complete", {
            "duration_sec": round(duration_sec, 3),
            "pick_count": pick_count,
        })

    def record_collision(self, object_name: str) -> None:
        self.record("collision", {"object": object_name})

    def record_erp_marker(self, marker_type: str, confidence: float) -> None:
        self.record("erp_marker", {
            "marker_type": marker_type,
            "confidence": round(confidence, 4),
        })

    def summary(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "variant": self.variant.name,
            "total_events": len(self._events),
            "event_types": list({e["event"] for e in self._events}),
        }


# ── 편의 함수 ─────────────────────────────────────────────────────────────────

def create_session(subject_id: str, log_dir: str = "data/ab_logs") -> ABEventTracker:
    """피험자 ID로 variant 할당 + 이벤트 추적기 반환."""
    variant = assign_variant(subject_id)
    log_path = os.path.join(log_dir, f"ab_{variant.name}_{subject_id}.csv")
    return ABEventTracker(
        subject_id=subject_id,
        variant=variant,
        log_path=log_path,
    )
