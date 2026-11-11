"""샘플 실험 로그 생성 — 가상 피험자 12명으로 A/B 할당 + 이벤트 시뮬레이션"""
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v2.ab_test import assign_variant, ABEventTracker
from v2.feature_flags import FeatureFlags

random.seed(42)

SUBJECTS = [f"S{i:03d}" for i in range(1, 13)]
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ab_logs")
os.makedirs(LOG_DIR, exist_ok=True)

print(f"{'피험자':<8} {'Variant':<10} {'큐브수':<8} {'이벤트수':<10} {'로그 파일'}")
print("-" * 65)

for subject_id in SUBJECTS:
    variant = assign_variant(subject_id)
    flags = FeatureFlags(subject_id=subject_id)
    log_path = os.path.join(LOG_DIR, f"ab_{variant.name}_{subject_id}.csv")

    tracker = ABEventTracker(
        subject_id=subject_id,
        variant=variant,
        log_path=log_path,
    )

    # 플래그 상태 기록
    tracker.record("flags_snapshot", flags.summary())

    # 큐브 집기 이벤트 시뮬레이션
    num_picks = variant.num_cubes + random.randint(0, 2)
    for i in range(num_picks):
        tracker.record("pick_attempt", {
            "cube_index": i,
            "success": random.random() > 0.15,
            "reaction_time_ms": round(random.uniform(300, 900), 1),
        })

    # 충돌 이벤트 (0~2회)
    for _ in range(random.randint(0, 2)):
        tracker.record_collision(f"panda_link{random.randint(1, 7)}")

    # ERP 마커 (ENABLE_ERP_LOGGING 기본 ON)
    for marker in ["P300", "N200", "ERN"]:
        tracker.record_erp_marker(marker, confidence=round(random.uniform(0.6, 0.99), 4))

    # 태스크 완료
    duration = round(random.uniform(25.0, 90.0), 2)
    tracker.record_task_complete(duration_sec=duration, pick_count=num_picks)

    print(f"{subject_id:<8} {variant.name:<10} {variant.num_cubes:<8} "
          f"{len(tracker._events):<10} {os.path.basename(log_path)}")

print(f"\n로그 저장 위치: {os.path.abspath(LOG_DIR)}")
print(f"총 {len(SUBJECTS)}명 처리 완료")
