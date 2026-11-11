"""
A/B 테스트 2주 운영 시뮬레이션 + 지표 분석 리포트 생성
Variant A(큐브 3개) vs Variant B(큐브 5개) 핵심 지표 비교
"""

import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2.ab_test import assign_variant, VARIANT_A, VARIANT_B

random.seed(2024)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ab_experiment")
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = datetime(2026, 5, 1)
NUM_SUBJECTS = 40       # 피험자 40명
SESSIONS_PER_SUBJECT = 3  # 1인당 3세션


def simulate_session(subject_id: str, variant, session_num: int, date: datetime) -> dict:
    rng = random.Random(hash(f"{subject_id}_{session_num}"))

    # Variant B는 큐브가 많아 태스크 시간 증가, 충돌 횟수도 증가
    is_b = variant.name == "B"
    base_duration = 45.0 if is_b else 35.0
    duration = rng.gauss(base_duration, 8.0)
    duration = max(15.0, duration)

    pick_count = variant.num_cubes + rng.randint(0, 2)
    collisions = rng.randint(1, 4) if is_b else rng.randint(0, 2)
    success = duration < (70 if is_b else 55)

    # 학습 효과: 세션이 거듭될수록 시간 단축
    duration *= (1.0 - 0.08 * (session_num - 1))

    return {
        "subject_id": subject_id,
        "variant": variant.name,
        "session_num": session_num,
        "date": date.strftime("%Y-%m-%d"),
        "duration_sec": round(duration, 2),
        "pick_count": pick_count,
        "collision_count": collisions,
        "task_success": success,
        "erp_p300_detected": rng.random() > 0.3,
    }


def compute_metrics(sessions: list, variant_name: str) -> dict:
    rows = [s for s in sessions if s["variant"] == variant_name]
    if not rows:
        return {}
    durations = [r["duration_sec"] for r in rows]
    collisions = [r["collision_count"] for r in rows]
    success_rate = sum(1 for r in rows if r["task_success"]) / len(rows)
    erp_rate = sum(1 for r in rows if r["erp_p300_detected"]) / len(rows)
    return {
        "variant": variant_name,
        "n_sessions": len(rows),
        "avg_duration_sec": round(sum(durations) / len(durations), 2),
        "min_duration_sec": round(min(durations), 2),
        "max_duration_sec": round(max(durations), 2),
        "avg_collisions": round(sum(collisions) / len(collisions), 2),
        "success_rate": round(success_rate, 3),
        "erp_detection_rate": round(erp_rate, 3),
    }


def main():
    subjects = [f"S{i:03d}" for i in range(1, NUM_SUBJECTS + 1)]
    all_sessions = []

    for i, sid in enumerate(subjects):
        variant = assign_variant(sid)
        for sess in range(1, SESSIONS_PER_SUBJECT + 1):
            days_offset = (i * 3 + sess * 2) % 14
            date = START_DATE + timedelta(days=days_offset)
            all_sessions.append(simulate_session(sid, variant, sess, date))

    # 세션 CSV 저장
    session_csv = os.path.join(OUT_DIR, "ab_sessions.csv")
    with open(session_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_sessions[0].keys())
        writer.writeheader()
        writer.writerows(all_sessions)

    # 지표 분석
    metrics_a = compute_metrics(all_sessions, "A")
    metrics_b = compute_metrics(all_sessions, "B")

    # 주차별 지표 (Week 1 vs Week 2)
    weekly = {}
    for week in (1, 2):
        start = START_DATE + timedelta(days=(week - 1) * 7)
        end = start + timedelta(days=7)
        week_sessions = [s for s in all_sessions
                         if start <= datetime.strptime(s["date"], "%Y-%m-%d") < end]
        weekly[f"week{week}"] = {
            "A": compute_metrics(week_sessions, "A"),
            "B": compute_metrics(week_sessions, "B"),
        }

    report = {
        "experiment": {
            "name": "SeRT VR A/B Test — Cube Count Variation",
            "period": "2026-05-01 ~ 2026-05-14 (2주)",
            "variant_a": VARIANT_A.description,
            "variant_b": VARIANT_B.description,
            "total_subjects": NUM_SUBJECTS,
            "total_sessions": len(all_sessions),
        },
        "overall": {"A": metrics_a, "B": metrics_b},
        "weekly": weekly,
        "conclusion": {
            "winner": "A" if metrics_a["avg_duration_sec"] < metrics_b["avg_duration_sec"] else "B",
            "key_finding": (
                f"Variant A 평균 완료 시간 {metrics_a['avg_duration_sec']}초 vs "
                f"Variant B {metrics_b['avg_duration_sec']}초. "
                f"성공률 A={metrics_a['success_rate']:.1%} / B={metrics_b['success_rate']:.1%}"
            ),
        },
    }

    report_path = os.path.join(OUT_DIR, "ab_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== A/B 테스트 2주 실험 결과 ===\n")
    for v, m in (("A", metrics_a), ("B", metrics_b)):
        print(f"Variant {v} ({m['n_sessions']}세션)")
        print(f"  평균 완료 시간: {m['avg_duration_sec']}초")
        print(f"  평균 충돌 횟수: {m['avg_collisions']}회")
        print(f"  태스크 성공률:  {m['success_rate']:.1%}")
        print(f"  ERP P300 검출:  {m['erp_detection_rate']:.1%}\n")
    print(f"우세 variant: {report['conclusion']['winner']}")
    print(f"리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
