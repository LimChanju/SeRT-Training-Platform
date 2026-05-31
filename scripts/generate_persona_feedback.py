"""
LLM 기반 10명 페르소나 피드백 생성
Claude API로 각기 다른 배경의 사용자가 SeRT VR 실험 시스템을 평가하는 피드백 생성
"""

import csv
import json
import os
import sys
import time

PERSONAS = [
    {"id": "P001", "name": "김정희", "role": "뇌졸중 재활 환자", "age": 63,
     "background": "6개월 전 뇌졸중 발병, 우측 팔 운동 장애, 기술 비숙련자"},
    {"id": "P002", "name": "이수민", "role": "신경과학 연구자", "age": 34,
     "background": "뇌-컴퓨터 인터페이스 전공 박사후연구원, Python/Isaac Sim 경험 있음"},
    {"id": "P003", "name": "박준호", "role": "재활 치료사", "age": 42,
     "background": "물리치료사 15년 경력, VR 재활 치료 관심, 기술 중간 수준"},
    {"id": "P004", "name": "최유진", "role": "VR 소프트웨어 엔지니어", "age": 28,
     "background": "메타버스 스타트업 재직, OpenXR/SteamVR 전문가"},
    {"id": "P005", "name": "강민서", "role": "작업치료사", "age": 36,
     "background": "손 재활 전문, 디지털 치료제에 관심 많음"},
    {"id": "P006", "name": "윤태양", "role": "파킨슨병 환자", "age": 71,
     "background": "파킨슨 3기, 손 떨림 증상, 스마트폰 사용 가능 수준"},
    {"id": "P007", "name": "정하은", "role": "의과대학 교수", "age": 52,
     "background": "신경재활의학과 교수, 임상시험 설계 전문가"},
    {"id": "P008", "name": "오성민", "role": "임상 심리사", "age": 44,
     "background": "인지재활 전문, 행동 관찰 및 측정 경험 풍부"},
    {"id": "P009", "name": "한지원", "role": "의료기기 개발자", "age": 31,
     "background": "의료기기 스타트업 CTO, FDA/CE 인증 경험"},
    {"id": "P010", "name": "송예린", "role": "병원 IT 관리자", "age": 47,
     "background": "대학병원 의료정보팀, 보안/인프라 담당"},
]

SYSTEM_PROMPT = """당신은 {role}입니다. 배경: {background}
SeRT VR Training Platform을 평가하고 있습니다. 이 시스템은:
- NVIDIA Isaac Sim 기반 VR 재활 훈련 시뮬레이션
- Feature Flag로 VR모드/ERP로깅/충돌감지/레이캐스트 토글
- A/B 테스트: Variant A(큐브 3개) vs Variant B(큐브 5개)
- 피험자별 실험 이벤트 CSV 자동 기록

당신의 페르소나 관점에서 솔직하고 구체적인 피드백을 주세요."""

USER_PROMPT = """다음 항목에 대해 평가해주세요 (각 50자 내외, 한국어):
1. 전반적 유용성 (1-5점 + 한줄 코멘트)
2. Feature Flag 시스템에 대한 의견
3. A/B 테스트 설계의 적절성
4. 가장 중요한 개선 제안 1가지
5. 이 시스템을 실제로 사용할 의향 (예/아니오 + 이유)

JSON 형식으로 응답하세요:
{{"score": 점수, "overall": "코멘트", "feature_flag": "의견", "ab_test": "의견", "improvement": "제안", "would_use": true/false, "use_reason": "이유"}}"""


def generate_with_api(persona: dict) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SYSTEM_PROMPT.format(**persona),
        messages=[{"role": "user", "content": USER_PROMPT}],
    )
    text = msg.content[0].text.strip()
    # JSON 블록 추출
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    return json.loads(text)


def generate_fallback(persona: dict) -> dict:
    """API 키 없을 때 페르소나 특성 기반 규칙으로 피드백 생성."""
    import hashlib, random
    rng = random.Random(int(hashlib.md5(persona["id"].encode()).hexdigest()[:8], 16))

    role = persona["role"]
    is_patient   = "환자" in role
    is_engineer  = "엔지니어" in role or "개발자" in role
    is_clinician = "치료사" in role or "교수" in role or "심리사" in role

    score = rng.choice([3, 4, 4, 5] if is_clinician else
                       [4, 4, 5, 5] if is_engineer else
                       [2, 3, 3, 4] if is_patient else [3, 4, 4])

    feedbacks = {
        "뇌졸중 재활 환자":  ("VR 화면이 어지러울 수 있어요", "너무 복잡해요", "큐브 수 차이가 실제로 느껴졌어요", "인터페이스를 더 단순하게", False, "기술이 어렵습니다"),
        "신경과학 연구자":   ("ERP 마커 연동이 인상적", "세밀한 제어가 가능해 연구에 적합", "해시 기반 할당이 통계적으로 타당", "EEG 동기화 기능 추가 필요", True, "연구 데이터 수집에 유용"),
        "재활 치료사":       ("환자 모니터링 측면에서 유용", "플래그 ON/OFF가 치료 계획 조정에 도움", "두 조건 비교가 치료 효과 측정에 적합", "환자 피로도 측정 지표 추가", True, "치료 프로토콜 개선에 활용 가능"),
        "VR 소프트웨어 엔지니어": ("OpenXR 호환성이 좋음", "환경변수 기반 토글은 DevOps 친화적", "SHA-256 해시 할당 견고함", "멀티플레이어 세션 지원 필요", True, "기술 스택이 현대적"),
        "작업치료사":        ("손 동작 추적 정밀도가 중요", "ERP 로깅 ON/OFF가 세션별 맞춤화에 유용", "큐브 수 변화가 난이도 조절로 적절", "손가락 개별 추적 기능 추가", True, "손 재활에 직접 활용 가능"),
        "파킨슨병 환자":     ("화면이 너무 빠르게 움직여요", "설정이 너무 많아 혼란스럽습니다", "차이를 크게 못 느꼈어요", "속도를 더 느리게 조절 가능하게", False, "사용하기 어렵습니다"),
        "의과대학 교수":     ("임상시험 설계 원칙에 부합", "플래그 기반 실험 조건 제어가 과학적", "피험자 무작위 배정이 통계적으로 적절", "윤리위원회 승인 절차 문서화 필요", True, "임상 연구 플랫폼으로 가능성 높음"),
        "임상 심리사":       ("행동 관찰 데이터 수집이 체계적", "실험 조건 전환이 심리 측정에 도움", "두 조건의 인지 부하 차이 측정 가능", "감정 상태 측정 도구 통합 필요", True, "인지재활 연구에 활용 가능"),
        "의료기기 개발자":   ("규제 요건 충족을 위한 로깅 체계 우수", "플래그로 CE 마킹 조건별 검증 가능", "A/B 데이터가 임상 증거로 활용 가능", "IEC 62304 준수 문서 필요", True, "의료기기 소프트웨어 개발에 참고 가능"),
        "병원 IT 관리자":    ("데이터 보안 및 익명화 처리 확인 필요", "환경변수 관리가 보안상 취약할 수 있음", "피험자 데이터 분리 저장 방식 적절", "PACS 연동 및 HL7 FHIR 지원 필요", False, "보안 인증 없이는 병원 도입 불가"),
    }

    f = feedbacks.get(role, ("유용한 시스템", "기능이 적절함", "테스트 설계 타당", "문서화 보완 필요", True, "활용 가능성 있음"))
    return {"score": score, "overall": f[0], "feature_flag": f[1], "ab_test": f[2],
            "improvement": f[3], "would_use": f[4], "use_reason": f[5]}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "persona_feedback")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "feedback_results.csv")
    json_path = os.path.join(out_dir, "feedback_results.json")

    use_api = bool(os.getenv("ANTHROPIC_API_KEY"))
    print(f"모드: {'Claude API' if use_api else '규칙 기반 (API 키 없음)'}\n")

    results = []
    fieldnames = ["id", "name", "role", "age", "score", "overall",
                  "feature_flag", "ab_test", "improvement", "would_use", "use_reason"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for persona in PERSONAS:
            print(f"[{persona['id']}] {persona['name']} ({persona['role']}) 평가 중...")
            try:
                fb = generate_with_api(persona) if use_api else generate_fallback(persona)
            except Exception as e:
                print(f"  → API 실패, 규칙 기반으로 대체: {e}")
                fb = generate_fallback(persona)

            row = {**{k: persona[k] for k in ("id", "name", "role", "age")}, **fb}
            writer.writerow(row)
            results.append(row)
            print(f"  → 점수: {fb['score']}/5 | 사용 의향: {'예' if fb['would_use'] else '아니오'}")
            if use_api:
                time.sleep(0.5)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg = sum(r["score"] for r in results) / len(results)
    would_use = sum(1 for r in results if r["would_use"])
    print(f"\n{'='*50}")
    print(f"평균 점수: {avg:.1f}/5.0")
    print(f"사용 의향: {would_use}/{len(results)}명")
    print(f"결과 저장: {csv_path}")


if __name__ == "__main__":
    main()
