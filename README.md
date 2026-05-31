# Isaac VR Project

## Description
VR 사람-로봇 협업 데이터 수집 및 분석 파이프라인 프로젝트입니다. 
NVIDIA Isaac Sim 4.5 기반 환경에서 로봇 팔(Panda)과 VR 트래킹을 통해 실시간으로 제어되는 사용자(손/팔 실린더 프록시) 간의 물체 조작(Pick and Place) 태스크를 시뮬레이션합니다. 
버전 관리 및 로깅된 실험 데이터를 통해 수정된 DORA 4대 지표를 자동 산출하고 대시보드로 구성하는 기능도 포함되어 있습니다.

## Installation
1. [NVIDIA Isaac Sim 4.5](https://developer.nvidia.com/isaac-sim) 이상 버전을 설치합니다.
2. 레포지토리를 Clone 합니다:
   ```bash
   git clone https://github.com/railabchan/isaac_vr_project.git
   cd isaac_vr_project
   ```

## Usage
메인 시뮬레이션 환경은 Isaac Sim의 파이썬 인터프리터를 사용해 실행합니다:
```bash
isaac ~/isaac_vr_project/v2/main.py
```
- VR 트래킹 데이터는 UDP 포트 `5555`를 통해 JSON 형태로 송신해야 로봇 환경 내에서 반영됩니다.

### DORA 지표 대시보드
자동화된 실험 파이프라인 관리를 위해 변경된 DORA 지표를 수집하고 시각화합니다.
- **Lead Time**: 시나리오/코드 변경부터 유효 데이터 추출까지의 시간
- **Deployment Frequency**: 수집 세션 성공 횟수 (일 단위)
- **Change Failure Rate**: 세션 실패 비율 (실패 / 전체)
- **MTTR**: 장애 발생부터 복구 완료까지의 평균 시간

*대시보드 HTML 출력 결과는 GitHub Actions 파이프라인 성공 후 `metrics/out/dashboard.html`에서 확인할 수 있습니다.*

<!-- DORA_SCREENSHOT -->
![DORA Metrics Dashboard](metrics/out/dashboard.png)

### 이벤트 로그 스키마
`metrics/session_events.csv`

컬럼: `timestamp,event,details`

이벤트 예시:
- `code_change`
- `session_start`
- `session_success`
- `session_failed`
- `incident_start`
- `incident_end`

`metrics/session_events.csv`

컬럼: `timestamp,event,details`
## Feature Flags

실험 조건을 코드 변경 없이 환경 변수 또는 피험자 ID 기준으로 토글합니다.

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `ENABLE_VR` | OFF | VR 모드 활성화 (환경변수 또는 `VR_SUBJECT_IDS` 목록 기반) |
| `ENABLE_ERP_LOGGING` | ON | ERP 마커 로깅 (`ERP_ROLLOUT_PCT`로 비율 조절 가능) |
| `ENABLE_ROBOT_COLLISION` | ON | 로봇 충돌 감지 활성화 |
| `ENABLE_RAYCAST` | OFF | 레이캐스트 시각화 (실험적 기능) |

설정은 [`flags.env`](flags.env)에서 관리하며, 실행 전 아래 명령으로 적용합니다:

```bash
export $(cat flags.env | xargs)
isaac ~/isaac_vr_project/v2/main.py
```

플래그는 [`v2/feature_flags.py`](v2/feature_flags.py)에 구현되어 있습니다.

## A/B 테스트

피험자 ID의 SHA-256 해시를 기반으로 variant를 결정합니다. 동일한 피험자 ID는 항상 같은 variant에 배정됩니다.

| | Variant A | Variant B |
|---|-----------|-----------|
| 큐브 수 | 3개 | 5개 |
| 배치 | 표준 간격 | 확장/좁은 간격 |
| 속도 임계값 | 0.5 m/s | 0.3 m/s |

이벤트 추적 로그는 `data/ab_logs/` 에 피험자별 CSV로 저장됩니다. 샘플 로그 생성:

```bash
python scripts/generate_sample_logs.py
```

A/B 테스트 구현은 [`v2/ab_test.py`](v2/ab_test.py)를 참고하세요.

## 테스트

### 단위 테스트 (pytest)

핵심 모듈에 대해 TDD(Red-Green-Refactor) 사이클로 작성된 단위 테스트입니다.

| TDD 사이클 | 대상 | 검증 내용 |
|-----------|------|-----------|
| 1 | `ENABLE_VR` 플래그 | 기본 OFF, 환경변수 ON, 피험자 목록 토글 |
| 2 | 피험자 bucket 해시 | 동일 ID → 항상 같은 그룹 배정 |
| 3 | A/B variant 할당 | 동일 피험자 → 항상 같은 variant |
| 4 | 이벤트 CSV 기록 | `record()` 호출 시 파일에 행 저장 |
| 5 | EventLogger | 충돌/드롭/스택 이벤트 기록 검증 |

```bash
pip install pytest pytest-cov numpy
python -m pytest tests/ --ignore=tests/e2e -v
```

현재 커버리지: **95%** (CI 최소 기준 80%)

### E2E 테스트 (Playwright)

DORA 대시보드 HTML을 실제 브라우저로 열어 시나리오를 검증합니다.

- 페이지 로드 및 제목 확인
- Lead Time, MTTR 등 지표 4개 텍스트 존재 확인
- 차트 요소(SVG/canvas) 렌더링 확인
- JavaScript 에러 없음 확인

테스트 실패 시 스크린샷이 `test-artifacts/screenshots/`에 자동 저장됩니다.

```bash
pip install pytest-playwright
playwright install chromium
python metrics/compute_dora.py --input metrics/session_events.csv --out metrics/out
python -m pytest tests/e2e/ -v
```

### CI

push 또는 PR 시 GitHub Actions에서 자동 실행됩니다.

1. 단위 테스트 → 커버리지 80% 미달 시 CI 실패
2. E2E 테스트 → 실패 시 스크린샷 아티팩트 업로드

## 사용자 피드백 및 실험 결과

### LLM 기반 10명 페르소나 피드백

뇌졸중 환자, 신경과학 연구자, 재활치료사, VR 엔지니어 등 10개 페르소나를 Claude API로 생성하여 시스템을 평가했습니다.

| 지표 | 결과 |
|------|------|
| 평균 만족도 | **4.2 / 5.0** |
| 사용 의향 | **7 / 10명 (70%)** |
| 주요 긍정 | ERP 마커 연동, 피험자별 조건 제어, 데이터 자동 기록 |
| 주요 개선 요청 | UI 단순화, EEG 동기화, 보안 인증 |

피드백 데이터: [`data/persona_feedback/`](data/persona_feedback/)

피드백 재생성:
```bash
ANTHROPIC_API_KEY=your_key python scripts/generate_persona_feedback.py
```

### A/B 테스트 2주 운영 결과

피험자 40명, 120세션을 대상으로 2주간 실험을 운영했습니다.

| 지표 | Variant A (큐브 3개) | Variant B (큐브 5개) |
|------|---------------------|---------------------|
| 평균 완료 시간 | **32.54초** | 42.21초 |
| 평균 충돌 횟수 | **1.0회** | 2.52회 |
| 태스크 성공률 | 100% | 100% |
| ERP P300 검출률 | 63.9% | 64.6% |

실험 데이터: [`data/ab_experiment/`](data/ab_experiment/)

### Pivot or Persevere 결정

**결정: Persevere (현행 유지 + 개선)**

Variant A가 완료 시간 23% 단축, 충돌 60% 감소로 우세하여 기본 실험 조건으로 확정했습니다. 상세 결정 근거 및 다음 스프린트 계획: [`docs/pivot-decision.md`](docs/pivot-decision.md)

### 주간 리포팅 자동화

매주 월요일 GitHub Actions가 자동으로 실험 지표를 수집하고 GitHub Issue로 리포트를 생성합니다.

## License
이 프로젝트는 [MIT License](LICENSE)에 따라 배포됩니다.
