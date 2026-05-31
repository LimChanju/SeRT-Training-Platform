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

## License
이 프로젝트는 [MIT License](LICENSE)에 따라 배포됩니다.
