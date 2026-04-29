# DORA 지표 (실험 파이프라인)

VR 사람-로봇 협업 데이터 수집 파이프라인에 맞게 DORA 지표를 재정의하고 시각화합니다.

## 시각화하는 지표
- **Lead Time**: 에피소드 시작(`episode_start`)부터 종료(`episode_end`)까지의 시간
- **Deployment Frequency**: 유효 데이터 수집 세션 성공 횟수(일 단위)
- **Change Failure Rate**: 세션 실패 비율 (실패 / 전체)
- **MTTR**: 장애 발생부터 복구 완료까지의 평균 시간

## 이벤트 로그 스키마
`metrics/session_events.csv`

컬럼: `timestamp,event,details`

이벤트 예시:
- `code_change`
- `session_start`
- `session_success`
- `session_failed`
- `incident_start`
- `incident_end`
- `episode_start`
- `episode_end`

## 대시보드
GitHub Actions로 생성:
- HTML: `metrics/out/dashboard.html`

HTML을 열면 Chart.js로 시각화된 대시보드를 확인할 수 있습니다.

<!-- DORA_SCREENSHOT -->
