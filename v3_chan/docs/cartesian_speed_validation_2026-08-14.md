# Cartesian Speed Validation - 2026-08-14

## 실행 조건

- Isaac Sim 4.5 headless, OpenXR/hand tracking/haptics 비활성
- session speed order: `slow,medium,fast`
- 각 profile에서 3-cube pick-and-place 1 episode
- 측정 기준: Lula `right_gripper` frame, simulation timestamp finite difference
- reset/첫 sample/비정상 `dt` 제외
- 실제 속도 허용오차: `max(0.02 m/s, active_limit * 0.05)`
- FK command 허용오차: `0.005 m/s`

테스트 파일은 `/tmp/v3_cartesian_speed_smoke.hdf5`에 생성했으며 production 데이터가 아니다.

## 결과 요약

| Profile | Phase | Limit (m/s) | Mean | P95 | Max | Valid samples | Violations |
|---|---|---:|---:|---:|---:|---:|---:|
| slow | transport | 0.30 | 0.1408 | 0.3027 | 0.3045 | 855 | 0 |
| slow | approach/place | 0.10 | 0.0914 | 0.1068 | 0.1100 | 1174 | 0 |
| medium | transport | 0.40 | 0.1424 | 0.3232 | 0.3752 | 847 | 0 |
| medium | approach/place | 0.15 | 0.1150 | 0.1552 | 0.1604 | 937 | 0 |
| fast | transport | 0.50 | 0.1423 | 0.3232 | 0.3744 | 849 | 0 |
| fast | approach/place | 0.20 | 0.1257 | 0.2042 | 0.2099 | 864 | 0 |

모든 episode가 3개 cube를 정상 배치해 성공률은 `3/3 (100%)`였다. Transport 상한은 목표 속도가 아니라 hard cap이므로 RMPFlow의 경로와 articulation dynamics에 따라 medium/fast가 항상 상한까지 가속하지는 않는다.

## 10-Event 측정

각 셀은 `mean / p95 / max`이며 단위는 m/s이다. Event 2/3/7은 limiter 비활성 단계이다.

| Event | Mode | slow | medium | fast |
|---:|---|---:|---:|---:|
| 0 | approach/place | 0.0945 / 0.1082 / 0.1100 | 0.1251 / 0.1585 / 0.1604 | 0.1433 / 0.2062 / 0.2099 |
| 1 | approach/place | 0.0899 / 0.1054 / 0.1062 | 0.1075 / 0.1540 / 0.1555 | 0.1133 / 0.2012 / 0.2029 |
| 2 | settling | 0.0086 / 0.0211 / 0.0213 | 0.0086 / 0.0212 / 0.0212 | 0.0086 / 0.0214 / 0.0217 |
| 3 | gripper close | 0.0003 / 0.0015 / 0.0018 | 0.0003 / 0.0016 / 0.0018 | 0.0003 / 0.0016 / 0.0018 |
| 4 | transport | 0.1214 / 0.2262 / 0.2399 | 0.1214 / 0.2260 / 0.2399 | 0.1211 / 0.2264 / 0.2400 |
| 5 | transport | 0.1716 / 0.3036 / 0.3045 | 0.1758 / 0.3555 / 0.3752 | 0.1755 / 0.3552 / 0.3744 |
| 6 | approach/place | 0.0863 / 0.1051 / 0.1056 | 0.1049 / 0.1540 / 0.1548 | 0.1104 / 0.2015 / 0.2023 |
| 7 | gripper open | 0.0213 / 0.0219 / 0.0219 | 0.0211 / 0.0213 / 0.0213 | 0.0221 / 0.0238 / 0.0241 |
| 8 | transport | 0.1038 / 0.2207 / 0.2397 | 0.1039 / 0.2207 / 0.2397 | 0.1039 / 0.2209 / 0.2396 |
| 9 | transport | 0.0217 / 0.0222 / 0.0223 | 0.0218 / 0.0224 / 0.0225 | 0.0218 / 0.0224 / 0.0225 |

## Distance Gate

Headless smoke에는 유효한 사람 손이 없어 full-gate 실측 sample은 0개였다. 따라서 다음 항목만 자동 검증했다.

- `surface_gap=0.13/0.09/0.05 m`에서 gate가 각각 `0/0.5/1`
- 0.13~0.05 m 구간에서 blended limit의 단조·연속 선형 감소
- full gate에서 slow/medium/fast FK command가 각각 `0.05/0.10/0.15 m/s` 이하

실제 VR 손을 사용한 full-gate simulation-time EE 속도 검증은 production 수집 전 별도 smoke session에서 확인해야 한다.

## 실행 결과

- 관련 및 비-E2E 전체 테스트: `134 passed`
- HDF5 validator (`--non-production`): passed
- 3-cube 성공: `3/3`
- Playwright dashboard E2E: Isaac Python에 `playwright`가 없어 collection 단계에서 미실행

## 안전 고지

이 결과는 Isaac Sim 데이터 수집 환경에만 적용된다. 실제 Franka 하드웨어 배포에는 별도의 위험성 평가, 안전정격 속도 제한, 정지거리 검증, 비상정지 검증 및 제조사 안전 기능 구성이 필요하다.
