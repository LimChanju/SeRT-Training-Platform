# 14주차 최종 제출 정리

## 프로젝트 개요

이 저장소는 인간-로봇 협업 환경을 위한 데이터 수집 플랫폼을 정리한 것이다.

본 프로젝트는 VR/Isaac Sim 기반 환경에서 사람과 로봇의 상호작용 데이터를
수집하고, 이를 AI 정책 학습 및 평가에 활용할 수 있도록 구성한 연구용
오픈소스 플랫폼이다. VR/hand tracking 입력, Panda 로봇 pick-and-place
시뮬레이션, 사람-로봇 근접/충돌 등 안전 관련 이벤트 로깅, trajectory 수집,
BC/PPO rollout 평가를 하나의 파이프라인으로 연결한다.

## 제출 범위

이 프로젝트의 최종 산출물은 일반적인 소비자용 웹 애플리케이션이 아니라,
인간-로봇 협업 데이터를 수집하고 AI 정책 학습/evaluation에 연결하기 위한
연구 플랫폼이다.

주요 기능은 다음과 같다.

- VR/Isaac Sim 기반 인간-로봇 협업 시뮬레이션
- 사람 손, 로봇, 큐브, 충돌, 근접 이벤트 로깅
- HRI 안전 관련 데이터 수집 및 피드백 라벨링 경로
- BC/PPO 정책 학습을 위한 trajectory schema
- rollout 평가 결과 JSON/CSV 저장
- 헬스체크 및 메트릭 API
- CI/CD, 보안 스캔, 테스트, 관측성 대시보드

범위에서 제외하거나 실험적 상태로 남아 있는 기능은 다음과 같다.

- 사람이 VR/햅틱 장갑으로 큐브를 직접 grab/release하는 기능은 구현
  시도 흔적(`v2/vr_grab.py`)은 있으나, 최종 제출의 완료 기능으로 주장하지
  않는다.
- 최종 제출에서 사람 입력은 주로 손/머리 위치, 근접/충돌, 상호작용 이벤트
  수집 및 안전 관련 피드백 라벨링을 위한 데이터 소스로 설명한다.

## 제출 요건 체크리스트

| 제출 요건 | 저장소 내 근거 | 상태 |
| --- | --- | --- |
| 공개 GitHub 저장소 | `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE` | 완료 |
| 동작 가능한 AI 기능 | `v2/train_rl.py`, `v2/evaluate_rollout_policy.py`, 안전 피드백 라벨링 코드 `v2/rl/pseudo_errp.py` | 완료 |
| API 또는 UI | `api/app.py`의 `/health`, `/metrics` | 완료 |
| PR 게이트 CI/CD | `.github/workflows/python-lint.yml`, `.github/workflows/test.yml` | 완료 |
| 미니 eval | `v2/eval_results/*.json`, `v2/eval_results/*.csv` | 완료 |
| main 배포 | GitHub Pages 대시보드, Cloud Run workflow | 완료 |
| 헬스체크 | `api/app.py`의 `GET /health` | 완료 |
| 롤백 계획 | `RUNBOOK.md` | 완료 |
| 관측성 | `metrics/README.md`, `metrics/out/dashboard.html`, `metrics/out/metrics.json` | 완료 |
| 테스트 | `tests/`, `tests/e2e/` | 완료 |
| 보안 | `.github/dependabot.yml`, `.github/workflows/security-scan.yml` | 완료 |
| 문서 | `docs/pipeline.md`, `docs/rl_trajectory_schema.md`, `docs/rl_progress.md` | 완료 |
| 릴리스 태그 | `v1.0.1` latest release | 완료 |
| 회고문 | `RETROSPECTIVE.md` | 완료 |
| 3분 이내 영상 데모 | 아래 데모 영상 링크 | 완료 |

## 데모 영상

[데모 영상 유튜브 링크](https://youtu.be/BBAQPnZaHSM?si=I-tZ5E8gA7BzGc7a)

데모 영상은 Isaac Sim 환경에서 VR/햅틱 장갑을 사용하는 사람이 로봇 팔과
같은 작업 공간에 참여하고, 로봇이 pick-and-place 작업을 수행하는 모습을
보여준다. 사람과 로봇의 그리퍼를 포함한 헤드 부분이 충돌할 때, 햅틱 피드백 장갑으로 진동 피드백이 온다.
이 과정에서 사람 상태, 로봇 상태, 근접/충돌 이벤트,
안전 관련 상호작용 로그를 수집할 수 있다.



## 프로젝트 범위 및 제한사항

이 프로젝트의 큰 범위는 인간-로봇 협업 환경에서 안전과 관련된 상호작용
데이터를 수집하고, 이를 AI 정책 학습 및 평가에 연결하는 플랫폼을 만드는
것이다.

현재 저장소에서 `pseudo-ErrP`라는 이름은 그 안전 관련 데이터 중 하나의
구체적인 라벨링 방식이다. 실제 EEG 기반 ErrP classifier가 완성된 것이
아니라, 사람-로봇 근접 및 충돌 이벤트를 안전 피드백 신호처럼 기록하기
위한 simulated feedback 경로로 사용한다. 향후에는 같은 인터페이스를 실제
EEG, 온라인 인간 피드백, 또는 다른 HRI safety signal로 대체할 수 있다.

또한 `v2/vr_grab.py`와 `VRGrabManager`는 사람이 큐브를 직접 잡는 기능을
위한 실험적 구현 흔적이다. 최종 제출의 핵심 기능은 로봇 pick-and-place,
VR/hand tracking 기반 사람 상태 수집, 근접/충돌 이벤트 로깅, 안전 관련
피드백 라벨링, 그리고 수집 데이터의 정책 학습/evaluation 연계이다.
