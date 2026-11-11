# Runbook

## 목적

이 문서는 인간-로봇 협업 데이터 수집 플랫폼의 실행, 헬스체크, 로그 확인,
관측성 확인, 장애 대응, 롤백 절차를 정리한다.

본 프로젝트는 VR/Isaac Sim 기반 데이터 수집 플랫폼이며, 핵심 기능은
사람-로봇 상호작용 데이터 수집, 안전 관련 이벤트 로깅, trajectory 생성,
BC/PPO 정책 학습 및 rollout 평가이다. 현재 `pseudo-ErrP`라는 이름의
구현은 안전 관련 피드백 라벨링 경로 중 하나로 사용된다.

## 주요 구성 요소

| 구성 요소 | 경로 | 설명 |
| --- | --- | --- |
| Isaac Sim runtime | `v2/main.py` | VR 입력, 로봇 제어, 충돌/근접 이벤트 로깅 |
| AI policy training | `v2/train_rl.py` | PPO 기반 정책 학습 |
| Rollout evaluation | `v2/evaluate_rollout_policy.py` | 학습 정책 평가 및 JSON/CSV 결과 저장 |
| Safety feedback labeling | `v2/rl/pseudo_errp.py` | 근접/충돌 이벤트를 안전 관련 피드백 신호로 변환 |
| Healthcheck API | `api/app.py` | `/health`, `/metrics` 제공 |
| Metrics dashboard | `metrics/out/dashboard.html` | DORA-style 메트릭 대시보드 |
| Tests | `tests/` | 단위 및 E2E 테스트 |

## 로컬 실행

### 1. 저장소 준비

```bash
git clone https://github.com/railabchan/isaac_vr_project.git
cd isaac_vr_project
```

### 2. Python 테스트 실행

```bash
python -m pytest
```

### 3. 헬스체크 API 실행

```bash
python api/app.py
```

다른 터미널에서 확인한다.

```bash
curl http://localhost:8080/health
curl http://localhost:8080/metrics
```

정상적인 `/health` 응답 예시는 다음과 같다.

```json
{
  "status": "ok",
  "version": "unknown",
  "env": "dev",
  "uptime_seconds": 10.0
}
```

### 4. Isaac Sim 데이터 수집 실행

Isaac Sim이 설치된 환경에서 실행한다.

```bash
isaac ~/isaac_vr_project/v2/main.py
```

또는 로컬 환경의 Isaac Sim launcher에 맞는 명령을 사용한다.

VR/hand tracking 데이터는 UDP 포트 `5555`를 통해 JSON 형태로 전달된다.

## 로그 및 산출물 확인

| 산출물 | 경로 | 설명 |
| --- | --- | --- |
| Safety/event markers | `v2/errp_markers.csv` | episode, collision, safety feedback 관련 이벤트 |
| Session samples | `v2/session_samples.csv` | 거리, 충돌 여부 등 frame/session sample |
| Rollout eval results | `v2/eval_results/*.json` | 정책 평가 요약 |
| Rollout eval CSV | `v2/eval_results/*.csv` | episode별 평가 결과 |
| Metrics JSON | `metrics/out/metrics.json` | 대시보드용 메트릭 |
| Metrics dashboard | `metrics/out/dashboard.html` | HTML 대시보드 |

## 관측성 확인

### 대시보드

브라우저에서 다음 파일을 열어 DORA-style 메트릭 대시보드를 확인한다.

```text
metrics/out/dashboard.html
```

대시보드에 포함되는 지표는 다음과 같다.

- Lead Time
- Deployment Frequency
- Change Failure Rate
- MTTR

### API 메트릭

헬스체크 API 실행 후 다음 엔드포인트로 메트릭을 확인한다.

```bash
curl http://localhost:8080/metrics
```

## CI/CD 확인

GitHub Actions에서 다음 workflow를 확인한다.

| Workflow | 목적 |
| --- | --- |
| `.github/workflows/python-lint.yml` | Python lint 및 matrix CI |
| `.github/workflows/test.yml` | 테스트 실행 |
| `.github/workflows/security-scan.yml` | pip-audit 기반 보안 스캔 |
| `.github/workflows/pages.yml` | metrics dashboard 배포 |
| `.github/workflows/docker.yml` | Docker image build 및 smoke test |
| `.github/workflows/cloud-run.yml` | Cloud Run 배포 및 healthcheck |

PR 제출 시 lint, test, security scan 중 하나 이상이 통과해야 merge 가능한
상태로 관리한다.

## 장애 대응

### 헬스체크 실패

1. `api/app.py` 프로세스가 실행 중인지 확인한다.
2. 포트 `8080`이 사용 가능한지 확인한다.
3. 환경 변수 `APP_VERSION`, `APP_ENV`, `PORT` 설정을 확인한다.
4. `/metrics`가 `204`를 반환하면 `metrics/out/metrics.json` 존재 여부를
   확인한다.

### 테스트 실패

1. 실패한 테스트 이름과 traceback을 확인한다.
2. 최근 변경 파일이 `v2/`, `api/`, `metrics/`, `tests/` 중 어디인지 확인한다.
3. 로컬에서 동일 명령으로 재현한다.

```bash
python -m pytest
```

4. Isaac Sim 의존 테스트와 일반 Python 테스트를 구분한다.

### 대시보드 갱신 실패

1. `metrics/session_events.csv`가 존재하는지 확인한다.
2. `metrics/compute_dora.py` 실행 결과를 확인한다.
3. `metrics/out/dashboard.html`, `metrics/out/metrics.json` 생성 여부를
   확인한다.
4. GitHub Pages workflow 로그를 확인한다.

### Rollout 평가 실패

1. checkpoint 경로가 올바른지 확인한다.
2. Isaac Sim 실행 환경과 GPU/CPU device 설정을 확인한다.
3. `v2/evaluate_rollout_policy.py`의 입력 인자와 output 경로를 확인한다.
4. 실패한 episode seed가 있으면 `v2/debug_rollout_seeds.py`로 재현한다.

## 롤백 계획

### 코드 롤백

문제가 발생한 변경이 main에 병합된 경우, 직전 안정 태그 또는 커밋으로
되돌린다.

```bash
git log --oneline
git revert <problem_commit_sha>
git push origin main
```

릴리스 기준으로 복구해야 하는 경우 `v1.0.0` 이상의 안정 태그를 기준으로
새 hotfix branch를 만든다.

```bash
git checkout -b hotfix/rollback-v1.0.0 v1.0.0
```

### Docker/GHCR 롤백

Docker 이미지 배포에 문제가 있으면 이전 stable tag 또는 release tag의
이미지를 다시 배포한다.

```bash
docker pull ghcr.io/limchanju/sert-training-platform:<previous_tag>
```

Cloud Run 또는 배포 환경에서는 해당 이전 이미지 태그를 선택해 재배포한다.

### Cloud Run 롤백

Cloud Run 배포 실패 또는 런타임 장애가 발생하면 Google Cloud Console에서
이전 정상 revision으로 트래픽을 되돌린다.

1. Cloud Run 서비스 페이지로 이동한다.
2. Revisions 탭에서 직전 정상 revision을 선택한다.
3. 해당 revision으로 traffic 100%를 이동한다.
4. `/health` 응답을 확인한다.

### GitHub Pages 대시보드 롤백

대시보드가 잘못 배포된 경우, 이전 정상 commit으로 `metrics/` 산출물을
복구한 뒤 Pages workflow를 다시 실행한다.

```bash
git revert <dashboard_problem_commit_sha>
git push origin main
```

## 제출 전 점검 목록

- `README.md`에서 14주차 최종 제출 문서로 이동할 수 있는가?
- `RETROSPECTIVE.md`가 루트에 존재하는가?
- `RUNBOOK.md`가 루트에 존재하는가?
- GitHub Actions의 lint/test/security scan이 통과했는가?
- `v1.0.0` 이상의 release tag가 GitHub 원격에 존재하는가?
- 데모 영상 링크가 README 또는 `docs/week14_final_submission.md`에 포함되어 있는가?
- `/health` 엔드포인트 설명이 README 또는 RUNBOOK에 포함되어 있는가?
- 로그, 메트릭, 대시보드 위치가 문서화되어 있는가?
