# Contributing to Isaac VR Project

저희 오픈소스 프로젝트에 기여해 주셔서 감사합니다! 원활한 코드 리뷰와 협업을 위해 아래 가이드라인을 따라 주세요.

## 1. Fork & Clone
프로젝트에 직접 푸시 권한이 없는 경우, 리포지토리를 Fork 한 뒤 로컬에 Clone 하여 작업합니다.
```bash
# 1. 오른쪽 상단의 'Fork' 버튼을 클릭하여 본인 계정으로 리포지토리 복사
# 2. 로컬로 Clone
git clone https://github.com/YourUsername/isaac_vr_project.git
cd isaac_vr_project

# 3. 원본(Upstream) 저장소 추가
git remote add upstream https://github.com/railabchan/isaac_vr_project.git
```

## 2. 브랜치 전략 (Branch Strategy)
작업의 종류를 직관적으로 파악할 수 있도록 아래 접두사를 사용하여 브랜치를 생성해 주세요.
* `feat/`: 새로운 기능 추가 (예: `feat/vr-cylinder-proxies`)
* `fix/`: 버그 수정 (예: `fix/config-import-error`)
* `docs/`: 문서 작성 및 마크다운 추가 (예: `docs/update-readme`)
* `refactor/`: 기능 변화 없는 코드 구조적 변경 (예: `refactor/event-logger`)
* `test/`: 테스트 코드 추가 및 수정

```bash
git checkout -b feat/your-feature-name
```

## 3. 커밋 메시지 컨벤션 (Commit Message Convention)
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 형식을 본떠 커밋 메시지를 작성합니다:
* `feat: add new gripper proxy`
* `fix: resolve UDP port binding error`
* `docs: initialize OSS structure`
* `refactor: centralize logging configurations`

## 4. Pull Request (PR) 절차
1. 로컬에서 변경된 코드가 에러 없이 정상 구동(Isaac Sim 모의 등)되는지 확인합니다.
2. 당신의 Fork 저장소로 브랜치를 Push 합니다. `git push origin 브랜치이름`
3. 원본 저장소의 `main` 브랜치로 **Pull Request**를 생성합니다.
4. **작업 요약**, **주요 변경 사항**을 PR 템플릿에 맞추어 자세히 작성합니다. (스크린샷이나 에러 로그가 있다면 포함합니다.)
5. 리뷰어의 피드백(MUST, SHOULD 등) 사항을 수용하고, 반영된 내용을 재커밋하여 PR에 연동시킵니다.
6. 충분한 리뷰를 거친 후 관리자에 의해 성공적으로 Merge 됩니다.
* `test/`: 테스트 코드 추가 및 수정

```bash
git checkout -b feat/your-feature-name
```

## 3. 커밋 메시지 컨벤션 (Commit Message Convention)
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 형식을 본떠 커밋 메시지를 작성합니다:
* `feat: add new gripper proxy`
* `fix: resolve UDP port binding error`
* `docs: initialize OSS structure`
* `refactor: centralize logging configurations`

## 4. Pull Request (PR) 절차
1. 로컬에서 변경된 코드가 에러 없이 정상 구동(Isaac Sim 모의 등)되는지 확인합니다.
2. 당신의 Fork 저장소로 브랜치를 Push 합니다. `git push origin 브랜치이름`
3. 원본 저장소의 `main` 브랜치로 **Pull Request**를 생성합니다.
4. **작업 요약**, **주요 변경 사항**을 PR 템플릿에 맞추어 자세히 작성합니다. (스크린샷이나 에러 로그가 있다면 포함합니다.)
5. 리뷰어의 피드백(MUST, SHOULD 등) 사항을 수용하고, 반영된 내용을 재커밋하여 PR에 연동시킵니다.
6. 충분한 리뷰를 거친 후 관리자에 의해 성공적으로 Merge 됩니다.
