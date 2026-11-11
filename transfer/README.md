# HRC 서버 이전 — 2026-11-11

이 브랜치는 `/home/railab/Desktop/Isaac_HRC`의 로컬 코드 snapshot이다. 원본 루트는 Git 저장소가 아니었고 두 내부 실험 저장소는 모두 LimChanju/SeRT-Training-Platform에 연결되어 있었다. 같은 원격 저장소의 별도 이전 브랜치로 보존한다.

## 코드 구성

| 경로 | 의미 |
|---|---|
| v3_chan/ | 실제 HRI 수집, Panda manipulation, 관측/기록/정책/피드백 코드 |
| v1/, v2/, v3_hoeng/ | 과거 장면·실험과 handover 구현 |
| SeRT-Training-Platform/ | BC/CBF handoff 실험의 로컬 변경 포함 snapshot |
| SeRT-Training-Platform-ac-selective-smoothing-v1/ | A/C selective smoothing 실험의 로컬 변경·미추적 코드 포함 snapshot |
| tests/, scripts/, api/, docs/, metrics/ | 검증 도구·API·문서·분석 코드 |

두 내부 실험 디렉터리는 gitlink/submodule이 아닌 일반 파일로 포함했다. 해당 .git와 데이터/정책은 올리지 않았다. 각 실험을 합친 새 알고리즘이 아니라 별도 경로의 코드 보존본이다. 실행할 디렉터리와 runtime contract를 구분한다.

원격 main과 기존 handoff 브랜치의 이력은 유지한다. 이 브랜치는 로컬 snapshot과 맞추기 위해 이전 main의 일부 생성 산출물을 tip에서 제외한다. force push나 기존 브랜치 덮어쓰기는 하지 않는다.

## 로컬에 보존한 별도 이전 자료

- trajectories/archive/release_assets의 HDF5·NPZ, 수집 CSV/JSON/영상과 logs/.speed_order_counter.
- policies의 PT/HDF5, 학습 이력 및 원래 runtime manifest·데이터 분할 manifest.
- 원본 ZIP·Git metadata·생성 결과·바이너리 asset과 별도 호스트 환경.
- 호스트 SteamVR/ALVR/OpenXR/방 설정과 복구한 Franka/Grid asset17파일.

GitHub에는 코드·기본 설정·전처리/검증 도구·라이선스를 포함한다. 바이너리 USD 장면은 별도 이전 자료에 남긴다. flags.env는 기존 feature 기본값 파일이며 실제 토큰/개인 키는 포함하지 않는다. 기존 문서의 LAN 주소나 머신 경로를 익명화한 것은 아니다. 수집 데이터와 학습 정책은 clone만으로 복원되지 않는다.

## 데이터 현황

손상된96바이트 HDF5 2개는 사용자의 이전 요청으로 삭제했다. 현재 HRC에는 읽히는 HDF5 60개가 있으며 3개는 episode가 비어 있다. 파일 분류는 원본39 / 보관12 / robot-only expert1 / 가공본8이다.

가공본24 episodes는 원본과 중복이다. 중복을 제외하면 사람 HRI125 episodes/289,332프레임과 robot expert100 episodes/62,651프레임이다. sessions17–24 가공본은52,224프레임 중51,924프레임이 training_valid_mask를 통과했다. 실제 사람이 착용한 측정은 머리·양손 및 sphere proxy 기반이며 어깨·팔꿈치 전관절 원시 skeleton이 아니다.

원본의 success 라벨과 grasp 신호, 시계·관측 버전을 구분해야 한다. HDF5의 ErrP 필드가 있다는 것만으로 실측 EEG 반응이 수집됐다고 해석하지 않는다. 상세 read-only 분석 자료는 기존 로컬 `vr_teleop/migration/hdf5_analysis_20261111`에 보존하며 원시 자료와 함께 별도 이전한다.

## 새 서버 환경

- 강화학습·Isaac Sim/Lab·MuJoCo는 Conda 계열(Miniconda 포함) 또는 Pixi로 구성한다. 최종 manager는 아직 정하지 않는다.
- VR/센서 트래킹과 실제 로봇 I/O는 각각 별도 독립 Python venv를 사용한다. Conda/Pixi Python이나 system-site-packages를 상속하지 않는다.
- 현재 HRC는 Isaac SimulationApp 안에서 VR 입력과 시뮬레이션을 함께 처리한다. 새 정책에 맞는 프로세스/IPC 분리는 이전 후 구현해야 하며 환경 생성만으로 완료되지 않는다.

현재4.5 런처·수집 코드는 byte 그대로 보존했다. 6.2의 Python/XR/API 이식은 이후 별도 작업이다. 기존 README의 과거 경로/원격 URL은 당시 기록으로 유지하며 현재 이전 기준은 이 안내를 따른다.

```bash
git clone --branch refactor/hrc-server-migration-20261111 https://github.com/LimChanju/SeRT-Training-Platform.git Isaac_HRC
```

새 서버 VR/장치·Isaac 실행 검증을 이 commit 작업에서 수행하지 않았다.
