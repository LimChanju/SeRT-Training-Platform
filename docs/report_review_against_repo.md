# 최종결과보고서 Repo 대조 검토

검토 대상:

- 보고서 PDF: `/home/railabchan/Downloads/26-1최종결과보고서.pdf`
- 코드/결과 디렉토리: `/home/railabchan/isaac_vr_project`

전체적으로 보고서의 큰 흐름은 실제 진행 내용과 맞다. 다만 몇몇 문장은 실제 구현보다 강하게 읽힐 수 있으므로, 아래 항목은 수정하거나 보완하는 것이 좋다.

## 핵심 수정 권고

### 1. EEG/EDL이 RL reward에 실제 적용된 것처럼 보이는 문장 완화 필요

보고서에는 다음 취지의 문장이 있다.

> EDL EEGNet에서 오류 확률과 불확실성을 추출하고, 이를 동적 보상 함수 모델을 적용한 강화학습 에이전트에 적용함으로써 EEG를 안전 피드백으로 사용할 수 있을지 검증하였다.

현재 repo 기준으로는 RL 환경에 실제 EEGNet/EDL classifier가 직접 연결되어 있지는 않다. `docs/rl_trajectory_schema.md`, `docs/rl_progress.md`에는 EDL/EEG replay를 향후 대체 가능한 구조로 설명하고 있고, 실제 RL reward 경로에는 `v2/rl/pseudo_errp.py`의 pseudo-ErrP가 연결되어 있다.

수정 권장:

> EDL EEGNet을 통해 ErrP 예측 불확실성을 추정하는 방향을 실험하고, 이를 향후 로봇 보상 신뢰도 조절에 활용할 수 있는 구조로 설계하였다. 실제 RL 환경에서는 우선 pseudo-ErrP를 사용하여 안전 피드백 경로를 구현하고 검증하였다.

## 2. EDL EEGNet 구현 근거는 이 repo 안에서 확인되지 않음

보고서의 `EDL 헤드로 변경한 EEGNet 구현` 항목은 내용상 자연스럽지만, 현재 `/home/railabchan/isaac_vr_project` 안에서는 EEGNet/EDL 학습 코드나 ECE/AUC/F1 결과 파일이 확인되지 않는다. repo 안에는 EDL 관련 문서 placeholder만 있다.

수정 방향은 둘 중 하나다.

- EDL 실험 코드/결과가 다른 디렉토리에 있다면 보고서 제출 자료에 함께 첨부하거나 경로를 남긴다.
- 첨부가 어렵다면 “본 repo에서 구현된 RL 파이프라인과는 별도 실험으로 수행하였다”는 톤으로 분리한다.

문장 보완 예:

> 별도 EEGNet 실험에서 softmax head를 EDL head로 변경하여 ErrP 예측 불확실성 추정을 시도하였다.

## 3. “실제 EEG 피드백”과 “pseudo-ErrP”를 명확히 구분해야 함

보고서 전반에서 EEG/ErrP라는 표현이 자주 나오는데, 실제 Isaac RL 결과는 EEG 신호가 아니라 pseudo-ErrP 기반이다.

repo 근거:

- `v2/rl/pseudo_errp.py`: `near_human`, `human_robot_collision` 등을 pseudo-ErrP feedback으로 변환
- `v2/eval_results/ppo_v7_synthetic_hand_pseudo_errp_eval_v2.json`: synthetic hand 조건에서 pseudo-ErrP 기록
- `docs/pseudo_errp_report_plan.md`: 실제 EEG가 아니라 simulated feedback이라고 정리

수정 권장:

> 실제 EEG 기반 ErrP가 아닌, 손-로봇 근접 및 충돌 이벤트를 ErrP가 발생했을 상황으로 가정한 pseudo-ErrP를 사용하였다.

피해야 할 표현:

- “EEG 피드백을 RL에 적용하였다”
- “실제 ErrP를 보상에 반영하였다”

추천 표현:

- “pseudo-ErrP 피드백을 보상 및 로그에 반영하였다”
- “향후 실제 EEG/ErrP로 대체 가능한 인터페이스를 마련하였다”

## 4. Human replay 결과와 synthetic hand 결과를 섞지 않기

보고서에는 6월에 human replay를 구현하고 인간 개입 상황을 RL 환경에 통합했다고 되어 있다. 구현 자체는 repo에 있다.

repo 근거:

- `v2/rl/human_replay.py`
- `v2/train_rl.py`, `v2/evaluate_rollout_policy.py`의 `--human-replay-data` 옵션

다만 현재 최종 수치로 강하게 사용할 수 있는 결과 파일은 human replay가 아니라 synthetic hand disturbance 결과다.

결과 파일의 config:

- `v2/eval_results/ppo_v7_synthetic_hand_pseudo_errp_eval.json`
  - `human_replay_data`: `""`
  - `synthetic_human`: 사용
- `v2/eval_results/ppo_v7_synthetic_hand_pseudo_errp_eval_v2.json`
  - `human_replay_data`: `""`
  - `synthetic_human`: 사용

따라서 보고서에서는 다음처럼 분리하는 것이 안전하다.

> human replay 구조는 구현하였다. 최종 정량 평가는 replay 데이터가 아니라 synthetic hand disturbance를 사용하여 pseudo-ErrP 경로를 stress test하는 방식으로 수행하였다.

## 5. “완성”이라는 표현은 “구성/구현/기반 마련”으로 낮추는 것이 안전함

보고서에 “몰입형 HRC 환경을 완성했습니다”라는 표현이 있다. 코드상 VR 입력, collision logging, haptic UDP bridge는 구현되어 있지만, 완성이라는 표현은 평가자가 실제 사용자 실험/정량 검증까지 기대할 수 있다.

repo 근거:

- `v2/haptics_udp.py`: Isaac 쪽 UDP client
- `bhaptics_udp_bridge.py`: bHaptics glove bridge
- `v2/main.py`: collision/haptic pulse path
- `docs/pipeline.md`: haptics data flow

수정 권장:

> VR 입력과 햅틱 피드백 장갑을 Isaac Sim과 연동하여, 충돌 이벤트 발생 시 촉각 피드백을 전달할 수 있는 HRC 실험 환경을 구성하였다.

사용자가 실제 촉각을 느낀 경험을 쓰고 싶다면:

> 충돌 이벤트가 발생했을 때 사용자가 촉각 피드백을 느낄 수 있어, 화면 기반 시뮬레이션보다 몰입감 있는 실험이 가능해졌다.

## 6. “협동 조립 상황”은 실제 구현과 다름

목표 부분에 “협동 조립 상황 등을 가정한 환경”이라는 표현이 있다. 실제 구현은 Franka Panda 기반 tabletop pick-and-place다.

수정 권장:

> tabletop pick-and-place 작업을 기반으로 한 인간-로봇 협업 시뮬레이션 환경

또는:

> 협동 조립으로 확장 가능한 기초 물체 조작 작업으로서 pick-and-place 환경

## 7. PPO baseline 표현은 수치와 함께 보완하면 더 설득력 있음

보고서의 “BC 및 PPO 기반 로봇 팔 강화학습 Baseline 확보”는 실제 결과와 잘 맞다. 다만 “안정적으로 수행”만 쓰기보다 수치를 넣으면 훨씬 강해진다.

repo 근거:

- `v2/eval_results/ppo_pick_place_v7_residual_rewardv4_strict_best_require_release_rollout_eval.json`

핵심 수치:

- episodes: 50
- success_rate: 1.0
- grasp_rate: 1.0
- mean_final_cube_target_dist: 0.0299 m
- strict release condition: `require_release_for_success=True`

추천 문장:

> 최종 PPO v7 residual 정책은 strict release 조건의 50-seed 평가에서 success rate 1.00, grasp rate 1.00, 평균 최종 target distance 약 0.0299 m를 기록하였다.

주의:

이 수치는 “사람 손 개입이 없는 normal task 조건”이다.

## 8. synthetic hand / pseudo-ErrP 결과를 도전 성과에 추가하면 좋음

현재 보고서에는 synthetic hand disturbance 실험 결과가 직접적으로 거의 드러나지 않는다. 실제 repo 결과 중 보고서에서 가장 좋은 “HRI-aware” 근거이므로 보완하면 좋다.

repo 근거:

- `v2/eval_results/ppo_v7_synthetic_hand_pseudo_errp_eval.json`
- `v2/eval_results/ppo_v7_synthetic_hand_pseudo_errp_eval_v2.json`

정량 결과:

| 조건 | 성공률 | grasp rate | 평균 최종 거리 | pseudo-ErrP |
|---|---:|---:|---:|---:|
| normal PPO v7 | 1.00 | 1.00 | 0.0299 m | 없음 |
| mild synthetic hand | 0.70 | 0.70 | 0.1621 m | 0 |
| strong synthetic hand | 0.10 | 0.70 | 0.4742 m | mean errp count 31.4 |

strong synthetic hand 세부:

- near-human source events: 868
- human-robot-collision source events: 314
- mean episode ErrP feedback: 0.0445
- max episode ErrP feedback: 1.0

추천 문장:

> synthetic hand disturbance 조건에서 사람 손이 로봇 주변에 접근하거나 충돌하는 상황을 만들고, 이를 pseudo-ErrP로 기록하였다. 강한 disturbance 조건에서는 near-human 이벤트 868회, collision-level 이벤트 314회가 기록되었고, 평균 episode당 31.4회의 pseudo-ErrP label이 발생하였다. 이때 성공률은 10%로 감소하여, 현재 task-only 정책이 HRI stress condition에 취약함을 확인하였다.

## 9. “VLA에 믿을 수 있는 데이터를 제공” 표현은 약간 추상적임

향후 계획에 “EEG의 불확실성을 이용해 VLA에 믿을 수 있는 데이터를 제공”이라는 표현이 있다. 의미는 맞지만 조금 모호하다.

수정 권장:

> ErrP 예측의 신뢰도를 추정하고, 신뢰도에 따라 VLA 또는 로봇 정책 입력/보상에 반영할 안전 피드백의 가중치를 조절하는 방향으로 고도화할 예정이다.

## 10. 오탈자/표현 수정

다음 표현은 다듬는 것이 좋다.

- `내제화` → `내재화`
- `매커니즘` → `메커니즘`
- `multi-modal` → 한국어 본문에서는 `멀티모달`
- `Replay 하는` → `replay하는` 또는 `재생하는`
- `Trajectory` → 한국어 본문에서는 `trajectory` 또는 `궤적 데이터`로 통일
- `결정 하였습니다` → `결정하였습니다`
- `바탕으로 EEG의 불확실성을 이용해 VLA에 믿을 수 있는 데이터` → 의미가 모호하므로 위 9번처럼 수정

## 보고서에 추가하면 좋은 짧은 보완 문단

아래 문단을 `도전 성과` 또는 `앞으로의 계획` 앞에 넣으면 실제 repo 결과와 더 잘 맞는다.

> 추가적으로, 사람 손이 로봇 주변에 무작위로 개입하는 synthetic hand disturbance 실험을 수행하였다. 사람 손이 없는 normal pick-and-place 조건에서는 PPO v7 정책이 50개 seed에서 100% 성공률을 보였지만, 손 개입이 강해진 조건에서는 성공률이 10%까지 감소하였다. 이때 손-로봇 근접 이벤트와 충돌 수준 이벤트가 pseudo-ErrP로 기록되어, 현재 정책이 HRI stress condition에 취약하며 향후 pseudo-ErrP 기반 안전 강화학습이 필요함을 확인하였다.

## 최종 판단

현재 보고서는 큰 방향은 맞다. 특히 다음 내용은 실제 repo와 잘 맞는다.

- Isaac Sim + Franka Panda pick-and-place 환경 구축
- VR 입력과 손 위치 기반 HRI 환경 구성
- haptic UDP bridge 및 collision feedback path 구현
- BC/PPO baseline 확보
- pseudo-ErrP reward/logging 구조 구현
- human replay 구조 구현
- HRI 데이터 기록 경로 구축

수정이 필요한 핵심은 하나다.

> 실제 EEG/EDL이 RL reward에 이미 직접 적용된 것처럼 읽히는 표현을 줄이고, “EDL은 별도 불확실성 추정 실험, RL 쪽은 pseudo-ErrP 기반 safety feedback path를 구현 및 검증”으로 분리해서 쓰는 것이 가장 안전하다.
