# A/C Selective-Smoothing VR Explicit-Feedback 수집환경

이 문서는 Frozen BC와 동일한 physical-safety/Recovery 계약을 사용하는
`A_reactive`와 `C_smooth`를 비교하기 위한 전용 VR 수집환경의 운영 문서다.
schema는 `bc_cbf_recovery_ac_explicit_feedback_v2`, protocol은
`ac_selective_smoothing_free_single_crossing_pilot_v3`다. 이 수집기는 사람의
feedback으로 현재 또는 이후 trial의 condition을 바꾸지 않으며 critic 학습,
BC fine-tuning, CBF parameter adaptation, ErrP, 햅틱을 실행하지 않는다.

기존 3-cube 수집환경은 보존한다. 이 환경에서 한 trial은 BC-feasible
single-pick 1회, participant가 편한 손으로 수행하는 완전한 free crossing 1회,
A 또는 C response 1회, HMD 안의 mandatory evaluation 1회다.

## 고정한 결정

| 항목 | 고정값 |
|---|---|
| Frozen BC | `v3_chan/policies/bc_pick_place_v2_release_settle.pt` |
| checkpoint SHA-256 | `d0fe9f9dc48a6049b77c8e1eab6de0905197207bb57ff858e0d7d332172b5230` |
| A | `A_reactive`, `joint_nominal`, `lambda_s=0` |
| C | `C_smooth`, `smooth_intervention`, `lambda_s=4.0` |
| 공통 control | 동일 CBF constraints/solver/fail-closed, strict task v5, Fixed State-aware Recovery v3 |
| 기본 설계 | practice 4 + evaluated 16 (`minimal_pilot`) |
| anchor 설계 | practice 4 + evaluated 20 (`pilot_with_anchors`, anchor-repeat 4) |
| 햅틱 | 항상 OFF; runner가 `BHAPTICS_ENABLED=0`, `HRI_HAPTIC_CONDITION=off`로 강제 |
| condition 변경 | trial `RESET`에서만 사전 배정값 적용; feedback 기반 변경 금지 |
| UI | condition을 숨긴 중앙 `XRSceneView` text panel |
| participant 시작 | BC-only 사전 시연 없이 첫 동작부터 Frozen BC + assigned CBF |
| 고정 layout | seed `10011`, layout ID `8b06c8502c07d1b3` |
| crossing detector | participant에게 보이지 않는 broad bilateral side gate와 interaction band |
| crossing 표시 | 중앙 panel의 `START`만 사용; corridor·green/red/yellow guide는 표시하지 않음 |

16개 core trial의 기존 task phase 4 × severity 2 × condition 2 schedule과 A/C
균형은 보존한다. 다만 v3에서 scheduled hand·direction·speed·gap은 participant
지시나 통과 판정이 아니다(`participant_prescribed=false`). participant/session
가명과 seed는 condition 순서를 사전에 고정할 뿐이며, 실제 선택 손·방향·속도와
minimum gap은 관측값으로 자동 기록한다.

Safety-response detector는 다음 phase를 구분한다.

```text
PRE_RESPONSE -> CBF_ACTIVE -> SMOOTH_TAIL -> RECOVERY_ACTIVE
             -> BC_RESUMED -> STABLE_TASK_RESUMPTION
```

onset은 active constraint와 `intervention_norm >= 0.05 rad/s`가 3 control
frame 연속일 때 confirm한다. stable resumption은 CBF/tail/Recovery가 모두
inactive이고 BC가 재개되었으며 `intervention_norm <= 0.01 rad/s`인 상태가
0.5초 유지될 때다. 설문은 이 stable 판정 뒤에만 표시한다.

Trial lifecycle은 다음 17개 상태를 한 방향으로만 통과한다.

```text
RESET -> LOAD_BC_FEASIBLE_SCENARIO -> START_FROZEN_BC
-> WAIT_FOR_TARGET_TASK_PHASE -> SHOW_FREE_CROSSING_START
-> DETECT_SINGLE_FREE_CROSSING -> RUN_ASSIGNED_RESPONSE_A_OR_C
-> TRACK_SAFETY_RESPONSE_EPISODE -> DETECT_STABLE_TASK_RESUMPTION
-> SHOW_RETURN_HAND_TO_NEUTRAL_CUE -> PAUSE_NOMINAL_TASK_PROGRESSION
-> SHOW_MANDATORY_FEEDBACK_UI -> SAVE_FEEDBACK
-> RESUME_TASK_TO_COMPLETION_OR_TERMINAL -> SAVE_TRIAL
-> VALIDATE_TRIAL -> NEXT_TRIAL_OR_END_SESSION
```

내부 crossing detector를 latch한 뒤 `START` 전 cue-lead와 설문 중에는 nominal task
progression을 safe hold로 멈추며 physics, CBF, emergency abort는 계속 켜 둔다.
`START`부터 Frozen BC를 재개하고, 설문 응답 후에도 task success/failure까지 진행한다.
설문 응답을 저장한 직후 terminal 설문 page를 닫고, HMD에는 남은
pick-and-place가 진행 중이며 양손을 로봇에서 15 cm 이상 떨어진 가슴 앞에 유지하고
다음 `START` 전까지 다시 넣지 말라는 안내를 표시한다. XR 부하로 task가 느리게
보이더라도 이 구간은 현재 trial의 자동 완료 구간이며 다음 trial 준비 구간이 아니다.

## VR controller 입력

실시간 marker는 crossing하는 손과 방향에 관계없이 오른쪽 controller만 사용한다.

| 버튼 | 의미 |
|---|---|
| 오른손 A | 이상한 궤적 (`realtime_behavior_anomaly`) |
| 오른손 B | 위험한 궤적 (`realtime_safety_concern`) |

설문 navigation은 `X=이전`, `Y=다음`, `A=선택`, `B=확인`, `X+Y=뒤로`다.
`A+B`를 2초 유지하면 emergency abort다. 실제 OpenXR input/gesture 이름은
startup에서 cached controller device로 discovery하므로, 첫 HMD smoke에서
Quest/ALVR runtime이 이 logical mapping을 제공하는지 반드시 확인한다.
수집기는 실제 ALVR/SteamVR controller binding을 유지하기 위해 Kit의 `vr`
tools layout은 유지하되, 기본 tool action map 대신 repo-local tool-free map을
선택한다. `tools/layout`을 빈 값으로 두면 실제 OpenXR 세션에서 `no actionmap`이
선택되어 controller pose/button wrapper가 무효화될 수 있으므로 사용하지 않는다.
이 설정과 custom map 등록은 XR extension보다 먼저 적용되며
수집 세션에서 persistent XR 사용자 설정을 격리한다. HMD view는 기존
느림/빠름/중간 pick-and-place 수집기와 동일하게
`active camera`에서 시작해 최초 physical HMD sample이 잡히면
`XRCore.schedule_set_camera()`로 avatar eye `[1, 0, 1.5]`와 테이블 중심을
정렬한다. 이 경로가 실패할 때만 `xrAnchor` 직접 보정을 fallback으로 사용하며,
user-height 자동 보정은 끈다. Tracking-clear 확인의 마지막 프레임 뒤에는 추가
render/sleep을 하지 않아 정상 샘플을 100 ms가 지난 cached pose로 오인하지 않는다.
제어 step에서는 controller input/e-stop 확인 뒤 live-pose preflight를 수행한다. 정상
XR source이지만 cached policy observation에 apply 경로용 30 ms 여유가 남지 않았으면
verified hold 상태에서 observation을 정확히 한 번 재생성한다. 100 ms 한계 자체는
늘리지 않으며 `pose_invalid`, source stale/switch, 추적 상태 오류는 재시도 없이
fail-closed 처리한다.
각 trial control step은 actuation 전과 feedback 처리 전에 양 controller와
X/Y/A/B의 known-state를 별도로 검사한다. pose tracking이 유효하더라도 button
channel이 끊기면 verified safe hold 후 `controller_input_unavailable...`로
abort하며 `.partial`에 이유를 남긴다. 이 장비 오류를 participant의
timeout/no-response와 합치지 않는다.

실시간 marker 두 종류는 독립 event다. 버튼을 누르지 않은 것은
`acceptable_as_is`가 아니다. Q1의 `uncertain`, timeout, no-response도
negative label로 바꾸지 않는다. 저장되는 `response_disposition`은 확정 응답
`answered`, 완료된 uncertain 응답 `uncertain_abstain`, timeout/no-response
`missing_abstain`으로 구분한다. Q1이
`needs_modification`이면 정해진 reason을 하나 이상 골라야 저장할 수 있다.
`query_answers`는 각 prompt cycle을 `answer_status`로 구분한다. 완료된 설문은
Q1–Q6 6개(필요할 때 reason까지 7개)가 모두 `confirmed`여야 한다. timeout이나
no-response는 이미 확정한 prefix 뒤에 현재 보이던 prompt를 `unconfirmed`
1행으로 남기고, 값은 JSON `null`, confirmed clock과 latency는 명시적 sentinel로
저장한다. 따라서 미확정 선택을 label로 승격하지 않으면서 prompt/first-input
audit은 보존된다.

## 기록과 validator

기본 출력은
`v3_chan/ac_selective_smoothing_feedback_data/<session>_<time>.hdf5`다. schedule,
trial/encounter/query/realtime-marker 식별자와 세 clock(simulation, monotonic,
Unix), raw hand tracking, robot/object/task state, BC raw 5D action, nominal·filtered·
applied command, A/C objective와 lambda, CBF correction memory/tail, Recovery,
gap/TTC/collision, response phase를 step별로 기록한다. trial summary에는 실제로
선택한 손과 방향, 실제 crossing 속도·minimum gap·완전 횡단 횟수, phase별
duration/jerk, intervention integral/TV, task outcome, Q1–Q6, reason과 입력
timestamp를 기록한다. `decision_context`에는 pre-solve 정보만 허용하고 future
outcome은 금지한다.
251개 synchronized control column은 30-row contiguous batch로 기록한다. realtime
marker, query, trial 종료 및 관리된 abort 경계에서는 남은 row를 즉시 write/flush하므로
정상적인 fail-closed 종료의 `.partial`에는 마지막 control row까지 보존된다.
`delta_path_m`과 `delta_completion_time_s`는 각각 direct-live active-task 측정값에서
같은 고정 layout의 A10 uninterrupted BC-only baseline을 뺀 값이다. summary에는
actual/baseline 원값과 comparison semantics를 함께 저장하고 validator가 delta를
재계산한다. cue-lead/query/pause hold는 active-task path/time에서 제외되며 validator가
raw EE velocity와 step state에서 actual 값을 다시 계산한다. scheduled
hand·direction·speed·gap은 v3 participant 행동의 ground truth로 사용하지 않는다.

완료 파일은 구조·provenance fail-closed validator를 통과해야만 유효하다.
구조적으로 유효하지만 crossing 미완료·두 번째 완전 횡단·onset 미검출 같은
protocol 사유가 있는
trial도 최종 `.hdf5` raw data로 보존하고 `analysis_exclude=true`, session
validator report의 `study_eligible=false`로 재수집 대상임을 표시한다. 실행 중단 또는 구조 검증
실패 파일만 `.partial`로 보존한다.

```bash
/home/railab/isaac-sim-4.5.0/python.sh \
  v3_chan/validate_ac_selective_smoothing_feedback.py \
  v3_chan/ac_selective_smoothing_feedback_data/<session>.hdf5 --json
```

## 실행 전 정적 확인

repository root에서 실행한다.

```bash
cd /home/railab/Desktop/Isaac_HRC/SeRT-Training-Platform-ac-selective-smoothing-v1

python v3_chan/runtime_contracts/ac_selective_smoothing_v1/validate_runtime_contract.py
python -m unittest -v \
  tests.test_v3_ac_selective_smoothing_feedback \
  tests.test_v3_ac_feedback_video

python v3_chan/collect_ac_selective_smoothing_feedback.py \
  --participant-id TEST_P01 --session-id dryrun_01 --dry-run

python v3_chan/collect_ac_selective_smoothing_feedback.py \
  --participant-id TEST_P01 --session-id dryrun_anchor_01 \
  --mode pilot_with_anchors --dry-run
```

Isaac Python 위치가 다르면 runner 호출 전에 절대경로를 지정한다.

```bash
ISAACSIM_PYTHON=/absolute/path/to/isaac-sim/python.sh \
  bash v3_chan/run_ac_selective_smoothing_feedback.sh --help
```

## HMD practice smoke

Participant 1명의 무작위 practice 1 trial:

```bash
bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id TEST_P01 --session-id hmd_smoke_random_01 \
  --practice-only --max-trials 1
```

실행 직후에는 BC-only layout replay나 participant-visible corridor 보정을 실행하지 않는다.
participant에게 별도의 `본래 행동`, BC-only 예시, A/C·CBF 기술 설명 화면도 보여주지
않는다. 시작 화면은 손을 가슴 앞 안전 위치에 유지하라는 안내만 표시하고, 5초의
tracking·안전거리 확인이 끝나면 현재 trial의 Frozen BC+assigned CBF가 바로 실행된다.
이 5초는 기준 행동 시연이 아니라 실제 제어를 시작하기 위한 장비·안전 gate다.
수집기는 BC task success와 release가 이미 확인된 고정 layout(seed `10011`, ID
`8b06c8502c07d1b3`)을 step 없이 pose만 복원한다. 이 근거는 A10 candidate 0의
BC-only 694-step strict success이며, 그 artifact는 participant crossing 경로의
qualification 근거로
사용하지 않는다. source artifact는
`hmd_smoke_A_10_20260905_115706.hdf5.partial`, SHA-256은
`665d3e8896a4d6b1db93bdc91b8d498b357f32afd4102a9e346e8aaf87bbe470`이다.
candidate 전체 source configuration도 canonical SHA-256
`3be5f0119d5826ebfc83b389f199b5fbbcb78d9b36581e4e64c1a914d69dd99f`로
고정해 reset 직후 cube/target/robot 초기 상태를 함께 검증한다. HDF5에는 이 구분과
source artifact/source configuration SHA-256을 함께 남긴다.

각 trial 시작 전 사용자는 양 controller를 편한 가슴 앞 안전 위치에 5초 동안
유지한다. 프로그램은 특정 가슴 좌표와 controller 위치를 비교하지 않는다. 짧은
pose dropout에도 wall-clock 진행률은 0으로 돌아가지 않고, 5초 뒤 같은 순간의
양손 live pose와 로봇에서 15 cm 이상 떨어진 안전거리만 최종 확인한다. 마지막
확인에서 tracking이 끊기면 5초를 반복하지 않고 회복만 기다린다. 이 gate 다음의
첫 participant-visible robot step부터 Frozen BC와 trial에 배정된 A/C CBF가 함께
적용된다.

로봇이 배정 task phase에 도달하면 participant에게 보이지 않는 broad bilateral
side gate와 로봇 주변 interaction band를 구성한다. 내부 detector는
`gate_half_span_m=0.20`, `gate_radius_m=0.35`,
`interaction_band_radius_m=0.22`를 사용한다. corridor, 시작점, 끝점, 이동 guide나
green/red/yellow marker는 생성하지 않는다. 내부 detector를 안정화하는
`cue_lead_s=1.0` monotonic wall-clock second 동안에는 current-EE protocol hold를 사용하고,
physics와 배정된 CBF는 계속 적용한다.
이 hold의 5차원 task action은 BC 진행을 막는 zero sentinel이며 물리적인 정지 명령
자체가 아니다. 실제 제어 목표는 cue 진입 순간 RMPFlow가 제어하는 `right_gripper`
frame의 FK full pose(position+orientation)를 latch한 값이다. 이 목표도 평상시와 같은
`RMPFlow -> assigned CBF -> robot.apply_action` 경로를 통과하므로, 손이 가까워져 CBF가
개입해야 할 때는 정지 목표보다 안전 회피가 우선한다.

cue lead가 끝나면 중앙 panel에는 다음 안내만 표시한다.

```text
START — 지금 바로 움직이세요
편한 손으로 로봇 팔의 한쪽에서 반대쪽까지 자연스럽게 한 번 통과하세요.
```

participant는 왼손/오른손, 좌→우/우→좌 방향, 속도를 자유롭게 선택한다. 같은 손이
한쪽 side gate에서 출발해 interaction band를 지난 뒤 반대 side gate에 도달해야
완전한 횡단 1회로 검출된다. 경계 부근의 작은 망설임이나 짧은 역동작은 허용하며
추가 횡단으로 세지 않는다. `START`부터 neutral 연속 확인이 끝날 때까지, 첫 완료 뒤
다시 gate→interaction band→반대 gate를 완전히 통과한 경우에만
`multiple_crossing`으로 기록한다. 이 경우 raw data는
보존하지만 `analysis_exclude=true`로 표시한다. START 뒤 8 simulation-second 안에
첫 완전 횡단이 없으면 crossing timeout으로 기록한다.
준비 조건이나 tracking이 제한 시간 안에 회복되지 않으면 종료 원인을 HMD에 3초간
표시하고 안전 정지한 뒤 `.partial` 파일을 보존한다.
XR UI의 한국어는 시작 전에 Noto Sans CJK의 필요한 glyph만 font atlas에 병합하며,
폰트가 없으면 읽을 수 없는 설문을 진행하지 않고 명확한 오류로 중단한다.

`START`의 전체 대기 시간은 배정 phase 도달 시점에 따라 달라진다. 순서는 startup tracking과
face-button release 확인, 5초 staging, BC+CBF task 시작, 배정 phase 도달, hidden
bilateral detector 구성, `cue_lead_s=1.0` monotonic wall-clock second 대기, 중앙 `START`
표시다. RTX/XR 처리 속도와 관계없이 detector 구성 뒤 cue lead는 실제 1초로 유지된다.
setup check 동안에는 X/Y/A/B나 joystick을 사용할
필요가 없으며 participant 절차에서는 모두 놓아 둔다. 이는 버튼이 위험해서가 아니라
초기 release 확인과 실험 입력 시점을 일정하게 하기 위한 것이다. X 또는 joystick을
한 번 만지는 것은 종료 입력이 아니다. 실제 trial 중 의도적 emergency abort는 오른쪽
A+B를 2초 유지할 때뿐이다. 별도의 장비
qualification 때만 setup 화면이 뜬 뒤 각 버튼을 한 번씩 눌러 충돌 부재를 확인한다.

Startup에서는 HMD pose와 XR camera anchor만 먼저 확인한 뒤 UI를 붙인다. 양 controller
pose의 연속 유효성 및 face-button release는 UI가 보이는 상태에서 확인한다. 따라서
한 controller tracking이 끊겨도 빈 화면에서 robot이 멈춰 있는 상태로 대기하지 않고,
어느 손을 HMD 앞에서 다시 움직여야 하는지 화면과 console에 함께 표시한다.

수집 runner는 repo-local `SeRT Feedback Oculus` action map을 XR 시작 전에 등록한다.
이 map은 유효한 `toolsLayout=vr`과 controller pose/raw input을 유지하지만 `tools=[]`,
`actions={}`로 Isaac의 beam/menu/teleport/grab 반응을 끈다. runtime에서 이 map과
빈 tool/action 목록을 확인하지 못하면 버튼 충돌이 있는 상태로 수집하지 않고
fail-closed 중단한다. 최초 확인 뒤에도 모든 tracking sample에서 같은 조건을 다시
검사하므로 controller reconnect나 dominant-hand 변경으로 stock map이 돌아오면 즉시
중단한다. 실제 선택된 map 이름, dominant hand, tool/action 수는 HDF5 root metadata의
`xr_feedback_action_map_json`에 저장한다. repo의 action-map JSON과 extension TOML도
`source_tree_sha256` 계산에 포함한다.

A practice 1 trial:

```bash
bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id TEST_P01 --session-id hmd_smoke_A_01 \
  --practice-only --force-condition A_reactive --max-trials 1
```

C practice 1 trial:

```bash
bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id TEST_P01 --session-id hmd_smoke_C_01 \
  --practice-only --force-condition C_smooth --max-trials 1
```

`--force-condition`과 `--max-trials`는 practice-only 장비 점검에만 허용된다.
pilot schedule을 강제로 바꾸는 데 사용할 수 없다.

## Full feasibility pilot

현재 handoff contract는 의도적으로 `production_collection_ready=false`다.
A/C 각각의 HMD practice와 생성 로그 validator가 통과하고, 아래 수동 항목을
확인한 뒤 runtime contract를 새 version으로 명시적으로 승인해야 한다. 그
전에는 다음 명령이 fail-closed로 종료되는 것이 정상이다.

Minimal pilot session:

```bash
HRI_LIVE_HMD_QUALIFIED=1 \
HRI_AC_SOURCE_LINEAGE_RECONCILED=1 \
  bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id PILOT_P01 --session-id pilot_p01_s01 \
  --mode minimal_pilot
```

Anchor-repeat pilot session:

```bash
HRI_LIVE_HMD_QUALIFIED=1 \
HRI_AC_SOURCE_LINEAGE_RECONCILED=1 \
  bash v3_chan/run_ac_selective_smoothing_feedback.sh \
  --participant-id PILOT_P01 --session-id pilot_p01_anchor_s01 \
  --mode pilot_with_anchors
```

Pilot/production은 추가로 clean Git HEAD와 검증 가능한 code provenance를
요구한다. `HRI_LIVE_HMD_QUALIFIED=1`은 체크리스트를 실제로 통과한 operator만
설정한다.

## Production 전 HMD 수동 체크리스트

- [ ] room calibration, controller handedness, `XR_SWAP_HANDS`가 실제 손과 일치한다.
- [ ] 시작 직후 시점이 avatar eye `[1, 0, 1.5]`에서 테이블 중심을 향하고 머리·손 표시와 일치한다.
- [ ] startup discovery에서 양쪽 controller와 X/Y/A/B의 press/click gesture가 보인다.
- [ ] startup log에서 `SeRT Feedback Oculus` action map과 `built-in-tools=off`가 확인된다.
- [ ] setup check 중 X/Y/A/B를 각각 눌러도 beam/menu가 뜨거나 session이 종료되지 않는다.
- [ ] 버튼을 누른 채 연결하거나 tracking이 끊겨도 phantom marker가 생기지 않는다.
- [ ] trial/query 중 한 controller 또는 face-button channel을 끊으면 즉시 verified hold 후 `.partial`로 종료되고 timeout label로 저장되지 않는다.
- [ ] 중앙 panel의 `START`와 free-crossing 한글 안내가 HMD 안에서 읽히고 A/C identity가 노출되지 않는다.
- [ ] corridor·green/red/yellow marker·moving guide가 보이지 않는다.
- [ ] 어느 손과 방향을 선택해도 오른손 A가 behavior anomaly, 오른손 B가 safety concern으로 각각 기록된다.
- [ ] no-marker가 acceptable로, uncertain/timeout이 negative로 변환되지 않는다.
- [ ] stable resumption 뒤 neutral-hand cue와 mandatory Q1–Q6이 정확히 한 번 나타난다.
- [ ] `needs_modification`은 reason 없이 확정되지 않고, back과 input timestamp가 기록된다.
- [ ] 설문 중 arm은 safe hold이고 physics/CBF는 ON이며 이후 task가 terminal까지 재개된다.
- [ ] A+B 2초 emergency abort가 동작하고 incomplete output은 `.partial`로만 남는다.
- [ ] gate→interaction band→반대 gate의 완전 횡단만 1회로 세고 작은 역동작은 허용한다.
- [ ] START부터 neutral 연속 확인 완료까지의 완전한 두 번째 횡단만 `multiple_crossing`이며 raw 보존과 `analysis_exclude`가 함께 적용된다.
- [ ] 실제 손·방향·속도·minimum gap이 자동 기록되고 scheduled 값으로 덮어쓰지 않는다.
- [ ] A 로그는 `joint_nominal/0`, C 로그는 `smooth_intervention/4.0`이고 공통 CBF/Recovery 값은 같다.
- [ ] 실제 햅틱 출력과 pulse가 없고 session metadata의 `haptics_enabled=false`와 일치한다.
- [ ] A practice와 C practice의 최종 HDF5가 각각 fail-closed validator를 통과한다.
- [ ] spectator video를 켰다면 camera pose/FPS가 고정되고 frame index와 simulation time이 맞는다.
- [ ] held-out qualification manifest와 현재 runtime handoff의 source-SHA lineage 차이를 해소했다.
- [ ] 승인된 runtime contract가 `production_collection_ready=true`이고 수집 checkout이 clean하다.

정적 unit test와 dry-run은 실제 controller binding, HMD panel 가시성, hidden
crossing detector, emergency stop을 대신 검증하지 않는다. 이 항목들은 A/C
각각의 in-HMD practice 결과와 validator report로 승인한다.
