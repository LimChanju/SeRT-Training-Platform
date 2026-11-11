# Frozen BC + fixed CBF online explicit-feedback collector

이 디렉터리의 새 수집 경로는 `bc_cbf_online_explicit_feedback_v1` 전용이다.
기존 3-cube collector와 기존 `bc_cbf_explicit_feedback_collection_v1` 파일은
변경하지 않는다.

## 동결 계약

- checkpoint: `v3_chan/policies/bc_pick_place_v1_100eps.pt`
- SHA-256: `fc4801dfdd7ae4e3edf8760277a1d87fd540c12bf27aeb0a06a3eefdbd5ba8c3`
- 84-D observation의 human field를 mask한 direct BC 5-D action
- policy update/online learning/feedback input/pseudo-ErrP: 모두 OFF
- single cube, strict task semantics v3, release 필수, place XY `0.05 m`
- CBF: safe `0.05 m`, activation `0.13 m`, gamma `8/s`, horizon
  `0.15 s`, max buffer `0.08 m`, max joint speed `2 rad/s`
- haptic과 participant-facing CBF active 표시: 모두 OFF

실행 시 policy/runtime/config hash가 맞지 않으면 시작하지 않는다. Pilot/production은
clean Git HEAD provenance와 수동 live-HMD 자격 확인
(`HRI_LIVE_HMD_QUALIFIED=1`)도 필수다. CBF가 실제로
사용하는 distal PhysX collider를 검색해 link/collider 목록을 출력하고 HDF5 root
metadata에 저장한다. proximal link 0--5 근처 crossing은 read-only AABB monitor로
`off_protocol` 처리할 뿐 CBF 범위를 확장하지 않는다.

## Trial FSM

`reset → wait_phase → pre_crossing → crossing → response → recovery(2 s) →
query_hold → optional resume_task → save/reset`

`wait_phase`에서는 frozen BC와 task FSM이 함께 진행된다. 지정 event에서 cue를
띄우면 실험 phase가 cue lead 동안 다른 phase로 바뀌지 않도록 event clock만 latch하고,
BC action은 `pre_crossing → crossing → response → recovery` 전체에서 계속 RMPFlow →
고정 CBF → articulation 경로를 통과한다. Recovery가 끝난 `query_hold`에서만 BC
action 대신 current-EE hold를 넣고 physics와 CBF는 계속 실행한다. 응답 뒤
`resume_task`에서 BC와 task FSM을 함께 재개한다. `world.pause()`는 reset/readiness
경계에만 사용한다.

`grasp_lift`는 close 명령만 제출되는 event 3이 아니라, grasp가 확인된 lift event
4에서 trigger한다. 따라서 네 phase 모두 crossing 동안 실제 RMPFlow/CBF arm command가
articulation에 제출된다.

각 trial에는 cube 1개, 지정 손 crossing 1회, query 1개만 있다. Query는 encounter와
별도 객체이므로 Control trial의 `encounter_id`는 빈 UTF-8(null encoding)일 수 있어도
`query_id`는 반드시 존재한다.

## 설계와 XR 입력

Production은 `4 phase × 3 severity × 2 direction = 24` factorial이며 slow/fast가
원본 24개 안에서 12/12다. 기본값은 practice 4개와 phase별 anchor repeat 4개를
추가하여 32개 실행, 28개 평가다. participant/session/seed로 순서를 안정적으로
randomize한다. Pilot은 `--pilot-evaluated-trials 12..16`을 사용한다.

Crossing corridor는 EE center 거리가 아니라 실제 protected PhysX surface와 hand
sphere 사이 gap query로 calibration한다. 먼저 accepted BC-only rollout에서 네 phase
trigger의 EE/arm-joint nominal reference를 저장하고, trial trigger가 EE `0.03 m`, arm
joint 최대 `0.10 rad` 안에서 일치할 때만 severity corridor를 유효하게 취급한다.

- Control: `0.16--0.20 m` (target `0.18`)
- Shallow: `0.08--0.12 m` (target `0.10`)
- Threat nominal corridor: `0.00--0.04 m` (target `0.02`); 실행 CBF는 `0.05 m` 보호

초록 start, 빨강 end, 파랑 corridor, 노랑 속도 guide를 VR scene에 표시한다. 기본
실시간 marker mapping은 crossing 반대편 controller를 쓴다.

Start/end radial gate, 최소 진행률, 최대 역방향 진행량, corridor/deviation 반경과
slow/fast 속도 허용치는 모두 `online_scenarios_v1.json`에서 읽는다. Detector와
crossing timeout은 화면의 `START` 시점에 arm하며, cue-lead row는 onset 전 window로
보존한다. START 전에 CBF/risk가 발생하거나 crossing을 끝내면 off-protocol이다.

- left-hand crossing: right A = safety concern, right B = behavior anomaly
- right-hand crossing: left X = safety concern, left Y = behavior anomaly

입력은 unknown/disconnected와 release를 구분하고, startup/reconnect/press 뒤 알려진
release 3 frame을 확인해야 다시 arm된다. Marker가 없다는 사실은 어떤 평가 label도
의미하지 않는다. Controller output/haptic API는 호출하지 않는다.

사후 설문은 XRSceneView 안에서 X=이전, Y=다음, A=선택, B=확인으로 조작한다.
이 매핑도 protocol config에서 읽고 화면 안내를 같은 값으로 생성한다. XR hand swap을
켜면 포즈와 버튼 모두 동일한 logical-hand resolver를 사용한다. Panel은 metre-stage에서
`0.9 × 0.7 m`가 되도록 명시적으로 scale한다.
Q1의 `needs_modification`, `acceptable_as_is`, `uncertain`, Q2--Q6의 개별 1--5 값,
그리고 Q1이 `needs_modification`일 때 복수 rejection reason을 저장한다. Timeout은
missing/abstain이며 label로 치환하지 않는다.

## 저장 구조

최종 HDF5의 주요 구조는 다음과 같다.

- root attrs: participant/session split key, policy/code/source/config SHA,
  frozen flags, actual protected links/colliders, full canonical config,
  runtime-vs-future feature provenance
- `/schedule`: planned phase/severity/direction/speed/crossing hand,
  practice/anchor/pilot/exclusion
- `/layout_prechecks`: exact source pose/state와 BC-only strict success evidence
- `/bc_infeasible_diagnostic`: 실제 BC-only 실패 layout만 별도 보존
- `/trials/trial_*`: planned/actual/CBF/task/feedback aggregate와 모든 step
- step `human`, `robot`, `object`, `task`, `actions`, `cbf`, `safety`, `control`,
  `crossing`, `provenance` groups
- action stages: BC raw 5-D, controller 5-D, nominal RMPFlow, CBF-filtered,
  submitted, pre-applied, applied full joint vectors + masks
- `/realtime_markers`, `/queries`, `/query_answers`, `/encounters`

Active constraint identity는 frozen CBF가 소비한 동일 `safety_result`에서 hand/link/
collider를 재구성하고 diagnostic constraint count와 일치할 때만 exact-available로
기록한다. Applied와 filtered는 bitwise equality를 요구하지 않으며 gripper merge와
articulation drive provenance/mask를 보존한다.

Encounter onset은 gap/TTC/same-space RMPFlow-vs-CBF norm 조건 중 하나가 3 frame
연속일 때다. 모든 clear 조건이 simulation time `0.5 s` 지속되어야 offset이며,
`1 s` 이내 re-entry는 merge한다. TTC onset/clear에는 유효한 dynamic measurement가
필수다. `8 s` 초과 timeout, onset 전 `1 s`와 offset 후
`2 s` window, risk/CBF candidate/3-frame confirmation step을 저장한다.

Trial의 `actual_minimum_gap_m`는 지정 crossing 손의 protected-surface gap이고,
`actual_minimum_any_hand_gap_m`는 양손 전체 진단값이다. Task completion은 관측된
경우에만 `completion_time_s >= 0`이며 query hold를 제외한 active simulation time이다.
미완료는 `task_completion_observed=0`, `completion_time_s=-1`로 구분한다. Scheduled
evaluated trial 중 하나라도 off-protocol/incomplete이면 validator가 최종 파일 공개를
거부하고 `.partial`을 남긴다.

## 실행

먼저 HMD 없이 schedule/config를 확인한다.

```bash
python v3_chan/collect_bc_cbf_explicit_feedback.py \
  --participant-id P001 --session-id P001_S01 --dry-run
```

Production:

```bash
HRI_LIVE_HMD_QUALIFIED=1 \
HRI_PARTICIPANT_ID=P001 HRI_SESSION_ID=P001_S01 \
  ./v3_chan/run_bc_cbf_explicit_feedback.sh
```

Pilot 16 evaluated trials:

```bash
HRI_LIVE_HMD_QUALIFIED=1 \
./v3_chan/run_bc_cbf_explicit_feedback.sh \
  --participant-id P001 --session-id P001_PILOT \
  --pilot-evaluated-trials 16
```

Practice 한 trial 장비 smoke test:

```bash
./v3_chan/run_bc_cbf_explicit_feedback.sh \
  --participant-id SMOKE01 --session-id SMOKE01_S01 \
  --practice-only --practice-trials 1 --no-include-anchor-repeats
```

검증:

```bash
$ISAACSIM_ROOT/python.sh v3_chan/validate_bc_cbf_explicit_feedback.py \
  v3_chan/explicit_feedback_data/P001_S01_YYYYMMDD_HHMMSS.hdf5 --json
```

Writer는 step마다 flush하고 `.partial`을 먼저 seal한다. Fail-closed validator를
통과한 경우에만 최종 `.hdf5`로 atomic rename한다. Tracking/CBF/action/query 계약
위반이나 중단 시 최종 study 파일을 만들지 않고 `.partial`을 보존한다.

Pilot/production 전에는 아래를 사람이 HMD 안에서 확인한 뒤에만
`HRI_LIVE_HMD_QUALIFIED=1`을 설정한다: 양 controller 연결/논리 손 매핑, X/Y/A/B
edge 입력, 반대 controller marker 두 종류, XRSceneView 문항 가독성, start/end/corridor/
speed guide 정렬, tracking dropout 시 controlled hold/abort, haptic 무출력, proximal
침범의 off-protocol 처리, 그리고 1-trial `.partial → validator → .hdf5` 흐름. Source-only
작업 복사본은 practice만 허용하며 pilot/production에는 검증 가능한 clean Git HEAD가
필요하다.
