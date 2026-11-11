# Handover 구현 보고서

작성일: 2026-07-07  
대상 경로: `/home/railab/Desktop/Isaac_HRC/v3_hoeng`

## 1. 요약

이번 작업의 목표는 Isaac Sim + VR 환경에서 사람이 손으로 잡고 있는 블럭을 Franka Panda 로봇이 실제 그리퍼로 접근해 잡고, 사람 손에서 블럭의 소유권을 넘겨받은 뒤 목표 위치에 가져다 놓는 handover task를 안정화하는 것이었다.

현재 구현은 학습 기반이 아니다. 강화학습, imitation learning, neural policy로 성공을 학습하는 구조가 아니라, 매 simulation step마다 손 위치, cube 위치, gripper finger 위치, gripper width, finger-cube 거리, jaw/depth/vertical offset 등을 계산하는 규칙 기반 FSM이다.

최종적으로 다음 흐름이 동작하도록 구현했다.

1. 사람이 VR 손으로 cube를 잡는다.
2. `VRGrabManager`가 손 physical proxy와 cube 사이에 human fixed joint를 만든다.
3. `HandoverTaskManager`가 사람이 잡은 cube를 active handover cube로 채택한다.
4. 손이 일정 시간 안정되면 `robot_takeover` phase로 전환한다.
5. 로봇이 oriented grasp FSM으로 cube 주변에 접근한다.
6. 그리퍼가 close 상태에서 cube를 finger 기준으로 충분히 포획했다고 판단하면, robot fingertip custody joint를 먼저 만든다.
7. 그 직후 human fixed joint와 grab state를 강제로 해제한다.
8. 짧은 verify 단계를 통과하면 로봇이 cube를 lift, carry, descend, release 순서로 목표 위치에 놓는다.
9. place 완료 후 다음 handover cycle로 돌아간다.

성공 로그 예시는 다음과 같다.

```text
[ErrP] handover_physical_place_complete | cube=cube_1,place=0.600,-0.250,0.976
[ErrP] handover_place_complete | cube=cube_1,place=0.600,-0.250,0.976
[Handover] robot_place_done cube=cube_1
```

## 2. 주요 수정 파일

### 2.1 `simulation/task/handover_task.py`

가장 많은 변경이 들어간 핵심 파일이다. Handover task의 phase 관리, oriented grasp FSM, gripper geometry 판정, human release, robot fingertip custody joint, physical carry/place까지 모두 이 파일에서 관리한다.

주요 추가/수정 영역은 다음과 같다.

- `HandoverTaskConfig`에 handover grasp 관련 옵션 추가
- 사람 손 안정성 판단 강화
- handover 전용 oriented grasp FSM 구현
- gripper finger 기반 grasp metric 계산
- close/attach 단계에서 strict, loose, force release 조건 구현
- robot fingertip custody joint 구현
- human fixed joint 강제 탐색/삭제 구현
- human re-grab suppression 구현
- human hand collision suppression 구현
- physical carry stage 구현
- terminal log noise 필터 구현

### 2.2 `vr/vr_grab.py`

사람이 VR hand physical proxy로 cube를 잡는 기존 구조를 분석하고, handover task에서 해제할 수 있도록 `force_release`, `holding_hand_for`, `set_grab_suppressed`, hand fixed joint path 구조를 활용했다.

현재 사람 손 잡기는 기본적으로 다음 방식이다.

```text
/World/HandoverJoints/{hand}_{cube}_fixed_joint
```

예:

```text
/World/HandoverJoints/right_cube_0_fixed_joint
```

### 2.3 `vr/vr_avatar.py`

사람 손 physical proxy collision을 handover 중 강제로 끄고 다시 켤 수 있는 기능을 사용했다. 로봇이 cube를 넘겨받는 동안 사람 손 proxy가 gripper와 계속 충돌하면 cube가 밀리거나 튕기기 때문에, robot grasp 순간에는 손 collision suppression이 필요하다.

관련 기능:

- `set_physical_proxy_collision_forced_disabled`
- `_set_physical_proxy_collision_enabled`
- `get_physical_hand_proxy_path`
- `get_physical_hand_proxy_pos`
- `get_physical_hand_proxy_radius`

### 2.4 `sim.md`

handover 실험 실행 환경변수를 정리했다. 현재 handover 실험은 `ENABLE_HANDOVER_TASK=1`, `ENABLE_PICK_PLACE=0` 경로에서 진행한다.

핵심 설정은 다음과 같다.

```bash
ENABLE_HANDOVER_TASK=1 ENABLE_PICK_PLACE=0
XR_PHYSICAL_HAND_PROXY=1
XR_PHYSICAL_HAND_CONTACT_GRAB=1
XR_PHYSICAL_HAND_FIXED_JOINT_GRAB=1
HANDOVER_ADOPT_GRABBED_CUBE=1
HANDOVER_HUMAN_CUBE_PHYSICS=1
HANDOVER_PHYSICAL_MODE=1
HANDOVER_ORIENTED_GRASP=1
HANDOVER_ROBOT_FIXED_JOINT_GRASP=0
HANDOVER_ROBOT_FINGERTIP_JOINT_TAKEOVER=1
HANDOVER_DISABLE_HUMAN_HAND_COLLISION_ON_ROBOT_GRASP=1
```

## 3. 현재 handover task 전체 흐름

## 3.1 Phase 단위 흐름

`HandoverTaskManager`의 큰 phase는 다음 흐름을 가진다.

```text
waiting_for_hand
-> spawned_to_hand 또는 human_holding
-> robot_takeover
-> place complete
-> waiting_for_hand
```

현재 실험 설정에서는 `HANDOVER_ADOPT_GRABBED_CUBE=1`이므로, task가 특정 cube만 기다리는 것이 아니라 사람이 잡은 handover candidate cube를 active cube로 채택한다.

대표 로그:

```text
[VRGrab] right fixed joint created: /World/HandoverJoints/right_cube_0_fixed_joint ...
[VRGrab] right grabbed 'cube_0' (dist=0.000 m via physical_proxy)
[ErrP] handover_active_cube_adopted | hand=right,cube=cube_0
[Handover] adopted grabbed cube 'cube_0' from right hand
[Handover] human_holding hand=right cube=cube_0
```

## 3.2 사람이 cube를 잡는 방식

사람 손이 cube를 잡을 때는 `vr/vr_grab.py`의 `VRGrabManager._attach()`가 동작한다.

현재 환경에서는 다음 조건이 켜져 있다.

```bash
XR_PHYSICAL_HAND_PROXY=1
XR_PHYSICAL_HAND_CONTACT_GRAB=1
XR_PHYSICAL_HAND_FIXED_JOINT_GRAB=1
XR_PHYSICAL_HAND_ALLOW_LOGICAL_FALLBACK=0
```

따라서 사람 손은 logical pose teleport가 아니라 physical proxy 기반 fixed joint로 cube를 잡는다.

동작 방식:

1. hand physical proxy가 cube에 접촉하거나 proximity 조건을 만족한다.
2. `VRGrabManager`가 nearest cube를 선택한다.
3. `/World/HandoverJoints/{hand}_{cube}_fixed_joint`를 만든다.
4. `grabbed[hand] = cube`로 grab state를 기록한다.
5. cube physics와 collision은 physical mode에 맞게 유지된다.

중요한 점은 사람이 잡은 cube가 실제로는 손 proxy와 fixed joint로 연결되어 있기 때문에, 로봇이 가져가려면 이 joint를 반드시 제거해야 한다는 것이다.

## 3.3 손 안정성 판단

사람 손은 계속 움직일 수 있으므로, 바로 robot takeover로 넘어가지 않는다. `human_holding` phase에서 일정 시간 잡고 있는지와 손 움직임이 안정적인지를 확인한다.

현재 설정:

```bash
HANDOVER_HUMAN_HOLD_STEPS=24
HANDOVER_STABLE_WINDOW_STEPS=12
HANDOVER_STABLE_MOTION_THRESHOLD_M=0.035
HANDOVER_STABLE_MAX_STEP_M=0.055
HANDOVER_STABLE_MEAN_STEP_M=0.010
HANDOVER_STABLE_JITTER_THRESHOLD_M=0.025
HANDOVER_STABLE_OUTLIER_TOLERANCE_M=0.055
HANDOVER_STABLE_REQUIRED_WINDOWS=3
```

로그 예:

```text
[Handover] holding_for_transfer 23/24
[Handover] waiting_for_stable_hand reason=window_motion stable=0/3
[Handover] waiting_for_stable_hand reason=confirming stable=1/3
[Handover] waiting_for_stable_hand reason=confirming stable=2/3
[ErrP] handover_transfer_ready | hand=right,cube=cube_0,pick=...
```

판단 기준은 최근 손 위치 히스토리에서 다음 값을 계산하는 방식이다.

- 전체 window motion
- step별 최대 이동량
- 평균 step 이동량
- jitter
- outlier 이동량

이 값들이 threshold 안에 들어오고 안정 window가 충분히 누적되면 `transfer_ready`가 된다.

## 3.4 Robot takeover 시작

손이 안정되면 phase가 `robot_takeover`로 바뀐다.

로그:

```text
[Handover] robot_takeover_started cube=cube_0
```

이후 일반 `PickPlaceController`만 쓰는 top-down pick-and-place가 아니라, handover 전용 oriented grasp FSM이 우선 사용된다.

## 4. Oriented grasp FSM

현재 handover grasp는 `simulation/task/handover_task.py`의 `_run_oriented_robot_takeover()`에서 처리한다.

FSM state는 다음과 같다.

```text
orient_reopen
orient_pregrasp
orient_approach
orient_close
orient_attach
orient_verify
orient_lift_place
```

## 4.1 `orient_pregrasp`

로봇이 cube 바로 위 또는 접근 방향 바깥쪽의 pregrasp pose로 이동한다. 이 단계에서는 gripper를 open 명령으로 유지한다.

기존에는 다음 로그가 매번 출력됐다.

```text
[Handover] orient_pregrasp steps=32
[Handover] orient_pregrasp steps=33
```

너무 시끄럽기 때문에 terminal 출력만 숨겼다. 기능은 그대로 유지된다.

구현:

```python
def _suppress_terminal_status_message(self, message: str) -> bool:
    return message.startswith(f"{ORIENT_STAGE_PREGRASP} steps=")
```

즉 내부 FSM message는 그대로 반환되지만 터미널에는 `orient_pregrasp steps=...`만 출력되지 않는다. 다른 Handover 로그는 그대로 출력된다.

## 4.2 `orient_approach`

pregrasp에서 더 가까운 approach pose로 이동한다. 이때 gripper는 계속 open 상태다.

close 시작 여부는 `_gripper_ready_to_start_close()`가 판단한다.

현재 close 시작 조건은 다음 metric을 본다.

- `center_dist`
- `min_dist`
- `jaw_offset`
- `vertical_offset`
- `depth_offset`

초기에는 `head_dist`도 close 판단에 사용했지만 제거했다. `head_dist`는 gripper palm 또는 panda_hand와 cube 사이 거리라서, 실제 finger capture 가능 여부와 맞지 않는 경우가 많았다. 특히 사람이 손으로 잡은 cube 주변에서는 palm/head가 가까워지는 것이 자연스러운데, 이 값을 실패 조건으로 쓰면 정상적인 close 기회를 놓쳤다.

## 4.3 `orient_close`

gripper를 닫는 단계다. 이 단계에서 cube가 finger 사이로 충분히 들어오면 early release를 시작할 수 있다.

이전 문제:

```text
[ErrP] handover_oriented_retry | reason=head_clearance_lost ...
```

과거에는 gripper가 cube를 거의 잡은 상태에서도 `head_clearance_lost`가 먼저 발생해서 retry로 빠졌다. 그러면 사람 손 fixed joint를 끊는 단계까지 가지 못했다.

수정 후 흐름:

1. close 단계에서 gripper metrics를 계산한다.
2. `_gripper_ready_to_force_detach_human(metrics)`가 true인지 먼저 본다.
3. true면 `handover_close_capture_ready`를 로그로 남긴다.
4. `head_clearance_lost` retry보다 먼저 `_try_begin_human_detach_for_physical_grasp()`를 호출한다.
5. detach가 시작되면 `orient_verify`로 넘어간다.

즉 현재는 finger 기준 포획 조건이 만족되면 palm/head clearance보다 human release가 우선이다.

## 4.4 `orient_attach`

close hold 이후, cube가 gripper 안에 들어왔다고 판단하면 attach 단계로 넘어간다.

여기서 다음 세 가지 grasp 판단을 사용한다.

### Strict attach

`_gripper_contains_cube(metrics)`가 true인 상태다.

조건:

- `center_dist <= handover_grasp_attach_center_dist_m`
- `min_dist <= handover_grasp_attach_dist_m`
- `width <= handover_grasp_attach_width_m`
- `jaw_offset <= handover_grasp_jaw_axis_max_m`
- `vertical_offset <= handover_grasp_vertical_max_m`
- `depth_offset <= handover_grasp_depth_axis_max_m`

현재 `head_dist`는 strict attach 조건에서 제거했다.

### Loose release

strict 조건에서 depth 등이 조금 벗어나도, finger 주변에 cube가 충분히 들어왔다고 볼 수 있으면 사람 손을 release하도록 하는 조건이다.

대표적으로 다음 로그에서 strict는 실패했지만 loose/force로 넘길 수 있는 상황이 있었다.

```text
center=0.059/0.065
dist=0.059/0.080
width=0.057/0.060
jaw=0.027/0.040
z=0.053/0.070
depth=0.053/0.040
```

여기서 depth만 strict 기준보다 크지만, 실제로는 gripper가 cube를 포획한 상태에 가까웠다.

### Force release

더 넓은 기준으로 cube가 finger 주변에 들어온 경우, `forced_attach_capture` 또는 `forced_close_capture` 모드로 사람 손을 떼도록 한다.

핵심은 “계속 기다리다가 attach_timeout으로 retry하지 않고, 충분히 잡을 수 있는 위치라면 human fixed joint를 끊고 검증 단계로 넘어가는 것”이다.

## 4.5 `orient_verify`

human release와 robot fingertip custody joint 생성 이후, 로봇이 실제로 cube를 넘겨받았는지 확인하는 단계다.

현재 fingertip custody joint가 있으면 긴 physical stability 검증 대신 짧게 verify한다.

```text
[Handover] verifying_fingertip_joint_grasp stable=... joint=...
```

사람 손이 여전히 cube를 잡고 있으면 `_force_release_active_cube()`를 다시 호출한다.

## 4.6 `orient_lift_place`

robot grasp가 확인되면 physical carry/place 단계로 넘어간다. 일반 `PickPlaceController`의 event 기반 place가 아니라, handover physical carry stage를 사용한다.

## 5. Grasp metric 계산

핵심 함수는 `_gripper_grasp_metrics()`다.

매 step마다 다음 값들을 계산한다.

| Metric | 의미 |
| --- | --- |
| `center_dist` | gripper finger center와 cube 중심 사이 거리 |
| `min_dist` | left/right finger 및 center 중 cube에 가장 가까운 점의 거리 |
| `width` | gripper joint position 합으로 계산한 gripper 열림 정도 |
| `jaw_offset` | finger 좌우축 기준 cube가 얼마나 옆으로 벗어났는지 |
| `vertical_offset` | finger center와 cube의 z 차이 |
| `depth_offset` | gripper 접근/깊이축 기준 cube가 얼마나 앞뒤로 벗어났는지 |
| `head_dist` | panda_hand 또는 EE 기준 거리. 현재는 주 성공 조건에서 제외 |

중요한 변화는 다음과 같다.

- 초기에는 `head_dist`를 grasp 성공/실패에 강하게 사용했다.
- 하지만 이 값은 “실제 손가락 사이에 잡혔는가”가 아니라 “palm/head가 cube와 가까운가”를 의미한다.
- 그래서 정상적인 finger capture 상황도 `head_clearance_lost`로 실패 처리했다.
- 현재는 finger 기반 metric을 우선 사용하고, `head_dist`는 로그와 일부 보호 조건에만 남겨둔 상태다.

## 6. Robot fingertip custody joint

초기 문제 중 하나는 로봇이 cube를 실제 gripper finger로 잡는 느낌이 아니라, panda hand/head 쪽에 cube가 붙는 것처럼 보인다는 점이었다.

이를 줄이기 위해 robot custody joint를 `/World/Franka/panda_hand`가 아니라 가장 가까운 finger link에 만들도록 변경했다.

현재 흐름:

1. `_nearest_finger_body_path()`가 left/right finger 중 cube에 가까운 finger를 고른다.
2. `_create_robot_cube_fingertip_custody_joint()`가 해당 finger와 cube 사이에 fixed joint를 만든다.
3. joint 이름은 `robot_finger_{cube}_fixed_joint` 형태다.

예상 로그:

```text
[Handover] robot fixed joint created: /World/HandoverJoints/robot_finger_cube_0_fixed_joint body0=/World/Franka/panda_leftfinger ...
[Handover] fingertip custody joint takeover cube=cube_0 body=/World/Franka/panda_leftfinger
```

이 방식은 완전한 마찰 기반 실제 접촉 grasp는 아니다. 하지만 기존의 panda_hand/head attachment보다 더 명확하게 gripper finger 쪽으로 custody를 넘긴다.

## 7. Human release 구현

사람 손에 cube가 “본드처럼” 붙어 있던 근본 원인은 human fixed joint가 남아 있거나, grab manager의 `grabbed[hand]` state가 해제되지 않는 경우였다.

이를 해결하기 위해 `_force_release_active_cube()`를 강화했다.

현재 수행 순서:

1. `/World/HandoverJoints` 아래에서 active cube를 target으로 가진 human fixed joint를 스캔한다.
2. human joint로 판정되는 joint를 제거한다.
3. `grab_manager.force_release(cube=active_cube)`를 호출한다.
4. 그래도 holding hand가 남아 있으면 `grab_manager._release(hand)`를 직접 호출한다.
5. `prev_grip[hand] = False`로 다시 즉시 grab되는 것을 줄인다.
6. 다시 human fixed joint를 스캔해 제거한다.
7. 그래도 남으면 `handover_human_release_failed` 로그를 남긴다.

관련 로그:

```text
[Handover] human fixed joint force-removed: /World/HandoverJoints/right_cube_0_fixed_joint
[ErrP] handover_human_joint_force_removed | cube=cube_0,count=...
```

## 8. Human re-grab suppression

robot takeover 직후 사람이 계속 grip 입력을 유지하면, cube가 손에서 떨어졌다가 같은 프레임 또는 다음 프레임에 다시 grab될 수 있다.

이를 막기 위해 `VRGrabManager.set_grab_suppressed()`를 사용한다.

흐름:

1. robot grasp/release 시작 전 active cube에 대해 re-grab suppression을 건다.
2. handover cycle이 끝나면 suppression을 복구한다.

로그:

```text
[ErrP] handover_human_regrab_suppressed | hand=right,cube=cube_0
[ErrP] handover_human_regrab_restored | hand=right,cube=cube_0
```

## 9. Human hand collision suppression

사람 손 physical proxy가 너무 크거나 gripper와 겹치면, 로봇이 cube를 잡으려는 순간 손 proxy와 gripper가 충돌해 cube가 튀거나 trajectory가 무너진다.

이를 줄이기 위해 robot grasp takeover 순간에는 active hand proxy collision을 강제로 끈다.

관련 설정:

```bash
HANDOVER_DISABLE_HUMAN_HAND_COLLISION_ON_ROBOT_GRASP=1
```

동작:

1. `_suppress_active_hand_collision()`이 active hand collision을 비활성화한다.
2. `_maintain_active_hand_collision_suppression()`이 매 update마다 계속 유지한다.
3. place 완료 또는 reset 시 `_restore_active_hand_collision()`으로 복구한다.

로그:

```text
[ErrP] handover_human_hand_collision_suppressed | hand=right,cube=cube_0
[ErrP] handover_human_hand_collision_restored | hand=right
```

## 10. Physical carry/place

robot grasp verify가 끝나면 `_run_physical_carry_place()`로 이동한다.

physical carry stage:

```text
lift
-> carry
-> descend
-> release
```

### `lift`

현재 EE 위치에서 z만 들어 올린다. 목표 z는 현재 위치, place 높이, workspace z max를 고려해 결정한다.

### `carry`

x, y를 place target 쪽으로 이동한다. z는 lift height를 유지한다.

### `descend`

place target 위쪽으로 내려간다. 이때 `handover_physical_place_above_m`를 사용한다.

### `release`

gripper open 명령을 유지하고, release hold steps가 지나면 place complete로 처리한다.

성공 로그:

```text
[ErrP] handover_physical_place_complete | cube=cube_1,place=0.600,-0.250,0.976
[ErrP] handover_place_complete | cube=cube_1,place=0.600,-0.250,0.976
[Handover] robot_place_done cube=cube_1
```

## 11. Oriented grasp plan

`_make_oriented_grasp_plan()`에서 cube 위치와 손 위치를 기준으로 로봇의 접근 방향과 EE orientation을 만든다.

현재 기본은 top-down이다.

```text
mode=top_down
approach=[0.0, 0.0, 1.0]
```

로그:

```text
[ErrP] handover_oriented_grasp_plan | cube=cube_0,hand=right,mode=top_down,approach=0.000,0.000,1.000,...
[Handover] oriented grasp plan mode=top_down approach=[0. 0. 1.]
```

옵션으로 손이 cube 아래쪽에 있을 때 side/upper-side grasp를 선택하는 코드가 있다.

관련 설정:

```bash
HANDOVER_SIDE_GRASP_WHEN_HAND_BELOW=0
```

현재 `sim.md`에서는 이 옵션이 꺼져 있으므로 기본적으로 top-down 접근을 사용한다. 필요하면 `HANDOVER_SIDE_GRASP_WHEN_HAND_BELOW=1`로 켤 수 있다.

## 12. RMPFlow controller 사용

학습된 policy가 아니라, 목표 end-effector pose를 계산한 뒤 `RMPFlowController.forward()`에 target position과 target orientation을 넘긴다.

즉 task logic은 다음 값을 계산한다.

- target EE position
- target EE orientation quaternion
- gripper open/close command

그리고 실제 Panda joint command는 RMPFlow가 계산한다.

정리하면 다음과 같다.

```text
Handover FSM
-> target_end_effector_position 계산
-> target_end_effector_orientation 계산
-> RMPFlowController.forward(...)
-> articulation action 적용
```

## 13. Fixed joint mode와 fingertip takeover 충돌 해결

중간에 큰 문제가 있었다. `HANDOVER_ROBOT_FIXED_JOINT_GRASP=1`로 실행되면 attach 단계에서 기존 fixed joint path가 우선되어, physical/fingertip takeover path가 막힐 수 있었다.

증상:

```text
[Handover] orient_attach waiting steps=...
[ErrP] handover_oriented_retry | reason=attach_timeout
```

이때 metrics는 대부분 좋은데 `depth`만 strict 기준보다 커서 strict attach가 안 되고, 원래는 loose/force detach로 넘어가야 했다. 하지만 `robot_fixed_joint_grasp` 조건 때문에 detach path가 막혔다.

해결:

`_use_fingertip_takeover_path()`를 추가했다.

```python
def _use_fingertip_takeover_path(self) -> bool:
    return bool(
        self.config.physical_mode
        and self.config.handover_fingertip_joint_takeover_enabled
    )
```

이제 `HANDOVER_ROBOT_FIXED_JOINT_GRASP=1`이어도 `HANDOVER_ROBOT_FINGERTIP_JOINT_TAKEOVER=1`이면 fingertip takeover path가 우선된다.

startup 로그에도 현재 상태가 보이도록 추가했다.

```text
[Handover] enabled | ... robot_fixed_joint_grasp=True fingertip_takeover=True ...
```

## 14. Log 개선

너무 많은 progress log가 터미널을 채우는 문제가 있었다.

요청에 따라 다음 로그만 숨겼다.

```text
[Handover] orient_pregrasp steps=32
[Handover] orient_pregrasp steps=33
```

기능은 제거하지 않았고, terminal 출력만 suppress한다. 다른 로그는 그대로 유지된다.

추가로 디버깅을 위해 다음 로그들을 남겼다.

- `handover_close_capture_ready`
- `attach_detach_decision`
- `handover_fingertip_joint_takeover`
- `handover_physical_grasp_release`
- `handover_physical_grasp_verified`
- `handover_human_joint_force_removed`
- `handover_human_release_failed`
- `handover_physical_carry_stage`
- `handover_physical_place_complete`

## 15. 현재 주요 환경변수 정리

### Handover task 활성화

| 변수 | 현재 값 | 의미 |
| --- | --- | --- |
| `ENABLE_HANDOVER_TASK` | `1` | handover task 활성화 |
| `ENABLE_PICK_PLACE` | `0` | 일반 pick-place 비활성화 |
| `HANDOVER_PHYSICAL_MODE` | `1` | physical handover 경로 사용 |
| `HANDOVER_ADOPT_GRABBED_CUBE` | `1` | 사람이 잡은 cube를 active cube로 채택 |
| `HANDOVER_RESPAWN_TO_HAND` | `0` | 손 위치로 cube respawn하지 않음 |

### VR hand grab

| 변수 | 현재 값 | 의미 |
| --- | --- | --- |
| `XR_PHYSICAL_HAND_PROXY` | `1` | 손 physical proxy 사용 |
| `XR_PHYSICAL_HAND_CONTACT_GRAB` | `1` | contact 기반 grab 사용 |
| `XR_PHYSICAL_HAND_FIXED_JOINT_GRAB` | `1` | 사람 손-cube fixed joint 사용 |
| `XR_PHYSICAL_HAND_JOINT_COLLISION` | `0` | hand fixed joint collision 비활성 |
| `XR_PHYSICAL_HAND_ALLOW_LOGICAL_FALLBACK` | `0` | logical fallback 금지 |
| `XR_PHYSICAL_HAND_CONTACT_MARGIN` | `0.025` | contact margin |
| `XR_PHYSICAL_HAND_PROXY_RADIUS` | `0.025` | hand proxy radius |

### Grasp geometry

| 변수 | 현재 값 | 의미 |
| --- | --- | --- |
| `HANDOVER_GRASP_ATTACH_DIST_M` | `0.08` | finger/cube 최소거리 기준 |
| `HANDOVER_GRASP_ATTACH_CENTER_DIST_M` | `0.065` | gripper center/cube 거리 기준 |
| `HANDOVER_GRASP_ATTACH_WIDTH_M` | `0.06` | gripper width 기준 |
| `HANDOVER_GRASP_JAW_AXIS_MAX_M` | `0.04` | jaw axis offset 기준 |
| `HANDOVER_GRASP_VERTICAL_MAX_M` | `0.07` | z offset 기준 |
| `HANDOVER_GRASP_DEPTH_AXIS_MAX_M` | `0.04` | strict depth 기준 |
| `HANDOVER_GRASP_HAND_TO_FINGERTIP_M` | `0.103` | panda hand에서 fingertip까지 보정 |
| `HANDOVER_GRASP_FINGERTIP_OFFSET_M` | `0.045` | finger pose depth 보정 |

### Takeover/release

| 변수 | 현재 값 | 의미 |
| --- | --- | --- |
| `HANDOVER_ROBOT_FIXED_JOINT_GRASP` | `0` | 기존 panda_hand fixed joint path 비활성 |
| `HANDOVER_ROBOT_FINGERTIP_JOINT_TAKEOVER` | `1` | robot finger custody joint 사용 |
| `HANDOVER_DISABLE_HUMAN_HAND_COLLISION_ON_ROBOT_GRASP` | `1` | robot grasp 중 human hand collision 억제 |
| `HANDOVER_GRASP_VERIFY_STEPS` | `18` | physical verify step |
| `HANDOVER_GRASP_VERIFY_TIMEOUT_STEPS` | `90` | verify timeout |
| `HANDOVER_GRASP_MAX_RETRIES` | `6` | oriented grasp retry 횟수 |

### Carry/place

| 변수 | 현재 값 | 의미 |
| --- | --- | --- |
| `HANDOVER_PHYSICAL_LIFT_M` | `0.14` | lift 단계 상승량 |
| `HANDOVER_PHYSICAL_PLACE_ABOVE_M` | `0.04` | release 목표의 place 위 높이 |
| `HANDOVER_PHYSICAL_RELEASE_HOLD_STEPS` | `30` | release 단계 open 유지 step |
| `HANDOVER_PHYSICAL_CARRY_LOST_STEPS` | `18` | carry lost 허용 step |
| `HANDOVER_PHYSICAL_CARRY_LOST_CENTER_M` | `0.12` | carry 중 cube-center 이탈 기준 |

## 16. 해결한 문제들

## 16.1 그리퍼가 cube 근처까지 가지만 손에서 못 떼어냄

원인:

- human fixed joint가 남아 있었다.
- 또는 `grab_manager.grabbed[hand]` state가 release되지 않았다.
- 또는 release path 자체가 strict attach 실패 때문에 호출되지 않았다.

해결:

- human joint scan/remove 구현
- `grab_manager.force_release(cube=...)` 호출
- `_release(hand)` 직접 fallback
- `prev_grip[hand] = False`
- re-grab suppression
- strict 실패 시 loose/force release 허용

## 16.2 gripper가 닫혔는데 잡은 것으로 처리하지 못함

원인:

- strict condition이 너무 좁았다.
- 특히 `depth_offset`과 `head_dist`가 정상 grasp를 막았다.

해결:

- close/attach에서 force detach 경로 추가
- `head_dist`를 strict success 조건에서 제거
- finger 기반 metric 우선
- `handover_close_capture_ready` 추가

## 16.3 cube가 robot head/palm 쪽에 붙는 것처럼 보임

원인:

- 기존 fixed joint body가 `/World/Franka/panda_hand` 중심이었다.

해결:

- nearest finger body를 선택한다.
- `/World/Franka/panda_leftfinger` 또는 `/World/Franka/panda_rightfinger`에 custody joint를 만든다.
- joint prefix를 `robot_finger`로 분리했다.

## 16.4 `attach_timeout` 반복

원인:

- strict attach가 실패했는데 physical detach 경로가 막히는 경우가 있었다.
- 특히 `HANDOVER_ROBOT_FIXED_JOINT_GRASP=1`이면 기존 fixed joint path가 우선되었다.

해결:

- `_use_fingertip_takeover_path()` 추가
- fingertip takeover가 켜져 있으면 fixed joint grasp 설정과 무관하게 physical/fingertip path 우선
- `attach_detach_decision` 로그 추가

## 16.5 손 proxy가 커서 gripper와 충돌

원인:

- 사람 손 physical proxy와 gripper/finger/panda link가 가까운 상황에서 충돌이 발생했다.

해결:

- robot grasp takeover 중 active hand collision suppression
- grasp 후 place complete 또는 reset 때 복구

## 16.6 로그가 너무 많음

원인:

- `orient_pregrasp steps=N`가 매 step 출력됐다.

해결:

- terminal status message filter 추가
- 해당 로그만 숨기고 나머지 로그는 유지

## 17. 현재 성공 기준

현재 성공으로 보는 조건은 다음과 같다.

1. 사람이 cube를 잡는다.
2. task가 active cube로 adopt한다.
3. 손 안정성 판단 후 `transfer_ready`가 발생한다.
4. robot takeover가 시작된다.
5. gripper가 close한다.
6. `physical grasp release started` 또는 `fingertip custody joint takeover`가 발생한다.
7. human fixed joint가 제거된다.
8. `physical grasp verified`가 발생한다.
9. physical carry stage가 `lift -> carry -> descend -> release`로 진행된다.
10. `handover_physical_place_complete`와 `handover_place_complete`가 발생한다.

최근 성공 로그:

```text
[ErrP] handover_physical_place_complete | cube=cube_1,place=0.600,-0.250,0.976
[ErrP] handover_place_complete | cube=cube_1,place=0.600,-0.250,0.976
[Handover] robot_place_done cube=cube_1
```

## 18. 현재 구현의 한계

## 18.1 학습 기반이 아니다

현재는 distance/geometry/FSM 기반이다. 상황이 바뀌어도 로봇이 스스로 정책을 학습하지 않는다. threshold와 rule을 사람이 조정해야 한다.

## 18.2 완전한 순수 물리 grasp는 아니다

사람 손은 fixed joint로 cube를 잡고, 로봇 takeover도 최종 custody 안정성을 위해 fingertip fixed joint를 사용한다. 그리퍼가 실제로 닫히고 finger 기준 geometry를 만족해야 release가 시작되지만, 마지막 carry 안정성은 custody joint에 의존한다.

즉 “실제 세계처럼 마찰과 접촉력만으로 들고 가는 grasp”는 아니다. Isaac Sim에서 안정적인 VR handover 실험을 위해 physical + fixed joint custody를 혼합한 방식이다.

## 18.3 top-down 접근이 기본이다

side grasp 코드가 있지만 현재 `HANDOVER_SIDE_GRASP_WHEN_HAND_BELOW=0`이다. 손이 cube 아래에 있는 경우 side/tilted grasp를 더 적극적으로 쓰려면 이 값을 켜고 추가 튜닝해야 한다.

## 18.4 자동 테스트가 제한적이다

`python3 -m compileall -q .`와 `SIM_MAX_STEPS` smoke는 가능하지만, 실제 VR handover 성공 여부는 수동 VR 실행에서 확인해야 한다.

## 19. 검증한 항목

수정 과정에서 반복적으로 다음을 확인했다.

```bash
python3 -m py_compile simulation/task/handover_task.py
python3 -m compileall -q .
SIM_MAX_STEPS=60 ... /home/railab/isaac-sim-4.5.0/python.sh main.py
SIM_MAX_STEPS=80 ... /home/railab/isaac-sim-4.5.0/python.sh main.py
```

Smoke test는 VR handover 자체를 완전히 재현하지는 못하지만, Isaac app boot, scene setup, Panda setup, HandoverTaskManager 초기화, env parsing, syntax/runtime import 문제를 확인하는 데 사용했다.

## 20. 앞으로 개선할 수 있는 방향

### 20.1 Contact sensor 기반 grasp 검증

현재는 finger pose와 cube pose의 기하 조건으로 grasp를 판단한다. 더 실제 물리와 가깝게 하려면 finger-cube contact sensor 또는 PhysX contact report 기반으로 left/right finger contact를 확인하는 방식이 좋다.

### 20.2 Pure physical grasp 실험

fixed joint custody를 줄이고 마찰/접촉만으로 들도록 하려면 다음이 필요하다.

- gripper finger collider 형상 개선
- cube material friction 튜닝
- contact solver iteration 증가
- gripper closing force/velocity 튜닝
- 손 proxy collision/size 재튜닝
- cube mass/damping 튜닝

### 20.3 Side grasp 활성화

손이 cube 아래에 있을 때 top-down으로만 접근하면 손과 충돌하거나 cube를 놓치기 쉽다. `HANDOVER_SIDE_GRASP_WHEN_HAND_BELOW=1`을 켜고 side/upper-side 접근을 더 튜닝할 수 있다.

### 20.4 실패 로그 자동 분류

현재 로그는 사람이 읽기 좋게 되어 있지만, 장기 실험에서는 실패 원인을 자동 집계하면 좋다.

예:

- `approach_not_ready`
- `head_clearance_lost`
- `attach_timeout`
- `human_still_holding`
- `cube_dropped`
- `physical_carry_failed`

### 20.5 학습 기반 정책으로 확장

현재 FSM이 충분히 안정화되었으므로, 이후에는 다음 중 하나로 확장할 수 있다.

- FSM을 expert policy로 사용해 demonstration 수집
- behavior cloning으로 grasp pose selection 학습
- RL로 approach pose 또는 release timing 학습
- hand/cube relative pose를 observation으로 하는 policy 구성

단, 현재 구현은 학습 전 baseline이자 data collection scaffold로 보는 것이 적절하다.

## 21. 디버깅 시 확인할 핵심 로그

정상 흐름:

```text
[Handover] human_holding hand=right cube=cube_0
[ErrP] handover_transfer_ready | ...
[Handover] robot_takeover_started cube=cube_0
[ErrP] handover_oriented_grasp_plan | ...
[ErrP] handover_close_capture_ready | ...
[Handover] fingertip custody joint takeover cube=...
[Handover] human fixed joint force-removed: ...
[Handover] physical grasp release started cube=...
[Handover] physical grasp verified cube=...
[Handover] physical carry stage=lift
[Handover] physical carry stage=carry
[Handover] physical carry stage=descend
[Handover] physical carry stage=release
[ErrP] handover_physical_place_complete | ...
[Handover] robot_place_done cube=...
```

문제가 있을 때 봐야 하는 로그:

```text
[Handover] attach_detach_decision fixed_joint=... fingertip_path=... strict=... loose=... force=...
[Handover] waiting_oriented_attach ...
[Handover] waiting_force_detach_ready ...
[Handover] grasp verify blocked: cube still held by ...
[Handover] human release still held by ...
[Handover] physical grasp failed: ...
[Handover] physical carry failed: ...
```

특히 다음 값이 중요하다.

- `fingertip_path=True`인지
- `strict`, `loose`, `force` 중 하나라도 true인지
- `human fixed joint force-removed`가 찍히는지
- `fingertip custody joint takeover`가 찍히는지
- `physical grasp verified`가 찍히는지

## 22. 결론

현재 handover 구현은 사람이 VR 손으로 잡은 cube를 로봇이 거리/기하 기반으로 인식하고, oriented grasp FSM으로 접근해, gripper close 후 robot fingertip custody joint를 생성하고, human fixed joint를 제거한 뒤, physical carry stage로 목표 위치에 놓는 구조다.

핵심적으로 해결한 문제는 다음이다.

- 로봇이 cube 근처까지 가지만 손에서 못 떼어내던 문제
- gripper가 닫혔는데도 release가 시작되지 않던 문제
- cube가 panda hand/head 쪽에 붙는 것처럼 보이던 문제
- `head_clearance_lost`가 정상 grasp를 방해하던 문제
- `attach_timeout`으로 계속 retry하던 문제
- 사람 손 proxy 충돌로 grasp가 튀던 문제
- 불필요한 pregrasp step log spam

현재 성공 로그상으로는 handover complete와 place complete까지 정상적으로 도달했다. 다음 단계는 순수 물리 grasp 품질 향상, side grasp 활성화, contact 기반 검증, 학습 기반 정책 확장 중 무엇을 목표로 할지 정하는 것이다.
