```bash
cd /home/railab/Desktop/Isaac_HRC

HRI_PARTICIPANT_ID=P01 \
HRI_PARTICIPANT_SESSION_INDEX=1 \
HRI_PARTICIPANT_HANDEDNESS=right \
HRI_IS_PRACTICE=0 \
HRI_EXPERIMENT_CONDITION=haptic_on_contact_multispeed_grasp_stable_v5 \
HRI_EXPERIMENT_BLOCK_ID=block_01 \
HRI_HAPTIC_CONDITION=on \
BHAPTICS_ENABLED=1 \
HRI_COLLECTION_LABEL=surface_twist_v8_dualclock_tracked_grasp_gate_v5 \
PICK_PLACE_GRASP_GATE_ENABLED=1 \
PICK_PLACE_GRASP_GATE_DISTANCE_M=0.075 \
PICK_PLACE_GRASP_GATE_CONSECUTIVE_STEPS=3 \
PICK_PLACE_GRASP_GATE_TIMEOUT_S=3.0 \
PICK_PLACE_MAX_ATTEMPTS_PER_CUBE=3 \
HRI_PROTOCOL_VERSION=errp_hri_collection_multispeed_grasp_stable_v5 \
HRI_ROOM_CALIBRATION_ID=vr_room_to_isaac_world_v1 \
bash v3_chan/run_pick_place.sh
```

`HRI_PARTICIPANT_SESSION_INDEX`는 P01의 누적 수집 번호로 매 session마다 증가시키고, `HRI_PARTICIPANT_HANDEDNESS`는 실제 self-reported 값으로 바꾼다. `HRI_IS_PRACTICE`, block, 햅틱 조건은 생략할 수 없으며 `HRI_HAPTIC_CONDITION=on/off`와 `BHAPTICS_ENABLED=1/0`은 서로 일치해야 한다. 한 실행은 `slow`, `medium`, `fast` 조건을 각각 한 episode씩 수집한다. 닫힘 직전에는 controller가 실제로 제어하는 `panda_rightfinger` end-effector 기준점이 cube center의 7.5 cm 이내에서 3 physics step 연속 유지되어야 하고, 3초 안에 도달하지 못하면 해당 attempt를 실패로 처리한다. 같은 cube의 실패는 최대 3회까지만 허용한다. HDF5 validator가 schema, metadata, timestamp, layout, 양손 pose 유효율 90%, RTF 유효율 95%, XR anchor 적용을 통과한 session만 다음 counterbalanced speed 순서로 자동 진행한다.

기본 파일 label은 `surface_twist_v8_dualclock_tracked_grasp_gate_v5`이다. fast grasp approach를 단축했던 v4 및 그 이전 guard 없는 `surface_twist_v8_dualclock_tracked_v1`, finger-link-origin gate의 `grasp_gate_v2`, controller-reference 7.0 cm gate의 `grasp_gate_v3` 파일과 학습 데이터로 섞지 않는다.
