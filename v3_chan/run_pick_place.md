```bash
cd /home/railab/Desktop/Isaac_HRC

HRI_PARTICIPANT_ID=P01 \
HRI_PARTICIPANT_SESSION_INDEX=1 \
HRI_PARTICIPANT_HANDEDNESS=right \
HRI_IS_PRACTICE=0 \
HRI_EXPERIMENT_CONDITION=haptic_on_contact_multispeed_v1 \
HRI_EXPERIMENT_BLOCK_ID=block_01 \
HRI_HAPTIC_CONDITION=on \
HRI_PROTOCOL_VERSION=errp_hri_collection_multispeed_v1 \
HRI_ROOM_CALIBRATION_ID=vr_room_to_isaac_world_v1 \
bash v3_chan/run_pick_place.sh
```

`HRI_PARTICIPANT_SESSION_INDEX`는 P01의 누적 수집 번호로 매 session마다 증가시키고, `HRI_PARTICIPANT_HANDEDNESS`는 실제 self-reported 값으로 바꾼다. 한 실행은 `slow`, `medium`, `fast` 조건을 각각 한 episode씩 수집한다. HDF5 validator가 schema, metadata, timestamp, layout을 통과한 session만 다음 counterbalanced speed 순서로 자동 진행한다.
