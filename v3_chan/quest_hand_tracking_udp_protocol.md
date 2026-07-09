```json
{
  "left": {
    "joints": {
      "palm": [0.55, 0.10, 1.20],
      "wrist": [0.60, 0.10, 1.15],
      "thumb_tip": [0.50, 0.15, 1.22],
      "index_tip": [0.48, 0.11, 1.24],
      "middle_tip": [0.47, 0.10, 1.25],
      "ring_tip": [0.48, 0.08, 1.24],
      "little_tip": [0.50, 0.06, 1.22]
    }
  },
  "right": {
    "joints": {
      "palm": [0.55, -0.10, 1.20],
      "wrist": [0.60, -0.10, 1.15],
      "thumb_tip": [0.50, -0.15, 1.22],
      "index_tip": [0.48, -0.11, 1.24]
    }
  }
}
```

```json
{
  "hand": "right",
  "joints": [
    [0.60, -0.10, 1.15],
    [0.55, -0.10, 1.20],
    [0.54, -0.13, 1.20],
    [0.52, -0.14, 1.21],
    [0.51, -0.145, 1.215],
    [0.50, -0.15, 1.22]
  ]
}
```

좌표는 첫 단계에서는 Isaac world 좌표 `[x, y, z]` meters로 보낸다고 가정한다.

UDP target:

```text
Linux PC IP: <linux-ip>
Port: 5555
Encoding: UTF-8 JSON
Rate: 30-90 Hz
```

지원하는 joint 이름:

```text
wrist, palm,
thumb_metacarpal, thumb_proximal, thumb_distal, thumb_tip,
index_metacarpal, index_proximal, index_intermediate, index_distal, index_tip,
middle_metacarpal, middle_proximal, middle_intermediate, middle_distal, middle_tip,
ring_metacarpal, ring_proximal, ring_intermediate, ring_distal, ring_tip,
little_metacarpal, little_proximal, little_intermediate, little_distal, little_tip
```

OpenXR 이름도 그대로 가능:

```text
XR_HAND_JOINT_INDEX_TIP_EXT
XR_HAND_JOINT_THUMB_TIP_EXT
```

수신 확인 로그:

```text
[HandTracking] listening on 0.0.0.0:5555
[HandTracking] first packet format=...
[Avatar] External left hand joints active: ...
```
