import json
import socket
import time
from typing import Optional

import numpy as np


HAND_JOINT_NAMES = (
    "XR_HAND_JOINT_WRIST_EXT",
    "XR_HAND_JOINT_PALM_EXT",
    "XR_HAND_JOINT_THUMB_METACARPAL_EXT",
    "XR_HAND_JOINT_THUMB_PROXIMAL_EXT",
    "XR_HAND_JOINT_THUMB_DISTAL_EXT",
    "XR_HAND_JOINT_THUMB_TIP_EXT",
    "XR_HAND_JOINT_INDEX_METACARPAL_EXT",
    "XR_HAND_JOINT_INDEX_PROXIMAL_EXT",
    "XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT",
    "XR_HAND_JOINT_INDEX_DISTAL_EXT",
    "XR_HAND_JOINT_INDEX_TIP_EXT",
    "XR_HAND_JOINT_MIDDLE_METACARPAL_EXT",
    "XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT",
    "XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT",
    "XR_HAND_JOINT_MIDDLE_DISTAL_EXT",
    "XR_HAND_JOINT_MIDDLE_TIP_EXT",
    "XR_HAND_JOINT_RING_METACARPAL_EXT",
    "XR_HAND_JOINT_RING_PROXIMAL_EXT",
    "XR_HAND_JOINT_RING_INTERMEDIATE_EXT",
    "XR_HAND_JOINT_RING_DISTAL_EXT",
    "XR_HAND_JOINT_RING_TIP_EXT",
    "XR_HAND_JOINT_LITTLE_METACARPAL_EXT",
    "XR_HAND_JOINT_LITTLE_PROXIMAL_EXT",
    "XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT",
    "XR_HAND_JOINT_LITTLE_DISTAL_EXT",
    "XR_HAND_JOINT_LITTLE_TIP_EXT",
)

JOINT_ALIASES = {
    "wrist": "XR_HAND_JOINT_WRIST_EXT",
    "palm": "XR_HAND_JOINT_PALM_EXT",
    "thumb_metacarpal": "XR_HAND_JOINT_THUMB_METACARPAL_EXT",
    "thumb_proximal": "XR_HAND_JOINT_THUMB_PROXIMAL_EXT",
    "thumb_distal": "XR_HAND_JOINT_THUMB_DISTAL_EXT",
    "thumb_tip": "XR_HAND_JOINT_THUMB_TIP_EXT",
    "index_metacarpal": "XR_HAND_JOINT_INDEX_METACARPAL_EXT",
    "index_proximal": "XR_HAND_JOINT_INDEX_PROXIMAL_EXT",
    "index_intermediate": "XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT",
    "index_distal": "XR_HAND_JOINT_INDEX_DISTAL_EXT",
    "index_tip": "XR_HAND_JOINT_INDEX_TIP_EXT",
    "middle_metacarpal": "XR_HAND_JOINT_MIDDLE_METACARPAL_EXT",
    "middle_proximal": "XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT",
    "middle_intermediate": "XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT",
    "middle_distal": "XR_HAND_JOINT_MIDDLE_DISTAL_EXT",
    "middle_tip": "XR_HAND_JOINT_MIDDLE_TIP_EXT",
    "ring_metacarpal": "XR_HAND_JOINT_RING_METACARPAL_EXT",
    "ring_proximal": "XR_HAND_JOINT_RING_PROXIMAL_EXT",
    "ring_intermediate": "XR_HAND_JOINT_RING_INTERMEDIATE_EXT",
    "ring_distal": "XR_HAND_JOINT_RING_DISTAL_EXT",
    "ring_tip": "XR_HAND_JOINT_RING_TIP_EXT",
    "little_metacarpal": "XR_HAND_JOINT_LITTLE_METACARPAL_EXT",
    "little_proximal": "XR_HAND_JOINT_LITTLE_PROXIMAL_EXT",
    "little_intermediate": "XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT",
    "little_distal": "XR_HAND_JOINT_LITTLE_DISTAL_EXT",
    "little_tip": "XR_HAND_JOINT_LITTLE_TIP_EXT",
}


def _canonical_joint_name(name: str) -> "str | None":
    key = str(name).strip()
    if not key:
        return None
    upper = key.upper()
    if upper in HAND_JOINT_NAMES:
        return upper
    lowered = key.lower()
    if lowered.startswith("xr_hand_joint_") and lowered.endswith("_ext"):
        return upper
    return JOINT_ALIASES.get(lowered)


class HandTrackingReceiver:
    """
    UDP receiver for Quest/ALVR hand-tracking bridge data.

    Expected JSON examples:
      {"right":{"index_tip":[x,y,z],"thumb_tip":[x,y,z]}}
      {"hand":"right","index_tip":[x,y,z],"thumb_tip":[x,y,z]}
      {"left":{"joints":{"palm":[x,y,z],"index_tip":[x,y,z]}}}
      {"hand":"left","joints":[[x,y,z], ... 26 OpenXR ordered joints ...]}
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        enabled: bool = True,
        stale_after: float = 0.25,
    ) -> None:
        self._sock: Optional[socket.socket] = None
        self._hands: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
        self._last_seen = {"left": 0.0, "right": 0.0}
        self._stale_after = stale_after
        self._format_logged = False
        self.enabled = enabled

        if not enabled:
            return

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((host, port))
            self._sock.setblocking(False)
            print(f"[HandTracking] listening on {host}:{port}")
        except OSError as exc:
            self.enabled = False
            self._sock = None
            print(f"[HandTracking] disabled: {exc}")

    def poll(self) -> None:
        if not self.enabled or self._sock is None:
            return

        while True:
            try:
                data, _ = self._sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError as exc:
                print(f"[HandTracking] recv failed: {exc}")
                break

            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            self._log_packet_format(payload)
            self._apply_payload(payload)

    def get_pinch_points(self) -> dict[str, dict[str, np.ndarray]]:
        now = time.monotonic()
        out = {}
        for hand, points in self._hands.items():
            if now - self._last_seen[hand] > self._stale_after:
                continue
            index_tip = self._point(points, "index_tip")
            thumb_tip = self._point(points, "thumb_tip")
            if index_tip is None or thumb_tip is None:
                continue
            out[hand] = {
                "index_tip": index_tip.copy(),
                "thumb_tip": thumb_tip.copy(),
            }
        return out

    def get_hand_joint_positions(self) -> dict[str, dict[str, np.ndarray]]:
        now = time.monotonic()
        return {
            hand: {name: pos.copy() for name, pos in points.items()}
            for hand, points in self._hands.items()
            if now - self._last_seen[hand] <= self._stale_after
        }

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _apply_payload(self, payload: dict) -> None:
        for hand in ("left", "right"):
            if hand in payload and isinstance(payload[hand], dict):
                self._update_hand(hand, payload[hand])

        hand = payload.get("hand")
        if hand in ("left", "right"):
            self._update_hand(hand, payload)

    def _log_packet_format(self, payload) -> None:
        if self._format_logged:
            return
        self._format_logged = True

        if not isinstance(payload, dict):
            print(f"[HandTracking] first packet format=unsupported root={type(payload).__name__}")
            print(f"[HandTracking] first packet raw={payload}")
            return

        nested = any(
            hand in payload
            and isinstance(payload[hand], dict)
            and (
                "joints" in payload[hand]
                or "index_tip" in payload[hand]
                or "XR_HAND_JOINT_INDEX_TIP_EXT" in payload[hand]
            )
            for hand in ("left", "right")
        )
        flat = (
            payload.get("hand") in ("left", "right")
            and (
                "joints" in payload
                or "index_tip" in payload
                or "XR_HAND_JOINT_INDEX_TIP_EXT" in payload
            )
        )

        if nested and flat:
            fmt = "both nested-hand and flat-hand"
        elif nested:
            fmt = "nested-hand"
        elif flat:
            fmt = "flat-hand"
        else:
            fmt = "unknown"

        print(f"[HandTracking] first packet format={fmt}")
        print(f"[HandTracking] first packet raw={payload}")

    def _update_hand(self, hand: str, payload: dict) -> None:
        updated = False
        joints = payload.get("joints")
        if isinstance(joints, dict):
            for key, value in joints.items():
                updated = self._set_joint(hand, key, value) or updated
        elif isinstance(joints, list):
            for idx, value in enumerate(joints[: len(HAND_JOINT_NAMES)]):
                updated = self._set_joint(hand, HAND_JOINT_NAMES[idx], value) or updated

        for key, value in payload.items():
            if key in ("hand", "joints", "timestamp", "time", "frame"):
                continue
            updated = self._set_joint(hand, key, value) or updated

        if updated:
            self._last_seen[hand] = time.monotonic()

    def _set_joint(self, hand: str, key: str, value) -> bool:
        joint_name = _canonical_joint_name(key)
        if joint_name is None:
            return False
        try:
            arr = np.array(value, dtype=float)
        except Exception:
            return False
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            return False
        self._hands[hand][joint_name] = arr
        return True

    def _point(self, points: dict[str, np.ndarray], name: str) -> "np.ndarray | None":
        canonical = _canonical_joint_name(name)
        if canonical is None:
            return None
        return points.get(canonical)
