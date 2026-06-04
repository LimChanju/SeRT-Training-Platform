import json
import socket
import time
from typing import Optional

import numpy as np


class HandTrackingReceiver:
    """
    UDP receiver for Quest/ALVR hand-tracking bridge data.

    Expected JSON examples:
      {"right":{"index_tip":[x,y,z],"thumb_tip":[x,y,z]}}
      {"hand":"right","index_tip":[x,y,z],"thumb_tip":[x,y,z]}
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        enabled: bool = True,
        stale_after: float = 0.25,
    ) -> None:
        self._sock: Optional[socket.socket] = None
        self._hands = {"left": {}, "right": {}}
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
        return {
            hand: points.copy()
            for hand, points in self._hands.items()
            if now - self._last_seen[hand] <= self._stale_after
            if "index_tip" in points and "thumb_tip" in points
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
            and "index_tip" in payload[hand]
            and "thumb_tip" in payload[hand]
            for hand in ("left", "right")
        )
        flat = (
            payload.get("hand") in ("left", "right")
            and "index_tip" in payload
            and "thumb_tip" in payload
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
        for key in ("index_tip", "thumb_tip"):
            value = payload.get(key)
            if value is None:
                continue
            arr = np.array(value, dtype=float)
            if arr.shape == (3,) and np.all(np.isfinite(arr)):
                self._hands[hand][key] = arr
                self._last_seen[hand] = time.monotonic()
