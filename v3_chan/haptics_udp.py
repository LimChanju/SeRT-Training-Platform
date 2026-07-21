import json
import os
import socket
import time
from typing import Optional

DEBUG_HAPTICS_UDP = os.environ.get("DEBUG_HAPTICS_UDP", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


class HapticsUdpClient:
    def __init__(
        self,
        host: str,
        port: int = 5005,
        min_interval: float = 0.15,
        enabled: bool = True,
    ) -> None:
        self._addr: Optional[tuple[str, int]] = (host, port) if host else None
        self._min_interval = min_interval
        self._last_send_at: dict[str, float] = {}
        self._sock: Optional[socket.socket] = None
        self.enabled = enabled and self._addr is not None

        if self.enabled:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setblocking(False)
            print(f"[HapticsUDP] enabled -> {host}:{port}")
        else:
            print("[HapticsUDP] disabled. Set BHAPTICS_NOTEBOOK_IP to enable.")

    def pulse(self, intensity: int = 100, hand: str = "right", event: str = "collision") -> bool:
        if not self.enabled or self._sock is None or self._addr is None:
            return False

        now = time.monotonic()
        key = f"{event}:{hand}"
        if now - self._last_send_at.get(key, 0.0) < self._min_interval:
            return False

        value = max(0, min(100, int(intensity)))
        payload = {
            "event": event,
            "hand": hand,
            "intensity": value,
        }
        try:
            self._sock.sendto(json.dumps(payload).encode("utf-8"), self._addr)
            self._last_send_at[key] = now
            if DEBUG_HAPTICS_UDP:
                print(f"[HapticsUDP] sent {payload} -> {self._addr}")
            return True
        except OSError as exc:
            print(f"[HapticsUDP] send failed: {exc}")
            return False

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
