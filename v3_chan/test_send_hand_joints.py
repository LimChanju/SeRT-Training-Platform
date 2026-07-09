import json
import math
import os
import socket
import time


HOST = os.environ.get("HAND_TRACKING_TEST_HOST", "127.0.0.1")
PORT = int(os.environ.get("HAND_TRACKING_UDP_PORT", "5555"))
RATE_HZ = float(os.environ.get("HAND_TRACKING_TEST_RATE_HZ", "30"))


def hand_packet(t: float) -> dict:
    wiggle = 0.05 * math.sin(t)
    left_palm = [0.72 + wiggle, 0.18, 1.25]
    right_palm = [0.72 - wiggle, -0.18, 1.25]
    return {
        "left": {
            "joints": {
                "palm": left_palm,
                "wrist": [left_palm[0] + 0.04, left_palm[1], left_palm[2] - 0.06],
                "thumb_tip": [left_palm[0] - 0.03, left_palm[1] + 0.055, left_palm[2]],
                "index_tip": [left_palm[0] - 0.08, left_palm[1] + 0.025, left_palm[2] + 0.02],
                "middle_tip": [left_palm[0] - 0.09, left_palm[1], left_palm[2] + 0.025],
                "ring_tip": [left_palm[0] - 0.08, left_palm[1] - 0.025, left_palm[2] + 0.02],
                "little_tip": [left_palm[0] - 0.06, left_palm[1] - 0.05, left_palm[2]],
            }
        },
        "right": {
            "joints": {
                "palm": right_palm,
                "wrist": [right_palm[0] + 0.04, right_palm[1], right_palm[2] - 0.06],
                "thumb_tip": [right_palm[0] - 0.03, right_palm[1] - 0.055, right_palm[2]],
                "index_tip": [right_palm[0] - 0.08, right_palm[1] - 0.025, right_palm[2] + 0.02],
                "middle_tip": [right_palm[0] - 0.09, right_palm[1], right_palm[2] + 0.025],
                "ring_tip": [right_palm[0] - 0.08, right_palm[1] + 0.025, right_palm[2] + 0.02],
                "little_tip": [right_palm[0] - 0.06, right_palm[1] + 0.05, right_palm[2]],
            }
        },
    }


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dt = 1.0 / max(1.0, RATE_HZ)
print(f"[HandTrackingTest] sending to {HOST}:{PORT} at {RATE_HZ:.1f} Hz")
start = time.monotonic()
while True:
    now = time.monotonic()
    payload = json.dumps(hand_packet(now - start)).encode("utf-8")
    sock.sendto(payload, (HOST, PORT))
    time.sleep(dt)
