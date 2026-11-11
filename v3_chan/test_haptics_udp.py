"""Send explicit bHaptics UDP test pulses without starting Isaac Sim."""

from __future__ import annotations

import argparse
import os
import sys
import time

from haptics_udp import HapticsUdpClient


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _interval(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send manual bHaptics UDP test pulses through the Windows bridge."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("BHAPTICS_NOTEBOOK_IP", "10.3.129.185"),
        help="Windows bridge IP address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("BHAPTICS_UDP_PORT", "5005")),
        help="Windows bridge UDP port",
    )
    parser.add_argument(
        "--hand",
        choices=("left", "right", "both"),
        default="both",
        help="glove to test",
    )
    parser.add_argument(
        "--pulses",
        type=_positive_int,
        default=2,
        help="number of pulses per selected hand",
    )
    parser.add_argument(
        "--intensity",
        type=int,
        default=int(os.environ.get("BHAPTICS_INTENSITY", "100")),
        help="requested bridge intensity from 0 to 100",
    )
    parser.add_argument(
        "--interval",
        type=_interval,
        default=0.4,
        help="seconds between pulse requests",
    )
    args = parser.parse_args()

    if not args.host:
        print("[HapticsTest] missing Windows bridge IP", file=sys.stderr)
        return 2

    hands = ("left", "right") if args.hand == "both" else (args.hand,)
    client = HapticsUdpClient(args.host, args.port, min_interval=0.0)
    if not client.enabled:
        print("[HapticsTest] UDP client is disabled", file=sys.stderr)
        return 2

    try:
        for pulse_index in range(1, args.pulses + 1):
            for hand in hands:
                sent = client.pulse(
                    args.intensity,
                    hand=hand,
                    event="manual_haptics_test",
                )
                print(
                    f"[HapticsTest] pulse={pulse_index}/{args.pulses} "
                    f"hand={hand} udp_send_success={int(sent)}",
                    flush=True,
                )
                if args.interval > 0.0:
                    time.sleep(args.interval)
    finally:
        client.close()

    print(
        "[HapticsTest] completed. UDP success confirms only local sendto(); "
        "verify the physical glove vibration directly.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
