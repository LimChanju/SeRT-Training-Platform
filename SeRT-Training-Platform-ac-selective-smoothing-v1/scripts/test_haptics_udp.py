import argparse
import json
import socket
import time


def main():
    parser = argparse.ArgumentParser(description="Send test UDP pulses to the bHaptics bridge.")
    parser.add_argument("host", help="Bridge host/IP, usually the Windows notebook IP.")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--intensity", type=int, default=100)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = {
        "event": "manual_test",
        "hand": args.hand,
        "intensity": max(0, min(100, args.intensity)),
    }
    try:
        for idx in range(args.count):
            sock.sendto(json.dumps(payload).encode("utf-8"), (args.host, args.port))
            print(f"sent {idx + 1}/{args.count}: {payload} -> {args.host}:{args.port}")
            time.sleep(args.interval)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
