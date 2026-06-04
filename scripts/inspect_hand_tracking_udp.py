import argparse
import json
import socket


def detect_format(payload):
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
        return "both nested-hand and flat-hand"
    if nested:
        return "nested-hand"
    if flat:
        return "flat-hand"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Print raw Quest/ALVR hand-tracking UDP packets."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))

    print(f"[InspectHandTracking] listening on UDP {args.host}:{args.port}")
    print("[InspectHandTracking] stop with Ctrl+C")

    try:
        while True:
            data, addr = sock.recvfrom(4096)
            text = data.decode("utf-8", errors="replace").strip()
            print(f"\nfrom {addr}:")
            print(text)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"format: invalid json ({exc})")
                continue

            if isinstance(payload, dict):
                print(f"format: {detect_format(payload)}")
            else:
                print(f"format: unsupported json root ({type(payload).__name__})")
    except KeyboardInterrupt:
        print("\n[InspectHandTracking] stopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
