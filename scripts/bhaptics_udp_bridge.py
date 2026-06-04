import argparse
import asyncio
import json
import os
import socket


POSITION_BY_HAND = {
    "left": 8,
    "right": 9,
}

GLOVE_MOTOR_COUNT = 8


def _clamp_intensity(value: int) -> int:
    return max(0, min(100, int(value)))


def _parse_packet(data: bytes, default_hand: str) -> tuple[str, int]:
    text = data.decode("utf-8").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return default_hand, _clamp_intensity(int(float(text)))

    if isinstance(payload, (int, float)):
        return default_hand, _clamp_intensity(int(payload))
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported packet type: {type(payload).__name__}")

    hand = str(payload.get("hand", default_hand)).lower()
    intensity = int(payload.get("intensity", 100))
    if hand not in POSITION_BY_HAND:
        hand = default_hand
    return hand, _clamp_intensity(intensity)


async def _play_glove(
    bhaptics_python,
    hand: str,
    intensity: int,
    playtime: int,
    shape: int,
    burst_count: int,
    burst_interval: float,
) -> None:
    position = POSITION_BY_HAND[hand]
    motors = [intensity] * GLOVE_MOTOR_COUNT
    playtimes = [playtime] * GLOVE_MOTOR_COUNT
    shapes = [shape] * GLOVE_MOTOR_COUNT
    for _ in range(max(1, burst_count)):
        await bhaptics_python.play_glove(position, motors, playtimes, shapes, 0)
        await asyncio.sleep(max(0.0, burst_interval))


async def main() -> None:
    parser = argparse.ArgumentParser(description="UDP bridge from Isaac/ALVR to bHaptics TactGlove.")
    parser.add_argument("--host", default="", help="Bind host. Empty string listens on all interfaces.")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--playtime", type=int, default=8, help="bHaptics playtime code. 8 is 40 ms.")
    parser.add_argument("--shape", type=int, default=2, help="0 constant, 1 falling, 2 rising.")
    parser.add_argument("--burst-count", type=int, default=5, help="Repeat glove pulses per UDP packet.")
    parser.add_argument("--burst-interval", type=float, default=0.04, help="Seconds between repeated glove pulses.")
    args = parser.parse_args()

    app_id = os.environ.get("BHAPTICS_APP_ID", "")
    api_key = os.environ.get("BHAPTICS_API_KEY", "")
    if not app_id or not api_key:
        raise SystemExit("Set BHAPTICS_APP_ID and BHAPTICS_API_KEY before running this bridge.")

    import bhaptics_python

    result = await bhaptics_python.registry_and_initialize(app_id, api_key, "")
    if not result:
        raise SystemExit("Failed to initialize bHaptics SDK. Check bHaptics Player and credentials.")
    print(f"[bHapticsBridge] initialized={result}")
    try:
        print(f"[bHapticsBridge] left connected={await bhaptics_python.is_bhaptics_device_connected(8)}")
        print(f"[bHapticsBridge] right connected={await bhaptics_python.is_bhaptics_device_connected(9)}")
    except Exception as exc:
        print(f"[bHapticsBridge] connection check skipped: {exc}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.setblocking(False)
    print(f"[bHapticsBridge] listening on UDP {args.host or '0.0.0.0'}:{args.port}")

    loop = asyncio.get_running_loop()
    try:
        while True:
            data, addr = await loop.sock_recvfrom(sock, 1024)
            try:
                hand, intensity = _parse_packet(data, args.hand)
            except (ValueError, UnicodeDecodeError) as exc:
                print(f"[bHapticsBridge] bad packet from {addr}: {exc}")
                continue
            print(f"[bHapticsBridge] {hand} intensity={intensity} from {addr}")
            await _play_glove(
                bhaptics_python,
                hand,
                intensity,
                args.playtime,
                args.shape,
                args.burst_count,
                args.burst_interval,
            )
    finally:
        sock.close()
        await bhaptics_python.stop_all()
        await bhaptics_python.close()


if __name__ == "__main__":
    asyncio.run(main())
