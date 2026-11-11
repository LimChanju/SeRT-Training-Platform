import argparse
import asyncio
import inspect
import json
import os


KEYWORDS = (
    "pose",
    "pos",
    "position",
    "track",
    "tracking",
    "rotation",
    "orientation",
    "quat",
    "imu",
    "sensor",
    "device",
    "glove",
    "hand",
)


async def _maybe_call(name, fn):
    try:
        if inspect.iscoroutinefunction(fn):
            return await fn()
        return fn()
    except TypeError as exc:
        return f"<needs args: {exc}>"
    except Exception as exc:
        return f"<error: {type(exc).__name__}: {exc}>"


async def main():
    parser = argparse.ArgumentParser(
        description="Inspect installed bhaptics_python APIs for pose/tracking support."
    )
    parser.add_argument(
        "--call-candidates",
        action="store_true",
        help="Try zero-argument candidate functions and print their result.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize bHaptics SDK first using BHAPTICS_APP_ID/BHAPTICS_API_KEY.",
    )
    args = parser.parse_args()

    import bhaptics_python

    print(f"module={bhaptics_python!r}")
    print(f"file={getattr(bhaptics_python, '__file__', '<unknown>')}")

    names = [name for name in dir(bhaptics_python) if not name.startswith("__")]
    print("\n== All public names ==")
    for name in names:
        obj = getattr(bhaptics_python, name)
        kind = "async" if inspect.iscoroutinefunction(obj) else type(obj).__name__
        print(f"{name}: {kind}")

    candidates = [
        name
        for name in names
        if any(keyword in name.lower() for keyword in KEYWORDS)
    ]
    print("\n== Pose/tracking/device-like candidates ==")
    for name in candidates:
        obj = getattr(bhaptics_python, name)
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(signature unavailable)"
        kind = "async" if inspect.iscoroutinefunction(obj) else type(obj).__name__
        print(f"{name}{sig}: {kind}")

    if args.init:
        app_id = os.environ.get("BHAPTICS_APP_ID", "")
        api_key = os.environ.get("BHAPTICS_API_KEY", "")
        if not app_id or not api_key:
            raise SystemExit("Set BHAPTICS_APP_ID and BHAPTICS_API_KEY before --init.")
        result = await bhaptics_python.registry_and_initialize(app_id, api_key, "")
        print(f"\ninitialized={result}")

    if hasattr(bhaptics_python, "get_device_info_json"):
        print("\n== get_device_info_json ==")
        info = await _maybe_call(
            "get_device_info_json",
            getattr(bhaptics_python, "get_device_info_json"),
        )
        if isinstance(info, str):
            try:
                print(json.dumps(json.loads(info), indent=2))
            except Exception:
                print(info)
        else:
            print(info)

    if args.call_candidates:
        print("\n== Calling zero-argument candidates ==")
        for name in candidates:
            obj = getattr(bhaptics_python, name)
            if not callable(obj):
                continue
            result = await _maybe_call(name, obj)
            print(f"{name} -> {result}")

    if args.init and hasattr(bhaptics_python, "close"):
        await bhaptics_python.close()


if __name__ == "__main__":
    asyncio.run(main())
