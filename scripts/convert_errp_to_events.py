import argparse
import csv
from datetime import datetime, timedelta, timezone


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iso_from_offset(start_dt: datetime, seconds: float) -> str:
    ts = start_dt + timedelta(seconds=seconds)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="v2/errp_markers.csv")
    parser.add_argument("--output", default="metrics/session_events.csv")
    parser.add_argument("--start-iso", default=None)
    parser.add_argument("--emit-session-success", action="store_true")
    args = parser.parse_args()

    if args.start_iso:
        start_dt = datetime.fromisoformat(args.start_iso.replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
    else:
        start_dt = datetime.now(timezone.utc)

    rows = []
    with open(args.input, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sim_time = _parse_float(row.get("sim_time", "0"))
            event = row.get("event", "").strip()
            details = row.get("details", "").strip()
            if not event:
                continue
            if event not in {"episode_start", "episode_end"}:
                continue
            rows.append(
                {
                    "timestamp": _iso_from_offset(start_dt, sim_time),
                    "event": event,
                    "details": details,
                }
            )
            if args.emit_session_success and event == "episode_end":
                rows.append(
                    {
                        "timestamp": _iso_from_offset(start_dt, sim_time),
                        "event": "session_success",
                        "details": details or "source=episode_end",
                    }
                )

    rows.sort(key=lambda r: r["timestamp"])

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "event", "details"])
        for row in rows:
            writer.writerow([row["timestamp"], row["event"], row["details"]])


if __name__ == "__main__":
    main()
