#!/usr/bin/env python3
"""Convert Isaac VR CSV logs into TensorBoard event files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _load_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter
    except ImportError:
        pass

    try:
        from tensorboardX import SummaryWriter

        return SummaryWriter
    except ImportError:
        pass

    raise RuntimeError(
        "TensorBoard writer is not installed. Install one of these first:\n"
        "  python -m pip install tensorboard tensorboardX\n"
        "or\n"
        "  python -m pip install tensorboard torch"
    )


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def sim_time_to_step(sim_time: float, scale: int) -> int:
    return int(round(sim_time * scale))


def write_session_samples(writer, path: Path, step_scale: int) -> int:
    if not path.exists():
        return 0

    rows_written = 0
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sim_time = _float_or_none(row.get("sim_time", ""))
            if sim_time is None:
                continue

            step = _int_or_none(row.get("step", "")) or sim_time_to_step(sim_time, step_scale)
            left = _float_or_none(row.get("left_hand_gripper_dist_m", ""))
            right = _float_or_none(row.get("right_hand_gripper_dist_m", ""))
            minimum = _float_or_none(row.get("min_hand_gripper_dist_m", ""))
            collision = _float_or_none(row.get("human_robot_collision", ""))

            writer.add_scalar("session/sim_time_sec", sim_time, step)
            if left is not None:
                writer.add_scalar("distance/left_hand_gripper_m", left, step)
            if right is not None:
                writer.add_scalar("distance/right_hand_gripper_m", right, step)
            if minimum is not None:
                writer.add_scalar("distance/min_hand_gripper_m", minimum, step)
            if collision is not None:
                writer.add_scalar("collision/human_robot_collision", collision, step)

            rows_written += 1

    return rows_written


def write_errp_markers(writer, path: Path, step_scale: int) -> int:
    if not path.exists():
        return 0

    event_counts: dict[str, int] = defaultdict(int)
    rows_written = 0
    event_lines = []

    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sim_time = _float_or_none(row.get("sim_time", ""))
            event = (row.get("event") or "unknown").strip() or "unknown"
            details = (row.get("details") or "").strip()
            if sim_time is None:
                continue

            step = sim_time_to_step(sim_time, step_scale)
            event_counts[event] += 1

            writer.add_scalar("events/all_markers", 1.0, step)
            writer.add_scalar(f"events/{event}", 1.0, step)
            writer.add_scalar(f"events_cumulative/{event}", event_counts[event], step)

            if rows_written < 200:
                event_lines.append(f"| {sim_time:.3f} | `{event}` | `{details}` |")

            rows_written += 1

    if event_lines and hasattr(writer, "add_text"):
        table = "\n".join(
            [
                "| sim_time | event | details |",
                "| ---: | --- | --- |",
                *event_lines,
            ]
        )
        writer.add_text("events/marker_table_first_200", table, 0)

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert v2/errp_markers.csv and v2/session_samples.csv to TensorBoard logs."
    )
    parser.add_argument("--markers", default="v2/errp_markers.csv")
    parser.add_argument("--samples", default="v2/session_samples.csv")
    parser.add_argument("--logdir", default="runs/isaac_vr_csv")
    parser.add_argument(
        "--step-scale",
        type=int,
        default=1000,
        help="Multiplier used to convert sim_time seconds into TensorBoard steps for event markers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        SummaryWriter = _load_summary_writer()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    markers_path = Path(args.markers)
    samples_path = Path(args.samples)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(logdir))
    try:
        sample_count = write_session_samples(writer, samples_path, args.step_scale)
        marker_count = write_errp_markers(writer, markers_path, args.step_scale)
    finally:
        writer.close()

    print(f"Wrote TensorBoard logs to: {logdir}")
    print(f"Session samples: {sample_count}")
    print(f"ErrP markers: {marker_count}")
    print(f"Open with: tensorboard --logdir {logdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
