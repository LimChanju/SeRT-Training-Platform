#!/usr/bin/env python3
"""Report simulation-time Cartesian EE speed metrics from one HRI HDF5."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import h5py
import numpy as np

try:
    from v3_chan.pick_place_speed import controller_phase_mode
    from v3_chan.validate_hri_collection import (
        MEASURED_SPEED_ABS_TOLERANCE_MPS,
        MEASURED_SPEED_REL_TOLERANCE,
    )
except ImportError:
    from pick_place_speed import controller_phase_mode
    from validate_hri_collection import (
        MEASURED_SPEED_ABS_TOLERANCE_MPS,
        MEASURED_SPEED_REL_TOLERANCE,
    )


EVENT_NAMES = {
    0: "move_above_pick",
    1: "descend_to_pick",
    2: "settle_before_grasp",
    3: "close_gripper",
    4: "lift_grasped_cube",
    5: "transport_to_place_xy",
    6: "descend_to_place",
    7: "open_gripper",
    8: "retract_after_place",
    9: "safe_return",
}


@dataclass(frozen=True)
class SpeedMetric:
    episode: str
    profile: str
    scope: str
    sample_count: int
    mean_mps: float
    p95_mps: float
    max_mps: float
    mean_active_limit_mps: float
    max_active_limit_mps: float
    violation_count: int


@dataclass(frozen=True)
class TrajectorySpeedReport:
    path: str
    episode_count: int
    successful_episodes: int
    metrics: tuple[SpeedMetric, ...]


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _metric(
    episode_name: str,
    profile: str,
    scope: str,
    speed: np.ndarray,
    active_limit: np.ndarray,
    mask: np.ndarray,
    *,
    enforce_limit: bool = True,
) -> SpeedMetric:
    selected_speed = speed[mask]
    selected_limit = active_limit[mask]
    if selected_speed.size:
        tolerance = np.maximum(
            MEASURED_SPEED_ABS_TOLERANCE_MPS,
            MEASURED_SPEED_REL_TOLERANCE * selected_limit,
        )
        violations = (
            int(np.count_nonzero(selected_speed > selected_limit + tolerance))
            if enforce_limit
            else 0
        )
        mean = float(np.mean(selected_speed))
        p95 = float(np.percentile(selected_speed, 95))
        maximum = float(np.max(selected_speed))
        mean_limit = float(np.mean(selected_limit))
        max_limit = float(np.max(selected_limit))
    else:
        violations = 0
        mean = p95 = maximum = mean_limit = max_limit = 0.0
    return SpeedMetric(
        episode=episode_name,
        profile=profile,
        scope=scope,
        sample_count=int(selected_speed.size),
        mean_mps=mean,
        p95_mps=p95,
        max_mps=maximum,
        mean_active_limit_mps=mean_limit,
        max_active_limit_mps=max_limit,
        violation_count=violations,
    )


def analyze_cartesian_speed(path: str) -> TrajectorySpeedReport:
    metrics: list[SpeedMetric] = []
    successful = 0
    with h5py.File(path, "r") as data:
        episodes = data.get("episodes")
        episode_names = tuple(sorted(episodes.keys())) if episodes else ()
        for episode_name in episode_names:
            episode = episodes[episode_name]
            if "controller/ee_speed_sim_mps" not in episode:
                continue
            successful += int(bool(episode.attrs.get("success", False)))
            profile = _text(episode.attrs.get("controller_speed_profile", "unknown"))
            event = np.asarray(episode["controller/controller_event"][()], dtype=int)
            gate = np.asarray(episode["controller/distance_gate"][()], dtype=float)
            speed = np.asarray(episode["controller/ee_speed_sim_mps"][()], dtype=float)
            active_limit = np.asarray(
                episode["controller/active_speed_limit_mps"][()], dtype=float
            )
            valid = (
                np.asarray(episode["controller/ee_speed_sim_valid"][()], dtype=bool)
                & np.asarray(episode["controller/control_dt_valid"][()], dtype=bool)
                & np.isfinite(speed)
                & np.isfinite(active_limit)
            )
            motion_event = np.asarray(
                [controller_phase_mode(value) != "settling_gripper" for value in event],
                dtype=bool,
            )
            for event_index in range(10):
                event_has_cartesian_limit = (
                    controller_phase_mode(event_index) != "settling_gripper"
                )
                metrics.append(
                    _metric(
                        episode_name,
                        profile,
                        f"event_{event_index}_{EVENT_NAMES[event_index]}",
                        speed,
                        active_limit,
                        valid & (event == event_index),
                        enforce_limit=event_has_cartesian_limit,
                    )
                )
            for phase in ("transport", "approach_place"):
                phase_events = np.asarray(
                    [controller_phase_mode(value) == phase for value in event],
                    dtype=bool,
                )
                metrics.append(
                    _metric(
                        episode_name,
                        profile,
                        phase,
                        speed,
                        active_limit,
                        valid & phase_events,
                    )
                )
            metrics.append(
                _metric(
                    episode_name,
                    profile,
                    "near_human_full_gate",
                    speed,
                    active_limit,
                    valid & motion_event & (gate >= 1.0 - 1e-6),
                )
            )
    return TrajectorySpeedReport(
        path=str(path),
        episode_count=len(episode_names),
        successful_episodes=successful,
        metrics=tuple(metrics),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Hide per-event rows and show only phase/full-gate aggregates.",
    )
    args = parser.parse_args()
    report = analyze_cartesian_speed(args.path)
    if args.json:
        print(json.dumps(asdict(report), indent=2))
        return
    print(
        f"[SpeedMetrics] path={report.path} episodes={report.episode_count} "
        f"success={report.successful_episodes}/{report.episode_count}"
    )
    for metric in report.metrics:
        if args.aggregate_only and metric.scope.startswith("event_"):
            continue
        print(
            f"{metric.episode} profile={metric.profile} scope={metric.scope} "
            f"n={metric.sample_count} mean={metric.mean_mps:.4f} "
            f"p95={metric.p95_mps:.4f} max={metric.max_mps:.4f} "
            f"mean_limit={metric.mean_active_limit_mps:.4f} "
            f"max_limit={metric.max_active_limit_mps:.4f} "
            f"violations={metric.violation_count}"
        )


if __name__ == "__main__":
    main()
