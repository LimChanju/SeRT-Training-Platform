from __future__ import annotations

import argparse
import glob
import os

from rl.encounter_manifest import EncounterBuildConfig, build_encounter_manifest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build phase-aware HRI encounter metadata from recorded HDF5 files."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="HDF5 paths or glob patterns. Split sessions before building manifests.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON manifest.",
    )
    parser.add_argument("--gate-start-m", type=float, default=0.13)
    parser.add_argument("--near-m", type=float, default=0.05)
    parser.add_argument("--near-miss-m", type=float, default=0.02)
    parser.add_argument("--collision-m", type=float, default=0.0)
    parser.add_argument("--clear-end-m", type=float, default=0.15)
    parser.add_argument("--onset-frames", type=int, default=3)
    parser.add_argument("--clear-frames", type=int, default=15)
    parser.add_argument("--margin-frames", type=int, default=30)
    parser.add_argument("--safe-window-frames", type=int, default=180)
    parser.add_argument("--safe-stride-frames", type=int, default=90)
    parser.add_argument("--safe-min-frames", type=int, default=120)
    parser.add_argument("--max-safe-per-episode", type=int, default=8)
    return parser.parse_args()


def _resolve_sources(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        expanded = os.path.expanduser(pattern)
        matches = sorted(glob.glob(expanded))
        if matches:
            paths.extend(matches)
        else:
            paths.append(expanded)
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


def main() -> None:
    args = _parse_args()
    config = EncounterBuildConfig(
        gate_start_m=args.gate_start_m,
        near_m=args.near_m,
        near_miss_m=args.near_miss_m,
        collision_m=args.collision_m,
        clear_end_m=args.clear_end_m,
        onset_frames=args.onset_frames,
        clear_frames=args.clear_frames,
        margin_frames=args.margin_frames,
        safe_window_frames=args.safe_window_frames,
        safe_stride_frames=args.safe_stride_frames,
        safe_min_frames=args.safe_min_frames,
        max_safe_per_episode=args.max_safe_per_episode,
    )
    manifest = build_encounter_manifest(
        _resolve_sources(args.sources),
        args.output,
        config=config,
    )
    print(
        f"[EncounterManifest] saved={os.path.abspath(args.output)} "
        f"scenarios={manifest['scenario_count']} "
        f"severity_counts={manifest['severity_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
