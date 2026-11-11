from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze avoidance behavior from existing safety-evaluation step logs."
    )
    parser.add_argument("--experiment-tag", default="surface_gap_new_sessions_v1")
    parser.add_argument("--sweep-tag", default="strong_residual_screen_v1")
    parser.add_argument("--folds", default="1,2,3,4")
    parser.add_argument("--configs", default="balanced,strong_update")
    parser.add_argument("--eval-seeds", default="11,1011,2011")
    parser.add_argument("--meaningful-applied-mm", type=float, default=0.5)
    parser.add_argument("--meaningful-away-cosine", type=float, default=0.3)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    folds = _parse_ints(args.folds)
    seeds = _parse_ints(args.eval_seeds)
    configs = tuple(value.strip() for value in args.configs.split(",") if value.strip())
    if not configs:
        raise ValueError("--configs must contain at least one configuration")

    baseline_root = SCRIPT_DIR / "eval_results" / "cv4" / args.experiment_tag
    sweep_root = SCRIPT_DIR / "eval_results" / "safety_sweeps" / args.sweep_tag
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else sweep_root / "intervention_analysis.json"
    )

    conditions = ("task", *configs)
    session_rows: list[dict[str, Any]] = []
    for fold in folds:
        for condition in conditions:
            step_paths = []
            for seed in seeds:
                if condition == "task":
                    path = (
                        baseline_root
                        / f"fold_{fold:02d}"
                        / f"task_seed_{seed}_steps.csv"
                    )
                else:
                    path = (
                        sweep_root
                        / condition
                        / f"fold_{fold:02d}"
                        / f"seed_{seed}_steps.csv"
                    )
                if not path.exists():
                    raise FileNotFoundError(path)
                step_paths.append(path)
            session_rows.append(
                _analyze_session(
                    fold,
                    condition,
                    step_paths,
                    meaningful_applied_m=float(args.meaningful_applied_mm) / 1000.0,
                    meaningful_cosine=float(args.meaningful_away_cosine),
                )
            )

    summaries = []
    for condition in conditions:
        selected = [row for row in session_rows if row["condition"] == condition]
        record: dict[str, Any] = {
            "condition": condition,
            "sessions": len(selected),
        }
        metric_names = [key for key in selected[0] if key not in ("fold", "condition")]
        for metric in metric_names:
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            record[metric] = float(np.mean(values)) if values else None
        summaries.append(record)

    payload = {
        "definition": {
            "meaningful_intervention": (
                "gate_active AND applied_position_m >= threshold AND "
                "away_cosine >= threshold"
            ),
            "meaningful_applied_mm": float(args.meaningful_applied_mm),
            "meaningful_away_cosine": float(args.meaningful_away_cosine),
            "latency_unit": "simulation steps after each gate onset",
            "away_cosine": "cosine(applied residual XYZ, EE - nearest hand)",
        },
        "session_rows": session_rows,
        "summary": summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_csv(output.with_suffix(".csv"), summaries)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[InterventionAnalysis] saved {output}", flush=True)


def _analyze_session(
    fold: int,
    condition: str,
    paths: list[Path],
    *,
    meaningful_applied_m: float,
    meaningful_cosine: float,
) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (int(row["seed"]), int(row["episode"]))
                groups.setdefault(key, []).append(row)

    all_rows = [row for rows in groups.values() for row in rows]
    gate_rows = [row for row in all_rows if _as_bool(row, "gate_active")]
    near_rows = [row for row in all_rows if _as_bool(row, "near_human")]
    near_miss_rows = [row for row in all_rows if _as_bool(row, "near_miss")]
    collision_rows = [row for row in all_rows if _as_bool(row, "human_collision")]

    directional: list[tuple[dict[str, str], float, float]] = []
    meaningful_ids: set[tuple[int, int]] = set()
    for key, rows in groups.items():
        for row in rows:
            if not _as_bool(row, "gate_active"):
                continue
            cosine = _away_cosine(row)
            if cosine is None:
                continue
            applied_m = float(row["applied_position_m"])
            directional.append((row, cosine, applied_m * cosine))
            if applied_m >= meaningful_applied_m and cosine >= meaningful_cosine:
                meaningful_ids.add(key)

    gate_events = 0
    responded_events = 0
    latencies = []
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["step"]))
        index = 0
        while index < len(rows):
            if not _as_bool(rows[index], "gate_active"):
                index += 1
                continue
            event_start = index
            gate_events += 1
            while index < len(rows) and _as_bool(rows[index], "gate_active"):
                cosine = _away_cosine(rows[index])
                if (
                    cosine is not None
                    and float(rows[index]["applied_position_m"]) >= meaningful_applied_m
                    and cosine >= meaningful_cosine
                ):
                    responded_events += 1
                    latencies.append(index - event_start)
                    while index < len(rows) and _as_bool(rows[index], "gate_active"):
                        index += 1
                    break
                index += 1

    meaningful_steps = sum(
        float(row["applied_position_m"]) >= meaningful_applied_m
        and cosine >= meaningful_cosine
        for row, cosine, _ in directional
    )
    return {
        "fold": fold,
        "condition": condition,
        "episodes": len(groups),
        "gate_active_rate": _rate(len(gate_rows), len(all_rows)),
        "collision_rate": _rate(len(collision_rows), len(all_rows)),
        "near_rate": _rate(len(near_rows), len(all_rows)),
        "near_miss_rate": _rate(len(near_miss_rows), len(all_rows)),
        "mean_applied_mm_gate": _mean(gate_rows, "applied_position_m", scale=1000.0),
        "mean_away_cosine": _mean_values([cosine for _, cosine, _ in directional]),
        "positive_away_rate": _rate(
            sum(cosine > 0.0 for _, cosine, _ in directional), len(directional)
        ),
        "meaningful_step_rate_gate": _rate(meaningful_steps, len(gate_rows)),
        "meaningful_episode_rate": _rate(len(meaningful_ids), len(groups)),
        "mean_projected_avoidance_mm": _mean_values(
            [projection * 1000.0 for _, _, projection in directional]
        ),
        "mean_gap_delta_mm_gate": _mean(gate_rows, "surface_gap_delta_m", scale=1000.0),
        "mean_gap_delta_mm_near": _mean(near_rows, "surface_gap_delta_m", scale=1000.0),
        "mean_gap_delta_mm_near_miss": _mean(
            near_miss_rows, "surface_gap_delta_m", scale=1000.0
        ),
        "gate_event_count": gate_events,
        "gate_event_response_rate": _rate(responded_events, gate_events),
        "mean_response_latency_steps": _mean_values(latencies),
    }


def _away_cosine(row: dict[str, str]) -> float | None:
    residual = np.asarray(
        [float(row[f"applied_residual_{axis}"]) for axis in "xyz"], dtype=np.float64
    )
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 1e-12:
        return None
    ee = np.asarray([float(row[f"post_ee_{axis}"]) for axis in "xyz"])
    hands = [
        np.asarray([float(row[f"post_{side}_hand_{axis}"]) for axis in "xyz"])
        for side in ("left", "right")
    ]
    nearest = min(hands, key=lambda hand: float(np.linalg.norm(ee - hand)))
    away = ee - nearest
    away_norm = float(np.linalg.norm(away))
    if away_norm <= 1e-12:
        return None
    return float(np.dot(residual, away) / (residual_norm * away_norm))


def _as_bool(row: dict[str, str], key: str) -> bool:
    return bool(int(row[key]))


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(rows: list[dict[str, str]], key: str, *, scale: float = 1.0) -> float | None:
    return _mean_values([float(row[key]) * scale for row in rows])


def _mean_values(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(
        dict.fromkeys(int(value.strip()) for value in text.split(",") if value.strip())
    )
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
