from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORTED_ALGORITHMS = ("ppo", "sac", "td3")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot learning curves and held-out safety metrics produced by "
            "run_encounter_benchmark.py."
        )
    )
    parser.add_argument("--experiment-tag", required=True)
    parser.add_argument("--algorithms", default="ppo,sac,td3")
    parser.add_argument("--meaningful-applied-mm", type=float, default=0.5)
    parser.add_argument("--meaningful-away-cosine", type=float, default=0.3)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    algorithms = _parse_algorithms(args.algorithms)
    tag = args.experiment_tag.strip()
    policy_dir = SCRIPT_DIR / "policies" / "encounter_benchmarks" / tag
    result_dir = SCRIPT_DIR / "eval_results" / "encounter_benchmarks" / tag
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    training = _load_training_histories(policy_dir, algorithms)
    episode_rows = _read_csv(result_dir / "episode_results.csv")
    models = ("task_only", *algorithms)
    _validate_episode_models(episode_rows, models)

    intervention = {
        model: _analyze_step_logs(
            sorted((result_dir / model).glob("seed_*_steps.csv")),
            meaningful_applied_m=float(args.meaningful_applied_mm) / 1000.0,
            meaningful_cosine=float(args.meaningful_away_cosine),
        )
        for model in models
    }
    eval_summary = _summarize_episodes_by_seed(episode_rows, models)
    paired = _paired_comparisons(episode_rows, algorithms)

    _plot_training_curves(training, figure_dir, args.dpi)
    _plot_final_metrics(eval_summary, models, figure_dir, args.dpi)
    _plot_interventions(intervention, algorithms, figure_dir, args.dpi)
    _plot_paired_effects(paired, algorithms, figure_dir, args.dpi)

    payload = {
        "experiment_tag": tag,
        "uncertainty_display": "mean +/- sample SD across evaluation seeds",
        "definitions": {
            "evaluation_collision_rate": (
                "arithmetic mean of each episode's collision-step rate; "
                "paired comparisons use the same episode-level quantity"
            ),
            "training_gate_sample_availability": (
                "PPO: gate-active samples divided by rollout environment steps; "
                "SAC/TD3: gate-active fraction currently stored in replay buffer"
            ),
            "away_cosine": (
                "cosine(applied residual XYZ, end-effector minus nearest hand)"
            ),
            "meaningful_intervention": (
                "gate active, applied position >= "
                f"{args.meaningful_applied_mm:g} mm, and away cosine >= "
                f"{args.meaningful_away_cosine:g}"
            ),
            "gate_event_response": (
                "at least one meaningful intervention during a contiguous "
                "gate-active interval"
            ),
        },
        "evaluation_by_model": eval_summary,
        "paired_vs_task_only": paired,
        "intervention_by_model": intervention,
    }
    with (figure_dir / "benchmark_plot_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    _write_flat_summary_csv(
        figure_dir / "benchmark_plot_summary.csv",
        models,
        eval_summary,
        intervention,
    )
    _write_paired_summary_csv(
        figure_dir / "paired_comparisons.csv",
        algorithms,
        paired,
    )
    print(f"[EncounterPlots] saved figures under {figure_dir}", flush=True)


def _parse_algorithms(value: str) -> tuple[str, ...]:
    algorithms = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    unknown = sorted(set(algorithms) - set(SUPPORTED_ALGORITHMS))
    if not algorithms or unknown:
        raise ValueError(
            f"Unknown algorithms {unknown}; supported={SUPPORTED_ALGORITHMS}"
        )
    return algorithms


def _load_training_histories(
    policy_dir: Path,
    algorithms: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    histories: dict[str, list[dict[str, Any]]] = {}
    for algorithm in algorithms:
        path = policy_dir / f"{algorithm}_safety_history.json"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        updates = payload.get("updates", [])
        if not updates:
            raise ValueError(f"Training history has no updates: {path}")
        histories[algorithm] = updates
    return histories


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_episode_models(
    rows: list[dict[str, str]],
    models: tuple[str, ...],
) -> None:
    available = {row["model"] for row in rows}
    missing = sorted(set(models) - available)
    if missing:
        raise ValueError(f"Missing models in episode_results.csv: {missing}")


def _summarize_episodes_by_seed(
    rows: list[dict[str, str]],
    models: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], int(row["eval_seed"]))].append(row)

    metrics = {
        "success_pct": ("success", 100.0),
        "collision_rate_pct": ("collision_rate", 100.0),
        "near_rate_pct": ("near_rate", 100.0),
        "near_miss_rate_pct": ("near_miss_rate", 100.0),
        "min_surface_gap_cm": ("min_surface_gap_m", 100.0),
        "mean_applied_position_mm": ("mean_applied_position_m", 1000.0),
        "total_reward": ("total_reward", 1.0),
    }
    summary: dict[str, dict[str, Any]] = {}
    for model in models:
        seeds = sorted(seed for candidate, seed in grouped if candidate == model)
        if not seeds:
            raise ValueError(f"No evaluation seeds found for {model}")
        record: dict[str, Any] = {
            "seeds": seeds,
            "episodes": sum(len(grouped[(model, seed)]) for seed in seeds),
            "seed_means": {},
        }
        for output_name, (input_name, scale) in metrics.items():
            seed_values = [
                float(
                    np.mean(
                        [float(row[input_name]) for row in grouped[(model, seed)]]
                    )
                    * scale
                )
                for seed in seeds
            ]
            record[output_name] = _mean_sd(seed_values)
            record["seed_means"][output_name] = seed_values
        summary[model] = record
    return summary


def _paired_comparisons(
    rows: list[dict[str, str]],
    algorithms: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    by_model: dict[str, dict[tuple[int, str], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["eval_seed"]), row["encounter_id"])
        if key in by_model[row["model"]]:
            raise ValueError(f"Duplicate evaluation pair for {row['model']}: {key}")
        by_model[row["model"]][key] = row
    baseline = by_model["task_only"]
    metrics = {
        "success_pp": ("success", 100.0, "higher"),
        "collision_rate_pp": ("collision_rate", 100.0, "lower"),
        "near_rate_pp": ("near_rate", 100.0, "lower"),
        "near_miss_rate_pp": ("near_miss_rate", 100.0, "lower"),
        "min_surface_gap_cm": ("min_surface_gap_m", 100.0, "higher"),
        "steps": ("steps", 1.0, "lower"),
        "total_reward": ("total_reward", 1.0, "higher"),
    }
    result: dict[str, dict[str, Any]] = {}
    for algorithm in algorithms:
        candidate = by_model[algorithm]
        if set(candidate) != set(baseline):
            missing = len(set(baseline) - set(candidate))
            extra = len(set(candidate) - set(baseline))
            raise ValueError(
                f"Unpaired evaluation rows for {algorithm}: "
                f"missing={missing}, extra={extra}"
            )
        layout_mismatches = [
            key
            for key in baseline
            if baseline[key].get("scene_layout_id", "")
            != candidate[key].get("scene_layout_id", "")
        ]
        if layout_mismatches:
            raise ValueError(
                f"Scene layout mismatch for {algorithm}: "
                f"{layout_mismatches[:5]}"
            )
        model_result: dict[str, Any] = {
            "pairs": len(baseline),
            "layout_pairing_verified": True,
            "metrics": {},
        }
        ordered_keys = sorted(baseline)
        for output_name, (field, scale, direction) in metrics.items():
            base_values = np.asarray(
                [float(baseline[key][field]) * scale for key in ordered_keys],
                dtype=np.float64,
            )
            model_values = np.asarray(
                [float(candidate[key][field]) * scale for key in ordered_keys],
                dtype=np.float64,
            )
            differences = model_values - base_values
            ci_low, ci_high = _bootstrap_mean_ci(differences)
            model_result["metrics"][output_name] = {
                "direction_better": direction,
                "task_only_mean": float(np.mean(base_values)),
                "model_mean": float(np.mean(model_values)),
                "paired_difference": float(np.mean(differences)),
                "bootstrap_95_ci": [ci_low, ci_high],
            }
        result[algorithm] = model_result
    return result


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int = 10000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return 0.0, 0.0
    rng = np.random.default_rng(24023)
    means = np.empty(samples, dtype=np.float64)
    batch_size = 250
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = np.mean(values[indices], axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def _analyze_step_logs(
    paths: list[Path],
    *,
    meaningful_applied_m: float,
    meaningful_cosine: float,
) -> dict[str, Any]:
    if not paths:
        raise FileNotFoundError("No seed_*_steps.csv files found")
    seed_records = [
        _analyze_one_step_log(
            path,
            meaningful_applied_m=meaningful_applied_m,
            meaningful_cosine=meaningful_cosine,
        )
        for path in paths
    ]
    metrics = (
        "gate_active_rate_pct",
        "collision_rate_pct",
        "near_rate_pct",
        "near_miss_rate_pct",
        "mean_applied_mm_gate",
        "mean_away_cosine",
        "positive_away_rate_pct",
        "meaningful_step_rate_gate_pct",
        "mean_projected_avoidance_mm",
        "mean_gap_delta_mm_gate",
        "gate_event_response_rate_pct",
        "mean_response_latency_steps",
    )
    summary: dict[str, Any] = {
        "seed_files": [str(path) for path in paths],
        "seed_records": seed_records,
    }
    for metric in metrics:
        values = [
            float(record[metric])
            for record in seed_records
            if record[metric] is not None and math.isfinite(float(record[metric]))
        ]
        summary[metric] = _mean_sd(values) if values else None
    summary["steps"] = int(sum(record["steps"] for record in seed_records))
    summary["gate_events"] = int(
        sum(record["gate_events"] for record in seed_records)
    )
    return summary


def _analyze_one_step_log(
    path: Path,
    *,
    meaningful_applied_m: float,
    meaningful_cosine: float,
) -> dict[str, Any]:
    counters = defaultdict(int)
    sums = defaultdict(float)
    current_episode: tuple[int, int] | None = None
    in_gate_event = False
    event_responded = False
    event_start_step = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counters["steps"] += 1
            gate = bool(int(row["gate_active"]))
            collision = bool(int(row["human_collision"]))
            near = bool(int(row["near_human"]))
            near_miss = bool(int(row["near_miss"]))
            counters["gate"] += int(gate)
            counters["collision"] += int(collision)
            counters["near"] += int(near)
            counters["near_miss"] += int(near_miss)

            episode = (int(row["seed"]), int(row["episode"]))
            if episode != current_episode:
                current_episode = episode
                in_gate_event = False
                event_responded = False

            meaningful = False
            if gate:
                applied_m = float(row["applied_position_m"])
                sums["applied_m_gate"] += applied_m
                sums["gap_delta_m_gate"] += float(row["surface_gap_delta_m"])
                cosine = _away_cosine(row)
                if cosine is not None:
                    counters["directional"] += 1
                    sums["away_cosine"] += cosine
                    counters["positive_away"] += int(cosine > 0.0)
                    sums["projected_avoidance_m"] += applied_m * cosine
                    meaningful = (
                        applied_m >= meaningful_applied_m
                        and cosine >= meaningful_cosine
                    )
                    counters["meaningful"] += int(meaningful)

            if gate and not in_gate_event:
                in_gate_event = True
                event_responded = False
                event_start_step = int(row["step"])
                counters["gate_events"] += 1
            elif not gate:
                in_gate_event = False
                event_responded = False
            if gate and meaningful and not event_responded:
                event_responded = True
                counters["responded_events"] += 1
                sums["response_latency_steps"] += (
                    int(row["step"]) - event_start_step
                )

    return {
        "path": str(path),
        "steps": counters["steps"],
        "gate_events": counters["gate_events"],
        "gate_active_rate_pct": _percent(counters["gate"], counters["steps"]),
        "collision_rate_pct": _percent(
            counters["collision"], counters["steps"]
        ),
        "near_rate_pct": _percent(counters["near"], counters["steps"]),
        "near_miss_rate_pct": _percent(
            counters["near_miss"], counters["steps"]
        ),
        "mean_applied_mm_gate": _ratio(
            sums["applied_m_gate"] * 1000.0, counters["gate"]
        ),
        "mean_away_cosine": _ratio(
            sums["away_cosine"], counters["directional"]
        ),
        "positive_away_rate_pct": _percent(
            counters["positive_away"], counters["directional"]
        ),
        "meaningful_step_rate_gate_pct": _percent(
            counters["meaningful"], counters["gate"]
        ),
        "mean_projected_avoidance_mm": _ratio(
            sums["projected_avoidance_m"] * 1000.0,
            counters["directional"],
        ),
        "mean_gap_delta_mm_gate": _ratio(
            sums["gap_delta_m_gate"] * 1000.0, counters["gate"]
        ),
        "gate_event_response_rate_pct": _percent(
            counters["responded_events"], counters["gate_events"]
        ),
        "mean_response_latency_steps": _ratio(
            sums["response_latency_steps"], counters["responded_events"]
        ),
    }


def _away_cosine(row: dict[str, str]) -> float | None:
    residual = np.asarray(
        [float(row[f"applied_residual_{axis}"]) for axis in "xyz"],
        dtype=np.float64,
    )
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 1e-12:
        return None
    ee = np.asarray(
        [float(row[f"post_ee_{axis}"]) for axis in "xyz"], dtype=np.float64
    )
    hands = [
        np.asarray(
            [float(row[f"post_{side}_hand_{axis}"]) for axis in "xyz"],
            dtype=np.float64,
        )
        for side in ("left", "right")
    ]
    nearest = min(hands, key=lambda hand: float(np.linalg.norm(ee - hand)))
    away = ee - nearest
    away_norm = float(np.linalg.norm(away))
    if away_norm <= 1e-12:
        return None
    return float(np.dot(residual, away) / (residual_norm * away_norm))


def _plot_training_curves(
    histories: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flat
    colors = {"ppo": "#2f6b9a", "sac": "#d17a22", "td3": "#2f8f5b"}
    for algorithm, updates in histories.items():
        steps = np.asarray(
            [float(row.get("total_steps", row.get("step", 0))) for row in updates]
        )
        success = np.asarray(
            [float(row.get("recent_success_rate", 0.0)) * 100.0 for row in updates]
        )
        returns = np.asarray(
            [float(row.get("recent_return", 0.0)) for row in updates]
        )
        episodes = np.asarray([float(row.get("episodes", 0.0)) for row in updates])
        if algorithm == "ppo":
            step_deltas = np.diff(np.concatenate(([0.0], steps)))
            gate_availability = np.asarray(
                [float(row.get("active_train_samples", 0.0)) for row in updates]
            ) / np.maximum(step_deltas, 1.0)
        else:
            gate_availability = np.asarray(
                [float(row.get("gate_fraction", 0.0)) for row in updates]
            )
        label = algorithm.upper()
        axes[0].plot(steps, success, label=label, color=colors[algorithm], lw=1.8)
        axes[1].plot(steps, returns, label=label, color=colors[algorithm], lw=1.8)
        axes[2].plot(steps, episodes, label=label, color=colors[algorithm], lw=1.8)
        axes[3].plot(
            steps,
            gate_availability * 100.0,
            label=label,
            color=colors[algorithm],
            lw=1.8,
        )
    axes[0].set(title="Recent task success", ylabel="Success (%)")
    axes[1].set(title="Recent environment return", ylabel="Return")
    axes[2].set(title="Completed training episodes", ylabel="Episodes")
    axes[3].set(title="Gate-active sample availability", ylabel="Samples (%)")
    for axis in axes:
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylim(-2, 102)
    axes[0].legend(frameon=False)
    _save_figure(fig, figure_dir / "training_curves", dpi)


def _plot_final_metrics(
    summary: dict[str, dict[str, Any]],
    models: tuple[str, ...],
    figure_dir: Path,
    dpi: int,
) -> None:
    metrics = (
        ("success_pct", "Task success", "%", True),
        (
            "collision_rate_pct",
            "Mean episode collision-step rate",
            "%",
            False,
        ),
        ("near_rate_pct", "Near step rate", "%", False),
        ("min_surface_gap_cm", "Minimum surface gap", "cm", True),
    )
    labels = [_display_name(model) for model in models]
    colors = ["#777777", "#2f6b9a", "#d17a22", "#2f8f5b"][: len(models)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, title, unit, _) in zip(axes.flat, metrics):
        means = [float(summary[model][metric]["mean"]) for model in models]
        errors = [float(summary[model][metric]["sd"]) for model in models]
        axis.bar(labels, means, yerr=errors, color=colors, capsize=4)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle("Held-out encounter evaluation (mean +/- SD across seeds)")
    _save_figure(fig, figure_dir / "heldout_safety_comparison", dpi)


def _plot_interventions(
    summary: dict[str, dict[str, Any]],
    algorithms: tuple[str, ...],
    figure_dir: Path,
    dpi: int,
) -> None:
    metrics = (
        ("mean_applied_mm_gate", "Applied correction during gate", "mm/step"),
        ("mean_away_cosine", "Correction direction", "Away cosine"),
        ("meaningful_step_rate_gate_pct", "Meaningful intervention steps", "%"),
        ("gate_event_response_rate_pct", "Gate events with a response", "%"),
    )
    labels = [_display_name(model) for model in algorithms]
    colors = ["#2f6b9a", "#d17a22", "#2f8f5b"][: len(algorithms)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (metric, title, unit) in zip(axes.flat, metrics):
        means = [
            _summary_value(summary[model].get(metric), "mean")
            for model in algorithms
        ]
        errors = [
            _summary_value(summary[model].get(metric), "sd")
            for model in algorithms
        ]
        axis.bar(labels, means, yerr=errors, color=colors, capsize=4)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Safety-policy intervention behavior (mean +/- SD across seeds)")
    _save_figure(fig, figure_dir / "intervention_behavior", dpi)


def _plot_paired_effects(
    paired: dict[str, dict[str, Any]],
    algorithms: tuple[str, ...],
    figure_dir: Path,
    dpi: int,
) -> None:
    metrics = (
        ("success_pp", "Task success difference", "percentage points"),
        ("collision_rate_pp", "Collision-rate difference", "percentage points"),
        ("near_rate_pp", "Near-rate difference", "percentage points"),
        ("min_surface_gap_cm", "Minimum-gap difference", "cm"),
    )
    labels = [_display_name(model) for model in algorithms]
    colors = ["#2f6b9a", "#d17a22", "#2f8f5b"][: len(algorithms)]
    x = np.arange(len(algorithms))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (metric, title, unit) in zip(axes.flat, metrics):
        records = [paired[model]["metrics"][metric] for model in algorithms]
        means = np.asarray(
            [float(record["paired_difference"]) for record in records]
        )
        intervals = np.asarray(
            [record["bootstrap_95_ci"] for record in records],
            dtype=np.float64,
        )
        errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
        axis.bar(x, means, yerr=errors, color=colors, capsize=4)
        axis.axhline(0.0, color="#333333", lw=1.0)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Paired effects versus task-only (bootstrap 95% CI)")
    _save_figure(fig, figure_dir / "paired_effects", dpi)


def _save_figure(fig: plt.Figure, path_without_suffix: Path, dpi: int) -> None:
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=dpi)
    fig.savefig(path_without_suffix.with_suffix(".pdf"))
    plt.close(fig)


def _write_flat_summary_csv(
    path: Path,
    models: tuple[str, ...],
    evaluation: dict[str, dict[str, Any]],
    intervention: dict[str, dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {"model": model}
        for metric, value in evaluation[model].items():
            if isinstance(value, dict) and "mean" in value:
                row[f"{metric}_mean"] = value["mean"]
                row[f"{metric}_sd"] = value["sd"]
        for metric, value in intervention[model].items():
            if isinstance(value, dict) and "mean" in value:
                row[f"{metric}_mean"] = value["mean"]
                row[f"{metric}_sd"] = value["sd"]
        rows.append(row)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_paired_summary_csv(
    path: Path,
    algorithms: tuple[str, ...],
    paired: dict[str, dict[str, Any]],
) -> None:
    rows = []
    for algorithm in algorithms:
        for metric, record in paired[algorithm]["metrics"].items():
            rows.append(
                {
                    "model": algorithm,
                    "metric": metric,
                    "direction_better": record["direction_better"],
                    "pairs": paired[algorithm]["pairs"],
                    "task_only_mean": record["task_only_mean"],
                    "model_mean": record["model_mean"],
                    "paired_difference": record["paired_difference"],
                    "ci_95_low": record["bootstrap_95_ci"][0],
                    "ci_95_high": record["bootstrap_95_ci"][1],
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"mean": 0.0, "sd": 0.0, "n": 0}
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def _ratio(numerator: float, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _percent(numerator: int, denominator: int) -> float:
    return (
        100.0 * float(numerator) / float(denominator) if denominator else 0.0
    )


def _summary_value(value: Any, key: str) -> float:
    if isinstance(value, dict) and value.get(key) is not None:
        return float(value[key])
    return 0.0


def _display_name(model: str) -> str:
    return "Task only" if model == "task_only" else model.upper()


if __name__ == "__main__":
    main()
