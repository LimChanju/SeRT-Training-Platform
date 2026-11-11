from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
COLORS = {
    "task": "#4D4D4D",
    "balanced": "#D55E00",
    "strong_update": "#0072B2",
}
LABELS = {
    "task": "Task only",
    "balanced": "Balanced PPO",
    "strong_update": "Strong-update PPO",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize safety intervention metrics from existing evaluation logs."
    )
    parser.add_argument("--experiment-tag", default="surface_gap_new_sessions_v1")
    parser.add_argument("--sweep-tag", default="strong_residual_screen_v1")
    parser.add_argument("--folds", default="1,2,3,4")
    parser.add_argument("--eval-seeds", default="11,1011,2011")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    folds = _parse_ints(args.folds)
    seeds = _parse_ints(args.eval_seeds)
    baseline_root = SCRIPT_DIR / "eval_results" / "cv4" / args.experiment_tag
    sweep_root = SCRIPT_DIR / "eval_results" / "safety_sweeps" / args.sweep_tag
    analysis_path = sweep_root / "intervention_analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Run analyze_safety_interventions.py first: {analysis_path}"
        )
    with analysis_path.open("r", encoding="utf-8") as f:
        analysis = json.load(f)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else sweep_root / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = _plot_summary(analysis, output_dir)
    timeline_paths, selection = _plot_representative_timeline(
        baseline_root, sweep_root, folds, seeds, output_dir
    )
    manifest = {
        "summary_figure": [str(path) for path in summary_paths],
        "timeline_figure": [str(path) for path in timeline_paths],
        "representative_episode": selection,
    }
    with (output_dir / "visualization_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


def _plot_summary(analysis: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    by_condition = {row["condition"]: row for row in analysis["summary"]}
    conditions = ("task", "balanced", "strong_update")
    policy_conditions = ("balanced", "strong_update")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    fig.suptitle(
        "Safety Intervention Behavior (4-fold held-out evaluation)", fontsize=16
    )

    ax = axes[0, 0]
    metrics = (
        ("positive_away_rate", "Away-positive steps"),
        ("meaningful_step_rate_gate", "Meaningful gate steps"),
        ("gate_event_response_rate", "Gate-event response"),
    )
    x = np.arange(len(metrics))
    width = 0.34
    for index, condition in enumerate(policy_conditions):
        values = [
            100.0 * float(by_condition[condition][metric]) for metric, _ in metrics
        ]
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=COLORS[condition],
            label=LABELS[condition],
        )
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 75)
    ax.set_ylabel("Rate (%)")
    ax.set_title("A. Did the policy attempt avoidance?")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    x = np.arange(len(policy_conditions))
    applied = [
        float(by_condition[c]["mean_applied_mm_gate"]) for c in policy_conditions
    ]
    projected = [
        float(by_condition[c]["mean_projected_avoidance_mm"]) for c in policy_conditions
    ]
    bars1 = ax.bar(
        x - width / 2,
        applied,
        width,
        color=[COLORS[c] for c in policy_conditions],
        alpha=0.45,
        label="Applied residual",
    )
    bars2 = ax.bar(
        x + width / 2,
        projected,
        width,
        color=[COLORS[c] for c in policy_conditions],
        label="Away-projected component",
    )
    ax.bar_label(bars1, fmt="%.2f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [LABELS[c] for c in policy_conditions])
    ax.set_ylabel("Mean displacement per gate step (mm)")
    ax.set_title("B. How much motion was actually useful?")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    gap_metrics = (
        ("mean_gap_delta_mm_gate", "Gate active"),
        ("mean_gap_delta_mm_near", "Near <= 5 cm"),
        ("mean_gap_delta_mm_near_miss", "Near-miss <= 2 cm"),
    )
    x = np.arange(len(gap_metrics))
    width = 0.25
    for index, condition in enumerate(conditions):
        values = [float(by_condition[condition][metric]) for metric, _ in gap_metrics]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            color=COLORS[condition],
            label=LABELS[condition],
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, [label for _, label in gap_metrics])
    ax.set_ylabel("Surface-gap change per step (mm)")
    ax.set_title("C. Did clearance increase after intervention?")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    collision = [100.0 * float(by_condition[c]["collision_rate"]) for c in conditions]
    bars = ax.bar(
        np.arange(len(conditions)),
        collision,
        color=[COLORS[c] for c in conditions],
    )
    ax.bar_label(bars, fmt="%.3f%%", fontsize=9, padding=2)
    ax.set_xticks(np.arange(len(conditions)), [LABELS[c] for c in conditions])
    ax.set_ylabel("Collision steps (%)")
    ax.set_title("D. Final safety outcome")
    ax.set_ylim(0, max(collision) * 1.25)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)

    png = output_dir / "safety_intervention_summary.png"
    pdf = output_dir / "safety_intervention_summary.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _plot_representative_timeline(
    baseline_root: Path,
    sweep_root: Path,
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
) -> tuple[tuple[Path, Path], dict[str, int]]:
    task_groups = _load_condition_steps("task", baseline_root, sweep_root, folds, seeds)
    selection = max(
        task_groups,
        key=lambda key: sum(int(row["human_collision"]) for row in task_groups[key]),
    )
    conditions = ("task", "balanced", "strong_update")
    groups = {
        condition: _load_condition_steps(
            condition, baseline_root, sweep_root, folds, seeds
        )[selection]
        for condition in conditions
    }

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 10), sharex=True, constrained_layout=True
    )
    fold, seed, episode = selection
    fig.suptitle(
        f"Representative paired episode: fold {fold}, seed {seed}, episode {episode}",
        fontsize=15,
    )

    for condition in conditions:
        rows = groups[condition]
        steps = np.asarray([int(row["step"]) for row in rows])
        gap_cm = np.asarray([float(row["post_surface_gap_m"]) * 100.0 for row in rows])
        axes[0].plot(
            steps,
            np.clip(gap_cm, -7.0, 20.0),
            color=COLORS[condition],
            linewidth=1.4,
            label=LABELS[condition],
        )
        gate = np.asarray([float(row["gate"]) for row in rows])
        axes[1].plot(steps, gate, color=COLORS[condition], linewidth=1.3)
        applied_mm = np.asarray(
            [float(row["applied_position_m"]) * 1000.0 for row in rows]
        )
        axes[2].plot(steps, applied_mm, color=COLORS[condition], linewidth=1.2)
        cosine = np.asarray(
            [
                value if (value := _away_cosine(row)) is not None else np.nan
                for row in rows
            ]
        )
        axes[3].plot(steps, cosine, color=COLORS[condition], linewidth=1.1)

    axes[0].axhspan(-7.0, 0.0, color="#D73027", alpha=0.09, label="Collision")
    axes[0].axhline(13.0, color="#6A3D9A", linestyle="--", linewidth=1.0)
    axes[0].axhline(5.0, color="#E69F00", linestyle="--", linewidth=1.0)
    axes[0].axhline(2.0, color="#D55E00", linestyle=":", linewidth=1.0)
    axes[0].axhline(0.0, color="#D73027", linestyle="-", linewidth=1.0)
    axes[0].set_ylabel("Surface gap (cm)\n(clipped at 20 cm)")
    axes[0].set_title("A. Human-robot clearance")
    axes[0].legend(frameon=False, ncol=4, loc="upper right")

    axes[1].set_ylabel("Gate [0, 1]")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("B. Distance-gate activation")
    axes[2].axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("Applied residual (mm)")
    axes[2].set_title("C. Intervention magnitude (0.5 mm threshold)")
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].axhline(0.3, color="#777777", linestyle="--", linewidth=1.0)
    axes[3].set_ylim(-1.05, 1.05)
    axes[3].set_ylabel("Away cosine")
    axes[3].set_xlabel("Simulation step")
    axes[3].set_title(
        "D. Intervention direction (+1 is directly away from nearest hand)"
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.6)
        axis.set_axisbelow(True)

    png = output_dir / "representative_avoidance_timeline.png"
    pdf = output_dir / "representative_avoidance_timeline.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return (png, pdf), {"fold": fold, "seed": seed, "episode": episode}


def _load_condition_steps(
    condition: str,
    baseline_root: Path,
    sweep_root: Path,
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[tuple[int, int, int], list[dict[str, str]]]:
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    for fold in folds:
        for seed in seeds:
            if condition == "task":
                path = (
                    baseline_root / f"fold_{fold:02d}" / f"task_seed_{seed}_steps.csv"
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
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    key = (fold, seed, int(row["episode"]))
                    groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["step"]))
    return groups


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


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(
        dict.fromkeys(int(value.strip()) for value in text.split(",") if value.strip())
    )
    if not values:
        raise ValueError("Expected at least one integer")
    return values


if __name__ == "__main__":
    main()
