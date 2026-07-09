from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two rollout evaluations and failure analyses.")
    parser.add_argument(
        "--baseline-rollout-json",
        default=os.path.join(SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval.json"),
        help="Baseline rollout evaluation JSON.",
    )
    parser.add_argument(
        "--candidate-rollout-json",
        default=os.path.join(SCRIPT_DIR, "eval_results", "ppo_pick_place_v1_rollout_eval.json"),
        help="Candidate rollout evaluation JSON.",
    )
    parser.add_argument(
        "--baseline-analysis-json",
        default=os.path.join(SCRIPT_DIR, "eval_results", "bc_pick_place_v1_failure_analysis.json"),
        help="Baseline failure analysis JSON.",
    )
    parser.add_argument(
        "--candidate-analysis-json",
        default=os.path.join(SCRIPT_DIR, "eval_results", "ppo_pick_place_v1_failure_analysis.json"),
        help="Candidate failure analysis JSON.",
    )
    parser.add_argument(
        "--baseline-name",
        default="BC",
        help="Human-readable baseline label.",
    )
    parser.add_argument(
        "--candidate-name",
        default="PPO",
        help="Human-readable candidate label.",
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(SCRIPT_DIR, "eval_results", "bc_vs_ppo_pick_place_v1_comparison.json"),
        help="Path to save machine-readable comparison.",
    )
    parser.add_argument(
        "--output-md",
        default=os.path.join(SCRIPT_DIR, "eval_results", "bc_vs_ppo_pick_place_v1_comparison.md"),
        help="Path to save Markdown comparison.",
    )
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    baseline_rollout = _load_json(args.baseline_rollout_json)
    candidate_rollout = _load_json(args.candidate_rollout_json)
    baseline_analysis = _load_json(args.baseline_analysis_json)
    candidate_analysis = _load_json(args.candidate_analysis_json)

    comparison = _compare(
        baseline_rollout=baseline_rollout,
        candidate_rollout=candidate_rollout,
        baseline_analysis=baseline_analysis,
        candidate_analysis=candidate_analysis,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )
    comparison["inputs"] = {
        "baseline_rollout_json": os.path.abspath(args.baseline_rollout_json),
        "candidate_rollout_json": os.path.abspath(args.candidate_rollout_json),
        "baseline_analysis_json": os.path.abspath(args.baseline_analysis_json),
        "candidate_analysis_json": os.path.abspath(args.candidate_analysis_json),
    }
    _write_json(args.output_json, comparison)
    _write_markdown(args.output_md, comparison)
    _print_summary(comparison)


def _compare(
    *,
    baseline_rollout: dict[str, Any],
    candidate_rollout: dict[str, Any],
    baseline_analysis: dict[str, Any],
    candidate_analysis: dict[str, Any],
    baseline_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    baseline_summary = baseline_rollout.get("summary", {})
    candidate_summary = candidate_rollout.get("summary", {})
    baseline_fail_summary = baseline_analysis.get("summary", {})
    candidate_fail_summary = candidate_analysis.get("summary", {})

    paired = _paired_episode_changes(
        baseline_rollout.get("episodes", []),
        candidate_rollout.get("episodes", []),
    )

    numeric_fields = (
        "success_rate",
        "truncated_rate",
        "grasp_rate",
        "mean_steps",
        "mean_total_reward",
        "mean_final_cube_target_dist",
        "mean_min_cube_target_dist",
    )
    metric_deltas = {
        field: _delta(candidate_summary.get(field), baseline_summary.get(field))
        for field in numeric_fields
    }
    failure_category_delta = _counter_delta(
        baseline_fail_summary.get("failure_categories", {}),
        candidate_fail_summary.get("failure_categories", {}),
    )
    return {
        "labels": {
            "baseline": baseline_name,
            "candidate": candidate_name,
        },
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "metric_deltas_candidate_minus_baseline": metric_deltas,
        "baseline_failure_summary": baseline_fail_summary,
        "candidate_failure_summary": candidate_fail_summary,
        "failure_category_delta_candidate_minus_baseline": failure_category_delta,
        "paired_episode_changes": paired,
        "interpretation": _interpret(metric_deltas, failure_category_delta, paired, baseline_name, candidate_name),
    }


def _paired_episode_changes(
    baseline_episodes: list[dict[str, Any]],
    candidate_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_seed = {int(row.get("seed", row.get("episode", -1))): row for row in baseline_episodes}
    candidate_by_seed = {int(row.get("seed", row.get("episode", -1))): row for row in candidate_episodes}
    shared_seeds = sorted(set(baseline_by_seed) & set(candidate_by_seed))
    changes = []
    counters: Counter[str] = Counter()
    for seed in shared_seeds:
        base = baseline_by_seed[seed]
        cand = candidate_by_seed[seed]
        base_success = bool(base.get("success", False))
        cand_success = bool(cand.get("success", False))
        if base_success and cand_success:
            category = "both_success"
        elif base_success and not cand_success:
            category = "candidate_regression"
        elif not base_success and cand_success:
            category = "candidate_recovery"
        else:
            category = "both_failure"
        counters[category] += 1
        changes.append(
            {
                "seed": seed,
                "episode_baseline": int(base.get("episode", -1)),
                "episode_candidate": int(cand.get("episode", -1)),
                "category": category,
                "baseline_success": base_success,
                "candidate_success": cand_success,
                "baseline_final_dist": float(base.get("final_cube_target_dist", float("nan"))),
                "candidate_final_dist": float(cand.get("final_cube_target_dist", float("nan"))),
                "baseline_grasped_any": bool(base.get("grasped_any", False)),
                "candidate_grasped_any": bool(cand.get("grasped_any", False)),
                "baseline_steps": int(base.get("steps", 0)),
                "candidate_steps": int(cand.get("steps", 0)),
                "step_delta": int(cand.get("steps", 0)) - int(base.get("steps", 0)),
            }
        )
    step_deltas = np.asarray([row["step_delta"] for row in changes], dtype=np.float32)
    dist_deltas = np.asarray(
        [row["candidate_final_dist"] - row["baseline_final_dist"] for row in changes],
        dtype=np.float32,
    )
    return {
        "shared_seed_count": len(shared_seeds),
        "counts": dict(counters),
        "mean_step_delta": float(np.mean(step_deltas)) if len(step_deltas) else 0.0,
        "mean_final_dist_delta": float(np.mean(dist_deltas)) if len(dist_deltas) else 0.0,
        "regressions": [row for row in changes if row["category"] == "candidate_regression"],
        "recoveries": [row for row in changes if row["category"] == "candidate_recovery"],
        "both_failures": [row for row in changes if row["category"] == "both_failure"],
    }


def _interpret(
    metric_deltas: dict[str, float | None],
    failure_delta: dict[str, int],
    paired: dict[str, Any],
    baseline_name: str,
    candidate_name: str,
) -> list[str]:
    notes = []
    success_delta = metric_deltas.get("success_rate")
    if success_delta is not None and abs(success_delta) < 1e-6:
        notes.append(f"{candidate_name} preserved the overall success rate of {baseline_name}.")
    elif success_delta is not None and success_delta > 0:
        notes.append(f"{candidate_name} improved overall success rate by {success_delta:.3f}.")
    elif success_delta is not None:
        notes.append(f"{candidate_name} reduced overall success rate by {abs(success_delta):.3f}.")

    step_delta = metric_deltas.get("mean_steps")
    if step_delta is not None and step_delta < 0:
        notes.append(f"{candidate_name} finished successful/failed rollouts faster on average by {-step_delta:.1f} steps.")
    elif step_delta is not None and step_delta > 0:
        notes.append(f"{candidate_name} took longer on average by {step_delta:.1f} steps.")

    if failure_delta.get("released_outside_success_radius", 0) < 0:
        notes.append(f"{candidate_name} reduced release-outside-target failures.")
    if failure_delta.get("grasp_never_established", 0) > 0:
        notes.append(f"{candidate_name} introduced more grasp-establishment failures.")

    counts = paired.get("counts", {})
    recoveries = counts.get("candidate_recovery", 0)
    regressions = counts.get("candidate_regression", 0)
    if recoveries or regressions:
        notes.append(
            f"Seed-level comparison shows {recoveries} recoveries and {regressions} regressions."
        )
    return notes


def _counter_delta(baseline: dict[str, int], candidate: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(baseline) | set(candidate))
    return {key: int(candidate.get(key, 0)) - int(baseline.get(key, 0)) for key in keys}


def _delta(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_markdown(path: str, comparison: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    labels = comparison["labels"]
    base = labels["baseline"]
    cand = labels["candidate"]
    base_summary = comparison["baseline_summary"]
    cand_summary = comparison["candidate_summary"]
    deltas = comparison["metric_deltas_candidate_minus_baseline"]
    paired = comparison["paired_episode_changes"]
    lines = [
        "# Rollout Comparison",
        "",
        f"Baseline: `{base}`",
        f"Candidate: `{cand}`",
        "",
        "## Summary",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for field in (
        "success_rate",
        "grasp_rate",
        "mean_steps",
        "mean_final_cube_target_dist",
        "mean_min_cube_target_dist",
        "mean_total_reward",
    ):
        lines.append(
            f"| `{field}` | {_fmt(base_summary.get(field))} | {_fmt(cand_summary.get(field))} | {_fmt(deltas.get(field))} |"
        )

    lines.extend(["", "## Failure Categories", "", "| Category | Baseline | Candidate | Delta |", "|---|---:|---:|---:|"])
    base_fail = comparison["baseline_failure_summary"].get("failure_categories", {})
    cand_fail = comparison["candidate_failure_summary"].get("failure_categories", {})
    fail_delta = comparison["failure_category_delta_candidate_minus_baseline"]
    for category in sorted(set(base_fail) | set(cand_fail)):
        lines.append(
            f"| `{category}` | {int(base_fail.get(category, 0))} | {int(cand_fail.get(category, 0))} | {int(fail_delta.get(category, 0)):+d} |"
        )

    lines.extend(
        [
            "",
            "## Seed-Level Changes",
            "",
            f"- Shared seeds: {paired['shared_seed_count']}",
            f"- Both success: {paired['counts'].get('both_success', 0)}",
            f"- Candidate recoveries: {paired['counts'].get('candidate_recovery', 0)}",
            f"- Candidate regressions: {paired['counts'].get('candidate_regression', 0)}",
            f"- Both failure: {paired['counts'].get('both_failure', 0)}",
            f"- Mean step delta: {paired['mean_step_delta']:.2f}",
            f"- Mean final distance delta: {paired['mean_final_dist_delta']:.4f} m",
            "",
            "## Interpretation",
            "",
        ]
    )
    for note in comparison["interpretation"]:
        lines.append(f"- {note}")

    lines.extend(["", "## Candidate Regressions", ""])
    if paired["regressions"]:
        for row in paired["regressions"]:
            lines.append(
                f"- seed={row['seed']} baseline_dist={row['baseline_final_dist']:.4f} "
                f"candidate_dist={row['candidate_final_dist']:.4f} "
                f"candidate_grasped={int(row['candidate_grasped_any'])}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Candidate Recoveries", ""])
    if paired["recoveries"]:
        for row in paired["recoveries"]:
            lines.append(
                f"- seed={row['seed']} baseline_dist={row['baseline_final_dist']:.4f} "
                f"candidate_dist={row['candidate_final_dist']:.4f} "
                f"candidate_grasped={int(row['candidate_grasped_any'])}"
            )
    else:
        lines.append("- None")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_summary(comparison: dict[str, Any]) -> None:
    labels = comparison["labels"]
    base = labels["baseline"]
    cand = labels["candidate"]
    base_s = comparison["baseline_summary"]
    cand_s = comparison["candidate_summary"]
    deltas = comparison["metric_deltas_candidate_minus_baseline"]
    print(
        f"[CompareRollout] {base} success={float(base_s.get('success_rate', 0.0)):.3f} "
        f"{cand} success={float(cand_s.get('success_rate', 0.0)):.3f} "
        f"delta={float(deltas.get('success_rate') or 0.0):+.3f}"
    )
    print(
        f"[CompareRollout] mean_steps delta={float(deltas.get('mean_steps') or 0.0):+.1f} "
        f"mean_final_dist delta={float(deltas.get('mean_final_cube_target_dist') or 0.0):+.4f}"
    )
    for note in comparison["interpretation"]:
        print(f"[CompareRollout] {note}")


if __name__ == "__main__":
    _main(_parse_args())
