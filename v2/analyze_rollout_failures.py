from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze failure modes in rollout evaluation JSON.")
    parser.add_argument(
        "--input-json",
        default=os.path.join(SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval.json"),
        help="Rollout evaluation JSON from v2/evaluate_rollout_policy.py.",
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(SCRIPT_DIR, "eval_results", "bc_pick_place_v1_failure_analysis.json"),
        help="Path to save machine-readable failure analysis.",
    )
    parser.add_argument(
        "--output-md",
        default=os.path.join(SCRIPT_DIR, "eval_results", "bc_pick_place_v1_failure_analysis.md"),
        help="Path to save a Markdown failure report.",
    )
    parser.add_argument(
        "--success-dist",
        type=float,
        default=None,
        help="Override success distance. Defaults to the value stored in the rollout config.",
    )
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    data = _load_json(args.input_json)
    episodes = list(data.get("episodes", []))
    if not episodes:
        raise SystemExit(f"No episodes found in {args.input_json}")

    config = data.get("config", {})
    success_dist = float(args.success_dist if args.success_dist is not None else config.get("success_dist", 0.06))
    enriched = [_enrich_episode(row, success_dist) for row in episodes]
    failures = [row for row in enriched if not row["success"]]
    successes = [row for row in enriched if row["success"]]
    failure_categories = Counter(row["failure_category"] for row in failures)

    analysis = {
        "input_json": os.path.abspath(args.input_json),
        "success_dist": success_dist,
        "summary": {
            "episodes": len(enriched),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": len(successes) / len(enriched),
            "failure_categories": dict(failure_categories),
        },
        "overall_stats": _group_stats(enriched),
        "success_stats": _group_stats(successes),
        "failure_stats": _group_stats(failures),
        "failure_by_cube": _failure_by_cube(enriched),
        "worst_failures": sorted(
            failures,
            key=lambda row: float(row["final_cube_target_dist"]),
            reverse=True,
        ),
        "all_failures": failures,
        "recommendations": _recommendations(failure_categories, failures),
    }
    _write_json(args.output_json, analysis)
    _write_markdown(args.output_md, analysis, data)
    _print_summary(analysis)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _enrich_episode(row: dict[str, Any], success_dist: float) -> dict[str, Any]:
    result = dict(row)
    final_dist = float(row.get("final_cube_target_dist", float("inf")))
    min_dist = float(row.get("min_cube_target_dist", float("inf")))
    grasped_any = bool(row.get("grasped_any", False))
    final_has_grasped = bool(row.get("final_has_grasped", False))
    final_event = int(row.get("final_controller_event", -1))
    truncated = bool(row.get("truncated", False))

    if row.get("success", False):
        category = "success"
    elif not grasped_any:
        category = "grasp_never_established"
    elif min_dist <= success_dist and final_dist > success_dist:
        category = "reached_target_then_drifted"
    elif final_event >= 7 and not final_has_grasped and final_dist > success_dist:
        category = "released_outside_success_radius"
    elif truncated:
        category = "timeout_before_success"
    else:
        category = "other_failure"

    result["failure_category"] = category
    result["placement_margin_m"] = final_dist - success_dist
    result["best_placement_margin_m"] = min_dist - success_dist
    return result


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    fields = (
        "steps",
        "total_reward",
        "final_cube_target_dist",
        "min_cube_target_dist",
        "final_ee_cube_dist",
        "min_ee_cube_dist",
        "placement_margin_m",
        "best_placement_margin_m",
    )
    stats: dict[str, float | int] = {"count": len(rows)}
    for field in fields:
        values = np.asarray([float(row.get(field, 0.0)) for row in rows], dtype=np.float32)
        stats[f"mean_{field}"] = float(np.mean(values))
        stats[f"std_{field}"] = float(np.std(values))
        stats[f"min_{field}"] = float(np.min(values))
        stats[f"max_{field}"] = float(np.max(values))
    stats["grasp_rate"] = float(np.mean([bool(row.get("grasped_any", False)) for row in rows]))
    stats["truncated_rate"] = float(np.mean([bool(row.get("truncated", False)) for row in rows]))
    return stats


def _failure_by_cube(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("active_cube", ""))].append(row)
    result = {}
    for cube, cube_rows in sorted(groups.items()):
        failures = [row for row in cube_rows if not row["success"]]
        result[cube] = {
            "episodes": len(cube_rows),
            "failures": len(failures),
            "failure_rate": len(failures) / len(cube_rows),
            "mean_final_cube_target_dist": float(
                np.mean([float(row["final_cube_target_dist"]) for row in cube_rows])
            ),
        }
    return result


def _recommendations(categories: Counter, failures: list[dict[str, Any]]) -> list[str]:
    recs = []
    if not failures:
        return ["No failures observed in this rollout evaluation."]
    if categories.get("released_outside_success_radius", 0) or categories.get("reached_target_then_drifted", 0):
        recs.append(
            "Prioritize placement refinement: add reward/termination pressure for final cube-target distance after release."
        )
        recs.append(
            "Consider a short post-release settle check or a tighter event-6/event-7 release gate before opening the gripper."
        )
    if any(not row.get("final_has_grasped", False) and row.get("grasped_any", False) for row in failures):
        recs.append(
            "The policy usually grasps but releases outside the success radius; grasp reward is less urgent than placement accuracy."
        )
    if categories.get("grasp_never_established", 0):
        recs.append("Some failures never grasped the cube; tune close timing or grasp-phase rewards.")
    if categories.get("timeout_before_success", 0):
        recs.append("Some episodes timed out; inspect whether event progression or action scaling is too slow.")
    return recs


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_markdown(path: str, analysis: dict[str, Any], rollout_data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = analysis["summary"]
    failure_stats = analysis["failure_stats"]
    success_stats = analysis["success_stats"]
    lines = [
        "# Rollout Failure Analysis",
        "",
        f"Input: `{analysis['input_json']}`",
        "",
        "## Summary",
        "",
        f"- Episodes: {summary['episodes']}",
        f"- Successes: {summary['successes']}",
        f"- Failures: {summary['failures']}",
        f"- Success rate: {summary['success_rate']:.3f}",
        f"- Success distance: {analysis['success_dist']:.3f} m",
        "",
        "## Failure Categories",
        "",
    ]
    for category, count in summary["failure_categories"].items():
        lines.append(f"- `{category}`: {count}")
    if not summary["failure_categories"]:
        lines.append("- No failures")

    lines.extend(
        [
            "",
            "## Success vs Failure Stats",
            "",
            "| Group | Count | Mean steps | Mean final dist (m) | Mean min dist (m) | Grasp rate |",
            "|---|---:|---:|---:|---:|---:|",
            _stats_row("success", success_stats),
            _stats_row("failure", failure_stats),
            "",
            "## Failure By Cube",
            "",
            "| Active cube | Episodes | Failures | Failure rate | Mean final dist (m) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cube, stats in analysis["failure_by_cube"].items():
        lines.append(
            f"| {cube} | {stats['episodes']} | {stats['failures']} | "
            f"{stats['failure_rate']:.3f} | {stats['mean_final_cube_target_dist']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Failed Episodes",
            "",
            "| Episode | Seed | Cube | Category | Steps | Final dist (m) | Min dist (m) | Grasped | Final event |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["all_failures"]:
        lines.append(
            f"| {row['episode']} | {row['seed']} | {row['active_cube']} | "
            f"`{row['failure_category']}` | {row['steps']} | "
            f"{row['final_cube_target_dist']:.4f} | {row['min_cube_target_dist']:.4f} | "
            f"{int(bool(row['grasped_any']))} | {row['final_controller_event']} |"
        )

    lines.extend(["", "## Recommendations", ""])
    for rec in analysis["recommendations"]:
        lines.append(f"- {rec}")

    rollout_summary = rollout_data.get("summary", {})
    if rollout_summary:
        lines.extend(
            [
                "",
                "## Source Rollout Summary",
                "",
                "```json",
                json.dumps(rollout_summary, indent=2),
                "```",
            ]
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _stats_row(label: str, stats: dict[str, Any]) -> str:
    if not stats or int(stats.get("count", 0)) == 0:
        return f"| {label} | 0 | - | - | - | - |"
    return (
        f"| {label} | {int(stats['count'])} | {stats['mean_steps']:.1f} | "
        f"{stats['mean_final_cube_target_dist']:.4f} | {stats['mean_min_cube_target_dist']:.4f} | "
        f"{stats['grasp_rate']:.3f} |"
    )


def _print_summary(analysis: dict[str, Any]) -> None:
    summary = analysis["summary"]
    print(
        f"[AnalyzeRollout] episodes={summary['episodes']} failures={summary['failures']} "
        f"success_rate={summary['success_rate']:.3f}"
    )
    for category, count in summary["failure_categories"].items():
        print(f"[AnalyzeRollout] failure_category {category}={count}")
    for row in analysis["all_failures"]:
        print(
            f"[AnalyzeRollout] fail ep={row['episode']:04d} seed={row['seed']} cube={row['active_cube']} "
            f"category={row['failure_category']} final_dist={row['final_cube_target_dist']:.4f} "
            f"min_dist={row['min_cube_target_dist']:.4f} event={row['final_controller_event']}"
        )
    print(f"[AnalyzeRollout] saved json={args.output_json}")
    print(f"[AnalyzeRollout] saved md={args.output_md}")


if __name__ == "__main__":
    args = _parse_args()
    _main(args)
