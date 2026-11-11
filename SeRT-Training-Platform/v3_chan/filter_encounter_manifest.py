from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEVERITY_ORDER = ("safe", "gate_only", "near", "near_miss", "collision")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an encounter manifest using safety risk realized by a "
            "frozen task-policy rollout."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--screen-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--min-gate-active-steps", type=int, default=10)
    parser.add_argument("--min-geometry-valid-steps", type=int, default=1)
    parser.add_argument("--previous-retained-count", type=int, default=-1)
    parser.add_argument("--previous-input-count", type=int, default=-1)
    parser.add_argument(
        "--allow-task-failures",
        action="store_true",
        help="Retain risky encounters even when the frozen task policy failed.",
    )
    return parser.parse_args()


def build_realized_risk_manifest(
    manifest_path: str | Path,
    screen_results_path: str | Path,
    output_path: str | Path,
    *,
    min_gate_active_steps: int = 10,
    min_geometry_valid_steps: int = 1,
    require_task_success: bool = True,
    report_path: str | Path | None = None,
    previous_retained_count: int | None = None,
    previous_input_count: int | None = None,
) -> dict[str, Any]:
    if min_gate_active_steps < 1:
        raise ValueError("min_gate_active_steps must be positive")
    if min_geometry_valid_steps < 1:
        raise ValueError("min_geometry_valid_steps must be positive")

    manifest_path = Path(manifest_path).expanduser().resolve()
    screen_results_path = Path(screen_results_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    manifest = _load_json(manifest_path)
    screen_results = _load_json(screen_results_path)
    if manifest.get("schema_version") != "hri_encounter_manifest_v2":
        raise ValueError(f"Unsupported encounter manifest: {manifest_path}")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Encounter manifest has no scenarios")
    episodes = screen_results.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Screen result has no episodes")

    episodes_by_id: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        encounter_id = str(episode.get("encounter_id", ""))
        if not encounter_id:
            raise ValueError("Screen result contains an episode without encounter_id")
        if encounter_id in episodes_by_id:
            raise ValueError(f"Duplicate screened encounter_id: {encounter_id}")
        episodes_by_id[encounter_id] = episode

    retained: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    realized_counts: Counter[str] = Counter()
    condition_counts: dict[str, Counter[str]] = {}
    source_cube_condition_counts: dict[str, Counter[str]] = {}
    target_severity_condition_counts: dict[str, Counter[str]] = {}
    for scenario in scenarios:
        encounter_id = str(scenario.get("id", ""))
        episode = episodes_by_id.get(encounter_id)
        reason = _rejection_reason(
            episode,
            min_gate_active_steps=min_gate_active_steps,
            min_geometry_valid_steps=min_geometry_valid_steps,
            require_task_success=require_task_success,
        )
        metrics = _screen_metrics(episode)
        if reason is None:
            selected = dict(scenario)
            selected["baseline_realized_risk"] = metrics
            retained.append(selected)
            realized_counts[metrics["realized_severity"]] += 1
            status = "retained"
        else:
            reason_counts[reason] += 1
            status = reason
        restoration_mode = str(metrics["restoration_mode"])
        condition_counts.setdefault(restoration_mode, Counter())[status] += 1
        source_cube_key = str(metrics["source_cube_index"])
        source_cube_condition_counts.setdefault(
            source_cube_key, Counter()
        )[status] += 1
        target_severity_key = str(scenario.get("target_severity", ""))
        target_severity_condition_counts.setdefault(
            target_severity_key, Counter()
        )[status] += 1
        report_rows.append(
            {
                "encounter_id": encounter_id,
                "target_severity": str(scenario.get("target_severity", "")),
                "status": status,
                "manifest_source_cube_index": scenario.get("cube_index"),
                **metrics,
            }
        )

    if not retained:
        raise ValueError("Realized-risk filter rejected every encounter")
    missing_results = sorted(
        set(episodes_by_id) - {str(item.get("id", "")) for item in scenarios}
    )
    if missing_results:
        raise ValueError(
            "Screen results contain encounter IDs absent from the manifest: "
            f"{missing_results[:5]}"
        )

    target_counts = Counter(str(item.get("target_severity", "")) for item in retained)
    filtered = dict(manifest)
    filtered["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    filtered["parent_manifest"] = str(manifest_path)
    filtered["scenarios"] = retained
    filtered["scenario_count"] = len(retained)
    filtered["severity_counts"] = {
        severity: int(target_counts.get(severity, 0)) for severity in SEVERITY_ORDER
    }
    filtered["sources"] = _filtered_sources(manifest.get("sources", []), retained)
    filtered["realized_risk_filter"] = {
        "screen_results": str(screen_results_path),
        "screen_seed": screen_results.get("config", {}).get("seed"),
        "task_checkpoint": screen_results.get("config", {}).get("checkpoint", ""),
        "require_task_success": bool(require_task_success),
        "min_gate_active_steps": int(min_gate_active_steps),
        "min_geometry_valid_steps": int(min_geometry_valid_steps),
        "input_scenarios": len(scenarios),
        "retained_scenarios": len(retained),
        "retention_rate": len(retained) / len(scenarios),
        "rejection_counts": dict(sorted(reason_counts.items())),
        "realized_severity_counts": {
            severity: int(realized_counts.get(severity, 0))
            for severity in SEVERITY_ORDER
        },
        "condition_counts": {
            mode: dict(sorted(counts.items()))
            for mode, counts in sorted(condition_counts.items())
        },
        "source_cube_condition_counts": {
            cube_index: dict(sorted(counts.items()))
            for cube_index, counts in sorted(source_cube_condition_counts.items())
        },
        "target_severity_condition_counts": {
            severity: dict(sorted(counts.items()))
            for severity, counts in sorted(
                target_severity_condition_counts.items()
            )
        },
    }

    if (
        previous_retained_count is not None
        and previous_input_count is not None
        and previous_retained_count >= 0
        and previous_input_count > 0
    ):
        filtered["realized_risk_filter"]["previous_result_comparison"] = {
            "previous_retained_scenarios": int(previous_retained_count),
            "previous_input_scenarios": int(previous_input_count),
            "previous_retention_rate": (
                float(previous_retained_count) / float(previous_input_count)
            ),
            "corrected_retained_scenarios": len(retained),
            "corrected_input_scenarios": len(scenarios),
            "corrected_retention_rate": len(retained) / len(scenarios),
            "interpretation": (
                "The previous result used non-source-aligned screening conditions "
                "and must not be interpreted as encounter-intrinsic validity."
            ),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, filtered)
    if report_path is not None:
        report_path = Path(report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            report_path,
            {
                "schema_version": "hri_realized_risk_filter_report_v2",
                "manifest": str(manifest_path),
                "screen_results": str(screen_results_path),
                "output_manifest": str(output_path),
                "filter": filtered["realized_risk_filter"],
                "scenarios": report_rows,
            },
        )
    return filtered


def _rejection_reason(
    episode: dict[str, Any] | None,
    *,
    min_gate_active_steps: int,
    min_geometry_valid_steps: int,
    require_task_success: bool,
) -> str | None:
    if episode is None:
        return "missing_screen_result"
    if not bool(episode.get("source_configuration_available", False)):
        return "source_configuration_unavailable"
    if str(episode.get("restoration_mode", "")) == "unavailable":
        return "source_configuration_unavailable"
    if bool(episode.get("pose_mismatch", False)):
        return "source_configuration_pose_mismatch"
    if require_task_success and not bool(episode.get("success", False)):
        return "task_failure"
    if int(episode.get("encounter_active_count", 0)) <= 0:
        return "encounter_inactive"
    if int(episode.get("geometry_valid_steps", 0)) < min_geometry_valid_steps:
        return "geometry_invalid"
    if int(episode.get("safety_gate_active_count", 0)) < min_gate_active_steps:
        return "insufficient_gate_steps"
    return None


def _screen_metrics(episode: dict[str, Any] | None) -> dict[str, Any]:
    if episode is None:
        return {
            "screen_episode": -1,
            "task_success": False,
            "encounter_active_steps": 0,
            "geometry_valid_steps": 0,
            "gate_active_steps": 0,
            "gate_activation_rate": 0.0,
            "collision_steps": 0,
            "collision_rate": 0.0,
            "near_steps": 0,
            "near_miss_steps": 0,
            "min_surface_gap_m": None,
            "realized_severity": "unknown",
            "source_configuration_available": False,
            "restoration_mode": "unavailable",
            "restoration_reason": "missing_screen_result",
            "source_cube_index": None,
            "screening_cube_index": None,
            "collection_seed": None,
            "source_layout_seed": None,
            "screening_seed": None,
            "cube_pose_restored": False,
            "target_pose_restored": False,
            "pose_mismatch": False,
            "pose_mismatch_reason": "",
            "source_configuration_missing_fields": [],
        }
    return {
        "screen_episode": int(episode.get("episode", -1)),
        "task_success": bool(episode.get("success", False)),
        "encounter_active_steps": int(episode.get("encounter_active_count", 0)),
        "geometry_valid_steps": int(episode.get("geometry_valid_steps", 0)),
        "gate_active_steps": int(episode.get("safety_gate_active_count", 0)),
        "gate_activation_rate": float(episode.get("gate_activation_rate", 0.0)),
        "collision_steps": int(episode.get("collision_steps", 0)),
        "collision_rate": float(episode.get("collision_rate", 0.0)),
        "near_steps": int(episode.get("near_steps", 0)),
        "near_miss_steps": int(episode.get("near_miss_steps", 0)),
        "min_surface_gap_m": float(episode.get("min_surface_gap", 10.0)),
        "realized_severity": str(
            episode.get("encounter_realized_severity", "unknown")
        ),
        "source_configuration_available": bool(
            episode.get("source_configuration_available", False)
        ),
        "restoration_mode": str(episode.get("restoration_mode", "unavailable")),
        "restoration_reason": str(episode.get("restoration_reason", "")),
        "source_cube_index": episode.get("source_cube_index"),
        "screening_cube_index": episode.get("screening_cube_index"),
        "collection_seed": episode.get("collection_seed"),
        "source_layout_seed": episode.get("source_layout_seed"),
        "screening_seed": episode.get("screening_seed", episode.get("seed")),
        "source_layout_id": episode.get("source_layout_id"),
        "screening_layout_id": episode.get(
            "screening_layout_id", episode.get("scene_layout_id")
        ),
        "cube_pose_restored": bool(episode.get("cube_pose_restored", False)),
        "target_pose_restored": bool(episode.get("target_pose_restored", False)),
        "robot_initial_state_restored": bool(
            episode.get("robot_initial_state_restored", False)
        ),
        "pose_mismatch": bool(episode.get("pose_mismatch", False)),
        "pose_mismatch_reason": str(episode.get("pose_mismatch_reason", "")),
        "source_configuration_missing_fields": list(
            episode.get("source_configuration_missing_fields", [])
        ),
        "max_cube_position_error_m": episode.get("max_cube_position_error_m"),
        "max_cube_orientation_error_rad": episode.get(
            "max_cube_orientation_error_rad"
        ),
        "target_position_error_m": episode.get("target_position_error_m"),
        "target_orientation_error_rad": episode.get(
            "target_orientation_error_rad"
        ),
    }


def _filtered_sources(
    sources: Any,
    retained: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(str(item.get("source_path", "")) for item in retained)
    filtered_sources = []
    if isinstance(sources, list):
        for source in sources:
            path = str(source.get("path", ""))
            count = int(counts.get(path, 0))
            if count <= 0:
                continue
            item = dict(source)
            item["scenario_count"] = count
            filtered_sources.append(item)
    return filtered_sources


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = _parse_args()
    output = Path(args.output).expanduser().resolve()
    report = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output.with_name(f"{output.stem}_filter_report.json")
    )
    filtered = build_realized_risk_manifest(
        args.manifest,
        args.screen_results,
        output,
        min_gate_active_steps=args.min_gate_active_steps,
        min_geometry_valid_steps=args.min_geometry_valid_steps,
        require_task_success=not args.allow_task_failures,
        report_path=report,
        previous_retained_count=(
            args.previous_retained_count
            if args.previous_retained_count >= 0
            else None
        ),
        previous_input_count=(
            args.previous_input_count if args.previous_input_count >= 0 else None
        ),
    )
    details = filtered["realized_risk_filter"]
    print(
        f"[RealizedRiskFilter] saved={output} "
        f"retained={details['retained_scenarios']}/{details['input_scenarios']} "
        f"realized={details['realized_severity_counts']} report={report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
