from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_DIR / "v3_chan" / "filter_encounter_manifest.py"
SPEC = importlib.util.spec_from_file_location("_v3_filter_encounters_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _episode(encounter_id: str, **overrides) -> dict:
    payload = {
        "episode": 0,
        "encounter_id": encounter_id,
        "success": True,
        "encounter_active_count": 40,
        "geometry_valid_steps": 40,
        "safety_gate_active_count": 20,
        "gate_activation_rate": 0.2,
        "collision_steps": 2,
        "collision_rate": 0.02,
        "near_steps": 10,
        "near_miss_steps": 4,
        "min_surface_gap": -0.01,
        "encounter_realized_severity": "collision",
        "source_configuration_available": True,
        "restoration_mode": "exact_pose",
        "restoration_reason": "",
        "source_cube_index": 1,
        "screening_cube_index": 1,
        "collection_seed": 100,
        "source_layout_seed": 101,
        "screening_seed": 7,
        "source_layout_id": "source-layout",
        "screening_layout_id": "source-layout",
        "cube_pose_restored": True,
        "target_pose_restored": True,
        "robot_initial_state_restored": True,
        "pose_mismatch": False,
        "pose_mismatch_reason": "",
        "source_configuration_missing_fields": [],
    }
    payload.update(overrides)
    return payload


def test_filters_by_realized_gate_and_preserves_target_labels(tmp_path):
    manifest_path = tmp_path / "source.json"
    results_path = tmp_path / "screen.json"
    output_path = tmp_path / "filtered.json"
    report_path = tmp_path / "report.json"
    source = str(tmp_path / "session.hdf5")
    _write_json(
        manifest_path,
        {
            "schema_version": "hri_encounter_manifest_v2",
            "sources": [{"path": source, "episode_count": 1, "scenario_count": 3}],
            "scenario_count": 3,
            "severity_counts": {"safe": 1, "near": 1, "collision": 1},
            "scenarios": [
                {"id": "keep", "source_path": source, "target_severity": "safe"},
                {"id": "far", "source_path": source, "target_severity": "collision"},
                {"id": "failed", "source_path": source, "target_severity": "near"},
            ],
        },
    )
    _write_json(
        results_path,
        {
            "config": {"seed": 7, "checkpoint": "task.pt"},
            "episodes": [
                _episode("keep"),
                _episode(
                    "far",
                    episode=1,
                    safety_gate_active_count=0,
                    gate_activation_rate=0.0,
                    collision_steps=0,
                    collision_rate=0.0,
                    min_surface_gap=0.4,
                    encounter_realized_severity="safe",
                ),
                _episode("failed", episode=2, success=False),
            ],
        },
    )

    filtered = MODULE.build_realized_risk_manifest(
        manifest_path,
        results_path,
        output_path,
        report_path=report_path,
        previous_retained_count=234,
        previous_input_count=400,
    )

    assert filtered["scenario_count"] == 1
    assert filtered["scenarios"][0]["id"] == "keep"
    assert filtered["scenarios"][0]["target_severity"] == "safe"
    assert (
        filtered["scenarios"][0]["baseline_realized_risk"]["realized_severity"]
        == "collision"
    )
    assert filtered["severity_counts"]["safe"] == 1
    assert filtered["sources"][0]["scenario_count"] == 1
    assert filtered["realized_risk_filter"]["rejection_counts"] == {
        "insufficient_gate_steps": 1,
        "task_failure": 1,
    }
    assert output_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [item["status"] for item in report["scenarios"]] == [
        "retained",
        "insufficient_gate_steps",
        "task_failure",
    ]
    assert report["schema_version"] == "hri_realized_risk_filter_report_v2"
    retained_report = report["scenarios"][0]
    assert retained_report["restoration_mode"] == "exact_pose"
    assert retained_report["source_cube_index"] == 1
    assert retained_report["screening_cube_index"] == 1
    assert retained_report["cube_pose_restored"] is True
    comparison = report["filter"]["previous_result_comparison"]
    assert comparison["previous_retained_scenarios"] == 234
    assert comparison["previous_input_scenarios"] == 400
    assert report["filter"]["source_cube_condition_counts"] == {
        "1": {
            "insufficient_gate_steps": 1,
            "retained": 1,
            "task_failure": 1,
        }
    }
    assert report["filter"]["target_severity_condition_counts"] == {
        "collision": {"insufficient_gate_steps": 1},
        "near": {"task_failure": 1},
        "safe": {"retained": 1},
    }


def test_unavailable_source_configuration_is_not_gate_rejection(tmp_path):
    manifest_path = tmp_path / "source.json"
    results_path = tmp_path / "screen.json"
    output_path = tmp_path / "filtered.json"
    source = str(tmp_path / "session.hdf5")
    _write_json(
        manifest_path,
        {
            "schema_version": "hri_encounter_manifest_v2",
            "sources": [{"path": source, "episode_count": 1, "scenario_count": 2}],
            "scenarios": [
                {"id": "keep", "source_path": source, "target_severity": "near"},
                {"id": "missing", "source_path": source, "target_severity": "near"},
            ],
        },
    )
    _write_json(
        results_path,
        {
            "episodes": [
                _episode("keep"),
                _episode(
                    "missing",
                    episode=1,
                    source_configuration_available=False,
                    restoration_mode="unavailable",
                    restoration_reason="source_configuration_missing",
                    safety_gate_active_count=0,
                ),
            ]
        },
    )

    filtered = MODULE.build_realized_risk_manifest(
        manifest_path,
        results_path,
        output_path,
    )

    assert filtered["scenario_count"] == 1
    assert filtered["realized_risk_filter"]["rejection_counts"] == {
        "source_configuration_unavailable": 1
    }


def test_rejects_duplicate_screen_results(tmp_path):
    manifest_path = tmp_path / "source.json"
    results_path = tmp_path / "screen.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "hri_encounter_manifest_v2",
            "scenarios": [{"id": "same", "target_severity": "near"}],
        },
    )
    _write_json(
        results_path,
        {"episodes": [_episode("same"), _episode("same", episode=1)]},
    )

    try:
        MODULE.build_realized_risk_manifest(
            manifest_path,
            results_path,
            tmp_path / "filtered.json",
        )
    except ValueError as exc:
        assert "Duplicate screened encounter_id" in str(exc)
    else:
        raise AssertionError("duplicate screen results should fail")
