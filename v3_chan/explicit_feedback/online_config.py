"""Configuration loading and frozen-contract checks for the online study."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .online_schema import ONLINE_SCHEMA_VERSION
from .schema import POLICY_RELATIVE_PATH, POLICY_SHA256


DEFAULT_PROTOCOL_CONFIG = Path(__file__).resolve().parent / "configs" / "online_explicit_feedback_v1.json"
DEFAULT_SCENARIO_CONFIG = Path(__file__).resolve().parent / "configs" / "online_scenarios_v1.json"


# This schema is a frozen study contract, not a general-purpose configuration
# surface.  In particular, pilot subset size is selected by the collector CLI;
# changing a JSON design field must never create a second, unaudited knob.
_FROZEN_PROTOCOL_GROUPS: dict[str, Any] = {
    "protocol_version": "single_pick_controlled_crossing_explicit_feedback_v1",
    "policy": {
        "checkpoint": POLICY_RELATIVE_PATH,
        "sha256": POLICY_SHA256,
        "mode": "direct_bc",
        "frozen": True,
        "online_updates": False,
        "human_observations_masked": True,
        "feedback_is_policy_input": False,
    },
    "task": {
        "cube_count": 1,
        "strict_semantics": "physical_event_driven_pick_place_v3",
        "release_required": True,
        "place_xy_tolerance_m": 0.05,
        "pseudo_errp_enabled": False,
        "phase_event_map": {
            "reach_approach": [0, 1, 2],
            "grasp_lift": [3, 4],
            "transport": [5],
            "place_release": [6, 7],
        },
        "crossing_trigger_event": {
            "reach_approach": 0,
            "grasp_lift": 4,
            "transport": 5,
            "place_release": 6,
        },
        "phase_clock_latched_during_pre_crossing_cue": True,
        "task_progression_during_crossing_and_recovery": True,
        "task_progression_paused_during_query": True,
        "complete_task_after_query": True,
    },
    "cbf": {
        "controller": "cbf",
        "safe_gap_m": 0.05,
        "activation_gap_m": 0.13,
        "gamma_per_s": 8.0,
        "prediction_horizon_s": 0.15,
        "max_prediction_buffer_m": 0.08,
        "max_joint_speed_rad_s": 2.0,
        "frozen": True,
        "feedback_adaptation": False,
        "participant_activation_display": False,
    },
    "feedback": {
        "haptics_enabled": False,
        "query_timeout_s": 90.0,
        "stable_release_frames": 3,
        "opposite_controller_buttons": {
            "left": {
                "controller": "right",
                "realtime_safety_concern": "right.a",
                "realtime_behavior_anomaly": "right.b",
            },
            "right": {
                "controller": "left",
                "realtime_safety_concern": "left.x",
                "realtime_behavior_anomaly": "left.y",
            },
        },
        "navigation": {
            "previous": "left.x",
            "next": "left.y",
            "select": "right.a",
            "submit": "right.b",
        },
    },
    "encounter": {
        "onset_confirmation_frames": 3,
        "activation_gap_m": 0.13,
        "ttc_onset_s": 0.75,
        "intervention_onset_rad_s": 0.05,
        "clear_gap_m": 0.15,
        "ttc_clear_s": 1.0,
        "intervention_clear_rad_s": 0.01,
        "clear_duration_s": 0.5,
        "merge_reentry_s": 1.0,
        "timeout_s": 8.0,
        "window_pre_s": 1.0,
        "window_post_s": 2.0,
    },
    "design": {
        "practice_trials": 4,
        "production_trials": 24,
        "anchor_repeats": 4,
        "pilot_evaluated_trials_default": 16,
        "pilot_evaluated_trials_min": 12,
        "pilot_evaluated_trials_max": 16,
        "speed_counts_production": {"slow": 12, "fast": 12},
    },
}


_FROZEN_SCENARIO_GROUPS: dict[str, Any] = {
    "schema_version": "bc_cbf_online_scenarios_v1",
    "surface_gap_reference": (
        "time_aligned_bc_only_nominal_cbf_protected_surface_to_hypothetical_hand_surface"
    ),
    "actual_gap_reference": (
        "actual_cbf_protected_robot_surface_to_tracked_hand_surface"
    ),
    "severity": {
        "control": {"minimum_gap_range_m": [0.16, 0.20], "target_gap_m": 0.18},
        "shallow": {"minimum_gap_range_m": [0.08, 0.12], "target_gap_m": 0.10},
        "threat": {"minimum_gap_range_m": [0.00, 0.04], "target_gap_m": 0.02},
    },
    "crossing": {
        "corridor_length_m": 0.48,
        "corridor_radius_m": 0.045,
        "start_radius_m": 0.075,
        "end_radius_m": 0.075,
        "maximum_path_deviation_m": 0.08,
        "minimum_progress_fraction": 0.90,
        "maximum_reverse_progress_m": 0.06,
        "speed_target_m_s": {"slow": 0.18, "fast": 0.36},
        "speed_tolerance_fraction": 0.45,
        "cue_lead_s": 1.0,
        "crossing_timeout_s": 8.0,
        "recovery_observation_s": 2.0,
    },
    "nominal_bc_reference": {
        "semantics": (
            "bc_only_cue_latched_then_task_advancing_time_aligned_protected_surface_sweep_v1"
        ),
        "trigger_state_semantics": (
            "bc_only_phase_trigger_state_on_accepted_layout_v1"
        ),
        "maximum_ee_position_error_m": 0.03,
        "maximum_arm_joint_error_rad": 0.10,
        "maximum_start_gap_error_m": 0.03,
        "candidate_outward_offset_range_m": [-0.20, 0.30],
        "candidate_outward_offset_step_m": 0.02,
        "sweep_sample_interval_s": 0.03,
        "maximum_sweep_sample_interval_s": 0.04,
        "minimum_trajectory_samples": 16,
        "minimum_pre_cue_start_gap_m": 0.15,
    },
    "layout_precheck": {
        "required": True,
        "controller": "bc_only",
        "maximum_candidate_attempts": 12,
        "maximum_steps": 4500,
        "hands_clear_gap_m": 0.15,
        "hands_clear_confirmation_frames": 6,
        "failed_group": "bc_infeasible_diagnostic",
        "accepted_group": "bc_feasible_main",
    },
    "protected_crossing": {
        "requested_distal_links": [
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ],
        "proximal_link_tokens": [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
        ],
        "proximal_crossing_action": "mark_off_protocol_and_analysis_exclude",
    },
}


def load_json_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {resolved}")
    return value


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_frozen_protocol_config(protocol: Mapping[str, Any], scenario: Mapping[str, Any]) -> None:
    required = {
        "schema_version": ONLINE_SCHEMA_VERSION,
        "policy.checkpoint": POLICY_RELATIVE_PATH,
        "policy.sha256": POLICY_SHA256,
        "policy.mode": "direct_bc",
        "policy.frozen": True,
        "policy.online_updates": False,
        "policy.human_observations_masked": True,
        "policy.feedback_is_policy_input": False,
        "task.cube_count": 1,
        "task.strict_semantics": "physical_event_driven_pick_place_v3",
        "task.release_required": True,
        "task.place_xy_tolerance_m": 0.05,
        "task.pseudo_errp_enabled": False,
        "task.crossing_trigger_event.reach_approach": 0,
        "task.crossing_trigger_event.grasp_lift": 4,
        "task.crossing_trigger_event.transport": 5,
        "task.crossing_trigger_event.place_release": 6,
        "cbf.controller": "cbf",
        "cbf.safe_gap_m": 0.05,
        "cbf.activation_gap_m": 0.13,
        "cbf.gamma_per_s": 8.0,
        "cbf.prediction_horizon_s": 0.15,
        "cbf.max_prediction_buffer_m": 0.08,
        "cbf.max_joint_speed_rad_s": 2.0,
        "cbf.frozen": True,
        "cbf.feedback_adaptation": False,
        "cbf.participant_activation_display": False,
        "feedback.haptics_enabled": False,
        "feedback.navigation.previous": "left.x",
        "feedback.navigation.next": "left.y",
        "feedback.navigation.select": "right.a",
        "feedback.navigation.submit": "right.b",
        "encounter.onset_confirmation_frames": 3,
        "encounter.activation_gap_m": 0.13,
        "encounter.ttc_onset_s": 0.75,
        "encounter.intervention_onset_rad_s": 0.05,
        "encounter.clear_gap_m": 0.15,
        "encounter.ttc_clear_s": 1.0,
        "encounter.intervention_clear_rad_s": 0.01,
        "encounter.clear_duration_s": 0.5,
        "encounter.merge_reentry_s": 1.0,
        "encounter.timeout_s": 8.0,
        "encounter.window_pre_s": 1.0,
        "encounter.window_post_s": 2.0,
    }
    for path, expected in required.items():
        actual = _get_path(protocol, path)
        if actual != expected:
            raise ValueError(f"frozen protocol mismatch at {path}: expected {expected!r}, got {actual!r}")
    for name, expected in _FROZEN_PROTOCOL_GROUPS.items():
        actual = protocol.get(name)
        if not _frozen_value_matches(actual, expected):
            raise ValueError(
                f"frozen protocol mismatch at {name}: "
                f"expected {expected!r}, got {actual!r}"
            )
    for name, expected in _FROZEN_SCENARIO_GROUPS.items():
        actual = scenario.get(name)
        if not _frozen_value_matches(actual, expected):
            raise ValueError(
                f"frozen scenario mismatch at {name}: "
                f"expected {expected!r}, got {actual!r}"
            )


def actual_cbf_config(cbf: Any) -> dict[str, Any]:
    config = getattr(cbf, "config", None)
    if config is None:
        raise RuntimeError("runtime CBF has no config")
    if is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        value = dict(vars(config))
    result = {str(key): _json_value(item) for key, item in value.items()}
    expected = {
        "safe_gap_m": 0.05,
        "activation_gap_m": 0.13,
        "gamma_per_s": 8.0,
        "prediction_horizon_s": 0.15,
        "max_prediction_buffer_m": 0.08,
        "max_joint_speed_rad_s": 2.0,
    }
    for name, wanted in expected.items():
        if float(result.get(name, float("nan"))) != wanted:
            raise RuntimeError(f"runtime CBF config drift at {name}")
    return result


def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"missing configuration key: {path}")
        current = current[part]
    return current


def _frozen_value_matches(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    try:
        return json.dumps(
            actual, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) == json.dumps(
            expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return False


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
