"""Focused fail-closed tests for the online explicit-feedback v1 artifact.

The baseline fixture is written through ``OnlineExplicitFeedbackRecorder`` so
the tests exercise the production HDF5 layout instead of maintaining a second
hand-written schema.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import pytest

from v3_chan.explicit_feedback.online_recorder import (
    OnlineExplicitFeedbackRecorder,
)
from v3_chan.explicit_feedback.online_rows import online_row_specs
from v3_chan.explicit_feedback.online_schema import Q1_ID, QueryRecord, TrialPlan
from v3_chan.explicit_feedback.online_scenario import (
    NOMINAL_BC_SWEEP_SEMANTICS,
)
from v3_chan.explicit_feedback.online_validator import (
    OnlineCollectionValidationError,
    canonical_json_sha256,
    validate_online_collection,
)
from v3_chan.explicit_feedback.policy import mask_human_observations
from v3_chan.explicit_feedback.schema import (
    POLICY_SHA256,
    RUNTIME_CONTRACT_SHA256,
    RUNTIME_HANDOFF_COMMIT,
)
from v3_chan.rl.observations import observation_slices


JOINT_NAMES = tuple(
    [f"panda_joint{index}" for index in range(1, 8)]
    + ["panda_finger_joint1", "panda_finger_joint2"]
)
ARM_JOINT_INDICES = tuple(range(7))
TRIAL_PATH = "trials/trial_000000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_COLLIDER = "/World/envs/env_0/panda/panda_hand/collisions"


def _load_json(relative_path: str) -> dict[str, object]:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def _blank_row() -> dict[str, object]:
    row: dict[str, object] = {}
    for path, spec in online_row_specs(
        len(JOINT_NAMES), len(ARM_JOINT_INDICES)
    ).items():
        if spec.dtype == "str":
            row[path] = ""
            continue
        value = np.zeros(spec.tail_shape, dtype=spec.dtype)
        row[path] = value.item() if spec.tail_shape == () else value
    return row


def _valid_row(*, base_monotonic_ns: int, base_unix_ns: int) -> dict[str, object]:
    row = _blank_row()
    raw_observation = np.zeros(84, dtype=np.float32)
    row.update(
        {
            "control/source_step": 0,
            "control/result_step": 1,
            "control/sim_time": 0.1,
            "control/observation_monotonic_ns": base_monotonic_ns + 10,
            "control/preflight_monotonic_ns": base_monotonic_ns + 11,
            "control/policy_inference_started_monotonic_ns": base_monotonic_ns + 12,
            "control/policy_output_monotonic_ns": base_monotonic_ns + 13,
            "control/rmpflow_output_monotonic_ns": base_monotonic_ns + 14,
            "control/cbf_output_monotonic_ns": base_monotonic_ns + 15,
            "control/action_command_monotonic_ns": base_monotonic_ns + 16,
            "control/next_observation_monotonic_ns": base_monotonic_ns + 17,
            "control/post_step_monotonic_ns": base_monotonic_ns + 18,
            "control/wall_time_unix_ns": base_unix_ns + 10,
            "control/unix_time_ns": base_unix_ns + 10,
            "control/control_mode": "nominal_bc",
            "control/trial_state": "crossing",
            "control/trial_id": "practice-000",
            "control/query_id": "query-practice-000",
            "observations/obs_raw": raw_observation,
            "observations/obs_policy": mask_human_observations(
                raw_observation
            ),
            "human/valid_mask": np.ones(3, dtype=np.int8),
            "human_next/valid_mask": np.ones(3, dtype=np.int8),
            "human/left_orientation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "human/right_orientation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "human_next/left_orientation_wxyz": np.asarray(
                [1.0, 0.0, 0.0, 0.0]
            ),
            "human_next/right_orientation_wxyz": np.asarray(
                [1.0, 0.0, 0.0, 0.0]
            ),
            "robot/ee_orientation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "object/cube_orientation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "object/goal_orientation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "safety/left_surface_gap_m": 0.18,
            "safety/right_surface_gap_m": 0.30,
            "safety/min_surface_gap_m": 0.18,
            "safety/left_proximal_surface_gap_m": 10.0,
            "safety/right_proximal_surface_gap_m": 10.0,
            "safety/left_proximal_collider_path": "",
            "safety/right_proximal_collider_path": "",
            "safety/geometry_valid": 1,
            "safety/min_ttc_s": 10.0,
            "safety/dynamic_measurement_valid": 1,
            "task/experimental_phase": "reach_approach",
            "task/phase_advance_enabled": 1,
            "actions/action_pipeline_complete": 1,
            "actions/arm_action_submitted": 1,
            "actions/applied_arm_joint_velocities_mask": np.ones(
                len(ARM_JOINT_INDICES), dtype=np.int8
            ),
            "actions/controller_task_action_reason": "policy",
            "cbf/status": "inactive",
            "cbf/projection_status": "inactive",
            "cbf/intervention_available": 1,
            "cbf/tracked_hand_count": 2,
            "cbf/valid_hand_count": 2,
            "cbf/feasible": 1,
            "cbf/solver_converged": 1,
            "cbf/failure_reasons_json": "[]",
            "cbf/active_constraint_identity_available": 0,
            "cbf/active_constraint_summary_json": json.dumps(
                {
                    "constraint_count": 0,
                    "exact_identity_available": False,
                    "constraints": [],
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            "cbf/identity_semantics": "inactive_or_identity_unavailable",
            "crossing/active": 1,
            "crossing/state": "completed",
            "crossing/intended_completed_now": 1,
            "crossing/progress_fraction": 1.0,
            "crossing/intended_hand": "left",
            "crossing/observed_hand": "left",
            "provenance/runtime_feature_cutoff_available": 1,
        }
    )
    arm = np.asarray(ARM_JOINT_INDICES, dtype=int)
    for stage in ("nominal_rmpflow", "cbf_filtered", "submitted"):
        mask = np.zeros(len(JOINT_NAMES), dtype=np.int8)
        mask[arm] = 1
        row[f"actions/{stage}_joint_velocities_rad_s_mask"] = mask
    for stage in ("pre_applied", "applied"):
        row[f"actions/{stage}_joint_positions_rad_mask"] = np.ones(
            len(JOINT_NAMES), dtype=np.int8
        )
        row[f"actions/{stage}_joint_velocities_rad_s_mask"] = np.ones(
            len(JOINT_NAMES), dtype=np.int8
        )
    return row


def _nominal_calibration_fixture(
    scenario: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    triggers = {
        "reach_approach": 0,
        "grasp_lift": 4,
        "transport": 5,
        "place_release": 6,
    }
    reference_config = scenario["nominal_bc_reference"]
    crossing = scenario["crossing"]
    offsets = [
        -0.20 + index * 0.02
        for index in range(26)
    ]
    references: dict[str, object] = {}
    trajectories: dict[str, object] = {}
    bank: dict[str, object] = {}
    maximum_relative_time = float(crossing["cue_lead_s"]) + float(
        crossing["corridor_length_m"]
    ) / float(crossing["speed_target_m_s"]["slow"])
    sample_interval = float(reference_config["sweep_sample_interval_s"])
    relative_times = [
        sample_interval * index
        for index in range(
            int(math.floor(maximum_relative_time / sample_interval)) + 1
        )
    ]
    if relative_times[-1] < maximum_relative_time:
        relative_times.append(maximum_relative_time)

    for phase, event in triggers.items():
        phase_start = event / 60.0
        reference = {
            "phase": phase,
            "controller_event": event,
            "controller": "bc_only",
            "ee_position_world_m": [0.4, 0.0, 0.4],
            "arm_joint_positions_rad": [0.0] * 7,
            "control_step": event,
            "simulation_time_s": phase_start,
        }
        references[phase] = reference
        samples = [
            {
                "relative_time_s": relative_time,
                "simulation_time_s": phase_start + relative_time,
                "control_step": event + sample_index,
                "controller_event": event,
                "ee_position_world_m": [0.4, 0.0, 0.4],
                "ee_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "arm_joint_positions_rad": [0.0] * 7,
                "arm_joint_velocities_rad_s": [0.0] * 7,
            }
            for sample_index, relative_time in enumerate(relative_times)
        ]
        trajectory = {
            "semantics": NOMINAL_BC_SWEEP_SEMANTICS,
            "phase": phase,
            "controller": "bc_only",
            "cbf_enabled": False,
            "cue_phase_clock_latched": True,
            "task_progression_during_crossing": True,
            "trigger_event": event,
            "sample_interval_target_s": sample_interval,
            "maximum_sample_interval_s": float(
                reference_config["maximum_sweep_sample_interval_s"]
            ),
            "samples": samples,
        }
        trajectory["sha256"] = canonical_json_sha256(trajectory)
        trajectories[phase] = trajectory
        phase_bank: dict[str, object] = {}
        for direction in ("left_to_right", "right_to_left"):
            hand = "left" if direction == "left_to_right" else "right"
            direction_bank: dict[str, object] = {}
            for speed_name in ("slow", "fast"):
                speed_m_s = float(
                    crossing["speed_target_m_s"][speed_name]
                )
                crossing_start = phase_start + float(crossing["cue_lead_s"])
                crossing_end = crossing_start + float(
                    crossing["corridor_length_m"]
                ) / speed_m_s
                crossing_start_index = next(
                    index
                    for index, relative_time in enumerate(relative_times)
                    if phase_start + relative_time >= crossing_start
                )
                pre_count = crossing_start_index
                speed_bank: dict[str, object] = {}
                for severity in ("control", "shallow", "threat"):
                    severity_config = scenario["severity"][severity]
                    target_gap = float(severity_config["target_gap_m"])
                    corridor = {
                        "start_world_m": [0.4, 0.24, 0.4],
                        "end_world_m": [0.4, -0.24, 0.4],
                        "center_world_m": [0.4, 0.0, 0.4],
                        "tangent_world": [0.0, -1.0, 0.0],
                        "outward_normal_world": [1.0, 0.0, 0.0],
                        "planned_minimum_surface_gap_m": target_gap,
                        "closest_link": "panda_hand",
                        "closest_collider_path": PROTECTED_COLLIDER,
                        "calibration_iterations": 1,
                        "geometry_query_semantics": (
                            "physx_protected_surface_to_hand_sphere"
                        ),
                    }
                    candidates = [
                        {
                            "normal_offset_m": offset,
                            "geometry_valid": True,
                            "nominal_minimum_surface_gap_m": target_gap,
                            "nominal_minimum_simulation_time_s": crossing_start,
                            "nominal_minimum_sample_index": crossing_start_index,
                            "nominal_minimum_progress_fraction": 0.0,
                            "closest_link": "panda_hand",
                            "closest_collider_path": PROTECTED_COLLIDER,
                            "crossing_start_surface_gap_m": target_gap,
                            "crossing_start_simulation_time_s": (
                                phase_start + relative_times[crossing_start_index]
                            ),
                            "crossing_start_sample_index": crossing_start_index,
                            "crossing_start_closest_link": "panda_hand",
                            "crossing_start_closest_collider_path": PROTECTED_COLLIDER,
                            "minimum_pre_cue_start_gap_m": 0.16,
                            "minimum_pre_cue_simulation_time_s": phase_start,
                            "minimum_pre_cue_sample_index": 0,
                            "pre_cue_sample_count": pre_count,
                            "crossing_sample_count": len(samples) - pre_count,
                            "invalid_geometry_query_count": 0,
                        }
                        for offset in offsets
                    ]
                    sweep = {
                        "semantics": NOMINAL_BC_SWEEP_SEMANTICS,
                        "phase": phase,
                        "severity": severity,
                        "hand": hand,
                        "crossing_direction": direction,
                        "speed_m_s": speed_m_s,
                        "cue_lead_s": float(crossing["cue_lead_s"]),
                        "corridor_length_m": float(
                            crossing["corridor_length_m"]
                        ),
                        "target_gap_m": target_gap,
                        "allowed_gap_range_m": list(
                            severity_config["minimum_gap_range_m"]
                        ),
                        "selected_normal_offset_m": offsets[0],
                        "nominal_minimum_surface_gap_m": target_gap,
                        "nominal_minimum_simulation_time_s": crossing_start,
                        "nominal_minimum_sample_index": crossing_start_index,
                        "nominal_minimum_progress_fraction": 0.0,
                        "closest_link": "panda_hand",
                        "closest_collider_path": PROTECTED_COLLIDER,
                        "crossing_start_surface_gap_m": target_gap,
                        "crossing_start_query_simulation_time_s": (
                            phase_start + relative_times[crossing_start_index]
                        ),
                        "crossing_start_sample_index": crossing_start_index,
                        "crossing_start_closest_link": "panda_hand",
                        "crossing_start_closest_collider_path": PROTECTED_COLLIDER,
                        "minimum_pre_cue_start_gap_m": 0.16,
                        "required_minimum_pre_cue_start_gap_m": 0.15,
                        "minimum_pre_cue_simulation_time_s": phase_start,
                        "minimum_pre_cue_sample_index": 0,
                        "coverage": {
                            "phase_start_simulation_time_s": phase_start,
                            "crossing_start_simulation_time_s": crossing_start,
                            "crossing_end_simulation_time_s": crossing_end,
                            "first_sample_simulation_time_s": phase_start,
                            "last_sample_simulation_time_s": (
                                phase_start + relative_times[-1]
                            ),
                            "first_crossing_sample_simulation_time_s": (
                                phase_start + relative_times[crossing_start_index]
                            ),
                            "pre_cue_sample_count": pre_count,
                            "crossing_sample_count": len(samples) - pre_count,
                            "time_sample_count": len(samples),
                            "geometry_query_count": len(samples) * len(offsets),
                            "candidate_count": len(offsets),
                            "pre_cue_coverage_fraction": 1.0,
                            "crossing_coverage_fraction": 1.0,
                            "maximum_observed_sample_interval_s": sample_interval,
                            "maximum_allowed_sample_interval_s": float(
                                reference_config[
                                    "maximum_sweep_sample_interval_s"
                                ]
                            ),
                        },
                        "geometry_valid": True,
                        "candidate_results": candidates,
                        "corridor": corridor,
                        "trajectory_sha256": trajectory["sha256"],
                        "trajectory_sample_count": len(samples),
                        "trigger_reference_control_step": event,
                        "trigger_reference_simulation_time_s": phase_start,
                    }
                    sweep["sweep_audit_sha256"] = canonical_json_sha256(
                        {"nominal_bc_sweep": sweep}
                    )
                    speed_bank[severity] = sweep
                direction_bank[speed_name] = speed_bank
            phase_bank[direction] = direction_bank
        bank[phase] = phase_bank
    return references, trajectories, bank


def _build_valid_practice_artifact(tmp_path: Path) -> Path:
    protocol = _load_json(
        "v3_chan/explicit_feedback/configs/online_explicit_feedback_v1.json"
    )
    scenario = _load_json(
        "v3_chan/explicit_feedback/configs/online_scenarios_v1.json"
    )
    cbf_config = {
        "safe_gap_m": 0.05,
        "activation_gap_m": 0.13,
        "gamma_per_s": 8.0,
        "prediction_horizon_s": 0.15,
        "max_prediction_buffer_m": 0.08,
        "max_joint_speed_rad_s": 2.0,
    }
    metadata = {
        "participant_id": "practice-participant",
        "session_id": "practice-session",
        "participant_split_key": "practice-participant",
        "code_commit_sha": "",
        "code_version": "source-sha256:" + "b" * 64,
        "code_version_source": "source_tree_sha256",
        "code_commit_verification": "git_metadata_unavailable",
        "git_worktree_clean": -1,
        "source_tree_sha256": "b" * 64,
        "cbf_config_sha256": canonical_json_sha256(cbf_config),
        "scenario_config_sha256": canonical_json_sha256(scenario),
        "protocol_config_sha256": canonical_json_sha256(protocol),
        "cbf_config_json": json.dumps(
            cbf_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "cbf_protected_links_json": json.dumps(["panda_hand"]),
        "cbf_protected_colliders_json": json.dumps(
            [PROTECTED_COLLIDER]
        ),
        "proximal_protocol_monitor": "usd_world_aabb_conservative_read_only_v1",
        "proximal_link_tokens_json": json.dumps(
            scenario["protected_crossing"]["proximal_link_tokens"]
        ),
        "proximal_collider_paths_json": json.dumps(
            [
                f"/World/envs/env_0/panda/{token}/collisions"
                for token in scenario["protected_crossing"]["proximal_link_tokens"]
            ]
        ),
        "proximal_missing_tokens_json": "[]",
        "runtime_handoff_ready": 1,
        "runtime_handoff_commit": RUNTIME_HANDOFF_COMMIT,
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "runtime_production_collection_ready": 0,
        "live_hmd_qualified": 0,
        "physics_dt_s": 1.0 / 60.0,
        "practice_only": 1,
    }
    output_path = tmp_path / "online-practice.h5"
    recorder = OnlineExplicitFeedbackRecorder(
        output_path,
        metadata=metadata,
        joint_names=JOINT_NAMES,
        arm_joint_indices=ARM_JOINT_INDICES,
        protocol_config=protocol,
        scenario_config=scenario,
    )
    plan = TrialPlan(
        trial_index=0,
        trial_id="practice-000",
        query_id="query-practice-000",
        task_phase="reach_approach",
        severity="control",
        direction="left_to_right",
        speed="slow",
        crossing_hand="left",
        practice=True,
        analysis_exclude=True,
    )
    recorder.write_schedule([plan])

    base_monotonic_ns = time.monotonic_ns()
    base_unix_ns = time.time_ns()
    nominal_references, nominal_trajectories, nominal_bank = (
        _nominal_calibration_fixture(scenario)
    )
    recorder.append_layout_precheck(
        {
            "candidate_index": 0,
            "layout_id": "bc-feasible-layout-000",
            "seed": 101,
            "controller": "bc_only",
            "policy_sha256": POLICY_SHA256,
            "cbf_enabled": False,
            "human_absent_or_clear": True,
            "strict_success": True,
            "release_observed": True,
            "nominal_phase_references_complete": True,
            "nominal_phase_references": nominal_references,
            "nominal_corridor_bank_complete": True,
            "nominal_phase_trajectories": nominal_trajectories,
            "nominal_corridor_bank": nominal_bank,
            "steps": 300,
            "diagnostic_group": "bc_feasible_main",
            "failure_reason": "",
            "started_monotonic_ns": base_monotonic_ns,
            "ended_monotonic_ns": base_monotonic_ns + 1,
            "source_restoration": {"verified": True},
        }
    )
    recorder.start_trial(
        plan,
        seed=202,
        layout_id="bc-feasible-layout-000",
        layout_precheck_index=0,
        planned_gap_min_m=0.16,
        planned_gap_max_m=0.20,
        planned_target_gap_m=0.18,
        start_simulation_time_s=0.0,
        start_monotonic_ns=base_monotonic_ns + 2,
        start_unix_ns=base_unix_ns + 2,
        initial_scene={
            "cube_position_world_m": np.asarray([0.45, 0.0, 0.03]),
            "goal_position_world_m": np.asarray([0.55, 0.2, 0.03]),
        },
    )
    recorder.append_transition(
        _valid_row(
            base_monotonic_ns=base_monotonic_ns,
            base_unix_ns=base_unix_ns,
        )
    )

    issued_monotonic_ns = base_monotonic_ns + 30
    completed_monotonic_ns = base_monotonic_ns + 40
    for question_id, value in (
        (Q1_ID, "acceptable_as_is"),
        ("q2_collision_risk", 1),
        ("q3_excessive_avoidance", 1),
        ("q4_abrupt_unpredictable", 1),
        ("q5_task_disruption", 1),
        ("q6_confidence", 5),
    ):
        recorder.append_query_answer(
            query_id=plan.query_id,
            trial_id=plan.trial_id,
            answer={
                "question_id": question_id,
                "value": value,
                "sim_time_s": 0.85,
                "monotonic_ns": base_monotonic_ns + 35,
                "unix_ns": base_unix_ns + 35,
                "control_step": 1,
            },
        )
    recorder.append_query(
        QueryRecord(
            query_id=plan.query_id,
            trial_id=plan.trial_id,
            encounter_id="",
            issued_simulation_time_s=0.8,
            issued_monotonic_ns=issued_monotonic_ns,
            issued_unix_ns=base_unix_ns + 30,
            completed_simulation_time_s=0.9,
            completed_monotonic_ns=completed_monotonic_ns,
            completed_unix_ns=base_unix_ns + 40,
            response_status="completed",
            q1_response="acceptable_as_is",
            q2_collision_risk=1,
            q3_excessive_avoidance=1,
            q4_abrupt_unpredictable=1,
            q5_task_disruption=1,
            q6_confidence=5,
            response_latency_ms=(completed_monotonic_ns - issued_monotonic_ns)
            / 1e6,
        )
    )
    selected_sweep = nominal_bank["reach_approach"]["left_to_right"][
        "slow"
    ]["control"]
    selected_corridor = selected_sweep["corridor"]
    expected_start_relative_time = (
        selected_sweep["crossing_start_query_simulation_time_s"]
        - selected_sweep["coverage"]["phase_start_simulation_time_s"]
    )
    recorder.end_trial(
        {
            "protocol_valid": True,
            "actual_crossing_count": 1,
            "actual_path_deviation_rms_m": 0.01,
            "actual_path_deviation_max_m": 0.02,
            "actual_crossing_speed_m_s": 0.18,
            "actual_minimum_gap_m": 0.18,
            "actual_minimum_any_hand_gap_m": 0.18,
            "actual_minimum_proximal_gap_m": 10.0,
            "closest_proximal_collider": "",
            "planned_corridor_json": json.dumps(
                {
                    **selected_corridor,
                    "nominal_bc_reference": {
                        "semantics": (
                            "bc_only_phase_trigger_state_on_accepted_layout_v1"
                        ),
                        "phase": "reach_approach",
                        "controller": "bc_only",
                        "controller_event": 0,
                        "precheck_control_step": 0,
                        "precheck_simulation_time_s": 0.0,
                        "ee_position_error_m": 0.0,
                        "maximum_arm_joint_error_rad": 0.0,
                        "match": True,
                    },
                    "nominal_bc_sweep": selected_sweep,
                    "start_recheck": {
                        "semantics": (
                            "actual_start_vs_time_aligned_nominal_sweep_v1"
                        ),
                        "actual_relative_time_s": expected_start_relative_time,
                        "expected_query_relative_time_s": (
                            expected_start_relative_time
                        ),
                        "scheduled_time_error_s": 0.0,
                        "nearest_state_time_error_s": 0.0,
                        "actual_start_surface_gap_m": 0.18,
                        "expected_start_surface_gap_m": 0.18,
                        "start_surface_gap_error_m": 0.0,
                        "ee_position_error_m": 0.0,
                        "maximum_arm_joint_error_rad": 0.0,
                        "actual_closest_link": "panda_hand",
                        "actual_closest_collider_path": PROTECTED_COLLIDER,
                        "geometry_valid": True,
                        "match": True,
                    },
                    "pre_cbf_timeline_audit": {
                        "semantics": (
                            "pre_cbf_actual_vs_bc_only_cue_latched_then_task_advancing_trajectory_v1"
                        ),
                        "compared_sample_count": 1,
                        "maximum_time_alignment_error_s": 0.0,
                        "maximum_ee_position_error_m": 0.0,
                        "maximum_arm_joint_error_rad": 0.0,
                        "comparison_closed_at_cbf_step": -1,
                        "match": True,
                    },
                }
            ),
            "off_protocol": False,
            "off_protocol_reasons": [],
            "cbf_activated": False,
            "cbf_total_intervention_duration_s": 0.0,
            "cbf_max_intervention_norm_rad_s": 0.0,
            "cbf_intervention_count": 0,
            "task_success": True,
            "task_failure_reason": "",
            "object_drop": False,
            "grasp_outcome": "grasped",
            "release_outcome": "released_and_settled",
            "task_completion_observed": True,
            "completion_time_s": 1.0,
            "completion_time_semantics": (
                "active_task_simulation_time_excluding_query_hold_v1"
            ),
            "trial_simulation_duration_s": 1.0,
            "trial_wall_duration_s": 1.0,
            "recovery_time_s": 0.1,
            "risk_onset_step": -1,
            "cbf_intervention_start_step": -1,
            "cbf_intervention_confirmed_step": -1,
            "end_simulation_time_s": 1.0,
            "end_monotonic_ns": base_monotonic_ns + 50,
            "end_unix_ns": base_unix_ns + 50,
        }
    )
    recorder.seal()
    return recorder.commit_validated()


def _mutated_copy(source: Path, tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copyfile(source, destination)
    return destination


def _issue_codes(path: Path) -> set[str]:
    return {
        issue.code
        for issue in validate_online_collection(
            path, raise_on_error=False
        ).issues
    }


def test_minimal_practice_only_artifact_is_valid_but_not_study_eligible(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)

    report = validate_online_collection(artifact, raise_on_error=False)

    assert report.valid is True
    assert report.study_eligible is False
    assert report.trial_count == 1
    assert report.query_count == 1
    assert report.encounter_count == 0


def test_validator_rejects_trial_with_missing_mandatory_query(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "missing-query.h5")
    with h5py.File(mutated, "r+") as handle:
        for dataset in handle["queries"].values():
            dataset.resize((0, *dataset.shape[1:]))
        handle.attrs["query_count"] = 0

    assert "feedback.query_count" in _issue_codes(mutated)


def test_validator_rejects_timeout_that_carries_inferred_labels(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "timeout-with-labels.h5")
    with h5py.File(mutated, "r+") as handle:
        handle["queries/response_status"][0] = "timeout"

    assert "feedback.missing_semantics" in _issue_codes(mutated)


def test_validator_rejects_filtered_to_submitted_action_divergence(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "action-divergence.h5")
    with h5py.File(mutated, "r+") as handle:
        trial = handle[TRIAL_PATH]
        trial["actions/cbf_filtered_joint_velocities_rad_s_mask"][0, 0] = 1
        trial["actions/submitted_joint_velocities_rad_s_mask"][0, 0] = 1
        trial["actions/cbf_filtered_joint_velocities_rad_s"][0, 0] = 0.1
        trial["actions/submitted_joint_velocities_rad_s"][0, 0] = 0.2

    assert "action.filtered_submitted" in _issue_codes(mutated)


def test_validator_recomputes_cbf_intervention_norm(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "forged-cbf-norm.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/cbf/intervention_norm_radps"][0] = 0.2

    assert "action.cbf_norm_mismatch" in _issue_codes(mutated)


def test_validator_rejects_unsafe_cbf_diagnostics(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "cbf-fallback.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/cbf/fallback_applied"][0] = 1

    assert "cbf.unsafe_diagnostics" in _issue_codes(mutated)


def test_validator_rejects_submitted_to_applied_divergence(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "applied-divergence.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/actions/applied_joint_velocities_rad_s"][
            0, 0
        ] = 0.1

    assert "action.submitted_applied_mismatch" in _issue_codes(mutated)


def test_validator_rejects_forged_severity_corridor(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "severity-drift.h5")
    with h5py.File(mutated, "r+") as handle:
        trial = handle[TRIAL_PATH]
        corridor = json.loads(trial.attrs["planned_corridor_json"])
        corridor["planned_minimum_surface_gap_m"] = 0.02
        trial.attrs["planned_corridor_json"] = json.dumps(corridor)

    assert "severity.realization" in _issue_codes(mutated)


def test_validator_rejects_nominal_reference_mismatch(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "nominal-reference-drift.h5")
    with h5py.File(mutated, "r+") as handle:
        trial = handle[TRIAL_PATH]
        corridor = json.loads(trial.attrs["planned_corridor_json"])
        corridor["nominal_bc_reference"]["match"] = False
        trial.attrs["planned_corridor_json"] = json.dumps(corridor)

    assert "severity.nominal_reference" in _issue_codes(mutated)


def test_validator_rejects_incomplete_nominal_corridor_bank(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "incomplete-nominal-bank.h5")
    with h5py.File(mutated, "r+") as handle:
        handle["layout_prechecks/nominal_corridor_bank_complete"][0] = 0

    assert "layout.precheck" in _issue_codes(mutated)


def test_validator_rejects_mutated_nominal_trajectory_hash(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "nominal-trajectory-hash.h5")
    with h5py.File(mutated, "r+") as handle:
        dataset = handle["layout_prechecks/nominal_phase_trajectories_json"]
        trajectories = json.loads(dataset[0])
        trajectories["reach_approach"]["samples"][1][
            "ee_position_world_m"
        ][0] += 0.01
        dataset[0] = json.dumps(trajectories)

    assert "layout.nominal_trajectory_hash" in _issue_codes(mutated)


def test_validator_rejects_trial_sweep_not_in_precheck_bank(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "trial-sweep-relation.h5")
    with h5py.File(mutated, "r+") as handle:
        trial = handle[TRIAL_PATH]
        corridor = json.loads(trial.attrs["planned_corridor_json"])
        corridor["nominal_bc_sweep"]["trajectory_sample_count"] += 1
        trial.attrs["planned_corridor_json"] = json.dumps(corridor)

    codes = _issue_codes(mutated)
    assert "severity.nominal_sweep_relation" in codes
    assert "severity.nominal_sweep_hash" in codes


def test_validator_rejects_failed_start_recheck_on_valid_trial(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "failed-start-recheck.h5")
    with h5py.File(mutated, "r+") as handle:
        trial = handle[TRIAL_PATH]
        corridor = json.loads(trial.attrs["planned_corridor_json"])
        corridor["start_recheck"]["match"] = False
        trial.attrs["planned_corridor_json"] = json.dumps(corridor)

    assert "severity.start_recheck_consistency" in _issue_codes(mutated)


def test_validator_rejects_missing_pre_cbf_comparison_on_valid_trial(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "missing-pre-cbf-audit.h5")
    with h5py.File(mutated, "r+") as handle:
        trial = handle[TRIAL_PATH]
        corridor = json.loads(trial.attrs["planned_corridor_json"])
        corridor["pre_cbf_timeline_audit"]["compared_sample_count"] = 0
        corridor["pre_cbf_timeline_audit"]["match"] = False
        trial.attrs["planned_corridor_json"] = json.dumps(corridor)

    assert "severity.nominal_timeline" in _issue_codes(mutated)


def test_validator_pins_questionnaire_and_vr_input_source(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "questionnaire-drift.h5")
    with h5py.File(mutated, "r+") as handle:
        handle.attrs["q1_text_ko"] = "변경된 질문"
        handle["queries/input_source"][0] = "web_ui"

    codes = _issue_codes(mutated)
    assert "feedback.questionnaire_contract" in codes
    assert "feedback.input_source" in codes


def test_validator_rejects_controlled_phase_clock_drift(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "phase-clock-drift.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/control/trial_state"][0] = "pre_crossing"
        handle[f"{TRIAL_PATH}/task/phase_advance_enabled"][0] = 0
        handle[f"{TRIAL_PATH}/task/controller_event_after"][0] = 1

    assert "task.phase_latch" in _issue_codes(mutated)


def test_validator_rejects_schedule_metadata_drift(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "schedule-metadata-drift.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[TRIAL_PATH].attrs["pilot"] = 1
        handle[TRIAL_PATH].attrs["source_trial_id"] = "forged-source"

    assert "schedule.mismatch" in _issue_codes(mutated)


def test_validator_recomputes_crossing_completion_events(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "crossing-event-drift.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/crossing/intended_completed_now"][0] = 0

    codes = _issue_codes(mutated)
    assert "crossing.summary_count" in codes
    assert "crossing.exactly_one" in codes


def test_validator_rejects_step_clock_forgery(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "step-clock-forgery.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/control/sim_time"][0] = 999.0
        handle[f"{TRIAL_PATH}/control/unix_time_ns"][0] = 1

    codes = _issue_codes(mutated)
    assert "clock.unix_step" in codes
    assert "clock.trial_bounds" in codes


def test_validator_recomputes_trial_safety_and_cbf_summaries(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "summary-forgery.h5")
    with h5py.File(mutated, "r+") as handle:
        trial = handle[TRIAL_PATH]
        trial.attrs["actual_minimum_gap_m"] = -99.0
        trial.attrs["cbf_total_intervention_duration_s"] = -2.0
        trial.attrs["cbf_max_intervention_norm_rad_s"] = 77.0
        trial.attrs["cbf_intervention_count"] = 99
        trial.attrs["recovery_time_s"] = -5.0

    codes = _issue_codes(mutated)
    assert "safety.summary_gap" in codes
    assert "cbf.summary" in codes
    assert "trial.duration_summary" in codes


def test_validator_rejects_fabricated_encounter_without_risk_rows(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "fabricated-encounter.h5")
    with h5py.File(mutated, "r+") as handle:
        encounter = handle["encounters"]
        values = {
            "encounter_id": "practice-000:encounter-01",
            "trial_id": "practice-000",
            "onset_step": 0,
            "onset_confirmed_step": 2,
            "offset_step_exclusive": 3,
            "onset_simulation_time_s": 0.1,
            "offset_simulation_time_s": 0.1,
            "onset_monotonic_ns": int(
                handle[f"{TRIAL_PATH}/control/post_step_monotonic_ns"][0]
            ),
            "offset_monotonic_ns": int(
                handle[f"{TRIAL_PATH}/control/post_step_monotonic_ns"][0]
            ),
            "window_start_step": 0,
            "window_end_step_exclusive": 3,
            "minimum_surface_gap_m": 0.18,
            "maximum_intervention_norm_rad_s": 0.0,
            "risk_onset_step": 0,
            "cbf_intervention_start_step": -1,
            "cbf_intervention_confirmed_step": -1,
            "encounter_timeout": 0,
            "merged_reentry_count": 0,
        }
        for name, dataset in encounter.items():
            dataset.resize((1,))
            dataset[0] = values[name]
        handle.attrs["encounter_count"] = 1
        handle["queries/encounter_id"][0] = values["encounter_id"]
        trial = handle[TRIAL_PATH]
        trial.attrs["encounter_id"] = values["encounter_id"]
        trial.attrs["risk_onset_step"] = 0

    assert "encounter.step_relation" in _issue_codes(mutated)


def test_nonpractice_artifact_requires_qualified_runtime(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "unqualified-study.h5")
    with h5py.File(mutated, "r+") as handle:
        handle.attrs["practice_only"] = 0

    codes = _issue_codes(mutated)
    assert "runtime.production_readiness" in codes
    assert "runtime.live_hmd_qualification" in codes
    assert "integrity.code_provenance" in codes


def test_off_protocol_evaluated_trial_is_retained_but_not_study_eligible(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "invalid-evaluated.h5")
    with h5py.File(mutated, "r+") as handle:
        handle.attrs["practice_only"] = 0
        handle.attrs["runtime_production_collection_ready"] = 1
        handle.attrs["live_hmd_qualified"] = 1
        handle.attrs["code_commit_sha"] = "a" * 40
        handle.attrs["code_version"] = "a" * 40
        handle.attrs["code_version_source"] = "verified_git_head_clean"
        handle.attrs["code_commit_verification"] = "verified_git_head_clean"
        handle.attrs["git_worktree_clean"] = 1
        handle["schedule/practice"][0] = 0
        handle[TRIAL_PATH].attrs["practice"] = 0
        handle[TRIAL_PATH].attrs["protocol_valid"] = 0
        handle[TRIAL_PATH].attrs["analysis_exclude"] = 1

    report = validate_online_collection(mutated, raise_on_error=False)
    assert report.study_eligible is False
    assert "design.valid_evaluated_count" not in {
        issue.code for issue in report.issues
    }


def test_validator_recomputes_exact_frozen_human_mask(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "unmasked-human-policy-input.h5")
    human_index = observation_slices()["human_left_hand_pos"].start
    with h5py.File(mutated, "r+") as handle:
        trial = handle[TRIAL_PATH]
        trial["observations/obs_raw"][0, human_index] = 0.4
        trial["observations/obs_policy"][0, human_index] = 0.4

    assert "policy.human_mask_mismatch" in _issue_codes(mutated)


def test_validator_rejects_query_table_trial_summary_disagreement(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "query-trial-disagreement.h5")
    with h5py.File(mutated, "r+") as handle:
        query = handle["queries"]
        query["response_status"][0] = "timeout"
        query["q1_response"][0] = ""
        query["q2_collision_risk"][0] = 0
        query["q3_excessive_avoidance"][0] = 0
        query["q4_abrupt_unpredictable"][0] = 0
        query["q5_task_disruption"][0] = 0
        query["q6_confidence"][0] = 0

    codes = _issue_codes(mutated)
    assert "feedback.trial_query_mismatch" in codes
    assert "feedback.incomplete_trial_validity" in codes


def test_missing_required_trial_dataset_returns_closed_report(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "missing-row-dataset.h5")
    with h5py.File(mutated, "r+") as handle:
        del handle[f"{TRIAL_PATH}/control/source_step"]

    report = validate_online_collection(mutated, raise_on_error=False)

    assert report.valid is False
    assert "schema.dataset" in {issue.code for issue in report.issues}


def test_missing_required_query_column_returns_closed_report(
    tmp_path: Path,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "missing-query-column.h5")
    with h5py.File(mutated, "r+") as handle:
        del handle["queries/trial_id"]

    with pytest.raises(OnlineCollectionValidationError):
        validate_online_collection(mutated)
    report = validate_online_collection(mutated, raise_on_error=False)

    assert report.valid is False
    assert "schema.dataset" in {issue.code for issue in report.issues}


def test_scalar_table_column_returns_closed_shape_report(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "scalar-query-column.h5")
    with h5py.File(mutated, "r+") as handle:
        query_group = handle["queries"]
        del query_group["q1_response"]
        query_group.create_dataset(
            "q1_response",
            data="acceptable_as_is",
            dtype=h5py.string_dtype("utf-8"),
        )

    report = validate_online_collection(mutated, raise_on_error=False)

    assert report.valid is False
    assert "schema.shape" in {issue.code for issue in report.issues}


def test_unknown_control_mode_is_rejected(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "unknown-control-mode.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/control/control_mode"][0] = "adaptive_policy"

    assert "control.mode" in _issue_codes(mutated)


def test_protocol_hold_is_rejected_outside_query_hold(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "hold-during-crossing.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/control/control_mode"][0] = "protocol_hold"
        handle[f"{TRIAL_PATH}/control/trial_state"][0] = "crossing"

    assert "control.mode_state" in _issue_codes(mutated)


def test_protocol_hold_is_allowed_during_query_hold(tmp_path: Path) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(artifact, tmp_path, "hold-during-query.h5")
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/control/control_mode"][0] = "protocol_hold"
        handle[f"{TRIAL_PATH}/control/trial_state"][0] = "query_hold"
        handle[f"{TRIAL_PATH}/task/phase_advance_enabled"][0] = 0

    assert "control.mode_state" not in _issue_codes(mutated)


@pytest.mark.parametrize("trial_state", ["crossing", "recovery"])
def test_nominal_bc_is_allowed_during_crossing_and_recovery(
    tmp_path: Path,
    trial_state: str,
) -> None:
    artifact = _build_valid_practice_artifact(tmp_path)
    mutated = _mutated_copy(
        artifact, tmp_path, f"nominal-bc-{trial_state}.h5"
    )
    with h5py.File(mutated, "r+") as handle:
        handle[f"{TRIAL_PATH}/control/control_mode"][0] = "nominal_bc"
        handle[f"{TRIAL_PATH}/control/trial_state"][0] = trial_state

    codes = _issue_codes(mutated)
    assert "control.mode" not in codes
    assert "control.mode_state" not in codes
