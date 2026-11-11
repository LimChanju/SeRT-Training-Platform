"""Atomic HDF5 recorder for the one-crossing online explicit-feedback study."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .online_rows import online_row_specs
from .online_schema import (
    FEATURE_PROVENANCE_VERSION,
    FUTURE_OUTCOME_FIELDS,
    LIKERT_QUESTIONS,
    MARKER_TYPES,
    ONLINE_PROTOCOL_VERSION,
    ONLINE_ROW_SEMANTICS,
    ONLINE_SCHEMA_VERSION,
    Q1_ID,
    Q1_RESPONSES,
    Q1_TEXT_KO,
    REJECTION_REASONS,
    RUNTIME_OBSERVABLE_FIELDS,
    EncounterRecordV1,
    QueryRecord,
    RealtimeMarkerRecord,
    TrialPlan,
    validate_query_record,
)
from .recorder import (
    DatasetSpec,
    _append_table_row,
    _attribute_value,
    _coerce_value,
    _create_extendable_dataset,
    _write_static_value,
)
from .schema import (
    POLICY_RELATIVE_PATH,
    POLICY_SHA256,
    POLICY_SIZE_BYTES,
    RUNTIME_CONTRACT_SHA256,
    RUNTIME_HANDOFF_COMMIT,
)


class OnlineExplicitFeedbackRecorder:
    """Write to ``.partial`` until the new fail-closed validator approves it."""

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any],
        joint_names: Sequence[str],
        arm_joint_indices: Sequence[int],
        protocol_config: Mapping[str, Any],
        scenario_config: Mapping[str, Any],
        flush_every_steps: int = 1,
    ) -> None:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("online collection requires h5py; NPZ fallback is forbidden") from error
        self.h5py = h5py
        self.output_path = Path(output_path).expanduser().resolve()
        if self.output_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("output_path must end in .h5 or .hdf5")
        self.partial_path = Path(str(self.output_path) + ".partial")
        if self.output_path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite collection output: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.joint_names = tuple(str(value) for value in joint_names)
        self.arm_joint_indices = tuple(int(value) for value in arm_joint_indices)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        if not self.arm_joint_indices or any(
            value < 0 or value >= len(self.joint_names) for value in self.arm_joint_indices
        ):
            raise ValueError("arm_joint_indices are invalid")
        self.flush_every_steps = max(1, int(flush_every_steps))
        self._file = h5py.File(self.partial_path, "x", libver="latest")
        self._datasets: dict[str, Any] = {}
        self._active_trial = None
        self._active_plan: TrialPlan | None = None
        self._trial_length = 0
        self._trial_count = 0
        self._marker_count = 0
        self._query_count = 0
        self._query_answer_count = 0
        self._encounter_count = 0
        self._layout_precheck_count = 0
        self._sealed = False
        self._closed = False
        self._write_root_metadata(metadata, protocol_config, scenario_config)
        self._file.create_group("trials")
        self._file.create_group("bc_infeasible_diagnostic")
        self._create_tables()
        self._file.flush()

    def write_schedule(self, plans: Sequence[TrialPlan]) -> None:
        self._require_open()
        table = self._file["schedule"]
        if table["trial_id"].shape[0] != 0:
            raise RuntimeError("schedule was already written")
        for plan in plans:
            _append_table_row(
                table,
                {
                    "trial_index": int(plan.trial_index),
                    "trial_id": plan.trial_id,
                    "query_id": plan.query_id,
                    "task_phase": plan.task_phase,
                    "severity": plan.severity,
                    "direction": plan.direction,
                    "speed": plan.speed,
                    "crossing_hand": plan.crossing_hand,
                    "practice": int(plan.practice),
                    "anchor_repeat": int(plan.anchor_repeat),
                    "pilot": int(plan.pilot),
                    "analysis_exclude": int(plan.analysis_exclude),
                    "source_trial_id": plan.source_trial_id,
                },
            )
        self._file.attrs["scheduled_trial_count"] = len(plans)
        self._file.flush()

    def append_layout_precheck(self, record: Mapping[str, Any]) -> None:
        self._require_open()
        fields = {
            "candidate_index": int(record["candidate_index"]),
            "layout_id": str(record["layout_id"]),
            "seed": int(record["seed"]),
            "controller": str(record.get("controller", "bc_only")),
            "policy_sha256": str(record["policy_sha256"]),
            "cbf_enabled": int(bool(record.get("cbf_enabled", False))),
            "human_absent_or_clear": int(bool(record.get("human_absent_or_clear", False))),
            "strict_success": int(bool(record.get("strict_success", False))),
            "release_observed": int(bool(record.get("release_observed", False))),
            "nominal_phase_references_complete": int(
                bool(record.get("nominal_phase_references_complete", False))
            ),
            "nominal_phase_references_json": _json(
                record.get("nominal_phase_references", {})
            ),
            "nominal_corridor_bank_complete": int(
                bool(record.get("nominal_corridor_bank_complete", False))
            ),
            "nominal_phase_trajectories_json": _json(
                record.get("nominal_phase_trajectories", {})
            ),
            "nominal_corridor_bank_json": _json(
                record.get("nominal_corridor_bank", {})
            ),
            "steps": int(record.get("steps", 0)),
            "diagnostic_group": str(record.get("diagnostic_group", "")),
            "failure_reason": str(record.get("failure_reason", "")),
            "started_monotonic_ns": int(record.get("started_monotonic_ns", 0)),
            "ended_monotonic_ns": int(record.get("ended_monotonic_ns", 0)),
            "source_restoration_json": _json(record.get("source_restoration", {})),
        }
        _append_table_row(self._file["layout_prechecks"], fields)
        if (
            not fields["strict_success"]
            and fields["diagnostic_group"] == "bc_infeasible_diagnostic"
        ):
            diagnostic = self._file["bc_infeasible_diagnostic"].create_group(
                f"candidate_{self._layout_precheck_count:06d}"
            )
            for name, value in fields.items():
                diagnostic.attrs[name] = value
        self._layout_precheck_count += 1
        self._file.attrs["layout_precheck_count"] = self._layout_precheck_count
        self._file.flush()

    def start_trial(
        self,
        plan: TrialPlan,
        *,
        seed: int,
        layout_id: str,
        layout_precheck_index: int,
        planned_gap_min_m: float,
        planned_gap_max_m: float,
        planned_target_gap_m: float,
        start_simulation_time_s: float,
        start_monotonic_ns: int,
        start_unix_ns: int,
        initial_scene: Mapping[str, Any],
    ) -> None:
        self._require_open()
        if self._active_trial is not None:
            raise RuntimeError("a trial is already active")
        if int(plan.trial_index) != self._trial_count:
            raise ValueError(
                f"trial indices must be contiguous; expected {self._trial_count}, got {plan.trial_index}"
            )
        group = self._file["trials"].create_group(f"trial_{plan.trial_index:06d}")
        attrs = {
            **plan.as_dict(),
            "trial_seed": int(seed),
            "trial_complete": 0,
            "protocol_valid": 0,
            "single_pick": 1,
            "cube_count": 1,
            "intended_crossing_count": 1,
            "actual_crossing_count": 0,
            "query_present": 0,
            "encounter_id": "",
            "layout_id": str(layout_id),
            "layout_precheck_index": int(layout_precheck_index),
            "layout_bc_feasible": 1,
            "planned_gap_min_m": float(planned_gap_min_m),
            "planned_gap_max_m": float(planned_gap_max_m),
            "planned_target_gap_m": float(planned_target_gap_m),
            "start_simulation_time_s": float(start_simulation_time_s),
            "start_monotonic_ns": int(start_monotonic_ns),
            "start_unix_ns": int(start_unix_ns),
            "end_simulation_time_s": 0.0,
            "end_monotonic_ns": 0,
            "end_unix_ns": 0,
            "step_count": 0,
            "off_protocol": 0,
            "off_protocol_reasons_json": "[]",
            "query_response_status": "",
            "q1_primary_response": "",
        }
        for name, value in attrs.items():
            group.attrs[name] = _attribute_value(value)
        action_group = group.create_group("actions")
        string_dtype = self.h5py.string_dtype("utf-8")
        action_group.create_dataset(
            "all_joint_names", data=np.asarray(self.joint_names, dtype=string_dtype)
        )
        action_group.create_dataset(
            "arm_joint_indices", data=np.asarray(self.arm_joint_indices, dtype=np.int32)
        )
        action_group.create_dataset(
            "arm_joint_names",
            data=np.asarray(
                [self.joint_names[index] for index in self.arm_joint_indices],
                dtype=string_dtype,
            ),
        )
        scene = group.create_group("initial_scene")
        for name, value in initial_scene.items():
            _write_static_value(scene, str(name), value, h5py=self.h5py)
        specs = online_row_specs(len(self.joint_names), len(self.arm_joint_indices))
        self._datasets = {
            path: _create_extendable_dataset(group, path, spec, h5py=self.h5py)
            for path, spec in specs.items()
        }
        self._active_trial = group
        self._active_plan = plan
        self._trial_length = 0
        self._file.flush()

    def append_transition(self, row: Mapping[str, Any]) -> None:
        self._require_active_trial()
        specs = online_row_specs(len(self.joint_names), len(self.arm_joint_indices))
        missing = sorted(set(specs) - set(row))
        extra = sorted(set(row) - set(specs))
        if missing or extra:
            raise RuntimeError(f"transition columns mismatch: missing={missing}, extra={extra}")
        converted = {
            path: _coerce_value(row[path], spec, path=path)
            for path, spec in specs.items()
        }
        index = self._trial_length
        for path, value in converted.items():
            dataset = self._datasets[path]
            dataset.resize((index + 1, *dataset.shape[1:]))
            dataset[index] = value
        self._trial_length += 1
        self._active_trial.attrs["step_count"] = self._trial_length
        if self._trial_length % self.flush_every_steps == 0:
            self._file.flush()

    def append_marker(self, record: RealtimeMarkerRecord) -> None:
        self._require_open()
        if record.marker_type not in MARKER_TYPES:
            raise ValueError(f"invalid marker type: {record.marker_type}")
        _append_table_row(self._file["realtime_markers"], record.as_dict())
        self._marker_count += 1
        self._file.attrs["realtime_marker_count"] = self._marker_count
        self._file.flush()

    def append_query(self, record: QueryRecord) -> None:
        self._require_open()
        validate_query_record(record)
        _append_table_row(self._file["queries"], record.as_dict())
        self._query_count += 1
        self._file.attrs["query_count"] = self._query_count
        if self._active_trial is not None and self._active_plan is not None:
            if record.trial_id != self._active_plan.trial_id:
                raise RuntimeError("query trial_id does not match active trial")
            self._active_trial.attrs["query_present"] = 1
            self._active_trial.attrs["query_response_status"] = record.response_status
            self._active_trial.attrs["q1_primary_response"] = record.q1_response
            self._active_trial.attrs["encounter_id"] = record.encounter_id
            self._active_trial.attrs["q2_collision_risk"] = record.q2_collision_risk
            self._active_trial.attrs["q3_excessive_avoidance"] = record.q3_excessive_avoidance
            self._active_trial.attrs["q4_abrupt_unpredictable"] = record.q4_abrupt_unpredictable
            self._active_trial.attrs["q5_task_disruption"] = record.q5_task_disruption
            self._active_trial.attrs["q6_confidence"] = record.q6_confidence
            self._active_trial.attrs["rejection_reasons_json"] = _json(record.rejection_reasons)
            self._active_trial.attrs["response_latency_ms"] = record.response_latency_ms
        self._file.flush()

    def append_query_answer(
        self, *, query_id: str, trial_id: str, answer: Mapping[str, Any]
    ) -> None:
        """Preserve controller answer events, including partial timeout audits."""

        self._require_open()
        _append_table_row(
            self._file["query_answers"],
            {
                "query_id": str(query_id),
                "trial_id": str(trial_id),
                "question_id": str(answer["question_id"]),
                "value_json": _json(answer.get("value")),
                "simulation_time_s": float(answer["sim_time_s"]),
                "monotonic_ns": int(answer["monotonic_ns"]),
                "unix_ns": int(answer["unix_ns"]),
                "control_step": int(answer["control_step"]),
            },
        )
        self._query_answer_count += 1
        self._file.attrs["query_answer_count"] = self._query_answer_count
        self._file.flush()

    def append_encounter(self, record: EncounterRecordV1) -> None:
        self._require_open()
        fields = record.as_dict()
        fields["encounter_timeout"] = int(record.encounter_timeout)
        _append_table_row(self._file["encounters"], fields)
        self._encounter_count += 1
        self._file.attrs["encounter_count"] = self._encounter_count
        self._file.flush()

    def end_trial(self, summary: Mapping[str, Any]) -> None:
        self._require_active_trial()
        required = {
            "protocol_valid",
            "actual_crossing_count",
            "actual_path_deviation_rms_m",
            "actual_path_deviation_max_m",
            "actual_crossing_speed_m_s",
            "actual_minimum_gap_m",
            "actual_minimum_any_hand_gap_m",
            "actual_minimum_proximal_gap_m",
            "closest_proximal_collider",
            "planned_corridor_json",
            "off_protocol",
            "off_protocol_reasons",
            "cbf_activated",
            "cbf_total_intervention_duration_s",
            "cbf_max_intervention_norm_rad_s",
            "cbf_intervention_count",
            "task_success",
            "task_failure_reason",
            "object_drop",
            "grasp_outcome",
            "release_outcome",
            "task_completion_observed",
            "completion_time_s",
            "completion_time_semantics",
            "trial_simulation_duration_s",
            "trial_wall_duration_s",
            "recovery_time_s",
            "risk_onset_step",
            "cbf_intervention_start_step",
            "cbf_intervention_confirmed_step",
            "end_simulation_time_s",
            "end_monotonic_ns",
            "end_unix_ns",
        }
        missing = sorted(required - set(summary))
        if missing:
            raise RuntimeError(f"trial summary missing fields: {missing}")
        off_protocol = bool(summary["off_protocol"])
        response_status = str(self._active_trial.attrs.get("query_response_status", ""))
        protocol_valid = bool(summary["protocol_valid"])
        if int(summary["actual_crossing_count"]) != 1 or off_protocol:
            protocol_valid = False
        if response_status != "completed":
            protocol_valid = False
        attrs = dict(summary)
        attrs["protocol_valid"] = int(protocol_valid)
        attrs["trial_complete"] = 1
        attrs["analysis_exclude"] = int(
            bool(self._active_plan.analysis_exclude) or not protocol_valid
        )
        attrs["off_protocol"] = int(off_protocol)
        attrs["off_protocol_reasons_json"] = _json(summary["off_protocol_reasons"])
        del attrs["off_protocol_reasons"]
        for name, value in attrs.items():
            self._active_trial.attrs[name] = _attribute_value(value)
        self._file.flush()
        self._active_trial = None
        self._active_plan = None
        self._datasets = {}
        self._trial_length = 0
        self._trial_count += 1
        self._file.attrs["trial_count"] = self._trial_count

    def seal(self) -> Path:
        self._require_open()
        if self._active_trial is not None:
            raise RuntimeError("cannot seal while a trial is active")
        if self._trial_count <= 0:
            raise RuntimeError("cannot seal an empty collection")
        self._file.attrs.update(
            {
                "collection_complete": 1,
                "trial_count": self._trial_count,
                "query_count": self._query_count,
                "query_answer_count": self._query_answer_count,
                "encounter_count": self._encounter_count,
                "realtime_marker_count": self._marker_count,
                "layout_precheck_count": self._layout_precheck_count,
                "session_end_monotonic_ns": time.monotonic_ns(),
                "session_end_unix_ns": time.time_ns(),
            }
        )
        self._file.flush()
        self._file.close()
        self._closed = True
        self._sealed = True
        return self.partial_path

    def commit_validated(self) -> Path:
        if not self._sealed or not self._closed:
            raise RuntimeError("seal and validate before commit")
        if self.output_path.exists():
            raise FileExistsError(self.output_path)
        os.replace(self.partial_path, self.output_path)
        return self.output_path

    def abort(self, reason: str) -> Path:
        if self._closed:
            return self.partial_path
        if self._active_trial is not None:
            self._active_trial.attrs.update(
                {
                    "trial_complete": 0,
                    "protocol_valid": 0,
                    "analysis_exclude": 1,
                    "abort_reason": str(reason),
                    "step_count": self._trial_length,
                }
            )
        self._file.attrs.update(
            {
                "collection_complete": 0,
                "collection_abort_reason": str(reason).strip() or "unspecified_abort",
                "session_end_monotonic_ns": time.monotonic_ns(),
                "session_end_unix_ns": time.time_ns(),
            }
        )
        self._file.flush()
        self._file.close()
        self._closed = True
        return self.partial_path

    def _write_root_metadata(
        self,
        metadata: Mapping[str, Any],
        protocol_config: Mapping[str, Any],
        scenario_config: Mapping[str, Any],
    ) -> None:
        required = (
            "participant_id",
            "session_id",
            "code_version",
            "code_version_source",
            "code_commit_verification",
            "source_tree_sha256",
            "cbf_config_sha256",
            "scenario_config_sha256",
            "protocol_config_sha256",
        )
        missing = [name for name in required if not str(metadata.get(name, "")).strip()]
        if missing:
            raise ValueError(f"missing required session metadata: {missing}")
        attrs: dict[str, Any] = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "protocol_version": ONLINE_PROTOCOL_VERSION,
            "row_semantics": ONLINE_ROW_SEMANTICS,
            "feature_provenance_version": FEATURE_PROVENANCE_VERSION,
            "collection_complete": 0,
            "production_mode": 1,
            "policy_path": POLICY_RELATIVE_PATH,
            "policy_sha256": POLICY_SHA256,
            "policy_size_bytes": POLICY_SIZE_BYTES,
            "policy_mode": "direct_bc",
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "runtime_handoff_commit": RUNTIME_HANDOFF_COMMIT,
            "policy_frozen": 1,
            "policy_online_updates": 0,
            "policy_observation_dim": 84,
            "policy_action_dim": 5,
            "human_observations_masked": 1,
            "feedback_is_policy_input": 0,
            "pseudo_errp_enabled": 0,
            "haptics_enabled": 0,
            "cbf_frozen": 1,
            "cbf_feedback_adaptation": 0,
            "cbf_participant_activation_display": 0,
            "cbf_safe_gap_m": 0.05,
            "cbf_activation_gap_m": 0.13,
            "cbf_gamma_per_s": 8.0,
            "cbf_prediction_horizon_s": 0.15,
            "cbf_max_prediction_buffer_m": 0.08,
            "cbf_max_joint_speed_rad_s": 2.0,
            "cbf_scope": "tracked_hands_to_runtime_discovered_distal_panda_colliders",
            "cbf_constraint_identity_semantics": "exact_reconstruction_from_frozen_cbf_filter_input_v1",
            "single_pick": 1,
            "cube_count": 1,
            "intended_crossings_per_trial": 1,
            "queries_per_trial": 1,
            "strict_task_semantics_schema_version": "physical_event_driven_pick_place_v3",
            "task_require_release": 1,
            "strict_place_xy_tolerance_m": 0.05,
            "task_max_episode_steps": 12000,
            "task_phase_clock_latched_during_pre_crossing_cue": 1,
            "task_progression_during_crossing_and_recovery": 1,
            "task_progression_paused_during_query": 1,
            "query_encounter_nullable": 1,
            "null_string_encoding": "empty_utf8",
            "q1_id": Q1_ID,
            "q1_text_ko": Q1_TEXT_KO,
            "q1_responses_json": _json(Q1_RESPONSES),
            "likert_questions_json": _json(LIKERT_QUESTIONS),
            "rejection_reasons_json": _json(REJECTION_REASONS),
            "realtime_marker_types_json": _json(MARKER_TYPES),
            "runtime_observable_fields_json": _json(RUNTIME_OBSERVABLE_FIELDS),
            "future_outcome_fields_json": _json(FUTURE_OUTCOME_FIELDS),
            "protocol_config_json": _json(protocol_config),
            "scenario_config_json": _json(scenario_config),
            "trial_count": 0,
            "query_count": 0,
            "query_answer_count": 0,
            "encounter_count": 0,
            "realtime_marker_count": 0,
            "layout_precheck_count": 0,
            "session_start_monotonic_ns": time.monotonic_ns(),
            "session_start_unix_ns": time.time_ns(),
        }
        attrs.update(dict(metadata))
        for name, value in attrs.items():
            self._file.attrs[str(name)] = _attribute_value(value)

    def _create_tables(self) -> None:
        tables: dict[str, dict[str, DatasetSpec]] = {
            "schedule": {
                "trial_index": DatasetSpec(np.int64),
                "trial_id": DatasetSpec("str"),
                "query_id": DatasetSpec("str"),
                "task_phase": DatasetSpec("str"),
                "severity": DatasetSpec("str"),
                "direction": DatasetSpec("str"),
                "speed": DatasetSpec("str"),
                "crossing_hand": DatasetSpec("str"),
                "practice": DatasetSpec(np.int8),
                "anchor_repeat": DatasetSpec(np.int8),
                "pilot": DatasetSpec(np.int8),
                "analysis_exclude": DatasetSpec(np.int8),
                "source_trial_id": DatasetSpec("str"),
            },
            "layout_prechecks": {
                "candidate_index": DatasetSpec(np.int64),
                "layout_id": DatasetSpec("str"),
                "seed": DatasetSpec(np.int64),
                "controller": DatasetSpec("str"),
                "policy_sha256": DatasetSpec("str"),
                "cbf_enabled": DatasetSpec(np.int8),
                "human_absent_or_clear": DatasetSpec(np.int8),
                "strict_success": DatasetSpec(np.int8),
                "release_observed": DatasetSpec(np.int8),
                "nominal_phase_references_complete": DatasetSpec(np.int8),
                "nominal_phase_references_json": DatasetSpec("str"),
                "nominal_corridor_bank_complete": DatasetSpec(np.int8),
                "nominal_phase_trajectories_json": DatasetSpec("str"),
                "nominal_corridor_bank_json": DatasetSpec("str"),
                "steps": DatasetSpec(np.int64),
                "diagnostic_group": DatasetSpec("str"),
                "failure_reason": DatasetSpec("str"),
                "started_monotonic_ns": DatasetSpec(np.int64),
                "ended_monotonic_ns": DatasetSpec(np.int64),
                "source_restoration_json": DatasetSpec("str"),
            },
            "realtime_markers": {
                "marker_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "marker_type": DatasetSpec("str"),
                "simulation_time_s": DatasetSpec(np.float64),
                "monotonic_ns": DatasetSpec(np.int64),
                "unix_ns": DatasetSpec(np.int64),
                "control_step": DatasetSpec(np.int64),
                "controller_hand": DatasetSpec("str"),
                "button": DatasetSpec("str"),
            },
            "queries": {
                "query_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "encounter_id": DatasetSpec("str"),
                "issued_simulation_time_s": DatasetSpec(np.float64),
                "issued_monotonic_ns": DatasetSpec(np.int64),
                "issued_unix_ns": DatasetSpec(np.int64),
                "completed_simulation_time_s": DatasetSpec(np.float64),
                "completed_monotonic_ns": DatasetSpec(np.int64),
                "completed_unix_ns": DatasetSpec(np.int64),
                "response_status": DatasetSpec("str"),
                "q1_response": DatasetSpec("str"),
                "q2_collision_risk": DatasetSpec(np.int8),
                "q3_excessive_avoidance": DatasetSpec(np.int8),
                "q4_abrupt_unpredictable": DatasetSpec(np.int8),
                "q5_task_disruption": DatasetSpec(np.int8),
                "q6_confidence": DatasetSpec(np.int8),
                "rejection_reasons_json": DatasetSpec("str"),
                "response_latency_ms": DatasetSpec(np.float64),
                "input_source": DatasetSpec("str"),
            },
            "query_answers": {
                "query_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "question_id": DatasetSpec("str"),
                "value_json": DatasetSpec("str"),
                "simulation_time_s": DatasetSpec(np.float64),
                "monotonic_ns": DatasetSpec(np.int64),
                "unix_ns": DatasetSpec(np.int64),
                "control_step": DatasetSpec(np.int64),
            },
            "encounters": {
                "encounter_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "onset_step": DatasetSpec(np.int64),
                "onset_confirmed_step": DatasetSpec(np.int64),
                "offset_step_exclusive": DatasetSpec(np.int64),
                "onset_simulation_time_s": DatasetSpec(np.float64),
                "offset_simulation_time_s": DatasetSpec(np.float64),
                "onset_monotonic_ns": DatasetSpec(np.int64),
                "offset_monotonic_ns": DatasetSpec(np.int64),
                "window_start_step": DatasetSpec(np.int64),
                "window_end_step_exclusive": DatasetSpec(np.int64),
                "minimum_surface_gap_m": DatasetSpec(np.float64),
                "maximum_intervention_norm_rad_s": DatasetSpec(np.float64),
                "risk_onset_step": DatasetSpec(np.int64),
                "cbf_intervention_start_step": DatasetSpec(np.int64),
                "cbf_intervention_confirmed_step": DatasetSpec(np.int64),
                "encounter_timeout": DatasetSpec(np.int8),
                "merged_reentry_count": DatasetSpec(np.int32),
            },
        }
        for group_name, specs in tables.items():
            group = self._file.require_group(group_name)
            for name, spec in specs.items():
                _create_extendable_dataset(group, name, spec, h5py=self.h5py)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("recorder is closed")

    def _require_active_trial(self) -> None:
        self._require_open()
        if self._active_trial is None or self._active_plan is None:
            raise RuntimeError("no active trial")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
