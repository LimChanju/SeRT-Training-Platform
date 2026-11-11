"""Atomic HDF5 recorder for the one-crossing online explicit-feedback study."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .online_rows import online_row_specs
from .online_schema import (
    ACTUAL_CROSSING_DIRECTIONS,
    ACTUAL_CROSSING_HANDS,
    FEATURE_PROVENANCE_VERSION,
    FUTURE_OUTCOME_FIELDS,
    LIKERT_QUESTIONS,
    MARKER_TYPES,
    MODIFICATION_REASONS,
    ONLINE_PROTOCOL_VERSION,
    ONLINE_ROW_SEMANTICS,
    ONLINE_SCHEMA_VERSION,
    Q1_ID,
    Q1_RESPONSES,
    Q1_TEXT_KO,
    RESPONSE_PHASES,
    RUNTIME_OBSERVABLE_FIELDS,
    EncounterRecordV1,
    QueryRecord,
    RealtimeMarkerRecord,
    TrialPlan,
    exact_bool,
    validate_query_record,
)
from .study import TrialState, validate_decision_context
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
        frozen_config: Mapping[str, Any],
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
        if (
            not self.arm_joint_indices
            or len(set(self.arm_joint_indices)) != len(self.arm_joint_indices)
            or any(
                value < 0 or value >= len(self.joint_names)
                for value in self.arm_joint_indices
            )
        ):
            raise ValueError("arm_joint_indices are invalid")
        self.flush_every_steps = max(1, int(flush_every_steps))
        self._file = h5py.File(self.partial_path, "x", libver="latest")
        self._datasets: dict[str, Any] = {}
        self._active_trial = None
        self._active_plan: TrialPlan | None = None
        self._trial_length = 0
        self._written_trial_length = 0
        self._row_specs: dict[str, DatasetSpec] = {}
        self._pending_transitions: list[dict[str, Any]] = []
        self._previous_crossing: dict[str, Any] | None = None
        self._trial_count = 0
        self._marker_count = 0
        self._marker_ids: set[str] = set()
        self._trial_marker_counts = {marker_type: 0 for marker_type in MARKER_TYPES}
        self._query_count = 0
        self._query_answer_count = 0
        self._encounter_count = 0
        self._layout_precheck_count = 0
        self._schedule: tuple[TrialPlan, ...] = ()
        self._participant_id = str(metadata.get("participant_id", "")).strip()
        self._session_id = str(metadata.get("session_id", "")).strip()
        direct_live = scenario_config.get("direct_live_scenario", {})
        self._direct_live = bool(
            isinstance(direct_live, Mapping)
            and direct_live.get("enabled", False)
        )
        self._direct_live_layout_id = (
            str(direct_live.get("expected_layout_id", ""))
            if isinstance(direct_live, Mapping)
            else ""
        )
        self._sealed = False
        self._closed = False
        self._write_root_metadata(
            metadata, protocol_config, scenario_config, frozen_config
        )
        self._file.create_group("trials")
        self._file.create_group("bc_infeasible_diagnostic")
        self._create_tables()
        self._file.flush()

    def write_schedule(self, plans: Sequence[TrialPlan]) -> None:
        self._require_open()
        table = self._file["schedule"]
        if table["trial_id"].shape[0] != 0:
            raise RuntimeError("schedule was already written")
        frozen_plans = tuple(plans)
        if not frozen_plans:
            raise ValueError("the preassigned trial schedule cannot be empty")
        if tuple(plan.trial_index for plan in frozen_plans) != tuple(
            range(len(frozen_plans))
        ):
            raise ValueError("scheduled trial indices must be contiguous from zero")
        for field in ("trial_id", "query_id", "encounter_id"):
            values = [str(getattr(plan, field)) for plan in frozen_plans]
            if len(set(values)) != len(values):
                raise ValueError(f"scheduled {field} values must be unique")
        if len({plan.schedule_seed for plan in frozen_plans}) != 1:
            raise ValueError("every trial must preserve the same preassigned schedule_seed")
        if len({plan.counterbalancing_group for plan in frozen_plans}) != 1:
            raise ValueError("counterbalancing_group cannot change within a session")
        for plan in frozen_plans:
            _append_table_row(
                table,
                {
                    "trial_index": int(plan.trial_index),
                    "trial_id": plan.trial_id,
                    "query_id": plan.query_id,
                    "encounter_id": plan.encounter_id,
                    "task_phase": plan.task_phase,
                    "severity": plan.severity,
                    "direction": plan.direction,
                    "speed": plan.speed,
                    "crossing_hand": plan.crossing_hand,
                    "condition_id": plan.condition_id,
                    "objective_mode": plan.objective_mode,
                    "lambda_s": float(plan.lambda_s),
                    "schedule_seed": int(plan.schedule_seed),
                    "block_id": plan.block_id,
                    "counterbalancing_group": plan.counterbalancing_group,
                    "practice": int(plan.practice),
                    "anchor_repeat": int(plan.anchor_repeat),
                    "pilot": int(plan.pilot),
                    "analysis_exclude": int(plan.analysis_exclude),
                    "source_trial_id": plan.source_trial_id,
                },
            )
        self._schedule = frozen_plans
        self._file.attrs["scheduled_trial_count"] = len(frozen_plans)
        self._file.attrs["schedule_seed"] = int(frozen_plans[0].schedule_seed)
        self._file.attrs["counterbalancing_group"] = str(
            frozen_plans[0].counterbalancing_group
        )
        self._file.attrs["trial_order_json"] = _json(
            [plan.trial_id for plan in frozen_plans]
        )
        self._file.flush()

    def append_layout_precheck(self, record: Mapping[str, Any]) -> None:
        self._require_open()
        if self._direct_live:
            raise RuntimeError(
                "free-crossing v3 forbids participant-session BC-only prechecks"
            )
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
        condition_config: Mapping[str, Any],
        actual_cbf_config: Mapping[str, Any],
    ) -> None:
        self._require_open()
        if self._active_trial is not None:
            raise RuntimeError("a trial is already active")
        if int(plan.trial_index) != self._trial_count:
            raise ValueError(
                f"trial indices must be contiguous; expected {self._trial_count}, got {plan.trial_index}"
            )
        if not self._schedule:
            raise RuntimeError("write the preassigned schedule before starting trials")
        if plan != self._schedule[self._trial_count]:
            raise RuntimeError(
                "runtime trial plan differs from the immutable preassigned schedule"
            )
        if self._direct_live and (
            int(layout_precheck_index) != -1
            or str(layout_id) != self._direct_live_layout_id
        ):
            raise RuntimeError(
                "direct-live trial must use the frozen layout and no-precheck "
                "sentinel -1"
            )
        condition_payload = dict(condition_config)
        if set(condition_payload) != {"condition_id", "objective_mode", "lambda_s"}:
            raise RuntimeError(
                "condition config must contain exactly "
                "condition_id/objective_mode/lambda_s"
            )
        if (
            str(condition_payload.get("condition_id", plan.condition_id))
            != plan.condition_id
            or str(condition_payload.get("objective_mode", ""))
            != plan.objective_mode
            or float(condition_payload.get("lambda_s", -1.0)) != float(plan.lambda_s)
        ):
            raise RuntimeError("condition config differs from the scheduled A/C assignment")
        actual_cbf_payload = dict(actual_cbf_config)
        if (
            str(actual_cbf_payload.get("objective_mode", "")) != plan.objective_mode
            or float(
                actual_cbf_payload.get("correction_smoothness_weight", -1.0)
            )
            != float(plan.lambda_s)
        ):
            raise RuntimeError("actual runtime CBF config differs from scheduled A/C assignment")
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
            "query_count": 0,
            "encounter_record_count": 0,
            "encounter_id": plan.encounter_id,
            "layout_id": str(layout_id),
            "layout_group": "bc_feasible_main",
            "layout_precheck_index": int(layout_precheck_index),
            "layout_bc_feasible": 1,
            "execution_semantics": (
                "fixed_bc_qualified_layout_without_participant_visible_precheck_v1"
            ),
            "corridor_reference_semantics": (
                "free_single_crossing_internal_bilateral_gate_v1"
            ),
            "crossing_detector_semantics": (
                "free_single_crossing_bilateral_gate_v1"
            ),
            "crossing_instruction_mode": "free_single_crossing",
            "scheduled_crossing_factors_applied": 0,
            "participant_visible_crossing_markers": 0,
            "participant_visible_bc_only_precheck": 0,
            "first_participant_visible_motion_controller": (
                "frozen_bc_plus_assigned_cbf"
            ),
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
            "haptics_enabled": 0,
            "haptic_event_count": 0,
            "query_response_status": "",
            "query_response_disposition": "",
            "q1_primary_response": "",
            "condition_config_json": _json(condition_payload),
            "condition_config_sha256": _sha256_json(condition_payload),
            "actual_cbf_config_json": _json(actual_cbf_payload),
            "actual_cbf_config_sha256": _sha256_json(actual_cbf_payload),
            "trial_fsm_history_json": "[]",
            "decision_context_json": "{}",
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
        self._row_specs = specs
        self._active_trial = group
        self._active_plan = plan
        self._trial_length = 0
        self._written_trial_length = 0
        self._pending_transitions = []
        self._previous_crossing = None
        self._trial_marker_counts = {marker_type: 0 for marker_type in MARKER_TYPES}
        self._file.flush()

    def append_transition(self, row: Mapping[str, Any]) -> None:
        self._require_active_trial()
        assert self._active_plan is not None
        specs = self._row_specs
        missing = sorted(set(specs) - set(row))
        extra = sorted(set(row) - set(specs))
        if missing or extra:
            raise RuntimeError(f"transition columns mismatch: missing={missing}, extra={extra}")
        if (
            str(row["control/trial_id"]) != self._active_plan.trial_id
            or str(row["control/query_id"]) != self._active_plan.query_id
            or str(row["controller/condition_id"]) != self._active_plan.condition_id
            or str(row["controller/objective_mode"]) != self._active_plan.objective_mode
            or float(row["controller/lambda_s"]) != float(self._active_plan.lambda_s)
            or str(row["policy/bc_checkpoint_sha256"]) != POLICY_SHA256
        ):
            raise RuntimeError("transition provenance differs from active trial assignment")
        _validate_free_crossing_transition(row, previous=self._previous_crossing)
        converted = {
            path: _coerce_value(row[path], spec, path=path)
            for path, spec in specs.items()
        }
        self._pending_transitions.append(converted)
        self._trial_length += 1
        self._previous_crossing = {
            "completed_count": int(row["crossing/completed_count"]),
            "first_completed_hand": str(row["crossing/first_completed_hand"]),
            "first_completed_direction": str(
                row["crossing/first_completed_direction"]
            ),
            "multiple_crossing": int(row["crossing/multiple_crossing"]),
        }
        if self._trial_length % self.flush_every_steps == 0:
            self._flush_transition_buffer()

    def _flush_transition_buffer(self) -> None:
        """Append queued control rows with one resize/write per dataset.

        The schema contains hundreds of synchronized columns.  Resizing each
        one for every 60 Hz row made simulation time run several times slower
        than wall time.  Batching changes neither row values nor ordering; all
        managed trial, marker, query, and abort boundaries call this method.
        """

        pending = getattr(self, "_pending_transitions", None)
        if not pending:
            return
        start = int(self._written_trial_length)
        stop = start + len(pending)
        for path, spec in self._row_specs.items():
            dataset = self._datasets[path]
            dataset.resize((stop, *dataset.shape[1:]))
            values = [row[path] for row in pending]
            if spec.dtype == "str":
                dataset[start:stop] = values
            elif spec.tail_shape:
                dataset[start:stop] = np.stack(values, axis=0)
            else:
                dataset[start:stop] = np.asarray(values, dtype=spec.dtype)
        self._active_trial.attrs["step_count"] = self._trial_length
        self._file.flush()
        # Commit the in-memory cursor only after HDF5 accepted the whole batch.
        # If an I/O error is raised, abort() can safely retry the same slice.
        self._written_trial_length = stop
        self._pending_transitions = []

    def append_marker(self, record: RealtimeMarkerRecord) -> None:
        self._require_active_trial()
        self._flush_transition_buffer()
        if record.marker_type not in MARKER_TYPES:
            raise ValueError(f"invalid marker type: {record.marker_type}")
        assert self._active_plan is not None
        expected_button = {
            "realtime_behavior_anomaly": ("right", "a", "right.a"),
            "realtime_safety_concern": ("right", "b", "right.b"),
        }[record.marker_type]
        if (
            record.participant_id != self._participant_id
            or record.session_id != self._session_id
            or record.trial_id != self._active_plan.trial_id
            or record.encounter_id != self._active_plan.encounter_id
            or record.condition_id != self._active_plan.condition_id
            or float(record.lambda_s) != float(self._active_plan.lambda_s)
            or (
                record.controller_hand,
                record.button,
                record.button_identifier,
            )
            != expected_button
        ):
            raise RuntimeError("realtime marker provenance does not match active trial")
        if record.marker_id in self._marker_ids:
            raise RuntimeError(f"duplicate realtime marker_id: {record.marker_id}")
        _append_table_row(self._file["realtime_markers"], record.as_dict())
        self._marker_ids.add(record.marker_id)
        self._marker_count += 1
        self._trial_marker_counts[record.marker_type] += 1
        self._file.attrs["realtime_marker_count"] = self._marker_count
        self._file.flush()

    def append_query(self, record: QueryRecord) -> None:
        self._require_active_trial()
        self._flush_transition_buffer()
        validate_query_record(record)
        assert self._active_plan is not None
        if (
            record.query_id != self._active_plan.query_id
            or record.trial_id != self._active_plan.trial_id
            or record.encounter_id != self._active_plan.encounter_id
        ):
            raise RuntimeError("query identifiers do not match the active trial plan")
        if int(self._active_trial.attrs.get("query_count", 0)) != 0:
            raise RuntimeError("exactly one mandatory query is allowed per trial")
        _append_table_row(self._file["queries"], record.as_dict())
        self._query_count += 1
        self._file.attrs["query_count"] = self._query_count
        self._active_trial.attrs["query_present"] = 1
        self._active_trial.attrs["query_count"] = 1
        self._active_trial.attrs["query_response_status"] = record.response_status
        self._active_trial.attrs["query_response_disposition"] = (
            record.response_disposition
        )
        self._active_trial.attrs["q1_primary_response"] = record.q1_response
        self._active_trial.attrs["encounter_id"] = record.encounter_id
        self._active_trial.attrs["q2_perceived_danger"] = record.q2_perceived_danger
        self._active_trial.attrs["q3_abruptness"] = record.q3_abruptness
        self._active_trial.attrs["q4_excessive_duration"] = record.q4_excessive_duration
        self._active_trial.attrs["q5_task_disruption"] = record.q5_task_disruption
        self._active_trial.attrs["q6_confidence"] = record.q6_confidence
        self._active_trial.attrs["modification_reasons_json"] = _json(
            record.modification_reasons
        )
        self._active_trial.attrs["response_latency_ms"] = record.response_latency_ms
        self._active_trial.attrs["prompt_shown_simulation_time_s"] = (
            record.issued_simulation_time_s
        )
        self._active_trial.attrs["first_input_simulation_time_s"] = (
            record.first_input_simulation_time_s
        )
        self._active_trial.attrs["confirmed_simulation_time_s"] = (
            record.completed_simulation_time_s
        )
        self._active_trial.attrs["feedback_input_device"] = record.input_device
        self._active_trial.attrs["back_correction_count"] = (
            record.back_correction_count
        )
        self._active_trial.attrs["accidental_input_count"] = (
            record.accidental_input_count
        )
        self._file.flush()

    def append_query_answer(
        self, *, query_id: str, trial_id: str, answer: Mapping[str, Any]
    ) -> None:
        """Preserve controller answer events, including partial timeout audits."""

        self._require_active_trial()
        self._flush_transition_buffer()
        assert self._active_plan is not None
        if (
            str(query_id) != self._active_plan.query_id
            or str(trial_id) != self._active_plan.trial_id
        ):
            raise RuntimeError("query-answer identifiers do not match active trial")
        _append_table_row(
            self._file["query_answers"],
            {
                "query_id": str(query_id),
                "trial_id": str(trial_id),
                "question_id": str(answer["question_id"]),
                "answer_status": str(answer["answer_status"]),
                "value_json": _json(answer.get("value")),
                "prompt_shown_simulation_time_s": float(
                    answer["prompt_shown_simulation_time_s"]
                ),
                "prompt_shown_monotonic_ns": int(
                    answer["prompt_shown_monotonic_ns"]
                ),
                "prompt_shown_unix_ns": int(
                    answer["prompt_shown_unix_ns"]
                ),
                "prompt_shown_control_step": int(
                    answer["prompt_shown_control_step"]
                ),
                "first_input_simulation_time_s": float(
                    answer["first_input_simulation_time_s"]
                ),
                "first_input_monotonic_ns": int(
                    answer["first_input_monotonic_ns"]
                ),
                "first_input_unix_ns": int(answer["first_input_unix_ns"]),
                "first_input_control_step": int(
                    answer["first_input_control_step"]
                ),
                "confirmed_simulation_time_s": float(
                    answer["confirmed_simulation_time_s"]
                ),
                "confirmed_monotonic_ns": int(
                    answer["confirmed_monotonic_ns"]
                ),
                "confirmed_unix_ns": int(answer["confirmed_unix_ns"]),
                "confirmed_control_step": int(
                    answer["confirmed_control_step"]
                ),
                "response_latency_ms": float(answer["response_latency_ms"]),
                "input_device": str(answer["input_device"]),
                "back_correction_count": int(answer["back_correction_count"]),
                "accidental_input_count": int(answer["accidental_input_count"]),
            },
        )
        self._query_answer_count += 1
        self._file.attrs["query_answer_count"] = self._query_answer_count
        self._file.flush()

    def append_encounter(self, record: EncounterRecordV1) -> None:
        self._require_active_trial()
        self._flush_transition_buffer()
        assert self._active_plan is not None
        if (
            record.trial_id != self._active_plan.trial_id
            or record.encounter_id != self._active_plan.encounter_id
        ):
            raise RuntimeError("encounter identifiers do not match active trial")
        if int(self._active_trial.attrs.get("encounter_record_count", 0)) != 0:
            raise RuntimeError("exactly one safety-response encounter is allowed per trial")
        fields = record.as_dict()
        fields["encounter_timeout"] = int(record.encounter_timeout)
        _append_table_row(self._file["encounters"], fields)
        self._active_trial.attrs["encounter_record_count"] = 1
        self._encounter_count += 1
        self._file.attrs["encounter_count"] = self._encounter_count
        self._file.flush()

    def end_trial(self, summary: Mapping[str, Any]) -> None:
        self._require_active_trial()
        self._flush_transition_buffer()
        required = {
            "protocol_valid",
            "actual_crossing_count",
            "actual_crossing_hand",
            "actual_crossing_direction",
            "crossing_start_step",
            "crossing_end_step",
            "crossing_start_simulation_time_s",
            "crossing_end_simulation_time_s",
            "crossing_detector_semantics",
            "crossing_monitoring_valid_rate",
            "actual_path_deviation_rms_m",
            "actual_path_deviation_max_m",
            "actual_crossing_speed_m_s",
            "tracking_validity_rate",
            "actual_minimum_gap_m",
            "actual_minimum_any_hand_gap_m",
            "actual_minimum_proximal_gap_m",
            "closest_proximal_collider",
            "planned_corridor_json",
            "off_protocol",
            "off_protocol_reasons",
            "cbf_activated",
            "response_onset_step",
            "response_onset_confirmed_step",
            "response_end_step_exclusive",
            "response_onset_simulation_time_s",
            "response_end_simulation_time_s",
            "recovery_onset_step",
            "recovery_end_step_exclusive",
            "recovery_onset_simulation_time_s",
            "recovery_end_simulation_time_s",
            "stable_task_resumption_step",
            "cbf_active_duration_s",
            "smooth_tail_duration_s",
            "recovery_duration_s",
            "bc_resumed_duration_s",
            "total_safety_response_duration_s",
            "integrated_intervention_rad",
            "correction_total_variation_rad_s",
            "cbf_max_intervention_norm_rad_s",
            "cbf_intervention_count",
            "windowed_jerk_json",
            "response_phase_history_json",
            "response_phase_durations_json",
            "delta_path_m",
            "path_comparison_semantics",
            "actual_task_ee_path_m",
            "baseline_bc_task_ee_path_m",
            "delta_completion_time_s",
            "completion_time_comparison_semantics",
            "actual_task_active_completion_time_s",
            "baseline_bc_task_completion_time_s",
            "decision_context_json",
            "trial_fsm_history_json",
            "realtime_safety_marker_count",
            "realtime_behavior_anomaly_marker_count",
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
        summary_bools = {
            name: bool(exact_bool(summary[name], name=f"trial summary {name}"))
            for name in (
                "protocol_valid", "off_protocol", "cbf_activated",
                "task_success", "object_drop", "task_completion_observed",
            )
        }
        raw_off_protocol_reasons = summary["off_protocol_reasons"]
        if not isinstance(raw_off_protocol_reasons, (list, tuple)) or any(
            not isinstance(reason, str) or not reason.strip()
            for reason in raw_off_protocol_reasons
        ):
            raise RuntimeError(
                "off_protocol_reasons must be a list of non-empty strings"
            )
        off_protocol_reasons = tuple(sorted(set(raw_off_protocol_reasons)))
        missing_onset = "response_onset_not_confirmed" in off_protocol_reasons
        missing_stable = (
            "stable_task_resumption_not_confirmed" in off_protocol_reasons
        )
        if missing_onset and not missing_stable:
            raise RuntimeError(
                "missing response onset also requires missing stable resumption"
            )
        if (missing_onset or missing_stable) and (
            not summary_bools["off_protocol"]
            or summary_bools["protocol_valid"]
        ):
            raise RuntimeError(
                "incomplete response episodes must be explicit invalid "
                "off-protocol trials"
            )
        assert self._active_trial is not None and self._active_plan is not None
        for name, expected in (
            ("condition_id", self._active_plan.condition_id),
            ("objective_mode", self._active_plan.objective_mode),
            ("lambda_s", float(self._active_plan.lambda_s)),
        ):
            if name in summary and summary[name] != expected:
                raise RuntimeError(f"final trial {name} differs from its assignment")
        if "actual_cbf_config_json" in summary:
            final_cbf_config = _decoded_json(
                summary["actual_cbf_config_json"], expected_type=dict
            )
            initial_cbf_config = json.loads(
                str(self._active_trial.attrs["actual_cbf_config_json"])
            )
            if final_cbf_config != initial_cbf_config:
                raise RuntimeError("actual CBF configuration changed within the trial")
        history = _decoded_json(summary["trial_fsm_history_json"], expected_type=list)
        expected_history = [state.value for state in TrialState]
        if history != expected_history:
            raise RuntimeError("trial FSM history is not the exact frozen 17-state sequence")
        decision_context = _decoded_json(
            summary["decision_context_json"], expected_type=dict
        )
        if decision_context:
            validate_decision_context(decision_context)
            if (
                str(decision_context.get("condition_id"))
                != self._active_plan.condition_id
                or float(decision_context.get("lambda_s", -1.0))
                != float(self._active_plan.lambda_s)
            ):
                raise RuntimeError(
                    "decision context condition differs from active trial"
                )
        elif summary_bools["protocol_valid"]:
            raise RuntimeError(
                "a protocol-valid trial requires a complete pre-response decision context"
            )
        response_phase_history = _decoded_json(
            summary["response_phase_history_json"], expected_type=list
        )
        history_phases: list[str] = []
        previous_step = -1
        previous_simulation_time_s = -math.inf
        for index, item in enumerate(response_phase_history):
            if not isinstance(item, Mapping):
                raise RuntimeError(
                    f"response phase history entry {index} must be an object"
                )
            phase_name = str(item.get("phase", ""))
            control_step = item.get("control_step")
            simulation_time_s = float(item.get("simulation_time_s", math.nan))
            if phase_name not in RESPONSE_PHASES:
                raise RuntimeError(
                    f"response phase history entry {index} has an invalid phase"
                )
            if (
                isinstance(control_step, bool)
                or not isinstance(control_step, (int, np.integer))
                or int(control_step) < previous_step
                or not math.isfinite(simulation_time_s)
                or simulation_time_s < previous_simulation_time_s
            ):
                raise RuntimeError(
                    f"response phase history entry {index} has invalid clocks"
                )
            if history_phases and phase_name == history_phases[-1]:
                raise RuntimeError(
                    "response phase history must contain transitions, not repeated phases"
                )
            history_phases.append(phase_name)
            previous_step = int(control_step)
            previous_simulation_time_s = simulation_time_s
        if not history_phases or history_phases[0] != "PRE_RESPONSE":
            raise RuntimeError("response phase history must start PRE_RESPONSE")
        if missing_stable:
            if "STABLE_TASK_RESUMPTION" in history_phases:
                raise RuntimeError(
                    "missing-stable trial cannot contain a stable response transition"
                )
        elif history_phases[-1] != "STABLE_TASK_RESUMPTION":
            raise RuntimeError(
                "complete response phase history must end "
                "STABLE_TASK_RESUMPTION"
            )
        windowed_jerk = _decoded_json(
            summary["windowed_jerk_json"], expected_type=dict
        )
        required_windows = {
            "PRE_RESPONSE",
            "CBF_ACTIVE",
            "SMOOTH_TAIL",
            "RECOVERY_ACTIVE",
            "BC_RESUMED_FIRST_0_5_S",
            "WHOLE_TASK",
        }
        if set(windowed_jerk) != required_windows:
            raise RuntimeError(
                "windowed jerk summary must contain exactly: "
                + ", ".join(sorted(required_windows))
            )
        jerk_channel_fields = {
            "ee": ("sample_count", "rms_norm_m_s3", "peak_norm_m_s3"),
            "joint_space": (
                "sample_count", "rms_norm_rad_s3", "peak_norm_rad_s3"
            ),
        }
        for window_name in required_windows:
            window = windowed_jerk.get(window_name)
            if not isinstance(window, Mapping) or set(window) != set(jerk_channel_fields):
                raise RuntimeError(
                    f"windowed jerk {window_name} must contain exact ee/joint_space channels"
                )
            for channel_name, field_names in jerk_channel_fields.items():
                channel = window[channel_name]
                if not isinstance(channel, Mapping) or set(channel) != set(field_names):
                    raise RuntimeError(
                        f"windowed jerk {window_name}/{channel_name} has invalid fields"
                    )
                sample_count = channel["sample_count"]
                if (
                    isinstance(sample_count, bool)
                    or not isinstance(sample_count, (int, np.integer))
                    or int(sample_count) < 0
                ):
                    raise RuntimeError(
                        f"windowed jerk {window_name}/{channel_name} sample_count is invalid"
                    )
                for field_name in field_names[1:]:
                    value = float(channel[field_name])
                    if not math.isfinite(value) or value < 0.0:
                        raise RuntimeError(
                            f"windowed jerk {window_name}/{channel_name}/{field_name} is invalid"
                        )
        numeric_nonnegative = (
            "actual_path_deviation_rms_m",
            "actual_path_deviation_max_m",
            "actual_crossing_speed_m_s",
            "cbf_active_duration_s",
            "smooth_tail_duration_s",
            "recovery_duration_s",
            "bc_resumed_duration_s",
            "total_safety_response_duration_s",
            "integrated_intervention_rad",
            "correction_total_variation_rad_s",
            "cbf_max_intervention_norm_rad_s",
            "trial_simulation_duration_s",
            "trial_wall_duration_s",
        )
        for name in numeric_nonnegative:
            value = float(summary[name])
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(f"trial summary {name} must be finite and non-negative")
        tracking_validity_rate = float(summary["tracking_validity_rate"])
        if not math.isfinite(tracking_validity_rate) or not 0.0 <= tracking_validity_rate <= 1.0:
            raise RuntimeError("tracking_validity_rate must be inside [0, 1]")
        for name in (
            "actual_minimum_gap_m", "actual_minimum_any_hand_gap_m",
            "actual_minimum_proximal_gap_m", "delta_path_m",
            "delta_completion_time_s", "actual_task_ee_path_m",
            "baseline_bc_task_ee_path_m",
            "actual_task_active_completion_time_s",
            "baseline_bc_task_completion_time_s",
        ):
            if not math.isfinite(float(summary[name])):
                raise RuntimeError(f"trial summary {name} must be finite")
        if (
            str(summary["path_comparison_semantics"])
            != "direct_live_active_task_path_minus_a10_fixed_layout_bc_only_baseline_v1"
            or str(summary["completion_time_comparison_semantics"])
            != "direct_live_active_task_time_minus_a10_fixed_layout_bc_only_baseline_v1"
            or not math.isclose(
                float(summary["delta_path_m"]),
                float(summary["actual_task_ee_path_m"])
                - float(summary["baseline_bc_task_ee_path_m"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(summary["delta_completion_time_s"]),
                float(summary["actual_task_active_completion_time_s"])
                - float(summary["baseline_bc_task_completion_time_s"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError(
                "direct-live task deltas are not bound to their raw A10 baseline values"
            )
        _validate_free_crossing_summary(
            summary,
            self._active_trial,
            protocol_valid=summary_bools["protocol_valid"],
        )
        actual_crossing_direction = str(summary["actual_crossing_direction"])
        if actual_crossing_direction not in ACTUAL_CROSSING_DIRECTIONS:
            raise RuntimeError("actual_crossing_direction is invalid")
        direction_not_observed = actual_crossing_direction == "not_observed"
        direction_not_observed_reason = (
            "actual_crossing_direction_not_observed" in off_protocol_reasons
        )
        if direction_not_observed != direction_not_observed_reason:
            raise RuntimeError(
                "not-observed crossing direction must exactly match its "
                "off-protocol reason"
            )
        if direction_not_observed and not summary_bools["off_protocol"]:
            raise RuntimeError(
                "not-observed crossing direction must be off-protocol"
            )
        if int(summary["actual_crossing_count"]) < 0:
            raise RuntimeError("actual_crossing_count must be non-negative")
        for name in (
            "cbf_intervention_count", "realtime_safety_marker_count",
            "realtime_behavior_anomaly_marker_count",
        ):
            if isinstance(summary[name], bool) or int(summary[name]) < 0:
                raise RuntimeError(f"trial summary {name} must be non-negative")
        if (
            int(summary["realtime_safety_marker_count"])
            != self._trial_marker_counts["realtime_safety_concern"]
            or int(summary["realtime_behavior_anomaly_marker_count"])
            != self._trial_marker_counts["realtime_behavior_anomaly"]
        ):
            raise RuntimeError("trial marker summary disagrees with recorded marker events")
        onset_sim = float(summary["response_onset_simulation_time_s"])
        end_sim = float(summary["response_end_simulation_time_s"])
        if not all(math.isfinite(value) for value in (onset_sim, end_sim)):
            raise RuntimeError("response timestamps must be finite")
        onset_step = int(summary["response_onset_step"])
        confirmed_step = int(summary["response_onset_confirmed_step"])
        end_step = int(summary["response_end_step_exclusive"])
        stable_step = int(summary["stable_task_resumption_step"])
        recovery_step = int(summary["recovery_onset_step"])
        recovery_end_step = int(summary["recovery_end_step_exclusive"])
        recovery_onset_sim = float(
            summary["recovery_onset_simulation_time_s"]
        )
        recovery_end_sim = float(
            summary["recovery_end_simulation_time_s"]
        )
        if not all(
            math.isfinite(value)
            for value in (recovery_onset_sim, recovery_end_sim)
        ):
            raise RuntimeError("Recovery timestamps must be finite")
        if missing_onset:
            if (
                (onset_sim, end_sim) != (-1.0, -1.0)
                or (recovery_onset_sim, recovery_end_sim) != (-1.0, -1.0)
                or (
                    onset_step,
                    confirmed_step,
                    end_step,
                    stable_step,
                    recovery_step,
                    recovery_end_step,
                ) != (-1, -1, -1, -1, -1, -1)
                or any(
                    float(summary[name]) != 0.0
                    for name in (
                        "total_safety_response_duration_s",
                        "integrated_intervention_rad",
                        "correction_total_variation_rad_s",
                        "recovery_duration_s",
                    )
                )
            ):
                raise RuntimeError(
                    "unconfirmed response onset requires exact absent-response sentinels"
                )
        elif missing_stable:
            if (
                onset_sim < 0.0
                or onset_step < 0
                or confirmed_step < onset_step
                or end_sim != -1.0
                or end_step != -1
                or stable_step != -1
                or any(
                    float(summary[name]) != 0.0
                    for name in (
                        "total_safety_response_duration_s",
                        "integrated_intervention_rad",
                        "correction_total_variation_rad_s",
                    )
                )
            ):
                raise RuntimeError(
                    "unstable response timeout requires valid onset and exact missing-end sentinels"
                )
        elif not (
            0.0 <= onset_sim <= end_sim
            and 0 <= onset_step <= confirmed_step < end_step
            and stable_step == end_step - 1
        ):
            raise RuntimeError("complete response bounds are invalid")

        recovery_active = np.asarray(
            self._active_trial["recovery/active"][()], dtype=np.int8
        )
        recovery_onset_rows = np.asarray(
            self._active_trial["recovery/onset_now"][()], dtype=np.int8
        )
        recovery_end_rows = np.asarray(
            self._active_trial["recovery/end_now"][()], dtype=np.int8
        )
        expected_recovery_onsets = (recovery_active == 1) & np.concatenate(
            (np.asarray([True]), recovery_active[:-1] == 0)
        )
        expected_recovery_ends = (recovery_active == 0) & np.concatenate(
            (np.asarray([False]), recovery_active[:-1] == 1)
        )
        if (
            np.any(recovery_onset_rows != expected_recovery_onsets.astype(np.int8))
            or np.any(recovery_end_rows != expected_recovery_ends.astype(np.int8))
        ):
            raise RuntimeError(
                "Recovery onset/end row evidence must exactly encode raw active-mask edges"
            )
        if not missing_onset:
            source_steps = np.asarray(
                self._active_trial["control/source_step"][()], dtype=np.int64
            )
            simulation_times = np.asarray(
                self._active_trial["control/sim_time"][()], dtype=np.float64
            )
            trial_states = np.asarray(
                self._active_trial["control/trial_state"].asstr()[()]
            )
            start_rows = np.flatnonzero(source_steps == onset_step)
            if start_rows.size != 1:
                raise RuntimeError("response onset step does not map to one control row")
            start_index = int(start_rows[0])
            if missing_stable:
                return_rows = np.flatnonzero(
                    trial_states
                    == TrialState.SHOW_RETURN_HAND_TO_NEUTRAL_CUE.value
                )
                window_end = (
                    int(return_rows[0]) if return_rows.size else len(source_steps)
                )
            else:
                stable_rows = np.flatnonzero(source_steps == stable_step)
                if stable_rows.size != 1:
                    raise RuntimeError(
                        "stable response step does not map to one control row"
                    )
                window_end = int(stable_rows[0]) + 1
            response_recovery = recovery_active[start_index:window_end]
            active_offsets = np.flatnonzero(response_recovery == 1)
            if active_offsets.size:
                first_recovery_index = start_index + int(active_offsets[0])
                last_recovery_index = start_index + int(active_offsets[-1])
                expected_recovery_step = int(source_steps[first_recovery_index])
                expected_recovery_onset_sim = float(
                    simulation_times[first_recovery_index]
                )
                if (
                    last_recovery_index + 1 < window_end
                    and recovery_active[last_recovery_index + 1] == 0
                ):
                    expected_recovery_end_step = int(
                        source_steps[last_recovery_index + 1]
                    )
                    expected_recovery_end_sim = float(
                        simulation_times[last_recovery_index + 1]
                    )
                else:
                    expected_recovery_end_step = -1
                    expected_recovery_end_sim = -1.0
            else:
                expected_recovery_step = -1
                expected_recovery_onset_sim = -1.0
                expected_recovery_end_step = -1
                expected_recovery_end_sim = -1.0
            expected_recovery_duration = float(
                np.sum(response_recovery)
                * float(self._file.attrs["physics_dt_s"])
            )
            if (
                recovery_step != expected_recovery_step
                or recovery_end_step != expected_recovery_end_step
                or not math.isclose(
                    recovery_onset_sim,
                    expected_recovery_onset_sim,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    recovery_end_sim,
                    expected_recovery_end_sim,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(summary["recovery_duration_s"]),
                    expected_recovery_duration,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise RuntimeError(
                    "Recovery summary does not exactly reconstruct from its raw mask"
                )

        all_trial_states = np.asarray(
            self._active_trial["control/trial_state"].asstr()[()]
        )
        required_hold = np.isin(
            all_trial_states,
            (
                TrialState.SHOW_HAND_CROSSING_CUE.value,
                TrialState.PAUSE_NOMINAL_TASK_PROGRESSION.value,
                TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            ),
        )
        if np.any(required_hold):
            hold_actions = np.asarray(
                self._active_trial["actions/controller_task_action"][()],
                dtype=np.float64,
            )
            hold_reasons = np.asarray(
                self._active_trial[
                    "actions/controller_task_action_reason"
                ].asstr()[()]
            )
            hold_modes = np.asarray(
                self._active_trial["control/control_mode"].asstr()[()]
            )
            hold_phase_advance = np.asarray(
                self._active_trial["task/phase_advance_enabled"][()],
                dtype=np.int8,
            )
            hold_recovery_authority = np.asarray(
                self._active_trial["recovery/control_authority"][()],
                dtype=np.int8,
            )
            hold_pipeline_complete = np.asarray(
                self._active_trial["actions/action_pipeline_complete"][()],
                dtype=np.int8,
            )
            hold_intervention_available = np.asarray(
                self._active_trial["cbf/intervention_available"][()],
                dtype=np.int8,
            )
            hold_geometry_valid = np.asarray(
                self._active_trial["safety/geometry_valid"][()],
                dtype=np.int8,
            )
            if (
                np.any(hold_actions[required_hold] != 0.0)
                or np.any(
                    hold_reasons[required_hold]
                    != "current_ee_protocol_hold"
                )
                or np.any(hold_modes[required_hold] != "protocol_hold")
                or np.any(hold_phase_advance[required_hold] != 0)
                or np.any(hold_recovery_authority[required_hold] != 0)
                or np.any(hold_pipeline_complete[required_hold] != 1)
                or np.any(hold_intervention_available[required_hold] != 1)
                or np.any(hold_geometry_valid[required_hold] != 1)
            ):
                raise RuntimeError(
                    "cue/query/pause rows do not prove the zero task-action "
                    "sentinel and motion-policy-frame hold mode through the "
                    "complete valid CBF pipeline"
                )
        off_protocol = summary_bools["off_protocol"]
        response_status = str(self._active_trial.attrs.get("query_response_status", ""))
        protocol_valid = summary_bools["protocol_valid"]
        if int(summary["actual_crossing_count"]) != 1 or off_protocol:
            protocol_valid = False
        if response_status != "completed":
            protocol_valid = False
        if int(self._active_trial.attrs.get("query_count", 0)) != 1:
            protocol_valid = False
        encounter_record_count = int(
            self._active_trial.attrs.get("encounter_record_count", 0)
        )
        if missing_onset:
            if encounter_record_count not in (0, 1):
                raise RuntimeError(
                    "unconfirmed response trial may contain at most one observed risk encounter"
                )
        elif encounter_record_count != 1:
            raise RuntimeError(
                "a confirmed response trial requires exactly one encounter"
            )
        if encounter_record_count != 1:
            protocol_valid = False
        if (
            onset_sim < 0.0
            or end_sim < onset_sim
            or int(summary["response_onset_step"]) < 0
            or int(summary["response_end_step_exclusive"])
            <= int(summary["response_onset_step"])
        ):
            protocol_valid = False
        attrs = dict(summary)
        attrs["protocol_valid"] = int(protocol_valid)
        attrs["trial_complete"] = 1
        attrs["analysis_exclude"] = int(
            bool(self._active_plan.analysis_exclude) or not protocol_valid
        )
        attrs["off_protocol"] = int(off_protocol)
        attrs["off_protocol_reasons_json"] = _json(off_protocol_reasons)
        del attrs["off_protocol_reasons"]
        attrs["trial_fsm_history_json"] = _json(history)
        attrs["decision_context_json"] = _json(decision_context)
        attrs["response_phase_history_json"] = _json(response_phase_history)
        attrs["windowed_jerk_json"] = _json(windowed_jerk)
        phase_durations = _decoded_json(
            summary["response_phase_durations_json"], expected_type=dict
        )
        if set(phase_durations) != set(RESPONSE_PHASES):
            raise RuntimeError(
                "response phase duration summary must contain exactly all six phases"
            )
        for phase_name in RESPONSE_PHASES:
            duration = float(phase_durations[phase_name])
            if not math.isfinite(duration) or duration < 0.0:
                raise RuntimeError(
                    f"response phase duration {phase_name} must be finite and non-negative"
                )
        for phase_name, summary_name, maximum_duration_s in (
            ("CBF_ACTIVE", "cbf_active_duration_s", None),
            ("SMOOTH_TAIL", "smooth_tail_duration_s", None),
            ("BC_RESUMED", "bc_resumed_duration_s", 0.5),
        ):
            expected_duration = float(phase_durations[phase_name])
            if maximum_duration_s is not None:
                expected_duration = min(maximum_duration_s, expected_duration)
            if not math.isclose(
                expected_duration,
                float(summary[summary_name]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    f"response phase duration {phase_name} disagrees with {summary_name}"
                )
        attrs["response_phase_durations_json"] = _json(phase_durations)
        for name, value in attrs.items():
            self._active_trial.attrs[name] = _attribute_value(value)
        self._file.flush()
        self._active_trial = None
        self._active_plan = None
        self._datasets = {}
        self._row_specs = {}
        self._pending_transitions = []
        self._previous_crossing = None
        self._trial_length = 0
        self._written_trial_length = 0
        self._trial_marker_counts = {marker_type: 0 for marker_type in MARKER_TYPES}
        self._trial_count += 1
        self._file.attrs["trial_count"] = self._trial_count

    def update_spectator_video_metadata(
        self, metadata: Mapping[str, Any]
    ) -> None:
        """Bind the finalized video/frame-clock summary before sealing."""

        self._require_open()
        payload = dict(metadata)
        if str(payload.get("schema_version", "")) != (
            "ac_spectator_video_sync_v1"
        ):
            raise ValueError("unexpected spectator video metadata schema")
        self._file.attrs["spectator_video_metadata_json"] = _json(payload)
        self._file.flush()

    def seal(self) -> Path:
        self._require_open()
        if self._active_trial is not None:
            raise RuntimeError("cannot seal while a trial is active")
        if self._trial_count <= 0:
            raise RuntimeError("cannot seal an empty collection")
        if not self._schedule or self._trial_count != len(self._schedule):
            raise RuntimeError(
                "cannot seal before every preassigned schedule entry is recorded"
            )
        trials = self._file["trials"]
        expected_encounter_count = sum(
            int(group.attrs.get("encounter_record_count", 0))
            for group in trials.values()
        )
        if (
            self._query_count != self._trial_count
            or not 0 <= self._encounter_count <= self._trial_count
            or self._encounter_count != expected_encounter_count
        ):
            raise RuntimeError(
                "cannot seal without one query per trial and an exact "
                "observed-encounter count"
            )
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
            self._flush_transition_buffer()
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
        frozen_config: Mapping[str, Any],
    ) -> None:
        required = (
            "participant_id",
            "session_id",
            "code_version",
            "code_version_source",
            "code_commit_verification",
            "source_tree_sha256",
            "cbf_config_json",
            "cbf_config_sha256",
            "scenario_config_sha256",
            "protocol_config_sha256",
            "xr_feedback_action_map_json",
            "xr_builtin_tools_enabled",
            "execution_semantics",
            "corridor_reference_semantics",
            "participant_visible_bc_only_precheck",
            "first_participant_visible_motion_controller",
            "fixed_layout_seed",
            "fixed_layout_id",
            "layout_qualification_json",
            "layout_qualification_sha256",
        )
        missing = [name for name in required if not str(metadata.get(name, "")).strip()]
        if missing:
            raise ValueError(f"missing required session metadata: {missing}")
        direct_live = scenario_config.get("direct_live_scenario", {})
        if not isinstance(direct_live, Mapping) or not bool(
            direct_live.get("enabled", False)
        ):
            raise ValueError("collector requires the free-crossing v3 scenario config")
        expected_direct_metadata = {
            "execution_semantics": (
                "fixed_bc_qualified_layout_without_participant_visible_precheck_v1"
            ),
            "corridor_reference_semantics": (
                "free_single_crossing_internal_bilateral_gate_v1"
            ),
            "participant_visible_bc_only_precheck": 0,
            "first_participant_visible_motion_controller": (
                "frozen_bc_plus_assigned_cbf"
            ),
            "fixed_layout_seed": int(direct_live.get("fixed_layout_seed", -1)),
            "fixed_layout_id": str(direct_live.get("expected_layout_id", "")),
        }
        for name, expected in expected_direct_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"direct-live session metadata mismatch at {name}"
                )
        layout_qualification = _decoded_json(
            metadata["layout_qualification_json"], expected_type=dict
        )
        if (
            layout_qualification
            != dict(direct_live.get("bc_layout_qualification", {}))
            or str(metadata["layout_qualification_sha256"])
            != _sha256_json(layout_qualification)
        ):
            raise ValueError(
                "layout qualification metadata is not bound to direct-live config"
            )
        protocol_config_json = _json(dict(protocol_config))
        scenario_config_json = _json(dict(scenario_config))
        cbf_config = _decoded_json(
            metadata["cbf_config_json"], expected_type=dict
        )
        cbf_config_json = _json(cbf_config)
        action_map_audit = _decoded_json(
            metadata["xr_feedback_action_map_json"], expected_type=dict
        )
        action_map_json = _json(action_map_audit)
        expected_dominant_hands = {
            "SeRT Feedback Oculus (Left)": "left",
            "SeRT Feedback Oculus (Right)": "right",
        }
        action_map_name = str(action_map_audit.get("name", ""))
        if (
            expected_dominant_hands.get(action_map_name)
            != str(action_map_audit.get("dominant_hand", ""))
            or str(action_map_audit.get("tools_layout", "")) != "vr"
            or int(action_map_audit.get("tool_count", -1)) != 0
            or int(action_map_audit.get("action_count", -1)) != 0
            or int(metadata["xr_builtin_tools_enabled"]) != 0
        ):
            raise ValueError(
                "session metadata does not prove the reviewed tool-free XR action map"
            )
        derived_configs = {
            "protocol": (protocol_config_json, _sha256_json(protocol_config)),
            "scenario": (scenario_config_json, _sha256_json(scenario_config)),
            "cbf": (cbf_config_json, _sha256_json(cbf_config)),
        }
        for prefix, (_, expected_sha256) in derived_configs.items():
            actual_sha256 = str(metadata.get(f"{prefix}_config_sha256", ""))
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"{prefix}_config_sha256 does not bind to the supplied config"
                )
        frozen_config_json = _json(dict(frozen_config))
        frozen_config_sha256 = hashlib.sha256(
            frozen_config_json.encode("utf-8")
        ).hexdigest()
        attrs: dict[str, Any] = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "protocol_version": ONLINE_PROTOCOL_VERSION,
            "row_semantics": ONLINE_ROW_SEMANTICS,
            "feature_provenance_version": FEATURE_PROVENANCE_VERSION,
            "collection_complete": 0,
            "production_mode": int(not bool(metadata.get("practice_only", False))),
            "pilot": 1,
            "policy_path": POLICY_RELATIVE_PATH,
            "policy_sha256": POLICY_SHA256,
            "policy_size_bytes": POLICY_SIZE_BYTES,
            "policy_mode": "frozen_direct_bc",
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "runtime_handoff_commit": RUNTIME_HANDOFF_COMMIT,
            "policy_frozen": 1,
            "policy_online_updates": 0,
            "policy_fine_tuning": 0,
            "policy_observation_dim": 84,
            "policy_observation_version": "obs_v1_state_controller_phase",
            "policy_action_dim": 5,
            "policy_action_version": "action_v1_controller_target_delta",
            "human_observations_masked": 1,
            "feedback_is_policy_input": 0,
            "feedback_changes_schedule": 0,
            "condition_schedule_preassigned": 1,
            "condition_blinding": 1,
            "condition_identity_visible": 0,
            "pseudo_errp_enabled": 0,
            "haptics_enabled": 0,
            "haptic_event_count": 0,
            "cbf_frozen": 1,
            "cbf_feedback_adaptation": 0,
            "cbf_parameter_adaptation": 0,
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
            "crossing_instruction_mode": "free_single_crossing",
            "crossing_detector_semantics": (
                "free_single_crossing_bilateral_gate_v1"
            ),
            "scheduled_crossing_factors_applied": 0,
            "participant_visible_crossing_markers": 0,
            "queries_per_trial": 1,
            "strict_task_semantics_schema_version": "physical_event_driven_pick_place_v5",
            "state_aware_recovery_enabled": 1,
            "state_aware_recovery_schema": "state_aware_pick_place_recovery_bridge_v3",
            "recovery_fixed": 1,
            "task_require_release": 1,
            "strict_place_xy_tolerance_m": 0.05,
            "task_max_episode_steps": 12000,
            "task_phase_clock_latched_during_pre_crossing_cue": 1,
            "task_progression_during_crossing_and_recovery": 1,
            "task_progression_paused_during_query": 1,
            "physics_remains_on_during_query": 1,
            "cbf_remains_on_during_query": 1,
            "safe_hold_during_query": 1,
            "emergency_stop_remains_on_during_query": 1,
            "query_encounter_nullable": 0,
            "null_string_encoding": "empty_utf8",
            "q1_id": Q1_ID,
            "q1_text_ko": Q1_TEXT_KO,
            "q1_responses_json": _json(Q1_RESPONSES),
            "no_marker_semantics": "missing_evidence_not_acceptable",
            "uncertain_semantics": "abstain",
            "timeout_semantics": "missing_abstain",
            "likert_questions_json": _json(LIKERT_QUESTIONS),
            "modification_reasons_json": _json(MODIFICATION_REASONS),
            "realtime_marker_types_json": _json(MARKER_TYPES),
            "runtime_observable_fields_json": _json(RUNTIME_OBSERVABLE_FIELDS),
            "future_outcome_fields_json": _json(FUTURE_OUTCOME_FIELDS),
            "protocol_config_json": protocol_config_json,
            "protocol_config_sha256": derived_configs["protocol"][1],
            "scenario_config_json": scenario_config_json,
            "scenario_config_sha256": derived_configs["scenario"][1],
            "cbf_config_json": cbf_config_json,
            "cbf_config_sha256": derived_configs["cbf"][1],
            "frozen_config_json": frozen_config_json,
            "frozen_config_sha256": frozen_config_sha256,
            "trial_count": 0,
            "query_count": 0,
            "query_answer_count": 0,
            "encounter_count": 0,
            "realtime_marker_count": 0,
            "layout_precheck_count": 0,
            "session_start_monotonic_ns": time.monotonic_ns(),
            "session_start_unix_ns": time.time_ns(),
        }
        locked = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "policy_path": POLICY_RELATIVE_PATH,
            "policy_sha256": POLICY_SHA256,
            "policy_size_bytes": POLICY_SIZE_BYTES,
            "policy_frozen": 1,
            "policy_online_updates": 0,
            "policy_fine_tuning": 0,
            "pilot": 1,
            "policy_mode": "frozen_direct_bc",
            "policy_observation_version": "obs_v1_state_controller_phase",
            "policy_action_version": "action_v1_controller_target_delta",
            "human_observations_masked": 1,
            "feedback_changes_schedule": 0,
            "condition_schedule_preassigned": 1,
            "condition_blinding": 1,
            "condition_identity_visible": 0,
            "haptics_enabled": 0,
            "haptic_event_count": 0,
            "cbf_frozen": 1,
            "cbf_feedback_adaptation": 0,
            "cbf_parameter_adaptation": 0,
            "recovery_fixed": 1,
            "state_aware_recovery_enabled": 1,
            "state_aware_recovery_schema": "state_aware_pick_place_recovery_bridge_v3",
            "strict_task_semantics_schema_version": "physical_event_driven_pick_place_v5",
            "physics_remains_on_during_query": 1,
            "cbf_remains_on_during_query": 1,
            "safe_hold_during_query": 1,
            "emergency_stop_remains_on_during_query": 1,
            "no_marker_semantics": "missing_evidence_not_acceptable",
            "uncertain_semantics": "abstain",
            "timeout_semantics": "missing_abstain",
            "frozen_config_json": frozen_config_json,
            "frozen_config_sha256": frozen_config_sha256,
            "protocol_config_json": protocol_config_json,
            "protocol_config_sha256": derived_configs["protocol"][1],
            "scenario_config_json": scenario_config_json,
            "scenario_config_sha256": derived_configs["scenario"][1],
            "cbf_config_json": cbf_config_json,
            "cbf_config_sha256": derived_configs["cbf"][1],
            "xr_feedback_action_map_json": action_map_json,
            "xr_builtin_tools_enabled": 0,
        }
        for name, expected in locked.items():
            if name in metadata and metadata[name] != expected:
                raise ValueError(f"session metadata attempts to override frozen {name}")
        attrs.update(dict(metadata))
        attrs.update(locked)
        for name, value in attrs.items():
            self._file.attrs[str(name)] = _attribute_value(value)

    def _create_tables(self) -> None:
        tables: dict[str, dict[str, DatasetSpec]] = {
            "schedule": {
                "trial_index": DatasetSpec(np.int64),
                "trial_id": DatasetSpec("str"),
                "query_id": DatasetSpec("str"),
                "encounter_id": DatasetSpec("str"),
                "task_phase": DatasetSpec("str"),
                "severity": DatasetSpec("str"),
                "direction": DatasetSpec("str"),
                "speed": DatasetSpec("str"),
                "crossing_hand": DatasetSpec("str"),
                "condition_id": DatasetSpec("str"),
                "objective_mode": DatasetSpec("str"),
                "lambda_s": DatasetSpec(np.float64),
                "schedule_seed": DatasetSpec(np.int64),
                "block_id": DatasetSpec("str"),
                "counterbalancing_group": DatasetSpec("str"),
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
                "participant_id": DatasetSpec("str"),
                "session_id": DatasetSpec("str"),
                "encounter_id": DatasetSpec("str"),
                "condition_id": DatasetSpec("str"),
                "lambda_s": DatasetSpec(np.float64),
                "marker_type": DatasetSpec("str"),
                "simulation_time_s": DatasetSpec(np.float64),
                "monotonic_ns": DatasetSpec(np.int64),
                "unix_ns": DatasetSpec(np.int64),
                "control_step": DatasetSpec(np.int64),
                "response_phase": DatasetSpec("str"),
                "time_since_response_onset_s": DatasetSpec(np.float64),
                "time_since_recovery_onset_s": DatasetSpec(np.float64),
                "controller_hand": DatasetSpec("str"),
                "button": DatasetSpec("str"),
                "button_identifier": DatasetSpec("str"),
            },
            "queries": {
                "query_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "encounter_id": DatasetSpec("str"),
                "issued_simulation_time_s": DatasetSpec(np.float64),
                "issued_monotonic_ns": DatasetSpec(np.int64),
                "issued_unix_ns": DatasetSpec(np.int64),
                "issued_control_step": DatasetSpec(np.int64),
                "completed_simulation_time_s": DatasetSpec(np.float64),
                "completed_monotonic_ns": DatasetSpec(np.int64),
                "completed_unix_ns": DatasetSpec(np.int64),
                "completed_control_step": DatasetSpec(np.int64),
                "response_status": DatasetSpec("str"),
                "response_disposition": DatasetSpec("str"),
                "q1_response": DatasetSpec("str"),
                "q2_perceived_danger": DatasetSpec(np.int8),
                "q3_abruptness": DatasetSpec(np.int8),
                "q4_excessive_duration": DatasetSpec(np.int8),
                "q5_task_disruption": DatasetSpec(np.int8),
                "q6_confidence": DatasetSpec(np.int8),
                "modification_reasons_json": DatasetSpec("str"),
                "response_latency_ms": DatasetSpec(np.float64),
                "input_device": DatasetSpec("str"),
                "first_input_simulation_time_s": DatasetSpec(np.float64),
                "first_input_monotonic_ns": DatasetSpec(np.int64),
                "first_input_unix_ns": DatasetSpec(np.int64),
                "first_input_control_step": DatasetSpec(np.int64),
                "back_correction_count": DatasetSpec(np.int32),
                "accidental_input_count": DatasetSpec(np.int32),
            },
            "query_answers": {
                "query_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "question_id": DatasetSpec("str"),
                "answer_status": DatasetSpec("str"),
                "value_json": DatasetSpec("str"),
                "prompt_shown_simulation_time_s": DatasetSpec(np.float64),
                "prompt_shown_monotonic_ns": DatasetSpec(np.int64),
                "prompt_shown_unix_ns": DatasetSpec(np.int64),
                "prompt_shown_control_step": DatasetSpec(np.int64),
                "first_input_simulation_time_s": DatasetSpec(np.float64),
                "first_input_monotonic_ns": DatasetSpec(np.int64),
                "first_input_unix_ns": DatasetSpec(np.int64),
                "first_input_control_step": DatasetSpec(np.int64),
                "confirmed_simulation_time_s": DatasetSpec(np.float64),
                "confirmed_monotonic_ns": DatasetSpec(np.int64),
                "confirmed_unix_ns": DatasetSpec(np.int64),
                "confirmed_control_step": DatasetSpec(np.int64),
                "response_latency_ms": DatasetSpec(np.float64),
                "input_device": DatasetSpec("str"),
                "back_correction_count": DatasetSpec(np.int32),
                "accidental_input_count": DatasetSpec(np.int32),
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


def _validate_free_crossing_transition(
    row: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None,
) -> None:
    """Validate cumulative free-crossing evidence before appending one row."""

    if str(row.get("crossing/intended_hand", "")).strip() or str(
        row.get("crossing/non_crossing_hand", "")
    ).strip():
        raise RuntimeError(
            "free-crossing v3 requires empty legacy intended/non-crossing hands"
        )
    monitoring_active = _strict_binary(
        row.get("crossing/monitoring_active"),
        "crossing/monitoring_active",
    )
    left_sign = _strict_completion_sign(
        row.get("crossing/left_completion_sign"),
        "crossing/left_completion_sign",
    )
    right_sign = _strict_completion_sign(
        row.get("crossing/right_completion_sign"),
        "crossing/right_completion_sign",
    )
    completed_count = _strict_nonnegative_integer(
        row.get("crossing/completed_count"),
        "crossing/completed_count",
    )
    first_hand = str(row.get("crossing/first_completed_hand", "")).strip()
    first_direction = str(
        row.get("crossing/first_completed_direction", "")
    ).strip()
    multiple = _strict_binary(
        row.get("crossing/multiple_crossing"),
        "crossing/multiple_crossing",
    )
    prior_count = (
        0
        if previous is None
        else _strict_nonnegative_integer(
            previous.get("completed_count"), "previous completed_count"
        )
    )
    completion_events = int(left_sign != 0) + int(right_sign != 0)
    if completed_count != prior_count + completion_events:
        raise RuntimeError(
            "crossing/completed_count must be cumulative completion-sign count"
        )
    if completion_events and not monitoring_active:
        raise RuntimeError("crossing completion must occur while monitoring is active")
    if multiple != int(completed_count > 1):
        raise RuntimeError(
            "crossing/multiple_crossing must exactly encode completed_count > 1"
        )
    if completed_count == 0:
        if first_hand or first_direction != "not_observed":
            raise RuntimeError(
                "pre-completion rows require empty hand and not_observed direction"
            )
        return
    if first_hand not in ACTUAL_CROSSING_HANDS[:2]:
        raise RuntimeError("completed crossing requires a left/right first hand")
    if first_direction not in ACTUAL_CROSSING_DIRECTIONS[:2]:
        raise RuntimeError("completed crossing requires an observed first direction")
    if previous is not None and prior_count:
        if (
            first_hand != str(previous.get("first_completed_hand", ""))
            or first_direction
            != str(previous.get("first_completed_direction", ""))
        ):
            raise RuntimeError(
                "first completed crossing identity cannot change within a trial"
            )
        if int(previous.get("multiple_crossing", 0)) and not multiple:
            raise RuntimeError("multiple-crossing evidence cannot clear")
        return
    if completion_events == 0:
        raise RuntimeError("first completed crossing has no completion-sign row")
    first_sign = left_sign if first_hand == "left" else right_sign
    expected_direction = (
        "right_to_left"
        if first_sign > 0
        else "left_to_right" if first_sign < 0 else ""
    )
    if not expected_direction or first_direction != expected_direction:
        raise RuntimeError(
            "first completed hand/direction disagree with its completion sign"
        )


def _validate_free_crossing_summary(
    summary: Mapping[str, Any],
    group: Any,
    *,
    protocol_valid: bool,
) -> None:
    """Bind free-crossing summary attributes to the per-step detector trace."""

    actual_count = _strict_nonnegative_integer(
        summary.get("actual_crossing_count"), "actual_crossing_count"
    )
    actual_hand_value = summary.get("actual_crossing_hand")
    actual_direction_value = summary.get("actual_crossing_direction")
    if not isinstance(actual_hand_value, (str, np.str_)):
        raise RuntimeError("actual_crossing_hand must be a string")
    if not isinstance(actual_direction_value, (str, np.str_)):
        raise RuntimeError("actual_crossing_direction must be a string")
    actual_hand = str(actual_hand_value).strip()
    actual_direction = str(actual_direction_value).strip()
    if actual_hand not in ACTUAL_CROSSING_HANDS:
        raise RuntimeError("actual_crossing_hand is invalid")
    if actual_direction not in ACTUAL_CROSSING_DIRECTIONS:
        raise RuntimeError("actual_crossing_direction is invalid")
    start_step = _strict_integer(
        summary.get("crossing_start_step"), "crossing_start_step", minimum=-1
    )
    end_step = _strict_integer(
        summary.get("crossing_end_step"), "crossing_end_step", minimum=-1
    )
    start_sim = _strict_finite_float(
        summary.get("crossing_start_simulation_time_s"),
        "crossing_start_simulation_time_s",
    )
    end_sim = _strict_finite_float(
        summary.get("crossing_end_simulation_time_s"),
        "crossing_end_simulation_time_s",
    )
    detector_semantics_value = summary.get("crossing_detector_semantics")
    if not isinstance(detector_semantics_value, (str, np.str_)):
        raise RuntimeError("crossing_detector_semantics must be a string")
    detector_semantics = str(detector_semantics_value).strip()
    if detector_semantics != "free_single_crossing_bilateral_gate_v1":
        raise RuntimeError("crossing_detector_semantics is not the free-v3 contract")
    monitoring_valid_rate = _strict_finite_float(
        summary.get("crossing_monitoring_valid_rate"),
        "crossing_monitoring_valid_rate",
    )
    if not 0.0 <= monitoring_valid_rate <= 1.0:
        raise RuntimeError("crossing_monitoring_valid_rate must be inside [0, 1]")

    monitoring = _integer_dataset(group, "crossing/monitoring_active")
    left_signs = _integer_dataset(group, "crossing/left_completion_sign")
    right_signs = _integer_dataset(group, "crossing/right_completion_sign")
    counts = _integer_dataset(group, "crossing/completed_count")
    multiple = _integer_dataset(group, "crossing/multiple_crossing")
    first_hands = _text_dataset(group, "crossing/first_completed_hand")
    first_directions = _text_dataset(
        group, "crossing/first_completed_direction"
    )
    source_steps = _integer_dataset(group, "control/source_step")
    simulation_times = _float_dataset(group, "control/sim_time")
    valid_masks = _numeric_dataset(group, "human_next/valid_mask", dtype=np.int8)
    row_count = len(counts)
    one_dimensional = (
        monitoring,
        left_signs,
        right_signs,
        multiple,
        first_hands,
        first_directions,
        source_steps,
        simulation_times,
    )
    if any(value.shape != (row_count,) for value in one_dimensional) or (
        valid_masks.shape != (row_count, 3)
    ):
        raise RuntimeError("free-crossing evidence datasets have inconsistent shapes")
    if (
        np.any((monitoring != 0) & (monitoring != 1))
        or np.any((multiple != 0) & (multiple != 1))
        or np.any(~np.isin(left_signs, (-1, 0, 1)))
        or np.any(~np.isin(right_signs, (-1, 0, 1)))
        or np.any(counts < 0)
    ):
        raise RuntimeError("free-crossing row evidence contains invalid enum values")
    increments = (left_signs != 0).astype(np.int64) + (
        right_signs != 0
    ).astype(np.int64)
    expected_counts = np.cumsum(increments, dtype=np.int64)
    if not np.array_equal(counts, expected_counts):
        raise RuntimeError(
            "crossing/completed_count does not reconstruct from completion signs"
        )
    if np.any((increments > 0) & (monitoring != 1)):
        raise RuntimeError("completion signs occur outside the monitoring window")
    expected_multiple = (counts > 1).astype(np.int64)
    if not np.array_equal(multiple, expected_multiple):
        raise RuntimeError("multiple-crossing mask disagrees with cumulative count")
    final_count = int(counts[-1]) if row_count else 0
    if actual_count != final_count:
        raise RuntimeError(
            "actual_crossing_count disagrees with per-step completion evidence"
        )

    monitoring_rows = monitoring == 1
    both_hands_valid = np.all(valid_masks[:, 1:3] == 1, axis=1)
    expected_valid_rate = (
        float(np.mean(both_hands_valid[monitoring_rows]))
        if np.any(monitoring_rows)
        else 0.0
    )
    if not math.isclose(
        monitoring_valid_rate,
        expected_valid_rate,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "crossing_monitoring_valid_rate does not reconstruct from hand masks"
        )

    completion_rows = np.flatnonzero(increments > 0)
    if actual_count == 0:
        if (
            actual_hand != "not_observed"
            or actual_direction != "not_observed"
            or completion_rows.size
        ):
            raise RuntimeError(
                "no-crossing summary requires exact not-observed sentinels"
            )
        if (start_step, end_step) != (-1, -1) or (
            start_sim,
            end_sim,
        ) != (-1.0, -1.0):
            raise RuntimeError(
                "unstarted crossing requires exact start/end sentinels"
            )
    else:
        if not completion_rows.size:
            raise RuntimeError("completed crossing lacks end row evidence")
        first_completion_index = int(completion_rows[0])
        start_matches = np.flatnonzero(
            (source_steps == start_step)
            & np.isclose(
                simulation_times,
                start_sim,
                rtol=0.0,
                atol=1e-9,
            )
            & monitoring_rows
        )
        if start_matches.size != 1:
            raise RuntimeError(
                "crossing start summary must map to one monitoring row"
            )
        first_start_index = int(start_matches[0])
        row_hand = str(first_hands[first_completion_index])
        row_direction = str(first_directions[first_completion_index])
        if (
            actual_hand != row_hand
            or actual_direction != row_direction
            or actual_hand not in ACTUAL_CROSSING_HANDS[:2]
            or actual_direction not in ACTUAL_CROSSING_DIRECTIONS[:2]
        ):
            raise RuntimeError(
                "actual crossing hand/direction disagree with first completion row"
            )
        sign = (
            int(left_signs[first_completion_index])
            if actual_hand == "left"
            else int(right_signs[first_completion_index])
        )
        expected_direction = (
            "right_to_left" if sign > 0 else "left_to_right" if sign < 0 else ""
        )
        if expected_direction != actual_direction:
            raise RuntimeError(
                "actual crossing direction disagrees with first-hand completion sign"
            )
        after_first = counts > 0
        if any(value != actual_hand for value in first_hands[after_first]) or any(
            value != actual_direction for value in first_directions[after_first]
        ):
            raise RuntimeError(
                "first completed crossing identity is not latched across rows"
            )
        if (
            start_step != int(source_steps[first_start_index])
            or end_step != int(source_steps[first_completion_index])
            or not math.isclose(
                start_sim,
                float(simulation_times[first_start_index]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                end_sim,
                float(simulation_times[first_completion_index]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or start_step > end_step
            or start_sim > end_sim
        ):
            raise RuntimeError(
                "crossing start/end summary does not bind to detector rows"
            )

    if protocol_valid and (
        actual_count != 1
        or actual_hand not in ACTUAL_CROSSING_HANDS[:2]
        or actual_direction not in ACTUAL_CROSSING_DIRECTIONS[:2]
        or bool(np.any(multiple))
    ):
        raise RuntimeError(
            "protocol-valid trial requires exactly one observed, unambiguous crossing"
        )


def _strict_binary(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    raise RuntimeError(f"{name} must be an exact boolean")


def _strict_completion_sign(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) not in (-1, 0, 1)
    ):
        raise RuntimeError(f"{name} must be exactly -1, 0, or 1")
    return int(value)


def _strict_integer(value: Any, name: str, *, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise RuntimeError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _strict_nonnegative_integer(value: Any, name: str) -> int:
    return _strict_integer(value, name, minimum=0)


def _strict_finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite")
    return result


def _dataset_value(dataset: Any) -> Any:
    return dataset[()] if hasattr(dataset, "__getitem__") else dataset


def _numeric_dataset(group: Any, path: str, *, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(_dataset_value(group[path]), dtype=dtype)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"required free-crossing dataset is invalid: {path}") from error


def _integer_dataset(group: Any, path: str) -> np.ndarray:
    return _numeric_dataset(group, path, dtype=np.int64)


def _float_dataset(group: Any, path: str) -> np.ndarray:
    values = _numeric_dataset(group, path, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"required free-crossing dataset is non-finite: {path}")
    return values


def _text_dataset(group: Any, path: str) -> np.ndarray:
    try:
        dataset = group[path]
        values = dataset.asstr()[()] if hasattr(dataset, "asstr") else _dataset_value(dataset)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"required free-crossing dataset is invalid: {path}") from error
    return np.asarray(
        [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in np.asarray(values).reshape(-1)
        ],
        dtype=object,
    )

def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _decoded_json(value: Any, *, expected_type: type) -> Any:
    if isinstance(value, expected_type):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise RuntimeError(
            f"expected {expected_type.__name__} or its JSON encoding, got {type(value).__name__}"
        )
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError("invalid JSON trial summary field") from error
    if not isinstance(decoded, expected_type):
        raise RuntimeError(
            f"JSON trial summary field must decode to {expected_type.__name__}"
        )
    return decoded


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(dict(value)).encode("utf-8")).hexdigest()
