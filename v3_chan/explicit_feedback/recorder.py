"""Incremental, atomic HDF5 writer for BC+CBF explicit-feedback sessions."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .schema import (
    ACTION_TRACE_SCHEMA_VERSION,
    CLOCK_SCHEMA_VERSION,
    COLLECTION_SCHEMA_VERSION,
    ENCOUNTER_SCHEMA_VERSION,
    EXPLICIT_FEEDBACK_SCHEMA_VERSION,
    POLICY_RELATIVE_PATH,
    POLICY_SHA256,
    POLICY_SIZE_BYTES,
    QUESTION_ID,
    QUESTION_TEXT_EN,
    QUESTION_TEXT_KO,
    RESPONSE_CODES,
    ROW_SEMANTICS,
    RUNTIME_BASE_COMMIT,
    RUNTIME_CONTRACT_SCHEMA_VERSION,
    RUNTIME_CONTRACT_SHA256,
    RUNTIME_HANDOFF_COMMIT,
    EncounterRecord,
    PromptRecord,
    ResponseRecord,
)


@dataclass(frozen=True)
class DatasetSpec:
    dtype: Any
    tail_shape: tuple[int, ...] = ()


class ExplicitFeedbackRecorder:
    """Write one session to a non-study-eligible partial file until validated."""

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any],
        joint_names: tuple[str, ...],
        arm_joint_indices: tuple[int, ...],
        flush_every_steps: int = 1,
    ) -> None:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError(
                "Production explicit-feedback collection requires h5py; NPZ fallback is forbidden"
            ) from error
        self.h5py = h5py
        self.output_path = Path(output_path).expanduser().resolve()
        if self.output_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("output_path must end in .h5 or .hdf5")
        self.partial_path = Path(str(self.output_path) + ".partial")
        if self.output_path.exists() or self.partial_path.exists():
            raise FileExistsError(
                f"refusing to overwrite collection output: {self.output_path}"
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.joint_names = tuple(str(name) for name in joint_names)
        self.arm_joint_indices = tuple(int(index) for index in arm_joint_indices)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        if (
            not self.arm_joint_indices
            or len(set(self.arm_joint_indices)) != len(self.arm_joint_indices)
            or any(index < 0 or index >= len(self.joint_names) for index in self.arm_joint_indices)
        ):
            raise ValueError("arm_joint_indices are invalid")
        self.flush_every_steps = max(1, int(flush_every_steps))
        self._file = h5py.File(self.partial_path, "x", libver="latest")
        self._datasets: dict[str, Any] = {}
        self._active_episode = None
        self._active_episode_index = -1
        self._episode_length = 0
        self._episode_count = 0
        self._encounter_count = 0
        self._prompt_count = 0
        self._response_count = 0
        self._sealed = False
        self._closed = False
        self._write_root_metadata(metadata)
        self._file.create_group("episodes")
        self._create_event_tables()
        self._file.flush()

    @property
    def active_episode_index(self) -> int:
        return self._active_episode_index

    def start_episode(
        self,
        episode_index: int,
        *,
        seed: int,
        start_monotonic_ns: int,
        start_unix_ns: int,
        initial_scene: Mapping[str, Any] | None = None,
    ) -> None:
        self._require_open()
        if self._active_episode is not None:
            raise RuntimeError("an episode is already active")
        index = int(episode_index)
        if index != self._episode_count:
            raise ValueError(
                f"episode indices must be contiguous; expected {self._episode_count}, got {index}"
            )
        group = self._file["episodes"].create_group(f"episode_{index:06d}")
        group.attrs.update(
            {
                "episode_index": index,
                "episode_seed": int(seed),
                "episode_length": 0,
                "episode_complete": 0,
                "success": 0,
                "terminated": 0,
                "truncated": 0,
                "terminal_reason": "",
                "abort_reason": "",
                "start_monotonic_ns": int(start_monotonic_ns),
                "start_unix_ns": int(start_unix_ns),
                "end_monotonic_ns": 0,
                "end_unix_ns": 0,
                "encounter_count": 0,
                "prompt_count": 0,
                "accepted_response_count": 0,
            }
        )
        action_group = group.create_group("actions")
        string_dtype = self.h5py.string_dtype(encoding="utf-8")
        action_group.create_dataset(
            "all_joint_names",
            data=np.asarray(self.joint_names, dtype=string_dtype),
        )
        action_group.create_dataset(
            "arm_joint_indices",
            data=np.asarray(self.arm_joint_indices, dtype=np.int32),
        )
        action_group.create_dataset(
            "arm_joint_names",
            data=np.asarray(
                [self.joint_names[index] for index in self.arm_joint_indices],
                dtype=string_dtype,
            ),
        )
        initial = group.create_group("initial_scene")
        for name, value in dict(initial_scene or {}).items():
            _write_static_value(initial, str(name), value, h5py=self.h5py)
        self._active_episode = group
        self._active_episode_index = index
        self._episode_length = 0
        self._datasets = {}
        for path, spec in _row_specs(
            len(self.joint_names), len(self.arm_joint_indices)
        ).items():
            self._datasets[path] = _create_extendable_dataset(
                group,
                path,
                spec,
                h5py=self.h5py,
            )
        self._file.flush()

    def append_transition(self, row: Mapping[str, Any]) -> None:
        self._require_active_episode()
        specs = _row_specs(len(self.joint_names), len(self.arm_joint_indices))
        missing = sorted(set(specs) - set(row))
        extra = sorted(set(row) - set(specs))
        if missing:
            raise RuntimeError(f"transition row is missing datasets: {missing}")
        if extra:
            raise RuntimeError(f"transition row has unknown datasets: {extra}")
        index = self._episode_length
        converted: dict[str, Any] = {}
        for path, spec in specs.items():
            converted[path] = _coerce_value(row[path], spec, path=path)
        # Validate everything before resizing any dataset.
        for path, value in converted.items():
            dataset = self._datasets[path]
            dataset.resize((index + 1, *dataset.shape[1:]))
            dataset[index] = value
        self._episode_length += 1
        self._active_episode.attrs["episode_length"] = self._episode_length
        if self._episode_length % self.flush_every_steps == 0:
            self._file.flush()

    def end_episode(
        self,
        *,
        success: bool,
        terminated: bool,
        truncated: bool,
        terminal_reason: str,
        end_monotonic_ns: int,
        end_unix_ns: int,
        abort_reason: str = "",
    ) -> None:
        self._require_active_episode()
        reason = str(terminal_reason).strip()
        abort = str(abort_reason).strip()
        complete = not bool(abort)
        if not reason:
            raise ValueError("terminal_reason must be non-empty")
        if bool(success) and not bool(terminated):
            raise ValueError("successful episodes must be terminated")
        if bool(terminated) and bool(truncated):
            raise ValueError("terminated and truncated cannot both be true")
        group = self._active_episode
        group.attrs.update(
            {
                "episode_length": self._episode_length,
                "episode_complete": int(complete),
                "success": int(bool(success)),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "terminal_reason": reason,
                "abort_reason": abort,
                "end_monotonic_ns": int(end_monotonic_ns),
                "end_unix_ns": int(end_unix_ns),
            }
        )
        self._file.flush()
        self._active_episode = None
        self._active_episode_index = -1
        self._datasets = {}
        self._episode_count += 1
        self._file.attrs["episode_count"] = self._episode_count

    def append_encounter(self, record: EncounterRecord) -> None:
        self._require_open()
        table = self._file["encounters"]
        values = {
            "encounter_id": record.encounter_id,
            "episode_index": record.episode_index,
            "start_step": record.start_step,
            "end_step_exclusive": record.end_step_exclusive,
            "start_monotonic_ns": record.start_monotonic_ns,
            "end_monotonic_ns": record.end_monotonic_ns,
            "minimum_surface_gap_m": record.minimum_surface_gap_m,
            "involved_hand_mask": record.involved_hand_mask,
            "eligible": int(record.eligible),
            "eligibility_reason": record.eligibility_reason,
            "terminal_status": record.terminal_status,
            "abort_reason": record.abort_reason,
            "prompt_id": record.prompt_id,
        }
        _append_table_row(table, values)
        self._encounter_count += 1
        if 0 <= record.episode_index < self._episode_count + 1:
            name = f"episodes/episode_{record.episode_index:06d}"
            if name in self._file:
                group = self._file[name]
                group.attrs["encounter_count"] = int(
                    group.attrs.get("encounter_count", 0)
                ) + 1
        self._file.attrs["encounter_count"] = self._encounter_count
        self._file.flush()

    def append_prompt(self, record: PromptRecord) -> None:
        self._require_open()
        table = self._file["explicit_feedback/prompts"]
        values = record.as_dict()
        _append_table_row(table, values)
        self._prompt_count += 1
        episode_index = self._episode_index_for_encounter(record.encounter_id)
        if episode_index is not None:
            group = self._file[f"episodes/episode_{episode_index:06d}"]
            group.attrs["prompt_count"] = int(group.attrs.get("prompt_count", 0)) + 1
        self._file.attrs["prompt_count"] = self._prompt_count
        self._file.flush()

    def append_response(self, record: ResponseRecord) -> None:
        self._require_open()
        table = self._file["explicit_feedback/responses"]
        _append_table_row(table, record.as_dict())
        self._response_count += 1
        if record.accepted:
            encounter_id = self._encounter_id_for_prompt(record.prompt_id)
            episode_index = (
                None
                if encounter_id is None
                else self._episode_index_for_encounter(encounter_id)
            )
            if episode_index is not None:
                group = self._file[f"episodes/episode_{episode_index:06d}"]
                group.attrs["accepted_response_count"] = int(
                    group.attrs.get("accepted_response_count", 0)
                ) + 1
        self._file.attrs["response_count"] = self._response_count
        self._file.flush()

    def seal(self) -> Path:
        self._require_open()
        if self._active_episode is not None:
            raise RuntimeError("cannot seal while an episode is active")
        if self._episode_count <= 0:
            raise RuntimeError("cannot seal an empty collection")
        self._file.attrs.update(
            {
                "collection_complete": 1,
                "episode_count": self._episode_count,
                "encounter_count": self._encounter_count,
                "prompt_count": self._prompt_count,
                "response_count": self._response_count,
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
            raise RuntimeError("seal and validate the partial file before commit")
        if self.output_path.exists():
            raise FileExistsError(self.output_path)
        os.replace(self.partial_path, self.output_path)
        return self.output_path

    def abort(self, reason: str) -> Path:
        if self._closed:
            return self.partial_path
        reason = str(reason).strip() or "unspecified_abort"
        if self._active_episode is not None:
            self._active_episode.attrs.update(
                {
                    "episode_length": self._episode_length,
                    "episode_complete": 0,
                    "terminal_reason": "aborted",
                    "abort_reason": reason,
                    "end_monotonic_ns": time.monotonic_ns(),
                    "end_unix_ns": time.time_ns(),
                }
            )
        self._file.attrs.update(
            {
                "collection_complete": 0,
                "collection_abort_reason": reason,
                "session_end_monotonic_ns": time.monotonic_ns(),
                "session_end_unix_ns": time.time_ns(),
            }
        )
        self._file.flush()
        self._file.close()
        self._closed = True
        return self.partial_path

    def close(self) -> None:
        if not self._closed:
            self.abort("recorder_closed_without_seal")

    def _write_root_metadata(self, metadata: Mapping[str, Any]) -> None:
        required = ("participant_id", "session_id", "code_version", "source_tree_sha256")
        missing = [name for name in required if not str(metadata.get(name, "")).strip()]
        if missing:
            raise ValueError(f"missing required session metadata: {missing}")
        attrs: dict[str, Any] = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "encounter_schema_version": ENCOUNTER_SCHEMA_VERSION,
            "explicit_feedback_schema_version": EXPLICIT_FEEDBACK_SCHEMA_VERSION,
            "action_trace_schema_version": ACTION_TRACE_SCHEMA_VERSION,
            "clock_schema_version": CLOCK_SCHEMA_VERSION,
            "row_semantics": ROW_SEMANTICS,
            "collection_complete": 0,
            "production_mode": 1,
            "runtime_contract_schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "runtime_handoff_commit": RUNTIME_HANDOFF_COMMIT,
            "runtime_base_commit": RUNTIME_BASE_COMMIT,
            "policy_path": POLICY_RELATIVE_PATH,
            "policy_sha256": POLICY_SHA256,
            "policy_size_bytes": POLICY_SIZE_BYTES,
            "policy_mode": "direct_bc",
            "policy_observation_dim": 84,
            "policy_observation_version": "obs_v1_state_controller_phase",
            "policy_action_dim": 5,
            "policy_action_version": "action_v1_controller_target_delta",
            "mask_human_observations": 1,
            "pseudo_errp_enabled": 0,
            "task_single_pick": 1,
            "task_max_steps": 4500,
            "task_gripper_mode": "event",
            "task_fixed_orientation": 1,
            "task_require_release_for_success": 1,
            "strict_task_semantics_schema_version": "physical_event_driven_pick_place_v3",
            "strict_place_xy_tolerance_m": 0.05,
            "evaluation_reward_version": "reward_v4_post_release_stability_hri_errp",
            "cbf_controller": "cbf",
            "cbf_scope": "tracked_hands_to_distal_panda_links",
            "cbf_safe_gap_m": 0.05,
            "cbf_activation_gap_m": 0.13,
            "cbf_gamma_per_s": 8.0,
            "cbf_prediction_horizon_s": 0.15,
            "cbf_max_prediction_buffer_m": 0.08,
            "cbf_max_joint_speed_rad_s": 2.0,
            "cbf_fail_closed_on_invalid_active_hand": 1,
            "cbf_stop_on_infeasible": 1,
            "required_tracking_mask": json.dumps([1, 1, 1]),
            "question_id": QUESTION_ID,
            "question_text_ko": QUESTION_TEXT_KO,
            "question_text_en": QUESTION_TEXT_EN,
            "response_codes_json": json.dumps(RESPONSE_CODES),
            "sample_interval_steps": 1,
            "episode_count": 0,
            "encounter_count": 0,
            "prompt_count": 0,
            "response_count": 0,
            "session_start_monotonic_ns": time.monotonic_ns(),
            "session_start_unix_ns": time.time_ns(),
        }
        for name, value in dict(metadata).items():
            attrs[str(name)] = _attribute_value(value)
        for name, value in attrs.items():
            self._file.attrs[name] = value

    def _create_event_tables(self) -> None:
        encounter_specs = {
            "encounter_id": DatasetSpec("str"),
            "episode_index": DatasetSpec(np.int64),
            "start_step": DatasetSpec(np.int64),
            "end_step_exclusive": DatasetSpec(np.int64),
            "start_monotonic_ns": DatasetSpec(np.int64),
            "end_monotonic_ns": DatasetSpec(np.int64),
            "minimum_surface_gap_m": DatasetSpec(np.float64),
            "involved_hand_mask": DatasetSpec(np.int8, (2,)),
            "eligible": DatasetSpec(np.int8),
            "eligibility_reason": DatasetSpec("str"),
            "terminal_status": DatasetSpec("str"),
            "abort_reason": DatasetSpec("str"),
            "prompt_id": DatasetSpec("str"),
        }
        prompt_specs = {
            "prompt_id": DatasetSpec("str"),
            "encounter_id": DatasetSpec("str"),
            "question_id": DatasetSpec("str"),
            "issued_monotonic_ns": DatasetSpec(np.int64),
            "issued_unix_ns": DatasetSpec(np.int64),
            "presentation_deadline_monotonic_ns": DatasetSpec(np.int64),
            "presented_monotonic_ns": DatasetSpec(np.int64),
            "deadline_monotonic_ns": DatasetSpec(np.int64),
            "status": DatasetSpec("str"),
        }
        response_specs = {
            "response_id": DatasetSpec("str"),
            "client_response_id": DatasetSpec("str"),
            "prompt_id": DatasetSpec("str"),
            "response_code": DatasetSpec("str"),
            "source": DatasetSpec("str"),
            "client_unix_ns": DatasetSpec(np.int64),
            "received_monotonic_ns": DatasetSpec(np.int64),
            "received_unix_ns": DatasetSpec(np.int64),
            "accepted": DatasetSpec(np.int8),
            "rejection_reason": DatasetSpec("str"),
        }
        for group_name, specs in (
            ("encounters", encounter_specs),
            ("explicit_feedback/prompts", prompt_specs),
            ("explicit_feedback/responses", response_specs),
        ):
            group = self._file.require_group(group_name)
            for name, spec in specs.items():
                _create_extendable_dataset(group, name, spec, h5py=self.h5py)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("recorder is closed")

    def _require_active_episode(self) -> None:
        self._require_open()
        if self._active_episode is None:
            raise RuntimeError("no active episode")

    def _episode_index_for_encounter(self, encounter_id: str) -> int | None:
        ids = self._file["encounters/encounter_id"].asstr()[()]
        matches = np.flatnonzero(np.asarray(ids) == str(encounter_id))
        if matches.size != 1:
            return None
        return int(self._file["encounters/episode_index"][int(matches[0])])

    def _encounter_id_for_prompt(self, prompt_id: str) -> str | None:
        ids = self._file["explicit_feedback/prompts/prompt_id"].asstr()[()]
        matches = np.flatnonzero(np.asarray(ids) == str(prompt_id))
        if matches.size != 1:
            return None
        return str(
            self._file["explicit_feedback/prompts/encounter_id"].asstr()[
                int(matches[0])
            ]
        )


def _row_specs(joint_count: int, arm_count: int) -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {
        "control/source_step": DatasetSpec(np.int64),
        "control/result_step": DatasetSpec(np.int64),
        "control/sim_time": DatasetSpec(np.float64),
        "control/observation_monotonic_ns": DatasetSpec(np.int64),
        "control/preflight_monotonic_ns": DatasetSpec(np.int64),
        "control/wall_time_unix_ns": DatasetSpec(np.int64),
        "control/policy_inference_started_monotonic_ns": DatasetSpec(np.int64),
        "control/policy_output_monotonic_ns": DatasetSpec(np.int64),
        "control/rmpflow_output_monotonic_ns": DatasetSpec(np.int64),
        "control/cbf_output_monotonic_ns": DatasetSpec(np.int64),
        "control/action_command_monotonic_ns": DatasetSpec(np.int64),
        "control/next_observation_monotonic_ns": DatasetSpec(np.int64),
        "control/post_step_monotonic_ns": DatasetSpec(np.int64),
        "observations/obs_raw": DatasetSpec(np.float32, (84,)),
        "observations/obs_policy": DatasetSpec(np.float32, (84,)),
        "observations/obs_next": DatasetSpec(np.float32, (84,)),
        "human/valid_mask": DatasetSpec(np.int8, (3,)),
        "human_next/valid_mask": DatasetSpec(np.int8, (3,)),
        "safety/left_surface_gap_m": DatasetSpec(np.float64),
        "safety/right_surface_gap_m": DatasetSpec(np.float64),
        "safety/min_surface_gap_m": DatasetSpec(np.float64),
        "safety/geometry_valid": DatasetSpec(np.int8),
        "safety/collision": DatasetSpec(np.int8),
        "safety/near": DatasetSpec(np.int8),
        "safety/near_miss": DatasetSpec(np.int8),
        "actions/bc_task_action": DatasetSpec(np.float32, (5,)),
        "actions/applied_arm_joint_velocities_rad_s": DatasetSpec(
            np.float64, (arm_count,)
        ),
        "actions/applied_arm_joint_velocities_mask": DatasetSpec(
            np.int8, (arm_count,)
        ),
        "actions/arm_action_submitted": DatasetSpec(np.int8),
        "actions/action_pipeline_complete": DatasetSpec(np.int8),
        "actions/gripper_command": DatasetSpec("str"),
        "robot/measured_joint_positions_before_rad": DatasetSpec(
            np.float64, (joint_count,)
        ),
        "robot/measured_joint_velocities_before_rad_s": DatasetSpec(
            np.float64, (joint_count,)
        ),
        "robot/measured_joint_positions_after_rad": DatasetSpec(
            np.float64, (joint_count,)
        ),
        "robot/measured_joint_velocities_after_rad_s": DatasetSpec(
            np.float64, (joint_count,)
        ),
        "task/controller_event_before": DatasetSpec(np.int32),
        "task/controller_event_after": DatasetSpec(np.int32),
        "task/controller_t_before": DatasetSpec(np.float64),
        "task/controller_t_after": DatasetSpec(np.float64),
        "task/success": DatasetSpec(np.int8),
        "task/terminated": DatasetSpec(np.int8),
        "task/truncated": DatasetSpec(np.int8),
        "task/terminal_reason": DatasetSpec("str"),
        "encounter/row_encounter_id": DatasetSpec("str"),
        "encounter/state": DatasetSpec("str"),
    }
    for namespace in ("human", "human_next"):
        for entity in ("head", "left", "right"):
            specs.update(
                {
                    f"{namespace}/{entity}_position_world_m": DatasetSpec(
                        np.float64, (3,)
                    ),
                    f"{namespace}/{entity}_pose_valid": DatasetSpec(np.int8),
                    f"{namespace}/{entity}_position_tracked": DatasetSpec(np.int8),
                    f"{namespace}/{entity}_tracking_status_known": DatasetSpec(np.int8),
                    f"{namespace}/{entity}_source_name": DatasetSpec("str"),
                    f"{namespace}/{entity}_source_path": DatasetSpec("str"),
                    f"{namespace}/{entity}_acquisition_monotonic_ns": DatasetSpec(
                        np.int64
                    ),
                    f"{namespace}/{entity}_pose_age_ms": DatasetSpec(np.float64),
                    f"{namespace}/{entity}_source_switched": DatasetSpec(np.int8),
                }
            )
    for stage in (
        "nominal_rmpflow",
        "cbf_filtered",
        "submitted",
        "pre_applied",
        "applied",
    ):
        for channel, unit in (
            ("joint_positions", "rad"),
            ("joint_velocities", "rad_s"),
            ("joint_efforts", "nm"),
        ):
            base = f"actions/{stage}_{channel}_{unit}"
            specs[base] = DatasetSpec(np.float64, (joint_count,))
            specs[base + "_mask"] = DatasetSpec(np.int8, (joint_count,))
    cbf_float = (
        "intervention_norm_radps",
        "nominal_velocity_norm_radps",
        "filtered_velocity_norm_radps",
        "max_constraint_violation_before_mps",
        "max_constraint_violation_after_mps",
        "slack_mps",
        "buffered_current_gap_m",
        "solve_time_ms",
    )
    cbf_int = (
        "active",
        "intervention_available",
        "constraint_count",
        "tracked_hand_count",
        "valid_hand_count",
        "feasible",
        "fallback_applied",
        "solver_converged",
        "infeasibility_proven",
        "relaxed_solution_available",
        "relaxed_solution_applied",
        "intentional_human_absence",
    )
    cbf_text = (
        "failure_reasons_json",
        "status",
        "solver_backend",
        "projection_status",
    )
    specs.update({f"cbf/{name}": DatasetSpec(np.float64) for name in cbf_float})
    specs.update({f"cbf/{name}": DatasetSpec(np.int32) for name in cbf_int})
    specs.update({f"cbf/{name}": DatasetSpec("str") for name in cbf_text})
    return specs


def _create_extendable_dataset(group, path: str, spec: DatasetSpec, *, h5py):
    parent = group
    parts = path.split("/")
    for part in parts[:-1]:
        parent = parent.require_group(part)
    dtype = h5py.string_dtype(encoding="utf-8") if spec.dtype == "str" else spec.dtype
    return parent.create_dataset(
        parts[-1],
        shape=(0, *spec.tail_shape),
        maxshape=(None, *spec.tail_shape),
        chunks=(max(1, min(256, 1)), *spec.tail_shape),
        dtype=dtype,
    )


def _coerce_value(value: Any, spec: DatasetSpec, *, path: str) -> Any:
    if spec.dtype == "str":
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)
    array = np.asarray(value)
    if array.shape != spec.tail_shape:
        if spec.tail_shape == () and array.size == 1:
            array = array.reshape(())
        else:
            raise RuntimeError(
                f"{path} has shape {array.shape}, expected {spec.tail_shape}"
            )
    if np.issubdtype(np.dtype(spec.dtype), np.integer):
        if np.issubdtype(np.dtype(spec.dtype), np.signedinteger) and path.endswith(
            "position_tracked"
        ):
            allowed = (-1, 0, 1)
        elif _is_binary_row_path(path):
            allowed = (0, 1)
        else:
            allowed = None
        numeric = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
            raise RuntimeError(f"{path} must contain finite integer values")
        if allowed is not None and not np.all(np.isin(numeric, allowed)):
            raise RuntimeError(f"{path} contains a value outside {allowed}")
    elif not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
        raise RuntimeError(f"{path} contains non-finite values")
    return np.asarray(array, dtype=spec.dtype)


def _is_binary_row_path(path: str) -> bool:
    return bool(
        path.endswith("_mask")
        or path.endswith("_pose_valid")
        or path.endswith("_tracking_status_known")
        or path.endswith("_source_switched")
        or path
        in {
            "safety/geometry_valid",
            "safety/collision",
            "safety/near",
            "safety/near_miss",
            "actions/arm_action_submitted",
            "actions/action_pipeline_complete",
            "task/success",
            "task/terminated",
            "task/truncated",
            "cbf/active",
            "cbf/intervention_available",
            "cbf/feasible",
            "cbf/fallback_applied",
            "cbf/solver_converged",
            "cbf/infeasibility_proven",
            "cbf/relaxed_solution_available",
            "cbf/relaxed_solution_applied",
            "cbf/intentional_human_absence",
        }
    )


def _append_table_row(group, values: Mapping[str, Any]) -> None:
    expected = set(group.keys())
    if set(values) != expected:
        raise RuntimeError(
            f"event row columns mismatch: missing={sorted(expected - set(values))}, "
            f"extra={sorted(set(values) - expected)}"
        )
    lengths = {dataset.shape[0] for dataset in group.values()}
    if len(lengths) != 1:
        raise RuntimeError("event table columns are not aligned")
    index = lengths.pop()
    converted: dict[str, Any] = {}
    for name, value in values.items():
        dataset = group[name]
        tail = dataset.shape[1:]
        if dataset.dtype.kind in "SUO":
            converted[name] = str(value)
        else:
            array = np.asarray(value)
            if array.shape != tail:
                if tail == () and array.size == 1:
                    array = array.reshape(())
                else:
                    raise RuntimeError(
                        f"event column {name} has shape {array.shape}, expected {tail}"
                    )
            if not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
                raise RuntimeError(f"event column {name} is non-finite")
            converted[name] = array
    for name, value in converted.items():
        dataset = group[name]
        dataset.resize((index + 1, *dataset.shape[1:]))
        dataset[index] = value


def _write_static_value(group, name: str, value: Any, *, h5py) -> None:
    if isinstance(value, str):
        group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
        return
    array = np.asarray(value)
    if array.dtype.kind in "SUO":
        group.create_dataset(
            name,
            data=np.asarray(array, dtype=h5py.string_dtype("utf-8")),
        )
        return
    if not np.all(np.isfinite(array.astype(np.float64))):
        raise ValueError(f"initial_scene/{name} contains non-finite values")
    group.create_dataset(name, data=array)


def _attribute_value(value: Any) -> Any:
    if isinstance(value, (str, bytes, int, float, bool, np.generic)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("metadata attributes must be finite")
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
