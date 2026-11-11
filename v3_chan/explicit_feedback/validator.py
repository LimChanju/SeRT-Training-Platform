"""Fail-closed validation for BC+CBF explicit-feedback collection files.

The recorder intentionally writes an HDF5 ``.partial`` file first.  A collector
may validate that sealed file with ``allow_partial_path=True`` immediately before
the atomic rename.  Public/offline validation never enables that escape hatch:
the path exception does not relax a single schema or study-eligibility check.

This module is Isaac-independent.  It validates the serialized evidence at the
BC -> RMPFlow -> CBF -> articulation boundary, live tracking provenance, and the
encounter/prompt/response relational and temporal contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

from .recorder import DatasetSpec, _is_binary_row_path, _row_specs
from .policy import mask_human_observations
from .runtime import EXPECTED_PANDA_JOINT_NAMES
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
    RESPONSE_CODES,
    ROW_SEMANTICS,
    RUNTIME_BASE_COMMIT,
    RUNTIME_CONTRACT_SCHEMA_VERSION,
    RUNTIME_CONTRACT_SHA256,
    RUNTIME_HANDOFF_COMMIT,
)
from .tracking import DEFAULT_ALLOWED_SOURCES, ENTITY_NAMES
from v3_chan.rl.observations import observation_slices


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_ATOL = 1e-7
_FLOAT_RTOL = 1e-6
_MASKED_ZERO_ATOL = 1e-12
_RESPONSE_REJECTION_REASONS = frozenset(
    {
        "broker_closed",
        "duplicate_response_id",
        "invalid_response_code",
        "prompt_already_answered",
        "prompt_expired",
        "prompt_not_presented",
        "response_precedes_presentation",
        "unknown_prompt_id",
    }
)


@dataclass(frozen=True)
class ValidationIssue:
    """One stable, machine-readable validation failure."""

    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    """Validation result.  Any issue makes the artifact non-study-eligible."""

    path: str
    valid: bool
    study_eligible: bool
    issues: tuple[ValidationIssue, ...]
    episode_count: int = 0
    encounter_count: int = 0
    prompt_count: int = 0
    response_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "valid": self.valid,
            "study_eligible": self.study_eligible,
            "episode_count": self.episode_count,
            "encounter_count": self.encounter_count,
            "prompt_count": self.prompt_count,
            "response_count": self.response_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class CollectionValidationError(RuntimeError):
    """Raised when a collection fails the production validator."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}"
            for issue in report.issues[:5]
        )
        if len(report.issues) > 5:
            detail += f"; and {len(report.issues) - 5} more issue(s)"
        super().__init__(f"collection is not study-eligible: {detail}")


_ROOT_EXACT: dict[str, Any] = {
    "schema_version": COLLECTION_SCHEMA_VERSION,
    "encounter_schema_version": ENCOUNTER_SCHEMA_VERSION,
    "explicit_feedback_schema_version": EXPLICIT_FEEDBACK_SCHEMA_VERSION,
    "action_trace_schema_version": ACTION_TRACE_SCHEMA_VERSION,
    "clock_schema_version": CLOCK_SCHEMA_VERSION,
    "row_semantics": ROW_SEMANTICS,
    "collection_complete": 1,
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
    "response_codes_json": json.dumps(RESPONSE_CODES),
    "sample_interval_steps": 1,
}

_REQUIRED_ROOT_TEXT = (
    "participant_id",
    "session_id",
    "code_version",
    "source_tree_sha256",
)

_EPISODE_ATTRIBUTES = (
    "episode_index",
    "episode_seed",
    "episode_length",
    "episode_complete",
    "success",
    "terminated",
    "truncated",
    "terminal_reason",
    "abort_reason",
    "start_monotonic_ns",
    "start_unix_ns",
    "end_monotonic_ns",
    "end_unix_ns",
    "encounter_count",
    "prompt_count",
    "accepted_response_count",
)

_EVENT_SPECS: dict[str, dict[str, DatasetSpec]] = {
    "encounters": {
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
    },
    "explicit_feedback/prompts": {
        "prompt_id": DatasetSpec("str"),
        "encounter_id": DatasetSpec("str"),
        "question_id": DatasetSpec("str"),
        "issued_monotonic_ns": DatasetSpec(np.int64),
        "issued_unix_ns": DatasetSpec(np.int64),
        "presentation_deadline_monotonic_ns": DatasetSpec(np.int64),
        "presented_monotonic_ns": DatasetSpec(np.int64),
        "deadline_monotonic_ns": DatasetSpec(np.int64),
        "status": DatasetSpec("str"),
    },
    "explicit_feedback/responses": {
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
    },
}


@dataclass
class _EpisodeData:
    index: int
    group: Any
    length: int
    joint_names: tuple[str, ...]
    arm_indices: np.ndarray
    arrays: dict[str, np.ndarray]


def validate_collection(
    path: str | Path,
    *,
    allow_partial_path: bool = False,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate one sealed production collection.

    ``allow_partial_path`` exists solely for the collector's seal/validate/rename
    transaction.  It does not permit an incomplete file or weaken any invariant.
    """

    validator = _CollectionValidator(Path(path), allow_partial_path=allow_partial_path)
    report = validator.run()
    if raise_on_error and not report.valid:
        raise CollectionValidationError(report)
    return report


def validate_collection_or_raise(
    path: str | Path, *, allow_partial_path: bool = False
) -> ValidationReport:
    """Explicit spelling for collector call sites."""

    return validate_collection(
        path,
        allow_partial_path=allow_partial_path,
        raise_on_error=True,
    )


class _CollectionValidator:
    def __init__(self, path: Path, *, allow_partial_path: bool) -> None:
        self.path = path.expanduser().resolve(strict=False)
        self.original_path = Path(path).expanduser()
        self.allow_partial_path = bool(allow_partial_path)
        self.issues: list[ValidationIssue] = []
        self.episodes: dict[int, _EpisodeData] = {}
        self.counts = {
            "episode_count": 0,
            "encounter_count": 0,
            "prompt_count": 0,
            "response_count": 0,
        }

    def issue(self, code: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(code, path, message))

    def run(self) -> ValidationReport:
        self._validate_path()
        if not self.path.is_file():
            self.issue("file.not_found", str(self.path), "path is not a regular file")
            return self._report()
        try:
            import h5py
        except ImportError:
            self.issue("file.h5py_unavailable", str(self.path), "h5py is required")
            return self._report()
        try:
            with h5py.File(self.path, "r") as data:
                self._validate_hdf5(data)
        except (OSError, ValueError) as error:
            self.issue("file.hdf5_open", str(self.path), str(error))
        except Exception as error:  # corrupted input must fail closed, not crash callers
            self.issue(
                "validator.internal_error",
                str(self.path),
                f"unexpected validation failure: {type(error).__name__}: {error}",
            )
        return self._report()

    def _report(self) -> ValidationReport:
        valid = not self.issues
        return ValidationReport(
            path=str(self.path),
            valid=valid,
            study_eligible=valid,
            issues=tuple(self.issues),
            **self.counts,
        )

    def _validate_path(self) -> None:
        name = self.original_path.name.lower()
        resolved_name = self.path.name.lower()
        is_partial = any(
            candidate.endswith(".h5.partial") or candidate.endswith(".hdf5.partial")
            for candidate in (name, resolved_name)
        )
        if is_partial and not self.allow_partial_path:
            self.issue(
                "file.partial_path",
                str(self.original_path),
                "public validation rejects .partial files",
            )
        if not is_partial and not (name.endswith(".h5") or name.endswith(".hdf5")):
            self.issue(
                "file.extension",
                str(self.original_path),
                "collection path must end in .h5 or .hdf5",
            )

    def _validate_hdf5(self, data: Any) -> None:
        self._validate_root(data)
        self._reject_errp(data)
        self._validate_episodes(data)
        tables = self._validate_event_tables(data)
        if tables is not None:
            self._validate_events(data, tables)

    def _validate_root(self, data: Any) -> None:
        for name, expected in _ROOT_EXACT.items():
            if name not in data.attrs:
                self.issue(
                    "root.attribute_missing", f"/@{name}", "required attribute is missing"
                )
                continue
            actual = _scalar(data.attrs[name])
            if not _same_scalar(actual, expected):
                self.issue(
                    "root.attribute_mismatch",
                    f"/@{name}",
                    f"expected {expected!r}, got {actual!r}",
                )
        for name in _REQUIRED_ROOT_TEXT:
            if name not in data.attrs or not _text(data.attrs.get(name, "")).strip():
                self.issue(
                    "root.attribute_missing",
                    f"/@{name}",
                    "required non-empty provenance attribute is missing",
                )
        source_sha = _text(data.attrs.get("source_tree_sha256", "")).strip()
        if source_sha and not _SHA256_RE.fullmatch(source_sha):
            self.issue(
                "root.attribute_mismatch",
                "/@source_tree_sha256",
                "source_tree_sha256 must be 64 lowercase hexadecimal characters",
            )
        for name in ("physics_dt_s", "max_pose_age_ms"):
            if name not in data.attrs:
                self.issue(
                    "root.attribute_missing",
                    f"/@{name}",
                    "required collection timing attribute is missing",
                )
                continue
            value = _finite_float(data.attrs[name])
            if value is None or value <= 0.0:
                self.issue(
                    "root.attribute_mismatch",
                    f"/@{name}",
                    "attribute must be finite and positive",
                )
        if "prompt_timeout_s" not in data.attrs:
            self.issue(
                "root.attribute_missing",
                "/@prompt_timeout_s",
                "required explicit-feedback timeout attribute is missing",
            )
        else:
            prompt_timeout_s = _finite_float(data.attrs["prompt_timeout_s"])
            if (
                prompt_timeout_s is None
                or not 0.1 <= prompt_timeout_s <= 3600.0
            ):
                self.issue(
                    "root.attribute_mismatch",
                    "/@prompt_timeout_s",
                    "attribute must be finite and in [0.1, 3600] seconds",
                )
        for name in (
            "episode_count",
            "encounter_count",
            "prompt_count",
            "response_count",
            "session_start_monotonic_ns",
            "session_start_unix_ns",
            "session_end_monotonic_ns",
            "session_end_unix_ns",
        ):
            if name not in data.attrs:
                self.issue("root.attribute_missing", f"/@{name}", "attribute is missing")
        start_mono = _integer(data.attrs.get("session_start_monotonic_ns"))
        end_mono = _integer(data.attrs.get("session_end_monotonic_ns"))
        start_unix = _integer(data.attrs.get("session_start_unix_ns"))
        end_unix = _integer(data.attrs.get("session_end_unix_ns"))
        if (
            start_mono is not None
            and end_mono is not None
            and (start_mono <= 0 or end_mono < start_mono)
        ):
            self.issue("root.temporal_order", "/", "session monotonic clock bounds are invalid")
        if (
            start_unix is not None
            and end_unix is not None
            and (start_unix <= 0 or end_unix < start_unix)
        ):
            self.issue("root.temporal_order", "/", "session Unix clock bounds are invalid")
        try:
            tracking_mask = json.loads(_text(data.attrs.get("required_tracking_mask", "")))
        except json.JSONDecodeError:
            tracking_mask = None
        if tracking_mask != [1, 1, 1]:
            self.issue(
                "root.attribute_mismatch",
                "/@required_tracking_mask",
                "required tracking mask must decode to [1, 1, 1]",
            )
        try:
            codes = json.loads(_text(data.attrs.get("response_codes_json", "")))
        except json.JSONDecodeError:
            codes = None
        if codes != list(RESPONSE_CODES):
            self.issue(
                "root.attribute_mismatch",
                "/@response_codes_json",
                f"response code order must be {list(RESPONSE_CODES)!r}",
            )

    def _reject_errp(self, data: Any) -> None:
        forbidden: list[str] = []

        def inspect(name: str, _value: Any) -> None:
            if any("errp" in part.lower() for part in name.split("/")):
                forbidden.append("/" + name)

        data.visititems(inspect)
        for path in forbidden:
            self.issue(
                "root.forbidden_errp",
                path,
                "legacy/pseudo ErrP datasets are forbidden in explicit-feedback v1",
            )

    def _validate_episodes(self, data: Any) -> None:
        if "episodes" not in data or not hasattr(data["episodes"], "keys"):
            self.issue("root.group_missing", "/episodes", "episodes group is missing")
            return
        root = data["episodes"]
        names = sorted(root.keys())
        expected_names = [f"episode_{index:06d}" for index in range(len(names))]
        if names != expected_names:
            self.issue(
                "episode.index_sequence",
                "/episodes",
                f"episode group names must be contiguous: expected {expected_names!r}, got {names!r}",
            )
        self.counts["episode_count"] = len(names)
        root_count = _integer(data.attrs.get("episode_count"))
        if root_count != len(names):
            self.issue(
                "root.count_mismatch",
                "/@episode_count",
                f"attribute says {root_count!r}, table has {len(names)}",
            )
        if not names:
            self.issue("episode.empty", "/episodes", "a sealed collection needs an episode")
            return
        for expected_index, name in enumerate(names):
            group = root[name]
            if not hasattr(group, "keys"):
                self.issue("episode.group_invalid", f"/episodes/{name}", "not a group")
                continue
            episode = self._validate_episode(group, expected_index)
            if episode is not None:
                self.episodes[expected_index] = episode

    def _validate_episode(self, group: Any, index: int) -> _EpisodeData | None:
        prefix = group.name
        for name in _EPISODE_ATTRIBUTES:
            if name not in group.attrs:
                self.issue(
                    "episode.attribute_missing",
                    f"{prefix}/@{name}",
                    "required episode attribute is missing",
                )
        recorded_index = _integer(group.attrs.get("episode_index"))
        if recorded_index != index:
            self.issue(
                "episode.index_sequence",
                f"{prefix}/@episode_index",
                f"expected {index}, got {recorded_index!r}",
            )
        length = _integer(group.attrs.get("episode_length"))
        if length is None or length <= 0:
            self.issue(
                "episode.length",
                f"{prefix}/@episode_length",
                "episode length must be positive",
            )
            return None
        if length > int(_scalar_or(group.file.attrs.get("task_max_steps"), 4500)):
            self.issue(
                "episode.length",
                f"{prefix}/@episode_length",
                "episode exceeds task_max_steps",
            )
        for name in ("episode_complete", "success", "terminated", "truncated"):
            value = _integer(group.attrs.get(name))
            if value not in (0, 1):
                self.issue(
                    "episode.attribute_value",
                    f"{prefix}/@{name}",
                    "attribute must be exactly 0 or 1",
                )
        if _integer(group.attrs.get("episode_complete")) != 1:
            self.issue(
                "episode.incomplete",
                f"{prefix}/@episode_complete",
                "sealed production episodes must be complete",
            )
        success = _integer(group.attrs.get("success")) == 1
        terminated = _integer(group.attrs.get("terminated")) == 1
        truncated = _integer(group.attrs.get("truncated")) == 1
        if terminated == truncated:
            self.issue(
                "episode.terminal_flags",
                prefix,
                "exactly one of terminated/truncated must be true",
            )
        if success and not terminated:
            self.issue("episode.terminal_flags", prefix, "success requires termination")
        terminal_reason = _text(group.attrs.get("terminal_reason", "")).strip()
        abort_reason = _text(group.attrs.get("abort_reason", "")).strip()
        if not terminal_reason:
            self.issue(
                "episode.terminal_reason", f"{prefix}/@terminal_reason", "must be non-empty"
            )
        if abort_reason:
            self.issue(
                "episode.abort",
                f"{prefix}/@abort_reason",
                "aborted episodes cannot enter a sealed production collection",
            )
        self._validate_episode_clock_attrs(group)
        static = self._read_action_static(group)
        if static is None:
            return None
        joint_names, arm_indices = static
        specs = _row_specs(len(joint_names), int(arm_indices.size))
        arrays: dict[str, np.ndarray] = {}
        for path, spec in specs.items():
            full = f"{prefix}/{path}"
            if path not in group:
                self.issue("episode.dataset_missing", full, "required transition dataset is missing")
                continue
            dataset = group[path]
            expected_shape = (length, *spec.tail_shape)
            if tuple(dataset.shape) != expected_shape:
                self.issue(
                    "episode.dataset_shape",
                    full,
                    f"expected {expected_shape}, got {tuple(dataset.shape)}",
                )
                continue
            if not _dtype_matches(dataset.dtype, spec.dtype):
                self.issue(
                    "episode.dataset_dtype",
                    full,
                    f"unexpected dtype {dataset.dtype}; expected {spec.dtype}",
                )
                continue
            try:
                value = _read_dataset(dataset, spec)
            except (OSError, UnicodeError, ValueError) as error:
                self.issue("episode.dataset_read", full, str(error))
                continue
            arrays[path] = value
            if spec.dtype != "str":
                numeric = np.asarray(value, dtype=np.float64)
                if not np.all(np.isfinite(numeric)):
                    self.issue("episode.dataset_nonfinite", full, "contains non-finite values")
            if np.issubdtype(np.dtype(spec.dtype), np.integer):
                numeric = np.asarray(value)
                if path.endswith("position_tracked"):
                    allowed = (-1, 0, 1)
                elif _is_binary_row_path(path):
                    allowed = (0, 1)
                else:
                    allowed = None
                if allowed is not None and not np.all(np.isin(numeric, allowed)):
                    self.issue(
                        "episode.dataset_value",
                        full,
                        f"values must be in {allowed}",
                    )
        episode = _EpisodeData(index, group, length, joint_names, arm_indices, arrays)
        if len(arrays) == len(specs):
            self._validate_transition_rows(episode)
        self._validate_episode_terminal_row(episode)
        return episode

    def _validate_episode_clock_attrs(self, group: Any) -> None:
        prefix = group.name
        start_mono = _integer(group.attrs.get("start_monotonic_ns"))
        end_mono = _integer(group.attrs.get("end_monotonic_ns"))
        start_unix = _integer(group.attrs.get("start_unix_ns"))
        end_unix = _integer(group.attrs.get("end_unix_ns"))
        if start_mono is None or end_mono is None or start_mono <= 0 or end_mono < start_mono:
            self.issue("episode.temporal_order", prefix, "episode monotonic bounds are invalid")
        if start_unix is None or end_unix is None or start_unix <= 0 or end_unix < start_unix:
            self.issue("episode.temporal_order", prefix, "episode Unix bounds are invalid")

    def _read_action_static(self, group: Any) -> tuple[tuple[str, ...], np.ndarray] | None:
        prefix = group.name
        required = ("actions/all_joint_names", "actions/arm_joint_indices", "actions/arm_joint_names")
        missing = [path for path in required if path not in group]
        for path in missing:
            self.issue("episode.dataset_missing", f"{prefix}/{path}", "static action metadata is missing")
        if missing:
            return None
        try:
            joint_names = tuple(_string_array(group["actions/all_joint_names"]))
            arm_names = tuple(_string_array(group["actions/arm_joint_names"]))
            arm_indices = np.asarray(group["actions/arm_joint_indices"][()], dtype=np.int64).reshape(-1)
        except (OSError, UnicodeError, ValueError) as error:
            self.issue("episode.dataset_read", f"{prefix}/actions", str(error))
            return None
        if joint_names != EXPECTED_PANDA_JOINT_NAMES:
            self.issue(
                "action.joint_contract",
                f"{prefix}/actions/all_joint_names",
                f"expected frozen Panda order {EXPECTED_PANDA_JOINT_NAMES!r}, got {joint_names!r}",
            )
        expected_indices = np.arange(7, dtype=np.int64)
        if not np.array_equal(arm_indices, expected_indices):
            self.issue(
                "action.joint_contract",
                f"{prefix}/actions/arm_joint_indices",
                f"expected {expected_indices.tolist()}, got {arm_indices.tolist()}",
            )
        if arm_indices.size and (
            np.any(arm_indices < 0)
            or np.any(arm_indices >= len(joint_names))
            or np.unique(arm_indices).size != arm_indices.size
        ):
            self.issue(
                "action.joint_contract",
                f"{prefix}/actions/arm_joint_indices",
                "arm indices are out of range or duplicated",
            )
        expected_arm_names = tuple(
            joint_names[int(value)]
            for value in arm_indices
            if 0 <= int(value) < len(joint_names)
        )
        if arm_names != expected_arm_names:
            self.issue(
                "action.joint_contract",
                f"{prefix}/actions/arm_joint_names",
                "arm names do not match all_joint_names at arm_joint_indices",
            )
        return joint_names, arm_indices

    def _validate_transition_rows(self, episode: _EpisodeData) -> None:
        a = episode.arrays
        prefix = episode.group.name
        length = episode.length
        source = a["control/source_step"].astype(np.int64)
        result = a["control/result_step"].astype(np.int64)
        expected = np.arange(length, dtype=np.int64)
        if not np.array_equal(source, expected):
            self.issue(
                "control.step_sequence",
                f"{prefix}/control/source_step",
                "source_step must be contiguous and zero-based",
            )
        if not np.array_equal(result, source + 1):
            self.issue(
                "control.step_sequence",
                f"{prefix}/control/result_step",
                "result_step must equal source_step + 1",
            )
        ordered_paths = (
            "control/observation_monotonic_ns",
            "control/preflight_monotonic_ns",
            "control/policy_inference_started_monotonic_ns",
            "control/policy_output_monotonic_ns",
            "control/rmpflow_output_monotonic_ns",
            "control/cbf_output_monotonic_ns",
            "control/action_command_monotonic_ns",
            "control/next_observation_monotonic_ns",
            "control/post_step_monotonic_ns",
        )
        clocks = [a[path].astype(np.int64) for path in ordered_paths]
        episode_start = _integer(episode.group.attrs.get("start_monotonic_ns"))
        episode_end = _integer(episode.group.attrs.get("end_monotonic_ns"))
        if (
            episode_start is not None
            and episode_start > int(clocks[0][0])
        ) or (
            episode_end is not None
            and episode_end < int(clocks[-1][-1])
        ):
            self.issue(
                "episode.temporal_order",
                prefix,
                "episode bounds do not contain all transition clocks",
            )
        for left_path, right_path, left, right in zip(
            ordered_paths, ordered_paths[1:], clocks, clocks[1:]
        ):
            bad = np.flatnonzero(right < left)
            if bad.size:
                self.issue(
                    "control.temporal_order",
                    f"{prefix}/{right_path}",
                    f"timestamp precedes {left_path} at row {int(bad[0])}",
                )
        for path in ("control/observation_monotonic_ns", "control/wall_time_unix_ns"):
            values = a[path].astype(np.int64)
            if np.any(values <= 0) or (length > 1 and np.any(np.diff(values) <= 0)):
                self.issue(
                    "control.temporal_order",
                    f"{prefix}/{path}",
                    "timestamps must be positive and strictly increasing",
                )
        preflight_clock = a["control/preflight_monotonic_ns"].astype(np.int64)
        if length > 1 and np.any(clocks[-1][:-1] > preflight_clock[1:]):
            self.issue(
                "control.temporal_order",
                f"{prefix}/control",
                "a row finishes after the next control preflight began",
            )
        raw_observations = a["observations/obs_raw"].astype(np.float32)
        policy_observations = a["observations/obs_policy"].astype(np.float32)
        for row in range(length):
            expected_policy = mask_human_observations(raw_observations[row])
            if not np.array_equal(policy_observations[row], expected_policy):
                self.issue(
                    "policy.human_mask_mismatch",
                    f"{prefix}/observations/obs_policy",
                    f"policy input is not the exact frozen human mask at row {row}",
                )
                break
        dt = _finite_float(episode.group.file.attrs.get("physics_dt_s"))
        sim_time = a["control/sim_time"].astype(np.float64)
        if dt is not None and length > 1 and not np.allclose(
            np.diff(sim_time), dt, rtol=0.0, atol=max(1e-9, dt * 1e-6)
        ):
            self.issue(
                "control.sim_time",
                f"{prefix}/control/sim_time",
                "sim_time increments do not match physics_dt_s",
            )
        self._validate_tracking(episode)
        self._validate_safety(episode)
        self._validate_actions(episode)
        self._validate_cbf(episode)

    def _validate_tracking(self, episode: _EpisodeData) -> None:
        a = episode.arrays
        prefix = episode.group.name
        max_age = _finite_float(episode.group.file.attrs.get("max_pose_age_ms"))
        slices = observation_slices()
        for namespace, observation_path, sample_clocks, validity_clocks in (
            (
                "human",
                "observations/obs_raw",
                a["control/observation_monotonic_ns"].astype(np.int64),
                a["control/action_command_monotonic_ns"].astype(np.int64),
            ),
            (
                "human_next",
                "observations/obs_next",
                a["control/next_observation_monotonic_ns"].astype(np.int64),
                a["control/post_step_monotonic_ns"].astype(np.int64),
            ),
        ):
            valid_mask = a[f"{namespace}/valid_mask"].astype(np.int8)
            computed = np.ones_like(valid_mask)
            for column, entity in enumerate(ENTITY_NAMES):
                pose_valid = a[f"{namespace}/{entity}_pose_valid"].astype(np.int8)
                tracked = a[f"{namespace}/{entity}_position_tracked"].astype(np.int8)
                known = a[f"{namespace}/{entity}_tracking_status_known"].astype(np.int8)
                switched = a[f"{namespace}/{entity}_source_switched"].astype(np.int8)
                sources = a[f"{namespace}/{entity}_source_name"]
                source_paths = a[f"{namespace}/{entity}_source_path"]
                acquisitions = a[
                    f"{namespace}/{entity}_acquisition_monotonic_ns"
                ].astype(np.int64)
                source_ages = a[f"{namespace}/{entity}_pose_age_ms"].astype(
                    np.float64
                )
                positions = a[f"{namespace}/{entity}_position_world_m"].astype(
                    np.float64
                )
                control_ages = (validity_clocks - acquisitions) / 1e6
                allowed = DEFAULT_ALLOWED_SOURCES[entity]
                for row in range(episode.length):
                    ok = bool(
                        pose_valid[row] == 1
                        # Physical XR may not expose POSITION_TRACKED.  Unknown
                        # is acceptable only with the explicit unknown sentinel.
                        and (
                            (known[row] == 0 and tracked[row] == -1)
                            or (known[row] == 1 and tracked[row] == 1)
                        )
                        and switched[row] == 0
                        and str(sources[row]) in allowed
                        and bool(str(source_paths[row]).strip())
                        and acquisitions[row] > 0
                        and acquisitions[row] <= sample_clocks[row]
                        and source_ages[row] >= 0.0
                        and np.all(np.isfinite(positions[row]))
                    )
                    if max_age is not None:
                        ok = bool(
                            ok
                            and source_ages[row] <= max_age + 1e-6
                            and control_ages[row] >= 0.0
                            and source_ages[row] + control_ages[row]
                            <= max_age + 1e-6
                        )
                    computed[row, column] = int(ok)
            mismatch = np.flatnonzero(np.any(valid_mask != computed, axis=1))
            if mismatch.size:
                self.issue(
                    "tracking.valid_mask_mismatch",
                    f"{prefix}/{namespace}/valid_mask",
                    f"valid_mask disagrees with provenance at row {int(mismatch[0])}",
                )
            nonvalid = np.flatnonzero(np.any(computed != 1, axis=1))
            if nonvalid.size:
                self.issue(
                    "tracking.invalid_row",
                    f"{prefix}/{namespace}",
                    f"required live tracking is invalid at row {int(nonvalid[0])}",
                )
            observations = a[observation_path].astype(np.float32)
            for entity, field in (
                ("head", "human_head_pos"),
                ("left", "human_left_hand_pos"),
                ("right", "human_right_hand_pos"),
            ):
                actual = observations[:, slices[field]]
                expected = a[
                    f"{namespace}/{entity}_position_world_m"
                ].astype(np.float32)
                mismatch = np.flatnonzero(np.any(actual != expected, axis=1))
                if mismatch.size:
                    self.issue(
                        "tracking.observation_mismatch",
                        f"{prefix}/{observation_path}",
                        f"{field} disagrees with {namespace} provenance at row "
                        f"{int(mismatch[0])}",
                    )

    def _validate_safety(self, episode: _EpisodeData) -> None:
        a = episode.arrays
        prefix = episode.group.name
        geometry_valid = a["safety/geometry_valid"].astype(np.int8)
        invalid_geometry = np.flatnonzero(geometry_valid != 1)
        if invalid_geometry.size:
            self.issue(
                "safety.geometry_invalid",
                f"{prefix}/safety/geometry_valid",
                f"required human-robot geometry is invalid at row "
                f"{int(invalid_geometry[0])}",
            )
        left = a["safety/left_surface_gap_m"].astype(np.float64)
        right = a["safety/right_surface_gap_m"].astype(np.float64)
        stored_minimum = a["safety/min_surface_gap_m"].astype(np.float64)
        expected_minimum = np.minimum(left, right)
        mismatch = np.flatnonzero(
            ~np.isclose(
                stored_minimum,
                expected_minimum,
                rtol=_FLOAT_RTOL,
                atol=_FLOAT_ATOL,
            )
        )
        if mismatch.size:
            row = int(mismatch[0])
            self.issue(
                "safety.minimum_gap_mismatch",
                f"{prefix}/safety/min_surface_gap_m",
                f"stored minimum gap disagrees with per-hand gaps at row {row}",
            )
        slices = observation_slices()
        observations = a["observations/obs_raw"].astype(np.float32)
        observation_minimum = observations[
            :, slices["min_hand_gripper_dist"]
        ].reshape(-1)
        minimum_mismatch = np.flatnonzero(
            ~np.isclose(
                observation_minimum,
                stored_minimum,
                rtol=_FLOAT_RTOL,
                atol=_FLOAT_ATOL,
            )
        )
        collision_mismatch = np.flatnonzero(
            observations[:, slices["human_robot_collision"]].reshape(-1)
            != a["safety/collision"].astype(np.float32)
        )
        near_mismatch = np.flatnonzero(
            observations[:, slices["near_human"]].reshape(-1)
            != a["safety/near"].astype(np.float32)
        )
        first_mismatch = min(
            (
                int(rows[0])
                for rows in (
                    minimum_mismatch,
                    collision_mismatch,
                    near_mismatch,
                )
                if rows.size
            ),
            default=None,
        )
        if first_mismatch is not None:
            self.issue(
                "safety.observation_mismatch",
                f"{prefix}/observations/obs_raw",
                "serialized safety values disagree with the raw observation at "
                f"row {first_mismatch}",
            )

    def _validate_actions(self, episode: _EpisodeData) -> None:
        a = episode.arrays
        prefix = episode.group.name
        arm = episode.arm_indices
        all_indices = np.arange(len(episode.joint_names), dtype=np.int64)
        fingers = np.setdiff1d(all_indices, arm, assume_unique=True)
        pipeline = a["actions/action_pipeline_complete"].astype(np.int8)
        bad_pipeline = np.flatnonzero(pipeline != 1)
        if bad_pipeline.size:
            self.issue(
                "action.pipeline_incomplete",
                f"{prefix}/actions/action_pipeline_complete",
                f"pipeline is incomplete at row {int(bad_pipeline[0])}",
            )
        for stage in ("nominal_rmpflow", "cbf_filtered", "submitted", "pre_applied", "applied"):
            command = np.zeros((episode.length, len(episode.joint_names)), dtype=bool)
            for channel, unit in (("joint_positions", "rad"), ("joint_velocities", "rad_s"), ("joint_efforts", "nm")):
                base = f"actions/{stage}_{channel}_{unit}"
                values = a[base].astype(np.float64)
                mask = a[base + "_mask"].astype(bool)
                command |= mask
                invalid_payload = np.abs(values[~mask]) > _MASKED_ZERO_ATOL
                if np.any(invalid_payload):
                    self.issue(
                        "action.masked_payload_nonzero",
                        f"{prefix}/{base}",
                        "uncommanded canonical slots must contain zero",
                    )
            empty = np.flatnonzero(~np.any(command, axis=1))
            if stage in ("nominal_rmpflow", "cbf_filtered", "submitted") and empty.size:
                self.issue(
                    "action.stage_incomplete",
                    f"{prefix}/actions/{stage}",
                    f"stage has no command at row {int(empty[0])}",
                )
            if stage in ("pre_applied", "applied"):
                for channel, unit in (("joint_positions", "rad"), ("joint_velocities", "rad_s")):
                    mask = a[f"actions/{stage}_{channel}_{unit}_mask"].astype(bool)
                    bad = np.flatnonzero(~np.all(mask, axis=1))
                    if bad.size:
                        self.issue(
                            "action.applied_target_incomplete",
                            f"{prefix}/actions/{stage}_{channel}_{unit}_mask",
                            f"full-DoF target is missing at row {int(bad[0])}",
                        )
                effort = a[f"actions/{stage}_joint_efforts_nm_mask"].astype(bool)
                partial = np.flatnonzero(np.any(effort, axis=1) & ~np.all(effort, axis=1))
                if partial.size:
                    self.issue(
                        "action.applied_target_incomplete",
                        f"{prefix}/actions/{stage}_joint_efforts_nm_mask",
                        f"effort target is partial at row {int(partial[0])}",
                    )
        for stage in ("nominal_rmpflow", "cbf_filtered"):
            mask = _command_mask(a, stage)
            bad_arm = np.flatnonzero(~np.all(mask[:, arm], axis=1))
            bad_fingers = np.flatnonzero(np.any(mask[:, fingers], axis=1))
            if bad_arm.size or bad_fingers.size:
                row = int(bad_arm[0] if bad_arm.size else bad_fingers[0])
                self.issue(
                    "action.stage_incomplete",
                    f"{prefix}/actions/{stage}",
                    f"stage must command exactly the seven arm joints at row {row}",
                )
        submitted = a["actions/arm_action_submitted"].astype(np.int8)
        gripper = a["actions/gripper_command"]
        for row in range(episode.length):
            if submitted[row] == 1:
                if str(gripper[row]).strip().lower() not in {"", "none"}:
                    self.issue(
                        "action.gripper_provenance",
                        f"{prefix}/actions/gripper_command",
                        f"arm submission row {row} unexpectedly records a gripper event",
                    )
                if not _stage_contains(a, "submitted", "cbf_filtered", row):
                    self.issue(
                        "action.submitted_filtered_mismatch",
                        f"{prefix}/actions/submitted",
                        f"submitted action does not preserve filtered CBF commands at row {row}",
                    )
                continue
            command = str(gripper[row]).strip()
            submitted_mask = _command_mask(a, "submitted")[row]
            if (
                command not in {"open", "close"}
                or np.any(submitted_mask[arm])
                or not np.all(submitted_mask[fingers])
            ):
                self.issue(
                    "action.sparse_gripper_carry_forward",
                    f"{prefix}/actions/submitted",
                    f"non-arm submission is not a complete sparse gripper event at row {row}",
                )
                continue
            carry_ok = True
            for channel, unit in (("joint_positions", "rad"), ("joint_velocities", "rad_s"), ("joint_efforts", "nm")):
                pre_base = f"actions/pre_applied_{channel}_{unit}"
                post_base = f"actions/applied_{channel}_{unit}"
                if not np.array_equal(
                    a[pre_base + "_mask"][row, arm], a[post_base + "_mask"][row, arm]
                ) or not np.allclose(
                    a[pre_base][row, arm],
                    a[post_base][row, arm],
                    rtol=0.0,
                    atol=_MASKED_ZERO_ATOL,
                ):
                    carry_ok = False
            for channel, unit in (("joint_positions", "rad"), ("joint_velocities", "rad_s"), ("joint_efforts", "nm")):
                submit_base = f"actions/submitted_{channel}_{unit}"
                post_base = f"actions/applied_{channel}_{unit}"
                mask = a[submit_base + "_mask"][row].astype(bool)
                if np.any(mask) and (
                    not np.all(a[post_base + "_mask"][row, mask] == 1)
                    or not np.allclose(
                        a[submit_base][row, mask],
                        a[post_base][row, mask],
                        rtol=0.0,
                        atol=_MASKED_ZERO_ATOL,
                    )
                ):
                    carry_ok = False
            if not carry_ok:
                self.issue(
                    "action.sparse_gripper_carry_forward",
                    f"{prefix}/actions/applied",
                    f"post-apply targets do not preserve arm drives and sparse gripper command at row {row}",
                )
        applied_velocity = a["actions/applied_arm_joint_velocities_rad_s"]
        applied_velocity_mask = a["actions/applied_arm_joint_velocities_mask"]
        full_velocity = a["actions/applied_joint_velocities_rad_s"][:, arm]
        full_velocity_mask = a["actions/applied_joint_velocities_rad_s_mask"][:, arm]
        if not np.array_equal(applied_velocity_mask, full_velocity_mask) or not np.allclose(
            applied_velocity, full_velocity, rtol=0.0, atol=_MASKED_ZERO_ATOL
        ):
            self.issue(
                "action.applied_arm_summary_mismatch",
                f"{prefix}/actions/applied_arm_joint_velocities_rad_s",
                "arm velocity summary differs from the full post-apply action",
            )

    def _validate_cbf(self, episode: _EpisodeData) -> None:
        a = episode.arrays
        prefix = episode.group.name
        dt = _finite_float(episode.group.file.attrs.get("physics_dt_s"))
        arm = episode.arm_indices
        if dt is None or dt <= 0.0:
            return
        nominal, nominal_ok = _stage_arm_velocity(a, "nominal_rmpflow", arm, dt)
        filtered, filtered_ok = _stage_arm_velocity(a, "cbf_filtered", arm, dt)
        incomplete = np.flatnonzero(~(nominal_ok & filtered_ok))
        if incomplete.size:
            self.issue(
                "action.stage_incomplete",
                f"{prefix}/actions",
                f"cannot reconstruct nominal/filtered arm velocity at row {int(incomplete[0])}",
            )
        nominal_norm = np.linalg.norm(nominal, axis=1)
        filtered_norm = np.linalg.norm(filtered, axis=1)
        intervention = np.linalg.norm(filtered - nominal, axis=1)
        diagnostics = (
            ("cbf/nominal_velocity_norm_radps", nominal_norm, "action.nominal_norm_mismatch"),
            ("cbf/filtered_velocity_norm_radps", filtered_norm, "action.filtered_norm_mismatch"),
            ("cbf/intervention_norm_radps", intervention, "action.intervention_norm_mismatch"),
        )
        for path, expected, code in diagnostics:
            actual = a[path].astype(np.float64)
            bad = np.flatnonzero(~np.isclose(actual, expected, rtol=_FLOAT_RTOL, atol=_FLOAT_ATOL))
            if bad.size:
                row = int(bad[0])
                self.issue(
                    code,
                    f"{prefix}/{path}",
                    f"diagnostic {actual[row]:.9g} disagrees with action trace {expected[row]:.9g} at row {row}",
                )
        active = a["cbf/active"].astype(np.int64)
        count = a["cbf/constraint_count"].astype(np.int64)
        inactive = np.flatnonzero(active == 0)
        for row in inactive:
            if not _stages_equal(a, "nominal_rmpflow", "cbf_filtered", int(row)):
                self.issue(
                    "action.inactive_inconsistent",
                    f"{prefix}/actions/cbf_filtered",
                    f"inactive CBF changed the nominal action at row {int(row)}",
                )
                break
        active_rows = np.flatnonzero(active == 1)
        max_speed = float(_scalar_or(episode.group.file.attrs.get("cbf_max_joint_speed_rad_s"), 2.0))
        if active_rows.size:
            values = np.abs(filtered[active_rows])
            bad_local = np.argwhere(values > max_speed + 1e-6)
            if bad_local.size:
                row = int(active_rows[int(bad_local[0, 0])])
                self.issue(
                    "action.filtered_speed_limit",
                    f"{prefix}/actions/cbf_filtered",
                    f"active filtered velocity exceeds {max_speed} rad/s at row {row}",
                )
        unsafe = np.zeros(episode.length, dtype=bool)
        unsafe |= a["cbf/intervention_available"].astype(np.int64) != 1
        unsafe |= a["cbf/feasible"].astype(np.int64) != 1
        unsafe |= a["cbf/fallback_applied"].astype(np.int64) != 0
        unsafe |= a["cbf/solver_converged"].astype(np.int64) != 1
        unsafe |= a["cbf/infeasibility_proven"].astype(np.int64) != 0
        unsafe |= a["cbf/relaxed_solution_applied"].astype(np.int64) != 0
        unsafe |= a["cbf/intentional_human_absence"].astype(np.int64) != 0
        unsafe |= count < 0
        unsafe |= count > 2
        unsafe |= active != (count > 0).astype(np.int64)
        unsafe |= a["cbf/tracked_hand_count"].astype(np.int64) != 2
        unsafe |= a["cbf/valid_hand_count"].astype(np.int64) != 2
        for path in (
            "cbf/intervention_norm_radps",
            "cbf/nominal_velocity_norm_radps",
            "cbf/filtered_velocity_norm_radps",
            "cbf/max_constraint_violation_before_mps",
            "cbf/max_constraint_violation_after_mps",
            "cbf/slack_mps",
            "cbf/solve_time_ms",
        ):
            unsafe |= a[path].astype(np.float64) < -_FLOAT_ATOL
        for row, raw in enumerate(a["cbf/failure_reasons_json"]):
            try:
                reasons = json.loads(str(raw))
            except json.JSONDecodeError:
                reasons = None
            if not isinstance(reasons, list) or reasons:
                unsafe[row] = True
        for row in inactive:
            if (
                count[row] != 0
                or abs(float(a["cbf/intervention_norm_radps"][row])) > _FLOAT_ATOL
                or abs(float(a["cbf/slack_mps"][row])) > _FLOAT_ATOL
                or str(a["cbf/status"][row]) != "inactive"
                or str(a["cbf/projection_status"][row]) != "inactive"
            ):
                unsafe[row] = True
        bad = np.flatnonzero(unsafe)
        if bad.size:
            self.issue(
                "cbf.unsafe_diagnostics",
                f"{prefix}/cbf",
                f"production CBF diagnostics are incomplete or unsafe at row {int(bad[0])}",
            )

    def _validate_episode_terminal_row(self, episode: _EpisodeData) -> None:
        a = episode.arrays
        required = ("task/success", "task/terminated", "task/truncated", "task/terminal_reason")
        if any(path not in a for path in required) or episode.length <= 0:
            return
        prefix = episode.group.name
        last = episode.length - 1
        mappings = (
            ("success", "task/success"),
            ("terminated", "task/terminated"),
            ("truncated", "task/truncated"),
        )
        for attr, path in mappings:
            if _integer(episode.group.attrs.get(attr)) != int(a[path][last]):
                self.issue(
                    "episode.terminal_row_mismatch",
                    f"{prefix}/{path}",
                    f"last row does not match episode {attr} attribute",
                )
        if _text(episode.group.attrs.get("terminal_reason", "")) != str(a["task/terminal_reason"][last]):
            self.issue(
                "episode.terminal_row_mismatch",
                f"{prefix}/task/terminal_reason",
                "last row does not match episode terminal_reason",
            )
        if episode.length > 1:
            early = (
                a["task/terminated"][:-1].astype(bool)
                | a["task/truncated"][:-1].astype(bool)
                | a["task/success"][:-1].astype(bool)
            )
            if np.any(early):
                self.issue(
                    "episode.terminal_row_mismatch",
                    f"{prefix}/task",
                    "terminal flags appear before the final transition",
                )

    def _validate_event_tables(self, data: Any) -> dict[str, dict[str, np.ndarray]] | None:
        tables: dict[str, dict[str, np.ndarray]] = {}
        complete = True
        for group_path, specs in _EVENT_SPECS.items():
            if group_path not in data or not hasattr(data[group_path], "keys"):
                self.issue("event.table_missing", "/" + group_path, "event table is missing")
                complete = False
                continue
            group = data[group_path]
            extras = sorted(set(group.keys()) - set(specs))
            missing = sorted(set(specs) - set(group.keys()))
            if extras:
                self.issue(
                    "event.column_extra",
                    group.name,
                    f"unknown event columns: {extras}",
                )
            for name in missing:
                self.issue(
                    "event.column_missing",
                    f"{group.name}/{name}",
                    "required event column is missing",
                )
            if missing:
                complete = False
                continue
            lengths = {int(group[name].shape[0]) for name in specs if group[name].ndim >= 1}
            if len(lengths) != 1:
                self.issue("event.column_length", group.name, "event columns are not aligned")
                complete = False
                continue
            length = lengths.pop()
            values: dict[str, np.ndarray] = {}
            for name, spec in specs.items():
                dataset = group[name]
                expected_shape = (length, *spec.tail_shape)
                if tuple(dataset.shape) != expected_shape:
                    self.issue(
                        "event.column_shape",
                        f"{group.name}/{name}",
                        f"expected {expected_shape}, got {tuple(dataset.shape)}",
                    )
                    complete = False
                    continue
                if not _dtype_matches(dataset.dtype, spec.dtype):
                    self.issue(
                        "event.column_dtype",
                        f"{group.name}/{name}",
                        f"unexpected dtype {dataset.dtype}",
                    )
                    complete = False
                    continue
                values[name] = _read_dataset(dataset, spec)
                if spec.dtype != "str" and not np.all(
                    np.isfinite(np.asarray(values[name], dtype=np.float64))
                ):
                    self.issue(
                        "event.column_nonfinite",
                        f"{group.name}/{name}",
                        "contains non-finite values",
                    )
            tables[group_path] = values
        return tables if complete else None

    def _validate_events(self, data: Any, tables: dict[str, dict[str, np.ndarray]]) -> None:
        encounters = tables["encounters"]
        prompts = tables["explicit_feedback/prompts"]
        responses = tables["explicit_feedback/responses"]
        encounter_count = len(encounters["encounter_id"])
        prompt_count = len(prompts["prompt_id"])
        response_count = len(responses["response_id"])
        self.counts.update(
            encounter_count=encounter_count,
            prompt_count=prompt_count,
            response_count=response_count,
        )
        for attr, actual in (
            ("encounter_count", encounter_count),
            ("prompt_count", prompt_count),
            ("response_count", response_count),
        ):
            stored = _integer(data.attrs.get(attr))
            if stored != actual:
                self.issue(
                    "root.count_mismatch",
                    f"/@{attr}",
                    f"attribute says {stored!r}, table has {actual}",
                )
        if encounter_count == 0:
            self.issue(
                "encounter.empty",
                "/encounters",
                "study-eligible explicit-feedback collection needs an encounter",
            )
        encounter_ids = tuple(str(value).strip() for value in encounters["encounter_id"])
        prompt_ids = tuple(str(value).strip() for value in prompts["prompt_id"])
        response_ids = tuple(str(value).strip() for value in responses["response_id"])
        self._validate_ids(encounter_ids, "/encounters/encounter_id", "encounter.id")
        self._validate_ids(prompt_ids, "/explicit_feedback/prompts/prompt_id", "prompt.id")
        self._validate_ids(response_ids, "/explicit_feedback/responses/response_id", "response.id")
        encounter_by_id = {value: index for index, value in enumerate(encounter_ids) if value}
        prompt_by_id = {value: index for index, value in enumerate(prompt_ids) if value}
        self._validate_encounters(encounters, encounter_by_id)
        prompt_timeout_s = _finite_float(data.attrs.get("prompt_timeout_s"))
        prompt_window_ns = (
            int(round(prompt_timeout_s * 1e9))
            if prompt_timeout_s is not None
            and 0.1 <= prompt_timeout_s <= 3600.0
            else None
        )
        self._validate_prompts(
            encounters,
            prompts,
            encounter_by_id,
            prompt_by_id,
            prompt_window_ns,
        )
        self._validate_responses(prompts, responses, prompt_by_id)
        self._validate_event_counts_by_episode(encounters, prompts, responses, encounter_by_id, prompt_by_id)

    def _validate_ids(self, values: tuple[str, ...], path: str, code: str) -> None:
        if any(not value for value in values):
            self.issue(code, path, "IDs must be non-empty")
        if len(set(values)) != len(values):
            self.issue(code, path, "IDs must be unique")

    def _validate_encounters(
        self, encounters: Mapping[str, np.ndarray], encounter_by_id: Mapping[str, int]
    ) -> None:
        intervals: dict[int, list[tuple[int, int, str]]] = {}
        activation = 0.13
        for row, encounter_id in enumerate(str(value).strip() for value in encounters["encounter_id"]):
            episode_index = int(encounters["episode_index"][row])
            episode = self.episodes.get(episode_index)
            if episode is None:
                self.issue(
                    "encounter.orphan_episode",
                    f"/encounters[{row}]",
                    f"unknown episode_index {episode_index}",
                )
                continue
            start = int(encounters["start_step"][row])
            end = int(encounters["end_step_exclusive"][row])
            if not 0 <= start < end <= episode.length:
                self.issue(
                    "encounter.interval",
                    f"/encounters[{row}]",
                    f"invalid [{start}, {end}) interval for episode length {episode.length}",
                )
                continue
            intervals.setdefault(episode_index, []).append((start, end, encounter_id))
            eligible = int(encounters["eligible"][row])
            reason = str(encounters["eligibility_reason"][row])
            status = str(encounters["terminal_status"][row])
            abort = str(encounters["abort_reason"][row]).strip()
            prompt_id = str(encounters["prompt_id"][row]).strip()
            if (
                eligible != 1
                or reason != "cbf_activation_band_entry"
                or status not in {"completed", "episode_end"}
                or abort
                or not prompt_id
            ):
                self.issue(
                    "encounter.ineligible",
                    f"/encounters[{row}]",
                    "sealed encounters must be eligible, non-aborted, and linked to a prompt",
                )
            a = episode.arrays
            needed = (
                "control/observation_monotonic_ns",
                "control/cbf_output_monotonic_ns",
                "control/post_step_monotonic_ns",
                "safety/left_surface_gap_m",
                "safety/right_surface_gap_m",
                "encounter/row_encounter_id",
                "human/valid_mask",
            )
            if any(path not in a for path in needed):
                continue
            start_clock = int(encounters["start_monotonic_ns"][row])
            end_clock = int(encounters["end_monotonic_ns"][row])
            start_ok = start_clock == int(a["control/cbf_output_monotonic_ns"][start])
            if status == "episode_end":
                episode_end = _integer(episode.group.attrs.get("end_monotonic_ns"))
                end_ok = bool(
                    episode_end is not None
                    and int(a["control/post_step_monotonic_ns"][end - 1])
                    <= end_clock
                    <= episode_end
                )
            else:
                end_ok = end_clock == int(a["control/cbf_output_monotonic_ns"][end - 1])
            if not start_ok or not end_ok:
                self.issue(
                    "encounter.temporal_order",
                    f"/encounters[{row}]",
                    "encounter clocks do not align with its transition interval",
                )
            gaps = np.column_stack(
                (a["safety/left_surface_gap_m"][start:end], a["safety/right_surface_gap_m"][start:end])
            ).astype(np.float64)
            observed_min = float(np.min(gaps))
            stored_min = float(encounters["minimum_surface_gap_m"][row])
            if not math.isclose(observed_min, stored_min, rel_tol=_FLOAT_RTOL, abs_tol=_FLOAT_ATOL):
                self.issue(
                    "encounter.minimum_gap",
                    f"/encounters[{row}]/minimum_surface_gap_m",
                    f"stored {stored_min:.9g}, rows imply {observed_min:.9g}",
                )
            observed_mask = np.any(gaps <= activation + _FLOAT_ATOL, axis=0).astype(np.int8)
            stored_mask = encounters["involved_hand_mask"][row].astype(np.int8)
            if not np.array_equal(observed_mask, stored_mask) or not np.any(stored_mask):
                self.issue(
                    "encounter.hand_mask",
                    f"/encounters[{row}]/involved_hand_mask",
                    "involved hands disagree with activation-band rows",
                )
            actual_ids = tuple(str(value) for value in a["encounter/row_encounter_id"][start:end])
            if any(value != encounter_id for value in actual_ids):
                self.issue(
                    "encounter.row_membership",
                    f"{episode.group.name}/encounter/row_encounter_id",
                    f"rows [{start}, {end}) do not all reference {encounter_id!r}",
                )
            tracking_ok = all(
                _tracking_row_is_valid(episode, tracking_row)
                for tracking_row in range(start, end)
            )
            if not tracking_ok:
                self.issue(
                    "tracking.invalid_during_eligible_encounter",
                    f"{episode.group.name}/human",
                    f"tracking invalid in eligible encounter {encounter_id!r}",
                )
        for episode_index, values in intervals.items():
            ordered = sorted(values)
            for previous, current in zip(ordered, ordered[1:]):
                if previous[1] > current[0]:
                    self.issue(
                        "encounter.overlap",
                        f"/episodes/episode_{episode_index:06d}",
                        f"encounters {previous[2]!r} and {current[2]!r} overlap",
                    )
            episode = self.episodes[episode_index]
            expected = np.full(episode.length, "", dtype=object)
            for start, end, encounter_id in ordered:
                if 0 <= start < end <= episode.length:
                    expected[start:end] = encounter_id
            actual = np.asarray(episode.arrays.get("encounter/row_encounter_id", []), dtype=object)
            if actual.shape == expected.shape and not np.array_equal(actual, expected):
                self.issue(
                    "encounter.row_membership",
                    f"{episode.group.name}/encounter/row_encounter_id",
                    "row encounter IDs do not exactly match event intervals",
                )

    def _validate_prompts(
        self,
        encounters: Mapping[str, np.ndarray],
        prompts: Mapping[str, np.ndarray],
        encounter_by_id: Mapping[str, int],
        prompt_by_id: Mapping[str, int],
        prompt_window_ns: int | None,
    ) -> None:
        prompts_by_encounter: dict[str, list[int]] = {}
        for row, encounter_id in enumerate(str(value).strip() for value in prompts["encounter_id"]):
            prompts_by_encounter.setdefault(encounter_id, []).append(row)
            if encounter_id not in encounter_by_id:
                self.issue(
                    "prompt.orphan_encounter",
                    f"/explicit_feedback/prompts[{row}]",
                    f"unknown encounter_id {encounter_id!r}",
                )
                continue
            encounter_row = encounter_by_id[encounter_id]
            prompt_id = str(prompts["prompt_id"][row]).strip()
            if str(encounters["prompt_id"][encounter_row]).strip() != prompt_id:
                self.issue(
                    "prompt.link_mismatch",
                    f"/explicit_feedback/prompts[{row}]",
                    "encounter prompt_id does not point back to this prompt",
                )
            if str(prompts["question_id"][row]) != QUESTION_ID:
                self.issue(
                    "prompt.question",
                    f"/explicit_feedback/prompts[{row}]/question_id",
                    f"expected {QUESTION_ID!r}",
                )
            issued = int(prompts["issued_monotonic_ns"][row])
            presentation_deadline = int(
                prompts["presentation_deadline_monotonic_ns"][row]
            )
            presented = int(prompts["presented_monotonic_ns"][row])
            deadline = int(prompts["deadline_monotonic_ns"][row])
            encounter_end = int(encounters["end_monotonic_ns"][encounter_row])
            if not (
                encounter_end <= issued < presentation_deadline
                and issued <= presented < presentation_deadline
                and presented < deadline
            ):
                self.issue(
                    "prompt.temporal_order",
                    f"/explicit_feedback/prompts[{row}]",
                    "expected encounter_end <= issued < presentation_deadline, "
                    "issued <= presented < presentation_deadline, and presented < deadline",
                )
            if prompt_window_ns is not None and (
                presentation_deadline - issued != prompt_window_ns
                or deadline - presented != prompt_window_ns
            ):
                self.issue(
                    "prompt.timeout_window",
                    f"/explicit_feedback/prompts[{row}]",
                    "presentation and response deadlines must each equal their "
                    "server timestamp plus prompt_timeout_s",
                )
            if int(prompts["issued_unix_ns"][row]) <= 0:
                self.issue(
                    "prompt.temporal_order",
                    f"/explicit_feedback/prompts[{row}]/issued_unix_ns",
                    "issued Unix time must be positive",
                )
            if str(prompts["status"][row]) != "accepted":
                self.issue(
                    "prompt.status",
                    f"/explicit_feedback/prompts[{row}]/status",
                    "sealed study prompt must have accepted status",
                )
        for encounter_id in encounter_by_id:
            rows = prompts_by_encounter.get(encounter_id, [])
            if len(rows) != 1:
                self.issue(
                    "prompt.count",
                    "/explicit_feedback/prompts",
                    f"encounter {encounter_id!r} has {len(rows)} prompts; expected one",
                )

    def _validate_responses(
        self,
        prompts: Mapping[str, np.ndarray],
        responses: Mapping[str, np.ndarray],
        prompt_by_id: Mapping[str, int],
    ) -> None:
        accepted_by_prompt: dict[str, list[int]] = {}
        seen_client_ids: set[str] = set()
        previous_received: int | None = None
        for row, prompt_id in enumerate(
            str(value).strip() for value in responses["prompt_id"]
        ):
            client_response_id = str(responses["client_response_id"][row]).strip()
            duplicate_client_id = client_response_id in seen_client_ids
            accepted = int(responses["accepted"][row])
            rejection = str(responses["rejection_reason"][row]).strip()
            response_code = str(responses["response_code"][row])
            source = str(responses["source"][row]).strip()
            received = int(responses["received_monotonic_ns"][row])
            received_unix = int(responses["received_unix_ns"][row])

            if not client_response_id:
                self.issue(
                    "response.client_id",
                    f"/explicit_feedback/responses[{row}]/client_response_id",
                    "client response ID must be non-empty",
                )
            elif duplicate_client_id and rejection not in {
                "duplicate_response_id",
                # ``broker_closed`` has priority over duplicate detection so
                # in-flight requests can be retained during the final drain.
                "broker_closed",
            }:
                self.issue(
                    "response.client_id",
                    f"/explicit_feedback/responses[{row}]",
                    "reused client response ID must be rejected as "
                    "duplicate_response_id (or broker_closed after shutdown)",
                )
            seen_client_ids.add(client_response_id)

            if (
                not source
                or received <= 0
                or received_unix <= 0
                or (previous_received is not None and received < previous_received)
            ):
                self.issue(
                    "response.audit_fields",
                    f"/explicit_feedback/responses[{row}]",
                    "source and server timestamps must be present, and response "
                    "monotonic timestamps must not decrease",
                )
            previous_received = received

            if accepted not in (0, 1):
                self.issue(
                    "response.accepted_value",
                    f"/explicit_feedback/responses[{row}]/accepted",
                    "must be exactly 0 or 1",
                )
                continue

            known_prompt = prompt_id in prompt_by_id
            if accepted == 1 and not known_prompt:
                self.issue(
                    "response.orphan_prompt",
                    f"/explicit_feedback/responses[{row}]",
                    f"unknown prompt_id {prompt_id!r}",
                )
                continue

            if accepted == 0:
                if rejection not in _RESPONSE_REJECTION_REASONS:
                    self.issue(
                        "response.rejection_reason",
                        f"/explicit_feedback/responses[{row}]",
                        "rejected response needs a recognized audit reason",
                    )
                    continue
                if not known_prompt:
                    allowed_unknown_reason = (
                        rejection == "broker_closed"
                        or (
                            rejection == "unknown_prompt_id"
                            and not duplicate_client_id
                        )
                        or (rejection == "duplicate_response_id" and duplicate_client_id)
                    )
                    if not allowed_unknown_reason:
                        self.issue(
                            "response.orphan_prompt",
                            f"/explicit_feedback/responses[{row}]",
                            f"unknown prompt_id {prompt_id!r} has inconsistent "
                            f"rejection reason {rejection!r}",
                        )
                    continue
                if rejection == "unknown_prompt_id":
                    self.issue(
                        "response.rejection_reason",
                        f"/explicit_feedback/responses[{row}]",
                        "known prompt cannot be rejected as unknown_prompt_id",
                    )
                elif rejection == "duplicate_response_id" and not duplicate_client_id:
                    self.issue(
                        "response.rejection_reason",
                        f"/explicit_feedback/responses[{row}]",
                        "duplicate_response_id requires an earlier matching client ID",
                    )
                elif rejection == "invalid_response_code" and response_code in RESPONSE_CODES:
                    self.issue(
                        "response.rejection_reason",
                        f"/explicit_feedback/responses[{row}]",
                        "invalid_response_code requires a code outside the protocol",
                    )
                elif rejection == "prompt_already_answered" and not accepted_by_prompt.get(
                    prompt_id
                ):
                    self.issue(
                        "response.rejection_reason",
                        f"/explicit_feedback/responses[{row}]",
                        "prompt_already_answered requires an earlier accepted response",
                    )
                elif rejection == "prompt_expired":
                    prompt_row = prompt_by_id[prompt_id]
                    deadline = int(prompts["deadline_monotonic_ns"][prompt_row])
                    if accepted_by_prompt.get(prompt_id) or received < deadline:
                        self.issue(
                            "response.rejection_reason",
                            f"/explicit_feedback/responses[{row}]",
                            "prompt_expired conflicts with prompt state or deadline",
                        )
                elif rejection == "broker_closed" and not accepted_by_prompt.get(
                    prompt_id
                ):
                    self.issue(
                        "response.rejection_reason",
                        f"/explicit_feedback/responses[{row}]",
                        "sealed known prompt must have been accepted before broker close",
                    )
                elif rejection in {
                    "prompt_not_presented",
                    "response_precedes_presentation",
                }:
                    prompt_row = prompt_by_id[prompt_id]
                    presented = int(prompts["presented_monotonic_ns"][prompt_row])
                    if received > presented:
                        self.issue(
                            "response.rejection_reason",
                            f"/explicit_feedback/responses[{row}]",
                            f"{rejection} must not be timestamped after presentation",
                        )
                continue

            if rejection:
                self.issue(
                    "response.accepted_payload",
                    f"/explicit_feedback/responses[{row}]",
                    "accepted response must not have a rejection reason",
                )
            accepted_by_prompt.setdefault(prompt_id, []).append(row)
            prompt_row = prompt_by_id[prompt_id]
            presented = int(prompts["presented_monotonic_ns"][prompt_row])
            deadline = int(prompts["deadline_monotonic_ns"][prompt_row])
            if not (presented <= received < deadline):
                self.issue(
                    "response.temporal_order",
                    f"/explicit_feedback/responses[{row}]",
                    "accepted response must arrive after presentation and before deadline",
                )
            if response_code not in RESPONSE_CODES or not source or rejection:
                self.issue(
                    "response.accepted_payload",
                    f"/explicit_feedback/responses[{row}]",
                    "accepted response code/source/timestamps/rejection fields are invalid",
                )
        for prompt_id in prompt_by_id:
            rows = accepted_by_prompt.get(prompt_id, [])
            if len(rows) != 1:
                self.issue(
                    "response.accepted_count",
                    "/explicit_feedback/responses",
                    f"prompt {prompt_id!r} has {len(rows)} accepted responses; expected one",
                )

    def _validate_event_counts_by_episode(
        self,
        encounters: Mapping[str, np.ndarray],
        prompts: Mapping[str, np.ndarray],
        responses: Mapping[str, np.ndarray],
        encounter_by_id: Mapping[str, int],
        prompt_by_id: Mapping[str, int],
    ) -> None:
        prompt_episode: dict[str, int] = {}
        encounter_counts = {index: 0 for index in self.episodes}
        prompt_counts = {index: 0 for index in self.episodes}
        accepted_counts = {index: 0 for index in self.episodes}
        for row, encounter_id in enumerate(str(value).strip() for value in encounters["encounter_id"]):
            episode_index = int(encounters["episode_index"][row])
            if episode_index in encounter_counts:
                encounter_counts[episode_index] += 1
        for row, prompt_id in enumerate(str(value).strip() for value in prompts["prompt_id"]):
            encounter_id = str(prompts["encounter_id"][row]).strip()
            encounter_row = encounter_by_id.get(encounter_id)
            if encounter_row is None:
                continue
            episode_index = int(encounters["episode_index"][encounter_row])
            prompt_episode[prompt_id] = episode_index
            if episode_index in prompt_counts:
                prompt_counts[episode_index] += 1
        for row, prompt_id in enumerate(str(value).strip() for value in responses["prompt_id"]):
            if int(responses["accepted"][row]) != 1:
                continue
            episode_index = prompt_episode.get(prompt_id)
            if episode_index in accepted_counts:
                accepted_counts[episode_index] += 1
        for index, episode in self.episodes.items():
            expected = {
                "encounter_count": encounter_counts[index],
                "prompt_count": prompt_counts[index],
                "accepted_response_count": accepted_counts[index],
            }
            for name, count in expected.items():
                stored = _integer(episode.group.attrs.get(name))
                if stored != count:
                    self.issue(
                        "episode.count_mismatch",
                        f"{episode.group.name}/@{name}",
                        f"attribute says {stored!r}, event relations imply {count}",
                    )


def _read_dataset(dataset: Any, spec: DatasetSpec) -> np.ndarray:
    if spec.dtype == "str":
        return np.asarray(dataset.asstr()[()], dtype=object)
    return np.asarray(dataset[()])


def _string_array(dataset: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in np.asarray(dataset.asstr()[()]).reshape(-1))


def _dtype_matches(actual: np.dtype, expected: Any) -> bool:
    if expected == "str":
        return np.dtype(actual).kind in "OSU"
    return np.dtype(actual) == np.dtype(expected)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(_scalar(value))


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar_or(value: Any, default: Any) -> Any:
    return default if value is None else _scalar(value)


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(_scalar(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or numeric != math.trunc(numeric):
        return None
    return int(numeric)


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(_scalar(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _same_scalar(actual: Any, expected: Any) -> bool:
    actual = _scalar(actual)
    if isinstance(expected, str):
        return _text(actual) == expected
    if isinstance(expected, float):
        value = _finite_float(actual)
        return value is not None and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
    value = _integer(actual)
    return value == int(expected)


def _command_mask(arrays: Mapping[str, np.ndarray], stage: str) -> np.ndarray:
    result = None
    for channel, unit in (("joint_positions", "rad"), ("joint_velocities", "rad_s"), ("joint_efforts", "nm")):
        mask = arrays[f"actions/{stage}_{channel}_{unit}_mask"].astype(bool)
        result = mask.copy() if result is None else result | mask
    assert result is not None
    return result


def _stage_contains(
    arrays: Mapping[str, np.ndarray], submitted: str, expected: str, row: int
) -> bool:
    for channel, unit in (("joint_positions", "rad"), ("joint_velocities", "rad_s"), ("joint_efforts", "nm")):
        submitted_base = f"actions/{submitted}_{channel}_{unit}"
        expected_base = f"actions/{expected}_{channel}_{unit}"
        expected_mask = arrays[expected_base + "_mask"][row].astype(bool)
        submitted_mask = arrays[submitted_base + "_mask"][row].astype(bool)
        if not np.all(submitted_mask[expected_mask]):
            return False
        if not np.array_equal(
            arrays[submitted_base][row, expected_mask],
            arrays[expected_base][row, expected_mask],
        ):
            return False
    return True


def _stages_equal(
    arrays: Mapping[str, np.ndarray], first: str, second: str, row: int
) -> bool:
    for channel, unit in (("joint_positions", "rad"), ("joint_velocities", "rad_s"), ("joint_efforts", "nm")):
        first_base = f"actions/{first}_{channel}_{unit}"
        second_base = f"actions/{second}_{channel}_{unit}"
        if not np.array_equal(
            arrays[first_base + "_mask"][row], arrays[second_base + "_mask"][row]
        ) or not np.array_equal(arrays[first_base][row], arrays[second_base][row]):
            return False
    return True


def _stage_arm_velocity(
    arrays: Mapping[str, np.ndarray],
    stage: str,
    arm_indices: np.ndarray,
    physics_dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows = arrays[f"actions/{stage}_joint_positions_rad"].shape[0]
    result = np.zeros((rows, arm_indices.size), dtype=np.float64)
    complete = np.zeros(rows, dtype=bool)
    velocity = arrays[f"actions/{stage}_joint_velocities_rad_s"][:, arm_indices].astype(np.float64)
    velocity_mask = arrays[f"actions/{stage}_joint_velocities_rad_s_mask"][:, arm_indices].astype(bool)
    positions = arrays[f"actions/{stage}_joint_positions_rad"][:, arm_indices].astype(np.float64)
    position_mask = arrays[f"actions/{stage}_joint_positions_rad_mask"][:, arm_indices].astype(bool)
    before = arrays["robot/measured_joint_positions_before_rad"][:, arm_indices].astype(np.float64)
    for row in range(rows):
        if np.all(velocity_mask[row]):
            result[row] = velocity[row]
            complete[row] = True
        elif np.all(position_mask[row]):
            result[row] = (positions[row] - before[row]) / physics_dt_s
            complete[row] = True
    return result, complete


def _tracking_row_is_valid(episode: _EpisodeData, row: int) -> bool:
    """Recompute the watchdog evidence for one serialized observation row."""

    arrays = episode.arrays
    observation_path = "control/observation_monotonic_ns"
    if (
        observation_path not in arrays
        or "control/action_command_monotonic_ns" not in arrays
        or "human/valid_mask" not in arrays
    ):
        return False
    required_paths = []
    for entity in ENTITY_NAMES:
        required_paths.extend(
            (
                f"human/{entity}_position_world_m",
                f"human/{entity}_pose_valid",
                f"human/{entity}_position_tracked",
                f"human/{entity}_tracking_status_known",
                f"human/{entity}_source_name",
                f"human/{entity}_source_path",
                f"human/{entity}_acquisition_monotonic_ns",
                f"human/{entity}_pose_age_ms",
                f"human/{entity}_source_switched",
            )
        )
    if any(path not in arrays for path in required_paths):
        return False
    if not np.all(arrays["human/valid_mask"][row] == 1):
        return False
    observation_ns = int(arrays[observation_path][row])
    action_command_ns = int(arrays["control/action_command_monotonic_ns"][row])
    max_age_ms = _finite_float(episode.group.file.attrs.get("max_pose_age_ms"))
    if max_age_ms is None or max_age_ms <= 0.0:
        return False
    for entity in ENTITY_NAMES:
        acquisition_ns = int(arrays[f"human/{entity}_acquisition_monotonic_ns"][row])
        source_age_ms = float(arrays[f"human/{entity}_pose_age_ms"][row])
        if not (
            int(arrays[f"human/{entity}_pose_valid"][row]) == 1
            and (
                (
                    int(arrays[f"human/{entity}_tracking_status_known"][row]) == 0
                    and int(arrays[f"human/{entity}_position_tracked"][row]) == -1
                )
                or (
                    int(arrays[f"human/{entity}_tracking_status_known"][row]) == 1
                    and int(arrays[f"human/{entity}_position_tracked"][row]) == 1
                )
            )
            and int(arrays[f"human/{entity}_source_switched"][row]) == 0
            and str(arrays[f"human/{entity}_source_name"][row])
            in DEFAULT_ALLOWED_SOURCES[entity]
            and bool(str(arrays[f"human/{entity}_source_path"][row]).strip())
            and 0 < acquisition_ns <= observation_ns
            and 0.0 <= source_age_ms <= max_age_ms + 1e-6
            and action_command_ns >= observation_ns
            and source_age_ms + (action_command_ns - acquisition_ns) / 1e6
            <= max_age_ms + 1e-6
            and np.all(
                np.isfinite(arrays[f"human/{entity}_position_world_m"][row])
            )
        ):
            return False
    return True


__all__ = [
    "CollectionValidationError",
    "ValidationIssue",
    "ValidationReport",
    "validate_collection",
    "validate_collection_or_raise",
]
