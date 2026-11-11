"""Fail-closed validator for ``bc_cbf_online_explicit_feedback_v1`` artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .online_config import validate_frozen_protocol_config
from .online_rows import online_row_specs
from .online_scenario import NOMINAL_BC_SWEEP_SEMANTICS
from .online_schema import (
    DIRECTIONS,
    MARKER_TYPES,
    ONLINE_PROTOCOL_VERSION,
    ONLINE_ROW_SEMANTICS,
    ONLINE_SCHEMA_VERSION,
    Q1_ID,
    Q1_RESPONSES,
    Q1_TEXT_KO,
    LIKERT_QUESTIONS,
    REJECTION_REASONS,
    SEVERITIES,
    SPEEDS,
    TASK_PHASES,
)
from .schema import (
    POLICY_RELATIVE_PATH,
    POLICY_SHA256,
    POLICY_SIZE_BYTES,
    RUNTIME_CONTRACT_SHA256,
    RUNTIME_HANDOFF_COMMIT,
)
from .policy import mask_human_observations


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "schedule": (
        "trial_index",
        "trial_id",
        "query_id",
        "task_phase",
        "severity",
        "direction",
        "speed",
        "crossing_hand",
        "practice",
        "anchor_repeat",
        "pilot",
        "analysis_exclude",
        "source_trial_id",
    ),
    "layout_prechecks": (
        "candidate_index",
        "layout_id",
        "seed",
        "controller",
        "policy_sha256",
        "cbf_enabled",
        "human_absent_or_clear",
        "strict_success",
        "release_observed",
        "nominal_phase_references_complete",
        "nominal_phase_references_json",
        "nominal_corridor_bank_complete",
        "nominal_phase_trajectories_json",
        "nominal_corridor_bank_json",
        "steps",
        "diagnostic_group",
        "failure_reason",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "source_restoration_json",
    ),
    "realtime_markers": (
        "marker_id",
        "trial_id",
        "marker_type",
        "simulation_time_s",
        "monotonic_ns",
        "unix_ns",
        "control_step",
        "controller_hand",
        "button",
    ),
    "queries": (
        "query_id",
        "trial_id",
        "encounter_id",
        "issued_simulation_time_s",
        "issued_monotonic_ns",
        "issued_unix_ns",
        "completed_simulation_time_s",
        "completed_monotonic_ns",
        "completed_unix_ns",
        "response_status",
        "q1_response",
        "q2_collision_risk",
        "q3_excessive_avoidance",
        "q4_abrupt_unpredictable",
        "q5_task_disruption",
        "q6_confidence",
        "rejection_reasons_json",
        "response_latency_ms",
        "input_source",
    ),
    "query_answers": (
        "query_id",
        "trial_id",
        "question_id",
        "value_json",
        "simulation_time_s",
        "monotonic_ns",
        "unix_ns",
        "control_step",
    ),
    "encounters": (
        "encounter_id",
        "trial_id",
        "onset_step",
        "onset_confirmed_step",
        "offset_step_exclusive",
        "onset_simulation_time_s",
        "offset_simulation_time_s",
        "onset_monotonic_ns",
        "offset_monotonic_ns",
        "window_start_step",
        "window_end_step_exclusive",
        "minimum_surface_gap_m",
        "maximum_intervention_norm_rad_s",
        "risk_onset_step",
        "cbf_intervention_start_step",
        "cbf_intervention_confirmed_step",
        "encounter_timeout",
        "merged_reentry_count",
    ),
}

_CONTROL_MODES = frozenset({"nominal_bc", "protocol_hold"})
_TRIAL_STATES = frozenset(
    {
        "wait_phase",
        "pre_crossing",
        "crossing",
        "response",
        "recovery",
        "query_hold",
        "resume_task",
    }
)


@dataclass(frozen=True)
class OnlineValidationIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineValidationReport:
    path: str
    valid: bool
    study_eligible: bool
    issues: tuple[OnlineValidationIssue, ...]
    trial_count: int = 0
    query_count: int = 0
    encounter_count: int = 0
    realtime_marker_count: int = 0

    # Compatibility with the old CLI/report surface.
    @property
    def episode_count(self) -> int:
        return self.trial_count

    @property
    def prompt_count(self) -> int:
        return self.query_count

    @property
    def response_count(self) -> int:
        return self.query_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "schema_version": ONLINE_SCHEMA_VERSION,
            "valid": self.valid,
            "study_eligible": self.study_eligible,
            "trial_count": self.trial_count,
            "query_count": self.query_count,
            "encounter_count": self.encounter_count,
            "realtime_marker_count": self.realtime_marker_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class OnlineCollectionValidationError(RuntimeError):
    def __init__(self, report: OnlineValidationReport) -> None:
        self.report = report
        detail = "; ".join(
            f"[{item.code}] {item.path}: {item.message}" for item in report.issues[:8]
        )
        super().__init__("online collection is not study-eligible: " + detail)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_online_collection(
    path: str | Path,
    *,
    allow_partial_path: bool = False,
    raise_on_error: bool = True,
) -> OnlineValidationReport:
    validator = _OnlineValidator(Path(path), allow_partial_path=allow_partial_path)
    report = validator.run()
    if raise_on_error and not report.valid:
        raise OnlineCollectionValidationError(report)
    return report


class _OnlineValidator:
    def __init__(self, path: Path, *, allow_partial_path: bool) -> None:
        self.path = path.expanduser().resolve(strict=False)
        self.allow_partial_path = bool(allow_partial_path)
        self.issues: list[OnlineValidationIssue] = []
        self.counts = {
            "trial_count": 0,
            "query_count": 0,
            "encounter_count": 0,
            "realtime_marker_count": 0,
        }
        self.file: Any = None
        self.protocol_config: dict[str, Any] = {}
        self.scenario_config: dict[str, Any] = {}
        self.cbf_config: dict[str, Any] = {}
        self.group_ready: dict[str, bool] = {}
        self.table_ready: dict[str, bool] = {}
        self.table_lengths: dict[str, int] = {}
        self.study_eligible = True

    def issue(self, code: str, path: str, message: str) -> None:
        self.issues.append(OnlineValidationIssue(code, path, message))

    def run(self) -> OnlineValidationReport:
        suffixes = [suffix.lower() for suffix in self.path.suffixes]
        final_suffix = suffixes[-1] if suffixes else ""
        if final_suffix == ".partial" and not self.allow_partial_path:
            self.issue("file.partial", str(self.path), "public validation rejects partial artifacts")
        if final_suffix != ".partial" and final_suffix not in {".h5", ".hdf5"}:
            self.issue("file.extension", str(self.path), "expected .h5 or .hdf5")
        if not self.path.is_file():
            self.issue("file.not_found", str(self.path), "path is not a regular file")
            return self._report()
        try:
            import h5py
        except ImportError:
            self.issue("file.h5py_unavailable", str(self.path), "h5py is required")
            return self._report()
        try:
            with h5py.File(self.path, "r") as handle:
                self.file = handle
                self._validate_root()
                self._validate_tables()
                self._validate_schedule_and_trials()
                self._validate_relations_and_timing()
                self._validate_design_counts()
        except OSError as error:
            self.issue("file.open", str(self.path), str(error))
        except Exception as error:
            # A malformed HDF5 file is untrusted input.  A validator must turn
            # unexpected h5py/numpy shape or lookup failures into a closed
            # report, never let them escape as a successful or crashed check.
            self.issue(
                "validator.malformed_artifact",
                "/",
                f"{type(error).__name__}: {error}",
            )
        finally:
            self.file = None
        return self._report()

    def _validate_root(self) -> None:
        attrs = self.file.attrs
        exact = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "protocol_version": ONLINE_PROTOCOL_VERSION,
            "row_semantics": ONLINE_ROW_SEMANTICS,
            "collection_complete": 1,
            "production_mode": 1,
            "policy_path": POLICY_RELATIVE_PATH,
            "policy_sha256": POLICY_SHA256,
            "policy_size_bytes": POLICY_SIZE_BYTES,
            "policy_mode": "direct_bc",
            "policy_frozen": 1,
            "policy_online_updates": 0,
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
            "q1_id": Q1_ID,
            "proximal_protocol_monitor": "usd_world_aabb_conservative_read_only_v1",
            "runtime_handoff_ready": 1,
            "runtime_handoff_commit": RUNTIME_HANDOFF_COMMIT,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        }
        for name, expected in exact.items():
            if name not in attrs:
                self.issue("root.missing", f"/@{name}", "required attribute is missing")
                continue
            actual = _scalar(attrs[name])
            if isinstance(expected, float):
                try:
                    matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                self.issue("root.value", f"/@{name}", f"expected {expected!r}, got {actual!r}")
        exact_json = {
            "q1_responses_json": list(Q1_RESPONSES),
            "likert_questions_json": LIKERT_QUESTIONS,
            "rejection_reasons_json": list(REJECTION_REASONS),
            "realtime_marker_types_json": list(MARKER_TYPES),
        }
        if str(_scalar(attrs.get("q1_text_ko", ""))) != Q1_TEXT_KO:
            self.issue(
                "feedback.questionnaire_contract",
                "/@q1_text_ko",
                "Q1 wording does not match the frozen Korean protocol",
            )
        for name, expected in exact_json.items():
            try:
                actual = json.loads(str(_scalar(attrs.get(name, ""))))
            except (TypeError, ValueError, json.JSONDecodeError):
                actual = None
            if actual != expected:
                self.issue(
                    "feedback.questionnaire_contract",
                    f"/@{name}",
                    "questionnaire metadata differs from the frozen protocol",
                )
        if (
            not _bool(attrs.get("practice_only", 0))
            and not _bool(attrs.get("runtime_production_collection_ready", 0))
        ):
            self.issue(
                "runtime.production_readiness",
                "/@runtime_production_collection_ready",
                "pilot/production artifacts require an explicitly qualified runtime contract",
            )
        if (
            not _bool(attrs.get("practice_only", 0))
            and not _bool(attrs.get("live_hmd_qualified", 0))
        ):
            self.issue(
                "runtime.live_hmd_qualification",
                "/@live_hmd_qualified",
                "pilot/production artifacts require explicit live-HMD equipment qualification",
            )
        for name in ("participant_id", "session_id"):
            if not str(_scalar(attrs.get(name, ""))).strip():
                self.issue("root.identity", f"/@{name}", "pseudonymous split identity is required")
        participant_id = str(_scalar(attrs.get("participant_id", "")))
        if str(_scalar(attrs.get("participant_split_key", ""))) != participant_id:
            self.issue(
                "split.participant_key",
                "/@participant_split_key",
                "participant split key must exactly equal the pseudonymous participant ID",
            )
        code_commit = str(_scalar(attrs.get("code_commit_sha", "")))
        source_hash = str(_scalar(attrs.get("source_tree_sha256", "")))
        code_version = str(_scalar(attrs.get("code_version", "")))
        code_version_source = str(
            _scalar(attrs.get("code_version_source", ""))
        )
        commit_verification = str(
            _scalar(attrs.get("code_commit_verification", ""))
        )
        practice_only = _bool(attrs.get("practice_only", 0))
        if code_commit and not _COMMIT_RE.fullmatch(code_commit):
            self.issue(
                "integrity.code_commit",
                "/@code_commit_sha",
                "non-empty commit must be a lowercase 40-hex Git SHA",
            )
        if code_version_source == "verified_git_head_clean":
            if (
                not _COMMIT_RE.fullmatch(code_commit)
                or code_version != code_commit
                or commit_verification != "verified_git_head_clean"
                or _int(attrs.get("git_worktree_clean", 0)) != 1
            ):
                self.issue(
                    "integrity.code_provenance",
                    "/@code_version",
                    "verified Git provenance fields are inconsistent",
                )
        elif code_version_source == "source_tree_sha256":
            if code_version != f"source-sha256:{source_hash}":
                self.issue(
                    "integrity.code_provenance",
                    "/@code_version",
                    "source-only version must be bound to source_tree_sha256",
                )
            if not practice_only:
                self.issue(
                    "integrity.code_provenance",
                    "/@code_version_source",
                    "source-only provenance is allowed only for practice runs",
                )
        else:
            self.issue(
                "integrity.code_provenance",
                "/@code_version_source",
                "unknown or unverified code provenance mode",
            )
        for name in (
            "source_tree_sha256",
            "cbf_config_sha256",
            "scenario_config_sha256",
            "protocol_config_sha256",
        ):
            value = str(_scalar(attrs.get(name, "")))
            if not _SHA256_RE.fullmatch(value):
                self.issue("integrity.sha256", f"/@{name}", "must be lowercase SHA-256")
        for label in ("protocol", "scenario", "cbf"):
            json_name = f"{label}_config_json"
            sha_name = f"{label}_config_sha256"
            try:
                value = json.loads(str(_scalar(attrs[json_name])))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.issue("integrity.config_json", f"/@{json_name}", "missing or invalid JSON")
                continue
            if not isinstance(value, dict):
                self.issue("integrity.config_json", f"/@{json_name}", "config must be a JSON object")
                continue
            actual_sha = canonical_json_sha256(value)
            expected_sha = str(_scalar(attrs.get(sha_name, "")))
            if actual_sha != expected_sha:
                self.issue("integrity.config_hash", f"/@{sha_name}", "hash does not match stored canonical config")
            if label == "protocol":
                self.protocol_config = value
            elif label == "scenario":
                self.scenario_config = value
            elif label == "cbf":
                self.cbf_config = value
        if self.protocol_config and self.scenario_config:
            try:
                validate_frozen_protocol_config(
                    self.protocol_config,
                    self.scenario_config,
                )
            except (KeyError, TypeError, ValueError) as error:
                self.issue(
                    "integrity.frozen_config",
                    "/",
                    str(error),
                )
        expected_cbf = {
            "safe_gap_m": 0.05,
            "activation_gap_m": 0.13,
            "gamma_per_s": 8.0,
            "prediction_horizon_s": 0.15,
            "max_prediction_buffer_m": 0.08,
            "max_joint_speed_rad_s": 2.0,
        }
        for name, expected in expected_cbf.items():
            try:
                actual = float(self.cbf_config[name])
            except (KeyError, TypeError, ValueError):
                actual = math.nan
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                self.issue(
                    "integrity.cbf_config",
                    f"/@cbf_config_json/{name}",
                    f"expected {expected!r}, got {actual!r}",
                )
        try:
            protected = json.loads(str(_scalar(attrs["cbf_protected_links_json"])))
            colliders = json.loads(str(_scalar(attrs["cbf_protected_colliders_json"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            protected, colliders = [], []
            self.issue("geometry.metadata", "/@cbf_protected_links_json", "runtime-discovered protected geometry is required")
        if not isinstance(protected, list) or not protected or not isinstance(colliders, list) or not colliders:
            self.issue("geometry.empty", "/", "protected links and collider paths must be non-empty")
        requested = set(
            self.scenario_config.get("protected_crossing", {}).get("requested_distal_links", [])
        )
        if requested and any(item not in requested for item in protected):
            self.issue("geometry.scope", "/@cbf_protected_links_json", "runtime CBF scope contains a non-distal/unrequested link")
        try:
            proximal_paths = json.loads(
                str(_scalar(attrs["proximal_collider_paths_json"]))
            )
            proximal_tokens = json.loads(
                str(_scalar(attrs["proximal_link_tokens_json"]))
            )
            proximal_missing = json.loads(
                str(_scalar(attrs["proximal_missing_tokens_json"]))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            proximal_paths, proximal_tokens, proximal_missing = [], [], ["invalid"]
            self.issue(
                "geometry.proximal_metadata",
                "/@proximal_collider_paths_json",
                "runtime-discovered proximal off-protocol geometry is required",
            )
        expected_proximal = list(
            self.scenario_config.get("protected_crossing", {}).get(
                "proximal_link_tokens", []
            )
        )
        if (
            not isinstance(proximal_paths, list)
            or not proximal_paths
            or proximal_tokens != expected_proximal
            or proximal_missing != []
        ):
            self.issue(
                "geometry.proximal_incomplete",
                "/@proximal_collider_paths_json",
                "every configured proximal link requires discovered off-protocol geometry",
            )
        elif any(
            not any(f"/{token}" in str(path) for path in proximal_paths)
            for token in expected_proximal
        ):
            self.issue(
                "geometry.proximal_scope",
                "/@proximal_collider_paths_json",
                "a configured proximal link has no discovered collider",
            )
        if isinstance(proximal_paths, list) and isinstance(colliders, list) and set(
            proximal_paths
        ).intersection(colliders):
            self.issue(
                "geometry.scope_overlap",
                "/",
                "distal CBF and proximal off-protocol collider scopes overlap",
            )
        root_start_mono = _int(attrs.get("session_start_monotonic_ns", 0))
        root_end_mono = _int(attrs.get("session_end_monotonic_ns", 0))
        root_start_unix = _int(attrs.get("session_start_unix_ns", 0))
        root_end_unix = _int(attrs.get("session_end_unix_ns", 0))
        if not (0 < root_start_mono <= root_end_mono and 0 < root_start_unix <= root_end_unix):
            self.issue("clock.session", "/", "session clocks are absent or non-monotonic")

    def _validate_tables(self) -> None:
        group_names = (
            *tuple(_TABLE_COLUMNS),
            "trials",
            "bc_infeasible_diagnostic",
        )
        for name in group_names:
            node = self.file.get(name)
            ready = node is not None and hasattr(node, "keys")
            self.group_ready[name] = ready
            if node is None:
                self.issue("schema.group", f"/{name}", "required group is missing")
            elif not ready:
                self.issue(
                    "schema.group_type",
                    f"/{name}",
                    "required group was replaced by a dataset",
                )

        for name, required_columns in _TABLE_COLUMNS.items():
            group = self.file.get(name)
            if not self.group_ready.get(name, False):
                self.table_ready[name] = False
                self.table_lengths[name] = 0
                continue
            ready = True
            lengths: set[int] = set()
            present_columns = set(group.keys())
            for column in required_columns:
                dataset = group.get(column)
                column_path = f"/{name}/{column}"
                if dataset is None:
                    self.issue(
                        "schema.dataset",
                        column_path,
                        "required table column is missing",
                    )
                    ready = False
                    continue
                shape = getattr(dataset, "shape", None)
                if shape is None:
                    self.issue(
                        "schema.dataset_type",
                        column_path,
                        "table column must be a dataset",
                    )
                    ready = False
                    continue
                if len(shape) != 1:
                    self.issue(
                        "schema.shape",
                        column_path,
                        f"table column must be one-dimensional, got {shape}",
                    )
                    ready = False
                    continue
                lengths.add(int(shape[0]))
            extra_columns = sorted(present_columns - set(required_columns))
            if extra_columns:
                self.issue(
                    "schema.extra_dataset",
                    f"/{name}",
                    f"unexpected table columns: {extra_columns}",
                )
                ready = False
            if len(lengths) != 1:
                self.issue(
                    "schema.table_alignment",
                    f"/{name}",
                    "columns have different lengths or no valid columns",
                )
                ready = False
                length = 0
            else:
                length = int(next(iter(lengths)))
            self.table_ready[name] = ready
            self.table_lengths[name] = length

        for name, counter in (
            ("queries", "query_count"),
            ("encounters", "encounter_count"),
            ("realtime_markers", "realtime_marker_count"),
        ):
            length = self.table_lengths.get(name, 0)
            self.counts[counter] = length
            if _int(self.file.attrs.get(counter, -1)) != length:
                self.issue(
                    "count.root",
                    f"/@{counter}",
                    "root count does not match table",
                )

    def _validate_schedule_and_trials(self) -> None:
        if not self.group_ready.get("trials", False):
            return
        trial_names = sorted(self.file["trials"].keys())
        self.counts["trial_count"] = len(trial_names)
        if _int(self.file.attrs.get("trial_count", -1)) != len(trial_names):
            self.issue("count.root", "/@trial_count", "root count does not match trial groups")
        schedule_ready = self.table_ready.get("schedule", False)
        schedule = self.file.get("schedule")
        scheduled = self.table_lengths.get("schedule", 0)
        if scheduled != len(trial_names):
            self.issue("schedule.count", "/schedule", "sealed session must contain every scheduled trial exactly once")
        queries_ready = self.table_ready.get("queries", False)
        query_ids = (
            _strings(self.file["queries/query_id"]) if queries_ready else []
        )
        query_trial_ids = (
            _strings(self.file["queries/trial_id"]) if queries_ready else []
        )
        query_indices_by_trial: dict[str, list[int]] = {}
        for query_index, query_trial_id in enumerate(query_trial_ids):
            query_indices_by_trial.setdefault(query_trial_id, []).append(
                query_index
            )
        encounters_by_trial: dict[str, list[str]] = {}
        if self.table_ready.get("encounters", False):
            for trial_id, encounter_id in zip(
                _strings(self.file.get("encounters/trial_id")),
                _strings(self.file.get("encounters/encounter_id")),
            ):
                encounters_by_trial.setdefault(trial_id, []).append(encounter_id)
        for index, trial_name in enumerate(trial_names):
            expected_name = f"trial_{index:06d}"
            if trial_name != expected_name:
                self.issue("trial.sequence", f"/trials/{trial_name}", f"expected {expected_name}")
            group = self.file["trials"][trial_name]
            path = f"/trials/{trial_name}"
            if not hasattr(group, "keys"):
                self.issue(
                    "schema.trial_type",
                    path,
                    "trial entry must be an HDF5 group",
                )
                continue
            attrs = group.attrs
            trial_id = str(_scalar(attrs.get("trial_id", "")))
            expected_values = {
                "trial_index": index,
                "single_pick": 1,
                "cube_count": 1,
                "intended_crossing_count": 1,
                "trial_complete": 1,
                "layout_bc_feasible": 1,
            }
            for name, expected in expected_values.items():
                if _scalar(attrs.get(name, None)) != expected:
                    self.issue("trial.integrity", f"{path}/@{name}", f"expected {expected}")
            phase = str(_scalar(attrs.get("task_phase", "")))
            severity = str(_scalar(attrs.get("severity", "")))
            direction = str(_scalar(attrs.get("direction", "")))
            speed = str(_scalar(attrs.get("speed", "")))
            crossing_hand = str(_scalar(attrs.get("crossing_hand", "")))
            if phase not in TASK_PHASES or severity not in SEVERITIES or direction not in DIRECTIONS or speed not in SPEEDS:
                self.issue("trial.factor", path, "unknown phase/severity/direction/speed")
            expected_hand = "left" if direction == "left_to_right" else "right"
            if crossing_hand != expected_hand:
                self.issue("trial.crossing_hand", f"{path}/@crossing_hand", "crossing hand must match direction")
            if schedule_ready and index < scheduled:
                for name in (
                    "trial_id",
                    "query_id",
                    "task_phase",
                    "severity",
                    "direction",
                    "speed",
                    "crossing_hand",
                    "practice",
                    "anchor_repeat",
                    "pilot",
                    "source_trial_id",
                ):
                    planned = _dataset_scalar(schedule[name], index)
                    actual = _scalar(attrs.get(name, ""))
                    if planned != actual:
                        self.issue("schedule.mismatch", f"{path}/@{name}", "trial does not match frozen schedule")
            practice = _bool(attrs.get("practice", 0))
            analysis_exclude = _bool(attrs.get("analysis_exclude", 0))
            protocol_valid = _bool(attrs.get("protocol_valid", 0))
            off_protocol = _bool(attrs.get("off_protocol", 0))
            if schedule_ready and index < scheduled:
                expected_exclude = bool(
                    _bool(schedule["analysis_exclude"][index])
                    or not protocol_valid
                )
                if analysis_exclude != expected_exclude:
                    self.issue(
                        "schedule.mismatch",
                        f"{path}/@analysis_exclude",
                        "derived analysis exclusion does not match schedule/protocol validity",
                    )
            if practice and not analysis_exclude:
                self.issue("split.practice", f"{path}/@analysis_exclude", "practice trial must be excluded")
            if off_protocol and not analysis_exclude:
                self.issue("tracking.off_protocol", f"{path}/@analysis_exclude", "off-protocol trial must be excluded")
            if protocol_valid and (off_protocol or _int(attrs.get("actual_crossing_count", 0)) != 1):
                self.issue("trial.validity", path, "valid trial must have exactly one compliant crossing")
            completion_observed = _bool(
                attrs.get("task_completion_observed", 0)
            )
            completion_time = _float(attrs.get("completion_time_s", math.nan))
            if str(
                _scalar(attrs.get("completion_time_semantics", ""))
            ) != "active_task_simulation_time_excluding_query_hold_v1":
                self.issue(
                    "task.completion_semantics",
                    f"{path}/@completion_time_semantics",
                    "task completion timing semantics are missing or unknown",
                )
            if completion_observed != (completion_time >= 0.0):
                self.issue(
                    "task.completion_semantics",
                    f"{path}/@completion_time_s",
                    "completion_time_s must be non-negative only when task completion was observed, otherwise -1",
                )
            if not completion_observed and not math.isclose(
                completion_time, -1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                self.issue(
                    "task.completion_semantics",
                    f"{path}/@completion_time_s",
                    "unobserved task completion must use the explicit -1 sentinel",
                )
            for duration_name in (
                "trial_simulation_duration_s",
                "trial_wall_duration_s",
            ):
                duration = _float(attrs.get(duration_name, math.nan))
                if not math.isfinite(duration) or duration < 0.0:
                    self.issue(
                        "trial.duration",
                        f"{path}/@{duration_name}",
                        "trial durations must be finite and non-negative",
                    )
            own_query_indices = query_indices_by_trial.get(trial_id, [])
            own_queries = [query_ids[item] for item in own_query_indices]
            if len(own_queries) != 1:
                self.issue("feedback.query_count", path, "every trial requires exactly one independent query")
            expected_query_id = str(_scalar(attrs.get("query_id", "")))
            if own_queries and own_queries[0] != expected_query_id:
                self.issue("feedback.query_id", f"{path}/@query_id", "query relation is inconsistent")
            if _int(attrs.get("query_present", 0)) != 1:
                self.issue(
                    "feedback.query_present",
                    f"{path}/@query_present",
                    "a mandatory query must be recorded",
                )
            if queries_ready and len(own_query_indices) == 1:
                query_index = own_query_indices[0]
                query_table = self.file["queries"]
                scalar_pairs = {
                    "query_response_status": "response_status",
                    "encounter_id": "encounter_id",
                    "q1_primary_response": "q1_response",
                    "q2_collision_risk": "q2_collision_risk",
                    "q3_excessive_avoidance": "q3_excessive_avoidance",
                    "q4_abrupt_unpredictable": "q4_abrupt_unpredictable",
                    "q5_task_disruption": "q5_task_disruption",
                    "q6_confidence": "q6_confidence",
                }
                for attr_name, column_name in scalar_pairs.items():
                    if _scalar(attrs.get(attr_name, None)) != _dataset_scalar(
                        query_table[column_name], query_index
                    ):
                        self.issue(
                            "feedback.trial_query_mismatch",
                            f"{path}/@{attr_name}",
                            f"does not match /queries/{column_name}[{query_index}]",
                        )
                try:
                    trial_reasons = json.loads(
                        str(_scalar(attrs.get("rejection_reasons_json", "")))
                    )
                    query_reasons = json.loads(
                        str(
                            _dataset_scalar(
                                query_table["rejection_reasons_json"],
                                query_index,
                            )
                        )
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    trial_reasons, query_reasons = None, None
                if trial_reasons != query_reasons:
                    self.issue(
                        "feedback.trial_query_mismatch",
                        f"{path}/@rejection_reasons_json",
                        "does not match the query table",
                    )
                query_status = str(
                    _dataset_scalar(
                        query_table["response_status"], query_index
                    )
                )
                if query_status != "completed" and (
                    protocol_valid or not analysis_exclude
                ):
                    self.issue(
                        "feedback.incomplete_trial_validity",
                        path,
                        "an incomplete mandatory query must invalidate and exclude the trial",
                    )
            own_encounters = encounters_by_trial.get(trial_id, [])
            if len(own_encounters) > 1:
                self.issue("encounter.count", path, "one controlled crossing may produce at most one merged encounter")
            recorded_encounter = str(_scalar(attrs.get("encounter_id", "")))
            if own_encounters and recorded_encounter != own_encounters[0]:
                self.issue("encounter.relation", f"{path}/@encounter_id", "trial encounter relation is inconsistent")
            if not own_encounters and recorded_encounter:
                self.issue("encounter.nullability", f"{path}/@encounter_id", "unknown encounter reference")
            if not own_encounters and any(
                _int(attrs.get(name, -2)) != -1
                for name in (
                    "risk_onset_step",
                    "cbf_intervention_start_step",
                    "cbf_intervention_confirmed_step",
                )
            ):
                self.issue(
                    "encounter.null_timestamps",
                    path,
                    "trial without an encounter must use -1 steering-step sentinels",
                )
            self._validate_trial_rows(
                group,
                path,
                protocol_valid=protocol_valid,
                planned_phase=phase,
            )
            self._validate_trial_summary(group, path)
            self._validate_precheck_reference(attrs, path)
            threshold = (
                self.scenario_config.get("crossing", {}).get("maximum_path_deviation_m")
            )
            if threshold is not None:
                deviation = _float(attrs.get("actual_path_deviation_max_m", math.inf))
                if deviation > float(threshold) and not off_protocol:
                    self.issue("tracking.path_deviation", f"{path}/@off_protocol", "excessive path deviation was not marked off-protocol")
            self._validate_planned_severity(
                attrs,
                path,
                severity=severity,
            )

    def _validate_planned_severity(
        self,
        attrs: Any,
        path: str,
        *,
        severity: str,
    ) -> None:
        severity_config = self.scenario_config.get("severity", {}).get(
            severity, {}
        )
        configured_range = severity_config.get("minimum_gap_range_m")
        configured_target = severity_config.get("target_gap_m")
        if (
            not isinstance(configured_range, list)
            or len(configured_range) != 2
            or configured_target is None
        ):
            self.issue(
                "severity.config",
                path,
                "severity range and target are missing from frozen scenario config",
            )
            return
        low, high = (float(configured_range[0]), float(configured_range[1]))
        target = float(configured_target)
        recorded_low = _float(attrs.get("planned_gap_min_m", math.nan))
        recorded_high = _float(attrs.get("planned_gap_max_m", math.nan))
        recorded_target = _float(attrs.get("planned_target_gap_m", math.nan))
        if not (
            math.isclose(recorded_low, low, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(recorded_high, high, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(recorded_target, target, rel_tol=0.0, abs_tol=1e-12)
        ):
            self.issue(
                "severity.planned_attributes",
                path,
                "trial severity attributes do not match the frozen scenario",
            )
        try:
            corridor = json.loads(
                str(_scalar(attrs.get("planned_corridor_json", "")))
            )
            planned_gap = float(corridor["planned_minimum_surface_gap_m"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.issue(
                "severity.corridor",
                f"{path}/@planned_corridor_json",
                "missing or invalid protected-surface corridor calibration",
            )
            return
        if not math.isfinite(planned_gap) or not (
            recorded_low <= planned_gap <= recorded_high
        ):
            self.issue(
                "severity.realization",
                f"{path}/@planned_corridor_json",
                "calibrated planned gap is outside the assigned severity range",
            )
        if corridor.get("geometry_query_semantics") != (
            "physx_protected_surface_to_hand_sphere"
        ):
            self.issue(
                "severity.geometry_semantics",
                f"{path}/@planned_corridor_json",
                "severity must be calibrated from the actual protected surface",
            )
        try:
            protected_colliders = set(
                json.loads(
                    str(
                        _scalar(
                            self.file.attrs.get(
                                "cbf_protected_colliders_json", "[]"
                            )
                        )
                    )
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            protected_colliders = set()
        closest_collider = str(corridor.get("closest_collider_path", ""))
        if not closest_collider or closest_collider not in protected_colliders:
            self.issue(
                "severity.collider_scope",
                f"{path}/@planned_corridor_json",
                "corridor calibration collider is outside the frozen CBF scope",
            )
        reference = corridor.get("nominal_bc_reference")
        reference_config = self.scenario_config.get(
            "nominal_bc_reference", {}
        )
        expected_event = (
            self.protocol_config.get("task", {})
            .get("crossing_trigger_event", {})
            .get(str(_scalar(attrs.get("task_phase", ""))))
        )
        if not isinstance(reference, Mapping):
            self.issue(
                "severity.nominal_reference",
                f"{path}/@planned_corridor_json",
                "corridor lacks its BC-only nominal trajectory reference audit",
            )
            return
        reference_exact = bool(
            reference.get("semantics")
            == reference_config.get("trigger_state_semantics")
            and reference.get("phase")
            == str(_scalar(attrs.get("task_phase", "")))
            and reference.get("controller") == "bc_only"
            and expected_event is not None
            and _int(reference.get("controller_event", -1))
            == int(expected_event)
            and _bool(reference.get("match", False))
        )
        ee_error = _float(reference.get("ee_position_error_m", math.inf))
        joint_error = _float(
            reference.get("maximum_arm_joint_error_rad", math.inf)
        )
        if (
            not reference_exact
            or not math.isfinite(ee_error)
            or ee_error
            > float(reference_config.get("maximum_ee_position_error_m", -1.0))
            or not math.isfinite(joint_error)
            or joint_error
            > float(
                reference_config.get(
                    "maximum_arm_joint_error_rad", -1.0
                )
            )
        ):
            self.issue(
                "severity.nominal_reference",
                f"{path}/@planned_corridor_json",
                "trial corridor is not anchored to the accepted BC-only phase trajectory state",
            )
        self._validate_trial_nominal_calibration(attrs, path, corridor)

    def _validate_trial_nominal_calibration(
        self,
        attrs: Any,
        path: str,
        corridor: Mapping[str, Any],
    ) -> None:
        """Bind a live trial to one immutable cell of its accepted BC replay."""

        audit_path = f"{path}/@planned_corridor_json"
        phase = str(_scalar(attrs.get("task_phase", "")))
        severity = str(_scalar(attrs.get("severity", "")))
        direction = str(_scalar(attrs.get("direction", "")))
        speed = str(_scalar(attrs.get("speed", "")))
        crossing_hand = str(_scalar(attrs.get("crossing_hand", "")))
        selected = corridor.get("nominal_bc_sweep")
        if not isinstance(selected, Mapping):
            self.issue(
                "severity.nominal_sweep",
                audit_path,
                "trial lacks its selected time-aligned BC-only sweep audit",
            )
            return
        index = _int(attrs.get("layout_precheck_index", -1))
        table = self.file.get("layout_prechecks")
        expected: Any = None
        try:
            bank = json.loads(
                str(_dataset_scalar(table["nominal_corridor_bank_json"], index))
            )
            expected = bank[phase][direction][speed][severity]
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            expected = None
        if not isinstance(expected, Mapping) or dict(selected) != dict(expected):
            self.issue(
                "severity.nominal_sweep_relation",
                audit_path,
                "trial sweep is not the exact scheduled cell of its layout bank",
            )
        sweep_body = {
            key: value
            for key, value in selected.items()
            if key != "sweep_audit_sha256"
        }
        if str(selected.get("sweep_audit_sha256", "")) != canonical_json_sha256(
            {"nominal_bc_sweep": sweep_body}
        ):
            self.issue(
                "severity.nominal_sweep_hash",
                audit_path,
                "trial nominal sweep audit hash does not match its content",
            )
        if not (
            selected.get("semantics") == NOMINAL_BC_SWEEP_SEMANTICS
            and selected.get("phase") == phase
            and selected.get("severity") == severity
            and selected.get("crossing_direction") == direction
            and selected.get("hand") == crossing_hand
        ):
            self.issue(
                "severity.nominal_sweep_condition",
                audit_path,
                "trial sweep condition labels do not match the scheduled trial",
            )
        selected_corridor = selected.get("corridor")
        corridor_fields = (
            "start_world_m",
            "end_world_m",
            "center_world_m",
            "tangent_world",
            "outward_normal_world",
            "planned_minimum_surface_gap_m",
            "closest_link",
            "closest_collider_path",
            "calibration_iterations",
            "geometry_query_semantics",
        )
        if not isinstance(selected_corridor, Mapping) or any(
            corridor.get(name) != selected_corridor.get(name)
            for name in corridor_fields
        ):
            self.issue(
                "severity.nominal_sweep_corridor",
                audit_path,
                "executed corridor differs from the selected nominal sweep corridor",
            )

        reference_config = self.scenario_config.get("nominal_bc_reference", {})
        start = corridor.get("start_recheck")
        if not isinstance(start, Mapping):
            self.issue(
                "severity.start_recheck",
                audit_path,
                "START-time actual-versus-nominal geometry audit is missing",
            )
        else:
            try:
                expected_start_gap = float(
                    selected["crossing_start_surface_gap_m"]
                )
                coverage = selected["coverage"]
                expected_relative_time = float(
                    selected["crossing_start_query_simulation_time_s"]
                ) - float(coverage["phase_start_simulation_time_s"])
                calculated_match = bool(
                    start.get("semantics")
                    == "actual_start_vs_time_aligned_nominal_sweep_v1"
                    and _bool(start.get("geometry_valid", False))
                    and math.isclose(
                        _float(start.get("expected_start_surface_gap_m", math.nan)),
                        expected_start_gap,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        _float(start.get("expected_query_relative_time_s", math.nan)),
                        expected_relative_time,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and _float(start.get("scheduled_time_error_s", math.inf))
                    <= float(reference_config.get("maximum_sweep_sample_interval_s", -1.0))
                    and _float(start.get("nearest_state_time_error_s", math.inf))
                    <= float(reference_config.get("maximum_sweep_sample_interval_s", -1.0))
                    and _float(start.get("start_surface_gap_error_m", math.inf))
                    <= float(reference_config.get("maximum_start_gap_error_m", -1.0))
                    and _float(start.get("ee_position_error_m", math.inf))
                    <= float(reference_config.get("maximum_ee_position_error_m", -1.0))
                    and _float(start.get("maximum_arm_joint_error_rad", math.inf))
                    <= float(reference_config.get("maximum_arm_joint_error_rad", -1.0))
                )
            except (KeyError, TypeError, ValueError):
                calculated_match = False
            if _bool(start.get("match", False)) != calculated_match:
                self.issue(
                    "severity.start_recheck_consistency",
                    audit_path,
                    "stored START recheck result disagrees with its measurements",
                )
            if _bool(attrs.get("protocol_valid", 0)) and not calculated_match:
                self.issue(
                    "severity.start_recheck",
                    audit_path,
                    "protocol-valid trial failed the frozen START recheck bounds",
                )

        timeline = corridor.get("pre_cbf_timeline_audit")
        if not isinstance(timeline, Mapping):
            self.issue(
                "severity.nominal_timeline",
                audit_path,
                "pre-CBF actual-versus-nominal trajectory audit is missing",
            )
        else:
            calculated_match = bool(
                timeline.get("semantics")
                == "pre_cbf_actual_vs_bc_only_cue_latched_then_task_advancing_trajectory_v1"
                and _int(timeline.get("compared_sample_count", 0)) >= 1
                and 0.0
                <= _float(timeline.get("maximum_time_alignment_error_s", math.inf))
                <= float(reference_config.get("maximum_sweep_sample_interval_s", -1.0))
                and 0.0
                <= _float(timeline.get("maximum_ee_position_error_m", math.inf))
                <= float(reference_config.get("maximum_ee_position_error_m", -1.0))
                and 0.0
                <= _float(timeline.get("maximum_arm_joint_error_rad", math.inf))
                <= float(reference_config.get("maximum_arm_joint_error_rad", -1.0))
                and _int(timeline.get("comparison_closed_at_cbf_step", -2)) >= -1
            )
            if _bool(timeline.get("match", False)) != calculated_match:
                self.issue(
                    "severity.nominal_timeline_consistency",
                    audit_path,
                    "stored pre-CBF trajectory result disagrees with its measurements",
                )
            if _bool(attrs.get("protocol_valid", 0)) and not calculated_match:
                self.issue(
                    "severity.nominal_timeline",
                    audit_path,
                    "protocol-valid trial lacks an in-bounds pre-CBF trajectory comparison",
                )

    def _validate_trial_rows(
        self,
        group: Any,
        path: str,
        *,
        protocol_valid: bool,
        planned_phase: str,
    ) -> None:
        if "actions/all_joint_names" not in group or "actions/arm_joint_indices" not in group:
            self.issue("action.names", f"{path}/actions", "joint provenance is missing")
            return
        joint_names = _strings(group["actions/all_joint_names"])
        arm_indices = np.asarray(group["actions/arm_joint_indices"][()], dtype=int).reshape(-1)
        try:
            specs = online_row_specs(len(joint_names), len(arm_indices))
        except Exception as error:
            self.issue("action.spec", path, str(error))
            return
        lengths: set[int] = set()
        structural_error = False
        for row_path, spec in specs.items():
            if row_path not in group:
                self.issue("schema.dataset", f"{path}/{row_path}", "required per-step dataset is missing")
                structural_error = True
                continue
            dataset = group[row_path]
            shape = getattr(dataset, "shape", None)
            if shape is None:
                self.issue(
                    "schema.dataset_type",
                    f"{path}/{row_path}",
                    "per-step field must be an HDF5 dataset",
                )
                structural_error = True
                continue
            if len(shape) < 1:
                self.issue(
                    "schema.shape",
                    f"{path}/{row_path}",
                    "per-step dataset must have a row dimension",
                )
                structural_error = True
                continue
            lengths.add(int(shape[0]))
            if shape[1:] != spec.tail_shape:
                self.issue("schema.shape", f"{path}/{row_path}", f"expected tail shape {spec.tail_shape}, got {shape[1:]}")
                structural_error = True
            dtype = getattr(dataset, "dtype", None)
            if dtype is None:
                self.issue(
                    "schema.dataset_type",
                    f"{path}/{row_path}",
                    "per-step field has no dataset dtype",
                )
                structural_error = True
            elif dtype.kind not in "SUO":
                try:
                    finite = np.all(np.isfinite(np.asarray(dataset[()], dtype=float)))
                except (OSError, TypeError, ValueError):
                    finite = False
                if not finite:
                    self.issue("schema.nonfinite", f"{path}/{row_path}", "numeric data must be finite")
        if len(lengths) != 1:
            self.issue("schema.row_alignment", path, "per-step datasets have different lengths")
            return
        length = next(iter(lengths), 0)
        if _int(group.attrs.get("step_count", -1)) != length or length <= 0:
            self.issue("trial.step_count", f"{path}/@step_count", "step count is empty or inconsistent")
        if length <= 0:
            return
        if structural_error:
            return
        monotonic_paths = (
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
        if all(name in group for name in monotonic_paths):
            clocks = np.stack([np.asarray(group[name][()], dtype=np.int64) for name in monotonic_paths], axis=1)
            if (
                np.any(clocks <= 0)
                or np.any(np.diff(clocks, axis=1) < 0)
                or np.any(np.diff(clocks[:, 0]) <= 0)
                or (
                    length > 1
                    and np.any(clocks[:-1, -1] > clocks[1:, 1])
                )
            ):
                self.issue("clock.step", f"{path}/control", "monotonic clocks are out of order")
        source = np.asarray(group.get("control/source_step", []), dtype=np.int64)
        result = np.asarray(group.get("control/result_step", []), dtype=np.int64)
        if source.size and (np.any(np.diff(source) != 1) or np.any(result != source + 1)):
            self.issue("clock.control_step", f"{path}/control", "control steps are not contiguous transitions")
        simulation_time = np.asarray(
            group["control/sim_time"][()], dtype=float
        )
        physics_dt_s = _float(self.file.attrs.get("physics_dt_s", math.nan))
        if (
            np.any(~np.isfinite(simulation_time))
            or np.any(simulation_time < 0.0)
            or (
                length > 1
                and (
                    not math.isfinite(physics_dt_s)
                    or physics_dt_s <= 0.0
                    or not np.allclose(
                        np.diff(simulation_time),
                        physics_dt_s,
                        rtol=0.0,
                        atol=max(1e-9, physics_dt_s * 1e-6),
                    )
                )
            )
        ):
            self.issue(
                "clock.simulation_step",
                f"{path}/control/sim_time",
                "simulation clock must advance by the frozen physics step",
            )
        wall_unix = np.asarray(
            group["control/wall_time_unix_ns"][()], dtype=np.int64
        )
        unix_time = np.asarray(
            group["control/unix_time_ns"][()], dtype=np.int64
        )
        if (
            np.any(wall_unix <= 0)
            or np.any(unix_time <= 0)
            or not np.array_equal(wall_unix, unix_time)
            or (length > 1 and np.any(np.diff(unix_time) <= 0))
        ):
            self.issue(
                "clock.unix_step",
                f"{path}/control/unix_time_ns",
                "Unix clocks must be positive, strictly increasing, and share one captured value",
            )
        if length > 1 and all(name in group for name in monotonic_paths):
            monotonic_delta = np.diff(clocks[:, -1]).astype(np.float64)
            unix_delta = np.diff(unix_time).astype(np.float64)
            if np.any(np.abs(monotonic_delta - unix_delta) > 50_000_000.0):
                self.issue(
                    "clock.cross_clock_step",
                    f"{path}/control",
                    "Unix and monotonic row-to-row elapsed times disagree by more than 50 ms",
                )
        trial_start_sim = _float(
            group.attrs.get("start_simulation_time_s", math.nan)
        )
        trial_end_sim = _float(
            group.attrs.get("end_simulation_time_s", math.nan)
        )
        trial_start_mono = _int(group.attrs.get("start_monotonic_ns", 0))
        trial_end_mono = _int(group.attrs.get("end_monotonic_ns", 0))
        trial_start_unix = _int(group.attrs.get("start_unix_ns", 0))
        trial_end_unix = _int(group.attrs.get("end_unix_ns", 0))
        clocks_in_bounds = bool(
            math.isfinite(trial_start_sim)
            and math.isfinite(trial_end_sim)
            and trial_start_sim <= simulation_time[0] <= simulation_time[-1] <= trial_end_sim
            and trial_start_mono <= int(clocks[0, 0])
            and int(clocks[-1, -1]) <= trial_end_mono
            and trial_start_unix <= int(unix_time[0])
            and int(unix_time[-1]) <= trial_end_unix
        )
        if not clocks_in_bounds:
            self.issue(
                "clock.trial_bounds",
                f"{path}/control",
                "step simulation/monotonic/Unix clocks must lie inside trial bounds",
            )
        row_trial_ids = _strings(group["control/trial_id"])
        row_query_ids = _strings(group["control/query_id"])
        if any(
            value != str(_scalar(group.attrs.get("trial_id", "")))
            for value in row_trial_ids
        ) or any(
            value != str(_scalar(group.attrs.get("query_id", "")))
            for value in row_query_ids
        ):
            self.issue(
                "trial.row_identity",
                f"{path}/control",
                "every step must carry the enclosing trial/query identity",
            )
        raw_observations = np.asarray(
            group["observations/obs_raw"][()], dtype=np.float32
        )
        policy_observations = np.asarray(
            group["observations/obs_policy"][()], dtype=np.float32
        )
        for row_index in range(length):
            expected_policy = mask_human_observations(
                raw_observations[row_index]
            )
            if not np.array_equal(
                policy_observations[row_index], expected_policy
            ):
                self.issue(
                    "policy.human_mask_mismatch",
                    f"{path}/observations/obs_policy",
                    "policy input is not the exact frozen human mask at "
                    f"row {row_index}",
                )
                break
        for stage in ("nominal_rmpflow", "cbf_filtered", "submitted", "applied"):
            values_name = f"actions/{stage}_joint_velocities_rad_s"
            mask_name = values_name + "_mask"
            if values_name not in group or mask_name not in group:
                continue
            values = np.asarray(group[values_name][()], dtype=float)
            masks = np.asarray(group[mask_name][()], dtype=int)
            if values.shape != masks.shape or values.shape[1:] != (len(joint_names),):
                self.issue("action.dimension", f"{path}/{values_name}", "command and mask dimensions must match full joint vector")
            if np.any(~np.isin(masks, (0, 1))):
                self.issue("action.mask", f"{path}/{mask_name}", "mask must be binary")
        self._validate_action_and_cbf_rows(
            group,
            path,
            length=length,
            joint_count=len(joint_names),
            arm_indices=arm_indices,
        )
        if protocol_valid:
            valid_mask = np.asarray(group["human/valid_mask"][()], dtype=int)
            if np.any(valid_mask != 1):
                self.issue("tracking.validity", f"{path}/human/valid_mask", "study-valid trial contains invalid tracked poses")
        phases = set(_strings(group["task/experimental_phase"]))
        if not phases.issubset(set(TASK_PHASES)):
            self.issue("task.phase", f"{path}/task/experimental_phase", "unknown experimental phase")
        control_modes = _strings(group["control/control_mode"])
        trial_states = _strings(group["control/trial_state"])
        for row_index, (control_mode, trial_state) in enumerate(
            zip(control_modes, trial_states)
        ):
            row_path = f"{path}/control[{row_index}]"
            if control_mode not in _CONTROL_MODES:
                self.issue(
                    "control.mode",
                    row_path,
                    f"unknown control mode: {control_mode!r}",
                )
                continue
            if trial_state not in _TRIAL_STATES:
                self.issue(
                    "control.trial_state",
                    row_path,
                    f"unknown trial state: {trial_state!r}",
                )
                continue
            if (control_mode == "protocol_hold") != (
                trial_state == "query_hold"
            ):
                self.issue(
                    "control.mode_state",
                    row_path,
                    "protocol_hold is allowed only and required during query_hold; "
                    "all task, crossing, response, recovery, and resume states require nominal_bc",
                )
        expected_trigger = (
            self.protocol_config.get("task", {})
            .get("crossing_trigger_event", {})
            .get(planned_phase)
        )
        if expected_trigger is None:
            self.issue(
                "task.phase_latch",
                path,
                "planned phase has no frozen crossing trigger event",
            )
        else:
            event_before = np.asarray(
                group["task/controller_event_before"][()], dtype=int
            )
            event_after = np.asarray(
                group["task/controller_event_after"][()], dtype=int
            )
            cue_latched = np.asarray(trial_states, dtype=object) == "pre_crossing"
            drift = cue_latched & (
                (event_before != int(expected_trigger))
                | (event_after != int(expected_trigger))
            )
            if np.any(drift):
                self.issue(
                    "task.phase_latch",
                    f"{path}/task/controller_event_before",
                    "pre-crossing cue left its assigned frozen phase event",
                )
            query_hold = np.asarray(trial_states, dtype=object) == "query_hold"
            if np.any(query_hold & (event_before != event_after)) or (
                np.sum(query_hold) > 1
                and len(set(event_before[query_hold].tolist())) != 1
            ):
                self.issue(
                    "task.query_pause",
                    f"{path}/task/controller_event_before",
                    "post-recovery questionnaire must pause nominal task progression",
                )
        advance_enabled = np.asarray(
            group["task/phase_advance_enabled"][()], dtype=int
        )
        expected_advance = np.isin(
            np.asarray(trial_states, dtype=object),
            ("wait_phase", "crossing", "response", "recovery", "resume_task"),
        ).astype(int)
        if (
            np.any(~np.isin(advance_enabled, (0, 1)))
            or not np.array_equal(advance_enabled, expected_advance)
        ):
            self.issue(
                "task.phase_progression_contract",
                f"{path}/task/phase_advance_enabled",
                "cue/query must pause and crossing/response/recovery must advance the nominal task",
            )
        relevant_states = np.isin(
            np.asarray(trial_states, dtype=object),
            ("pre_crossing", "crossing", "response", "recovery"),
        )
        for hand in ("left", "right"):
            gaps = np.asarray(
                group[f"safety/{hand}_proximal_surface_gap_m"][()],
                dtype=float,
            )
            paths = _strings(group[f"safety/{hand}_proximal_collider_path"])
            unsafe = relevant_states & (gaps <= 0.13)
            if any(flag and not paths[index] for index, flag in enumerate(unsafe)):
                self.issue(
                    "geometry.proximal_identity",
                    f"{path}/safety/{hand}_proximal_collider_path",
                    "unsafe proximal rows require exact collider provenance",
                )
            if protocol_valid and np.any(unsafe):
                self.issue(
                    "geometry.proximal_protocol",
                    f"{path}/safety/{hand}_proximal_surface_gap_m",
                    "study-valid trial entered unprotected proximal geometry",
                )
        active = np.asarray(group["cbf/active"][()], dtype=int).astype(bool)
        identity = np.asarray(
            group["cbf/active_constraint_identity_available"][()], dtype=int
        ).astype(bool)
        if np.any(active & ~identity):
            self.issue(
                "cbf.constraint_identity",
                f"{path}/cbf/active_constraint_identity_available",
                "every active CBF row requires exact hand/link/collider reconstruction",
            )

    def _validate_trial_summary(self, group: Any, path: str) -> None:
        """Recompute trial-level crossing, safety, and CBF aggregates."""

        attrs = group.attrs
        states = np.asarray(
            _strings(group["control/trial_state"]), dtype=object
        )
        relevant = np.isin(
            states, ("pre_crossing", "crossing", "response", "recovery")
        )
        crossing_states = _strings(group["crossing/state"])
        allowed_crossing_states = {
            "not_observed",
            "awaiting_start",
            "crossing",
            "completed",
            "extra_traversal",
        }
        binary_paths = (
            "crossing/active",
            "crossing/started_now",
            "crossing/intended_completed_now",
            "crossing/reverse_completed_now",
        )
        binary_values = {
            name: np.asarray(group[name][()], dtype=int) for name in binary_paths
        }
        if any(
            np.any(~np.isin(values, (0, 1)))
            for values in binary_values.values()
        ) or any(value not in allowed_crossing_states for value in crossing_states):
            self.issue(
                "crossing.row_contract",
                f"{path}/crossing",
                "crossing state/event rows contain an unknown or non-binary value",
            )
        intended_count = int(
            np.sum(binary_values["crossing/intended_completed_now"])
        )
        reverse_count = int(
            np.sum(binary_values["crossing/reverse_completed_now"])
        )
        if _int(attrs.get("actual_crossing_count", -1)) != intended_count:
            self.issue(
                "crossing.summary_count",
                f"{path}/@actual_crossing_count",
                "crossing count does not equal timestamped completion events",
            )
        planned_hand = str(_scalar(attrs.get("crossing_hand", "")))
        active = binary_values["crossing/active"].astype(bool)
        progress = np.asarray(
            group["crossing/progress_fraction"][()], dtype=float
        )
        intended_hands = _strings(group["crossing/intended_hand"])
        observed_hands = _strings(group["crossing/observed_hand"])
        crossing_evidence = relevant & active
        minimum_progress = _float(
            self.scenario_config.get("crossing", {}).get(
                "minimum_progress_fraction", math.nan
            )
        )
        valid_crossing_rows = bool(
            np.any(crossing_evidence)
            and all(
                intended_hands[index] == planned_hand
                and observed_hands[index] == planned_hand
                for index in np.flatnonzero(crossing_evidence)
            )
            and np.max(progress[crossing_evidence]) + 1e-12
            >= minimum_progress
        )
        protocol_valid = _bool(attrs.get("protocol_valid", 0))
        if protocol_valid and (
            intended_count != 1 or reverse_count != 0 or not valid_crossing_rows
        ):
            self.issue(
                "crossing.exactly_one",
                f"{path}/crossing",
                "protocol-valid trial lacks one complete, forward, intended-hand traversal",
            )

        for name in (
            "actual_path_deviation_rms_m",
            "actual_path_deviation_max_m",
            "actual_crossing_speed_m_s",
        ):
            value = _float(attrs.get(name, math.nan))
            if not math.isfinite(value) or value < 0.0:
                self.issue(
                    "crossing.summary_metric",
                    f"{path}/@{name}",
                    "crossing summary metrics must be finite and non-negative",
                )
        rms_deviation = _float(
            attrs.get("actual_path_deviation_rms_m", math.nan)
        )
        maximum_deviation = _float(
            attrs.get("actual_path_deviation_max_m", math.nan)
        )
        if rms_deviation > maximum_deviation + 1e-12:
            self.issue(
                "crossing.summary_metric",
                f"{path}/@actual_path_deviation_rms_m",
                "RMS path deviation cannot exceed maximum path deviation",
            )
        expected_speed = _float(
            self.scenario_config.get("crossing", {})
            .get("speed_target_m_s", {})
            .get(str(_scalar(attrs.get("speed", ""))), math.nan)
        )
        tolerance = _float(
            self.scenario_config.get("crossing", {}).get(
                "speed_tolerance_fraction", math.nan
            )
        )
        actual_speed = _float(attrs.get("actual_crossing_speed_m_s", math.nan))
        if protocol_valid and (
            not math.isfinite(expected_speed)
            or not math.isfinite(tolerance)
            or abs(actual_speed - expected_speed) / expected_speed
            > tolerance + 1e-12
        ):
            self.issue(
                "crossing.speed",
                f"{path}/@actual_crossing_speed_m_s",
                "protocol-valid crossing speed is outside its frozen category tolerance",
            )

        if np.any(relevant):
            intended_gap_rows = np.asarray(
                group[f"safety/{planned_hand}_surface_gap_m"][()], dtype=float
            )[relevant]
            all_gap_rows = np.asarray(
                group["safety/min_surface_gap_m"][()], dtype=float
            )[relevant]
            left_proximal = np.asarray(
                group["safety/left_proximal_surface_gap_m"][()], dtype=float
            )
            right_proximal = np.asarray(
                group["safety/right_proximal_surface_gap_m"][()], dtype=float
            )
            proximal_rows = np.minimum(left_proximal, right_proximal)[relevant]
            expected_gaps = {
                "actual_minimum_gap_m": float(np.min(intended_gap_rows)),
                "actual_minimum_any_hand_gap_m": float(np.min(all_gap_rows)),
                "actual_minimum_proximal_gap_m": float(np.min(proximal_rows)),
            }
        else:
            expected_gaps = {
                "actual_minimum_gap_m": 10.0,
                "actual_minimum_any_hand_gap_m": 10.0,
                "actual_minimum_proximal_gap_m": 10.0,
            }
        for name, expected in expected_gaps.items():
            actual = _float(attrs.get(name, math.nan))
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, rel_tol=0.0, abs_tol=1e-9
            ):
                self.issue(
                    "safety.summary_gap",
                    f"{path}/@{name}",
                    "trial gap summary does not equal its protected step rows",
                )

        cbf_active = np.asarray(group["cbf/active"][()], dtype=int).astype(bool)
        intervention = np.asarray(
            group["cbf/intervention_norm_radps"][()], dtype=float
        )
        relevant_active = cbf_active[relevant]
        relevant_intervention = intervention[relevant]
        intervention_mask = relevant_intervention > 0.01
        expected_activated = bool(np.any(relevant_active))
        expected_duration = float(np.sum(intervention_mask)) * _float(
            self.file.attrs.get("physics_dt_s", math.nan)
        )
        expected_maximum = (
            float(np.max(relevant_intervention))
            if relevant_intervention.size
            else 0.0
        )
        expected_count = int(
            np.sum(
                intervention_mask
                & ~np.concatenate(
                    (
                        np.asarray([False]),
                        intervention_mask[:-1],
                    )
                )
            )
        )
        cbf_checks = (
            _bool(attrs.get("cbf_activated", 0)) == expected_activated,
            math.isclose(
                _float(
                    attrs.get("cbf_total_intervention_duration_s", math.nan)
                ),
                expected_duration,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            math.isclose(
                _float(attrs.get("cbf_max_intervention_norm_rad_s", math.nan)),
                expected_maximum,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            _int(attrs.get("cbf_intervention_count", -1)) == expected_count,
        )
        if not all(cbf_checks):
            self.issue(
                "cbf.summary",
                path,
                "trial CBF aggregates do not equal relevant per-step interventions",
            )

        try:
            reasons = json.loads(
                str(_scalar(attrs.get("off_protocol_reasons_json", "")))
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            reasons = None
        off_protocol = _bool(attrs.get("off_protocol", 0))
        if (
            not isinstance(reasons, list)
            or len(reasons) != len(set(reasons))
            or any(not isinstance(reason, str) or not reason for reason in (reasons or ()))
            or off_protocol != bool(reasons)
            or (protocol_valid and off_protocol)
        ):
            self.issue(
                "trial.off_protocol_summary",
                f"{path}/@off_protocol_reasons_json",
                "off-protocol flag/reasons/protocol validity are inconsistent",
            )
        recovery_time = _float(attrs.get("recovery_time_s", math.nan))
        trial_simulation_duration = _float(
            attrs.get("trial_simulation_duration_s", math.nan)
        )
        start_simulation = _float(
            attrs.get("start_simulation_time_s", math.nan)
        )
        end_simulation = _float(attrs.get("end_simulation_time_s", math.nan))
        if not (
            math.isfinite(recovery_time)
            and 0.0 <= recovery_time <= trial_simulation_duration
            and math.isclose(
                trial_simulation_duration,
                end_simulation - start_simulation,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            self.issue(
                "trial.duration_summary",
                path,
                "simulation/recovery duration summaries are inconsistent",
            )
        grasp_outcome = str(_scalar(attrs.get("grasp_outcome", "")))
        release_outcome = str(_scalar(attrs.get("release_outcome", "")))
        task_success = _bool(attrs.get("task_success", 0))
        failure_reason = str(_scalar(attrs.get("task_failure_reason", "")))
        object_drop = _bool(attrs.get("object_drop", 0))
        if (
            grasp_outcome not in {"grasped", "not_observed"}
            or release_outcome
            not in {"released_and_settled", "reset_after_query", "not_settled"}
            or (
                task_success
                and (
                    failure_reason
                    or object_drop
                    or grasp_outcome != "grasped"
                    or release_outcome != "released_and_settled"
                    or not _bool(attrs.get("task_completion_observed", 0))
                )
            )
        ):
            self.issue(
                "task.summary",
                path,
                "task success/failure/drop/grasp/release outcomes are inconsistent",
            )

    def _validate_action_and_cbf_rows(
        self,
        group: Any,
        path: str,
        *,
        length: int,
        joint_count: int,
        arm_indices: np.ndarray,
    ) -> None:
        channels = (
            ("joint_positions", "rad"),
            ("joint_velocities", "rad_s"),
            ("joint_efforts", "nm"),
        )
        stages = (
            "nominal_rmpflow",
            "cbf_filtered",
            "submitted",
            "pre_applied",
            "applied",
        )
        arrays: dict[str, np.ndarray] = {}
        for stage in stages:
            commanded = np.zeros((length, joint_count), dtype=bool)
            for channel, unit in channels:
                base = f"actions/{stage}_{channel}_{unit}"
                values = np.asarray(group[base][()], dtype=float)
                mask = np.asarray(group[base + "_mask"][()], dtype=int)
                arrays[base] = values
                arrays[base + "_mask"] = mask
                commanded |= mask.astype(bool)
                if np.any(np.abs(values[~mask.astype(bool)]) > 1e-12):
                    self.issue(
                        "action.masked_payload_nonzero",
                        f"{path}/{base}",
                        "uncommanded canonical action slots must be zero",
                    )
            if stage in {"nominal_rmpflow", "cbf_filtered", "submitted"}:
                empty = np.flatnonzero(~np.any(commanded, axis=1))
                if empty.size:
                    self.issue(
                        "action.stage_incomplete",
                        f"{path}/actions/{stage}",
                        f"stage has no command at row {int(empty[0])}",
                    )
        pipeline = np.asarray(
            group["actions/action_pipeline_complete"][()], dtype=int
        )
        bad_pipeline = np.flatnonzero(pipeline != 1)
        if bad_pipeline.size:
            self.issue(
                "action.pipeline_incomplete",
                f"{path}/actions/action_pipeline_complete",
                f"pipeline is incomplete at row {int(bad_pipeline[0])}",
            )

        arm_set = set(int(value) for value in arm_indices)
        finger_indices = np.asarray(
            [index for index in range(joint_count) if index not in arm_set],
            dtype=int,
        )
        for stage in ("nominal_rmpflow", "cbf_filtered"):
            command_mask = np.zeros((length, joint_count), dtype=bool)
            for channel, unit in channels:
                command_mask |= arrays[
                    f"actions/{stage}_{channel}_{unit}_mask"
                ].astype(bool)
            bad_arm = np.flatnonzero(~np.all(command_mask[:, arm_indices], axis=1))
            bad_finger = (
                np.flatnonzero(np.any(command_mask[:, finger_indices], axis=1))
                if finger_indices.size
                else np.asarray([], dtype=int)
            )
            if bad_arm.size or bad_finger.size:
                row = int(bad_arm[0] if bad_arm.size else bad_finger[0])
                self.issue(
                    "action.stage_incomplete",
                    f"{path}/actions/{stage}",
                    "nominal and filtered stages must command exactly all arm "
                    f"joints at row {row}",
                )

        arm_submitted = np.asarray(
            group["actions/arm_action_submitted"][()], dtype=int
        ).astype(bool)
        gripper_commands = _strings(group["actions/gripper_command"])
        for row in range(length):
            if arm_submitted[row]:
                if gripper_commands[row].strip().lower() not in {"", "none"}:
                    self.issue(
                        "action.gripper_provenance",
                        f"{path}/actions/gripper_command",
                        f"arm row {row} unexpectedly records a gripper event",
                    )
                for channel, unit in channels:
                    filtered_base = f"actions/cbf_filtered_{channel}_{unit}"
                    submitted_base = f"actions/submitted_{channel}_{unit}"
                    expected_mask = arrays[filtered_base + "_mask"][row].astype(bool)
                    observed_mask = arrays[submitted_base + "_mask"][row].astype(bool)
                    if not np.all(observed_mask[expected_mask]) or not np.array_equal(
                        arrays[submitted_base][row, expected_mask],
                        arrays[filtered_base][row, expected_mask],
                    ):
                        self.issue(
                            "action.filtered_submitted",
                            f"{path}/actions/submitted",
                            f"submitted action does not preserve filtered CBF command at row {row}",
                        )
                        break
            for channel, unit in channels:
                submitted_base = f"actions/submitted_{channel}_{unit}"
                applied_base = f"actions/applied_{channel}_{unit}"
                selected = arrays[submitted_base + "_mask"][row].astype(bool)
                if np.any(selected) and (
                    not np.all(arrays[applied_base + "_mask"][row, selected] == 1)
                    or not np.allclose(
                        arrays[submitted_base][row, selected],
                        arrays[applied_base][row, selected],
                        rtol=0.0,
                        atol=1e-12,
                    )
                ):
                    self.issue(
                        "action.submitted_applied_mismatch",
                        f"{path}/actions/applied",
                        f"post-apply state does not contain submitted command at row {row}",
                    )
                    break

        applied_arm = np.asarray(
            group["actions/applied_arm_joint_velocities_rad_s"][()],
            dtype=float,
        )
        applied_arm_mask = np.asarray(
            group["actions/applied_arm_joint_velocities_mask"][()],
            dtype=int,
        )
        full_applied = arrays[
            "actions/applied_joint_velocities_rad_s"
        ][:, arm_indices]
        full_applied_mask = arrays[
            "actions/applied_joint_velocities_rad_s_mask"
        ][:, arm_indices]
        if not np.array_equal(applied_arm_mask, full_applied_mask) or not np.allclose(
            applied_arm, full_applied, rtol=0.0, atol=1e-12
        ):
            self.issue(
                "action.applied_arm_summary_mismatch",
                f"{path}/actions/applied_arm_joint_velocities_rad_s",
                "applied arm summary differs from the full post-apply action",
            )

        physics_dt_s = _float(self.file.attrs.get("physics_dt_s", math.nan))
        if not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
            self.issue(
                "runtime.physics_dt",
                "/@physics_dt_s",
                "positive physics_dt_s is required for action reconstruction",
            )
            return
        before = np.asarray(
            group["robot/measured_joint_positions_before_rad"][()], dtype=float
        )[:, arm_indices]

        def stage_velocity(stage: str) -> tuple[np.ndarray, np.ndarray]:
            velocity = arrays[
                f"actions/{stage}_joint_velocities_rad_s"
            ][:, arm_indices]
            velocity_mask = arrays[
                f"actions/{stage}_joint_velocities_rad_s_mask"
            ][:, arm_indices].astype(bool)
            positions = arrays[
                f"actions/{stage}_joint_positions_rad"
            ][:, arm_indices]
            position_mask = arrays[
                f"actions/{stage}_joint_positions_rad_mask"
            ][:, arm_indices].astype(bool)
            result = np.zeros_like(velocity, dtype=float)
            complete = np.zeros(length, dtype=bool)
            velocity_rows = np.all(velocity_mask, axis=1)
            position_rows = ~velocity_rows & np.all(position_mask, axis=1)
            result[velocity_rows] = velocity[velocity_rows]
            result[position_rows] = (
                positions[position_rows] - before[position_rows]
            ) / physics_dt_s
            complete |= velocity_rows | position_rows
            return result, complete

        nominal, nominal_complete = stage_velocity("nominal_rmpflow")
        filtered, filtered_complete = stage_velocity("cbf_filtered")
        incomplete = np.flatnonzero(~(nominal_complete & filtered_complete))
        if incomplete.size:
            self.issue(
                "action.stage_incomplete",
                f"{path}/actions",
                f"cannot reconstruct nominal/filtered arm velocity at row {int(incomplete[0])}",
            )
        expected_diagnostics = {
            "cbf/nominal_velocity_norm_radps": np.linalg.norm(nominal, axis=1),
            "cbf/filtered_velocity_norm_radps": np.linalg.norm(filtered, axis=1),
            "cbf/intervention_norm_radps": np.linalg.norm(filtered - nominal, axis=1),
        }
        for dataset_path, expected in expected_diagnostics.items():
            actual = np.asarray(group[dataset_path][()], dtype=float)
            bad = np.flatnonzero(
                ~np.isclose(actual, expected, rtol=1e-6, atol=1e-7)
            )
            if bad.size:
                self.issue(
                    "action.cbf_norm_mismatch",
                    f"{path}/{dataset_path}",
                    f"diagnostic disagrees with canonical action trace at row {int(bad[0])}",
                )

        active = np.asarray(group["cbf/active"][()], dtype=int)
        count = np.asarray(group["cbf/constraint_count"][()], dtype=int)
        unsafe = np.zeros(length, dtype=bool)
        unsafe |= np.asarray(group["cbf/intervention_available"][()], dtype=int) != 1
        unsafe |= np.asarray(group["cbf/feasible"][()], dtype=int) != 1
        unsafe |= np.asarray(group["cbf/fallback_applied"][()], dtype=int) != 0
        unsafe |= np.asarray(group["cbf/solver_converged"][()], dtype=int) != 1
        unsafe |= np.asarray(group["cbf/infeasibility_proven"][()], dtype=int) != 0
        unsafe |= np.asarray(group["cbf/relaxed_solution_applied"][()], dtype=int) != 0
        unsafe |= np.asarray(group["cbf/intentional_human_absence"][()], dtype=int) != 0
        unsafe |= (count < 0) | (count > 2)
        unsafe |= active != (count > 0).astype(int)
        unsafe |= np.asarray(group["cbf/tracked_hand_count"][()], dtype=int) != 2
        unsafe |= np.asarray(group["cbf/valid_hand_count"][()], dtype=int) != 2
        for diagnostic in (
            "cbf/intervention_norm_radps",
            "cbf/nominal_velocity_norm_radps",
            "cbf/filtered_velocity_norm_radps",
            "cbf/max_constraint_violation_before_mps",
            "cbf/max_constraint_violation_after_mps",
            "cbf/slack_mps",
            "cbf/solve_time_ms",
        ):
            unsafe |= np.asarray(group[diagnostic][()], dtype=float) < -1e-7
        statuses = _strings(group["cbf/status"])
        projection_statuses = _strings(group["cbf/projection_status"])
        failure_reasons = _strings(group["cbf/failure_reasons_json"])
        for row in range(length):
            try:
                failures = json.loads(failure_reasons[row])
            except (TypeError, ValueError, json.JSONDecodeError):
                failures = None
            if not isinstance(failures, list) or failures:
                unsafe[row] = True
            if active[row] == 0:
                for channel, unit in channels:
                    nominal_base = f"actions/nominal_rmpflow_{channel}_{unit}"
                    filtered_base = f"actions/cbf_filtered_{channel}_{unit}"
                    if not np.array_equal(
                        arrays[nominal_base + "_mask"][row],
                        arrays[filtered_base + "_mask"][row],
                    ) or not np.array_equal(
                        arrays[nominal_base][row], arrays[filtered_base][row]
                    ):
                        unsafe[row] = True
                if (
                    count[row] != 0
                    or abs(float(group["cbf/intervention_norm_radps"][row])) > 1e-7
                    or abs(float(group["cbf/slack_mps"][row])) > 1e-7
                    or statuses[row] != "inactive"
                    or projection_statuses[row] != "inactive"
                ):
                    unsafe[row] = True
        max_speed = float(self.file.attrs.get("cbf_max_joint_speed_rad_s", 2.0))
        active_rows = np.flatnonzero(active == 1)
        if active_rows.size and np.any(
            np.abs(filtered[active_rows]) > max_speed + 1e-6
        ):
            unsafe[int(active_rows[np.argwhere(
                np.any(np.abs(filtered[active_rows]) > max_speed + 1e-6, axis=1)
            )[0, 0]])] = True
        bad = np.flatnonzero(unsafe)
        if bad.size:
            self.issue(
                "cbf.unsafe_diagnostics",
                f"{path}/cbf",
                f"CBF/action diagnostics are incomplete or unsafe at row {int(bad[0])}",
            )

    def _validate_nominal_trajectory(
        self,
        payload: Any,
        *,
        phase: str,
        trigger_event: int,
        path: str,
    ) -> bool:
        config = self.scenario_config.get("nominal_bc_reference", {})
        valid = isinstance(payload, Mapping)
        samples = payload.get("samples") if valid else None
        valid = bool(
            valid
            and payload.get("semantics") == NOMINAL_BC_SWEEP_SEMANTICS
            and payload.get("phase") == phase
            and payload.get("controller") == "bc_only"
            and not _bool(payload.get("cbf_enabled", True))
            and _bool(payload.get("cue_phase_clock_latched", False))
            and _bool(payload.get("task_progression_during_crossing", False))
            and _int(payload.get("trigger_event", -1)) == trigger_event
            and math.isclose(
                _float(payload.get("sample_interval_target_s", math.nan)),
                float(config.get("sweep_sample_interval_s", -1.0)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                _float(payload.get("maximum_sample_interval_s", math.nan)),
                float(config.get("maximum_sweep_sample_interval_s", -1.0)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and isinstance(samples, list)
            and len(samples)
            >= int(config.get("minimum_trajectory_samples", 10**9))
        )
        if valid:
            previous_relative = -math.inf
            first_simulation_time = math.nan
            for sample_index, sample in enumerate(samples):
                if not isinstance(sample, Mapping):
                    valid = False
                    break
                relative = _float(sample.get("relative_time_s", math.nan))
                simulation_time = _float(
                    sample.get("simulation_time_s", math.nan)
                )
                if sample_index == 0:
                    first_simulation_time = simulation_time
                interval = relative - previous_relative
                sample_valid = bool(
                    math.isfinite(relative)
                    and math.isfinite(simulation_time)
                    and (sample_index != 0 or abs(relative) <= 1e-9)
                    and (sample_index == 0 or interval > 0.0)
                    and (
                        sample_index == 0
                        or interval
                        <= float(
                            config.get(
                                "maximum_sweep_sample_interval_s", -1.0
                            )
                        )
                        + 1e-9
                    )
                    and math.isclose(
                        simulation_time - first_simulation_time,
                        relative,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                    and _int(sample.get("control_step", -1)) >= 0
                    and 0 <= _int(sample.get("controller_event", -1)) <= 7
                    and (
                        relative
                        > float(
                            self.scenario_config.get("crossing", {}).get(
                                "cue_lead_s", math.nan
                            )
                        )
                        or _int(sample.get("controller_event", -1))
                        == trigger_event
                    )
                    and _finite_vector(sample.get("ee_position_world_m"), 3)
                    and _finite_vector(sample.get("ee_orientation_wxyz"), 4)
                    and _finite_vector(
                        sample.get("arm_joint_positions_rad"), 7
                    )
                    and _finite_vector(
                        sample.get("arm_joint_velocities_rad_s"), 7
                    )
                )
                if not sample_valid:
                    valid = False
                    break
                previous_relative = relative
        if not valid:
            self.issue(
                "layout.nominal_trajectory",
                path,
                "BC-only replay trajectory violates its frozen temporal/state contract",
            )
            return False
        stored_hash = str(payload.get("sha256", ""))
        body = {key: value for key, value in payload.items() if key != "sha256"}
        if not _SHA256_RE.fullmatch(stored_hash) or stored_hash != canonical_json_sha256(body):
            self.issue(
                "layout.nominal_trajectory_hash",
                path,
                "BC-only replay trajectory SHA-256 does not match its content",
            )
            return False
        return True

    def _validate_nominal_sweep(
        self,
        payload: Any,
        *,
        phase: str,
        severity: str,
        direction: str,
        speed: str,
        trajectory: Mapping[str, Any],
        reference: Mapping[str, Any],
        protected_colliders: set[str],
        path: str,
    ) -> bool:
        scenario_reference = self.scenario_config.get("nominal_bc_reference", {})
        crossing = self.scenario_config.get("crossing", {})
        severity_config = self.scenario_config.get("severity", {}).get(
            severity, {}
        )
        hand = "left" if direction == "left_to_right" else "right"
        allowed = severity_config.get("minimum_gap_range_m", ())
        coverage = payload.get("coverage") if isinstance(payload, Mapping) else None
        corridor = payload.get("corridor") if isinstance(payload, Mapping) else None
        candidates = payload.get("candidate_results") if isinstance(payload, Mapping) else None
        offset_range = scenario_reference.get("candidate_outward_offset_range_m", ())
        try:
            expected_candidate_count = int(
                round(
                    (float(offset_range[1]) - float(offset_range[0]))
                    / float(scenario_reference["candidate_outward_offset_step_m"])
                )
            ) + 1
            low, high = float(allowed[0]), float(allowed[1])
            speed_m_s = float(crossing["speed_target_m_s"][speed])
            nominal_gap = _float(
                payload.get("nominal_minimum_surface_gap_m", math.nan)
            )
            phase_start = _float(
                coverage.get("phase_start_simulation_time_s", math.nan)
            )
            crossing_start = _float(
                coverage.get("crossing_start_simulation_time_s", math.nan)
            )
            crossing_end = _float(
                coverage.get("crossing_end_simulation_time_s", math.nan)
            )
            maximum_interval = _float(
                coverage.get("maximum_observed_sample_interval_s", math.inf)
            )
            sample_count = _int(coverage.get("time_sample_count", -1))
            candidate_count = _int(coverage.get("candidate_count", -1))
            geometry_count = _int(coverage.get("geometry_query_count", -1))
        except (KeyError, IndexError, TypeError, ValueError):
            expected_candidate_count = -1
            low = high = speed_m_s = nominal_gap = math.nan
            phase_start = crossing_start = crossing_end = math.nan
            maximum_interval = math.inf
            sample_count = candidate_count = geometry_count = -1
        valid = bool(
            isinstance(payload, Mapping)
            and payload.get("semantics") == NOMINAL_BC_SWEEP_SEMANTICS
            and payload.get("phase") == phase
            and payload.get("severity") == severity
            and payload.get("crossing_direction") == direction
            and payload.get("hand") == hand
            and math.isclose(
                _float(payload.get("speed_m_s", math.nan)),
                speed_m_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                _float(payload.get("cue_lead_s", math.nan)),
                float(crossing.get("cue_lead_s", math.nan)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                _float(payload.get("corridor_length_m", math.nan)),
                float(crossing.get("corridor_length_m", math.nan)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and payload.get("allowed_gap_range_m") == list(allowed)
            and math.isclose(
                _float(payload.get("target_gap_m", math.nan)),
                float(severity_config.get("target_gap_m", math.nan)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and low <= nominal_gap <= high
            and _bool(payload.get("geometry_valid", False))
            and isinstance(coverage, Mapping)
            and candidate_count == expected_candidate_count
            and isinstance(candidates, list)
            and len(candidates) == expected_candidate_count
            and sample_count > 0
            and geometry_count == sample_count * candidate_count
            and _int(coverage.get("pre_cue_sample_count", 0)) > 0
            and _int(coverage.get("crossing_sample_count", 0)) > 0
            and math.isclose(
                crossing_start,
                phase_start + float(crossing.get("cue_lead_s", math.nan)),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                crossing_end,
                crossing_start
                + float(crossing.get("corridor_length_m", math.nan))
                / speed_m_s,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and 0.0 <= maximum_interval
            <= float(
                scenario_reference.get(
                    "maximum_sweep_sample_interval_s", -1.0
                )
            )
            + 1e-9
            and math.isclose(
                _float(
                    coverage.get("maximum_allowed_sample_interval_s", math.nan)
                ),
                float(
                    scenario_reference.get(
                        "maximum_sweep_sample_interval_s", math.nan
                    )
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and _float(coverage.get("pre_cue_coverage_fraction", -1.0)) > 0.0
            and _float(coverage.get("crossing_coverage_fraction", -1.0)) > 0.0
            and str(payload.get("trajectory_sha256", ""))
            == str(trajectory.get("sha256", ""))
            and _int(payload.get("trajectory_sample_count", -1))
            == len(trajectory.get("samples", ()))
            and _int(payload.get("trigger_reference_control_step", -1))
            == _int(reference.get("control_step", -2))
            and math.isclose(
                _float(
                    payload.get(
                        "trigger_reference_simulation_time_s", math.nan
                    )
                ),
                _float(reference.get("simulation_time_s", math.inf)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and _float(payload.get("minimum_pre_cue_start_gap_m", -math.inf))
            >= float(
                scenario_reference.get("minimum_pre_cue_start_gap_m", math.inf)
            )
            and math.isclose(
                _float(
                    payload.get(
                        "required_minimum_pre_cue_start_gap_m", math.nan
                    )
                ),
                float(
                    scenario_reference.get(
                        "minimum_pre_cue_start_gap_m", math.nan
                    )
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and isinstance(corridor, Mapping)
            and _finite_vector(corridor.get("start_world_m"), 3)
            and _finite_vector(corridor.get("end_world_m"), 3)
            and _finite_vector(corridor.get("center_world_m"), 3)
            and _finite_vector(corridor.get("tangent_world"), 3)
            and _finite_vector(corridor.get("outward_normal_world"), 3)
            and math.isclose(
                _float(
                    corridor.get("planned_minimum_surface_gap_m", math.nan)
                ),
                nominal_gap,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and corridor.get("geometry_query_semantics")
            == "physx_protected_surface_to_hand_sphere"
            and str(corridor.get("closest_collider_path", ""))
            in protected_colliders
        )
        if valid:
            selected_offset = _float(
                payload.get("selected_normal_offset_m", math.nan)
            )
            selected_candidates = [
                candidate
                for candidate in candidates
                if isinstance(candidate, Mapping)
                and math.isclose(
                    _float(candidate.get("normal_offset_m", math.nan)),
                    selected_offset,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            valid = bool(
                len(selected_candidates) == 1
                and _bool(selected_candidates[0].get("geometry_valid", False))
                and _int(
                    selected_candidates[0].get(
                        "invalid_geometry_query_count", -1
                    )
                )
                == 0
                and math.isclose(
                    _float(
                        selected_candidates[0].get(
                            "nominal_minimum_surface_gap_m", math.nan
                        )
                    ),
                    nominal_gap,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        if not valid:
            self.issue(
                "layout.nominal_sweep",
                path,
                "nominal corridor sweep violates condition, geometry, or coverage invariants",
            )
            return False
        stored_hash = str(payload.get("sweep_audit_sha256", ""))
        body = {
            key: value
            for key, value in payload.items()
            if key != "sweep_audit_sha256"
        }
        if not _SHA256_RE.fullmatch(stored_hash) or stored_hash != canonical_json_sha256(
            {"nominal_bc_sweep": body}
        ):
            self.issue(
                "layout.nominal_sweep_hash",
                path,
                "nominal corridor sweep SHA-256 does not match its content",
            )
            return False
        return True

    def _validate_precheck_reference(self, attrs: Any, path: str) -> None:
        if not self.table_ready.get("layout_prechecks", False):
            return
        index = _int(attrs.get("layout_precheck_index", -1))
        table = self.file["layout_prechecks"]
        count = table["layout_id"].shape[0] if "layout_id" in table else 0
        if index < 0 or index >= count:
            self.issue("layout.reference", f"{path}/@layout_precheck_index", "invalid BC-only precheck reference")
            return
        checks = {
            "strict_success": 1,
            "release_observed": 1,
            "nominal_phase_references_complete": 1,
            "nominal_corridor_bank_complete": 1,
            "cbf_enabled": 0,
            "human_absent_or_clear": 1,
            "controller": "bc_only",
            "policy_sha256": POLICY_SHA256,
        }
        for name, expected in checks.items():
            actual = _dataset_scalar(table[name], index)
            if actual != expected:
                self.issue("layout.precheck", f"/layout_prechecks/{name}[{index}]", f"expected {expected!r}, got {actual!r}")
        if _dataset_scalar(table["layout_id"], index) != str(_scalar(attrs.get("layout_id", ""))):
            self.issue("layout.identity", path, "trial layout does not match its passed precheck")
        payloads: dict[str, Any] = {}
        for name in (
            "nominal_phase_references_json",
            "nominal_phase_trajectories_json",
            "nominal_corridor_bank_json",
        ):
            try:
                value = json.loads(str(_dataset_scalar(table[name], index)))
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
            payloads[name] = value
            if not isinstance(value, Mapping):
                self.issue(
                    "layout.nominal_json",
                    f"/layout_prechecks/{name}[{index}]",
                    "accepted precheck nominal payload must be a JSON object",
                )
        references = payloads["nominal_phase_references_json"]
        trajectories = payloads["nominal_phase_trajectories_json"]
        bank = payloads["nominal_corridor_bank_json"]
        triggers = self.protocol_config.get("task", {}).get(
            "crossing_trigger_event", {}
        )
        valid_references = isinstance(references, Mapping) and set(
            references
        ) == set(TASK_PHASES)
        if valid_references:
            for phase in TASK_PHASES:
                reference = references.get(phase)
                if not isinstance(reference, Mapping) or not (
                    reference.get("phase") == phase
                    and reference.get("controller") == "bc_only"
                    and _int(reference.get("controller_event", -1))
                    == int(triggers.get(phase, -2))
                    and _finite_vector(reference.get("ee_position_world_m"), 3)
                    and _finite_vector(reference.get("arm_joint_positions_rad"), 7)
                    and _int(reference.get("control_step", -1)) >= 0
                    and math.isfinite(
                        _float(reference.get("simulation_time_s", math.nan))
                    )
                ):
                    valid_references = False
                    break
        if not valid_references:
            self.issue(
                "layout.nominal_references",
                f"/layout_prechecks/nominal_phase_references_json[{index}]",
                "accepted BC-only precheck must record all four valid phase references",
            )
            return
        valid_trajectories = isinstance(trajectories, Mapping) and set(
            trajectories
        ) == set(TASK_PHASES)
        valid_bank = isinstance(bank, Mapping) and set(bank) == set(TASK_PHASES)
        try:
            protected_colliders = set(
                json.loads(
                    str(
                        _scalar(
                            self.file.attrs.get(
                                "cbf_protected_colliders_json", "[]"
                            )
                        )
                    )
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            protected_colliders = set()
        for nominal_phase in TASK_PHASES:
            trajectory = (
                trajectories.get(nominal_phase)
                if isinstance(trajectories, Mapping)
                else None
            )
            valid_trajectories = bool(
                self._validate_nominal_trajectory(
                    trajectory,
                    phase=nominal_phase,
                    trigger_event=int(triggers.get(nominal_phase, -2)),
                    path=(
                        f"/layout_prechecks/nominal_phase_trajectories_json"
                        f"[{index}]/{nominal_phase}"
                    ),
                )
                and valid_trajectories
            )
            phase_bank = (
                bank.get(nominal_phase)
                if isinstance(bank, Mapping)
                else None
            )
            if not isinstance(phase_bank, Mapping) or set(phase_bank) != set(DIRECTIONS):
                valid_bank = False
                self.issue(
                    "layout.nominal_bank",
                    f"/layout_prechecks/nominal_corridor_bank_json[{index}]/{nominal_phase}",
                    "phase bank must contain both directions",
                )
                continue
            for direction in DIRECTIONS:
                direction_bank = phase_bank.get(direction)
                if not isinstance(direction_bank, Mapping) or set(direction_bank) != set(SPEEDS):
                    valid_bank = False
                    self.issue(
                        "layout.nominal_bank",
                        f"/layout_prechecks/nominal_corridor_bank_json[{index}]/{nominal_phase}/{direction}",
                        "direction bank must contain both speeds",
                    )
                    continue
                for speed in SPEEDS:
                    speed_bank = direction_bank.get(speed)
                    if not isinstance(speed_bank, Mapping) or set(speed_bank) != set(SEVERITIES):
                        valid_bank = False
                        self.issue(
                            "layout.nominal_bank",
                            f"/layout_prechecks/nominal_corridor_bank_json[{index}]/{nominal_phase}/{direction}/{speed}",
                            "speed bank must contain all three severities",
                        )
                        continue
                    for severity in SEVERITIES:
                        valid_bank = bool(
                            self._validate_nominal_sweep(
                                speed_bank.get(severity),
                                phase=nominal_phase,
                                severity=severity,
                                direction=direction,
                                speed=speed,
                                trajectory=(
                                    trajectory
                                    if isinstance(trajectory, Mapping)
                                    else {}
                                ),
                                reference=references[nominal_phase],
                                protected_colliders=protected_colliders,
                                path=(
                                    f"/layout_prechecks/nominal_corridor_bank_json"
                                    f"[{index}]/{nominal_phase}/{direction}/{speed}/{severity}"
                                ),
                            )
                            and valid_bank
                        )
        if not valid_trajectories or not valid_bank:
            self.issue(
                "layout.nominal_calibration",
                f"/layout_prechecks[{index}]",
                "accepted precheck lacks a complete validated trajectory/corridor bank",
            )
        phase = str(_scalar(attrs.get("task_phase", "")))
        try:
            corridor = json.loads(
                str(_scalar(attrs.get("planned_corridor_json", "")))
            )
            audit = corridor["nominal_bc_reference"]
            reference = references[phase]
            matches_precheck = bool(
                _int(audit.get("precheck_control_step", -1))
                == _int(reference.get("control_step", -2))
                and math.isclose(
                    _float(audit.get("precheck_simulation_time_s", math.nan)),
                    _float(reference.get("simulation_time_s", math.inf)),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            matches_precheck = False
        if not matches_precheck:
            self.issue(
                "layout.nominal_reference_relation",
                path,
                "trial nominal-reference audit does not match its BC-only precheck",
            )

    def _validate_relations_and_timing(self) -> None:
        if not self.table_ready.get("queries", False):
            self._validate_encounters()
            return
        queries = self.file["queries"]
        qids = _strings(queries.get("query_id"))
        query_trial_ids = _strings(queries.get("trial_id"))
        trial_groups: dict[str, Any] = {}
        if self.group_ready.get("trials", False):
            for group in self.file["trials"].values():
                if not hasattr(group, "keys"):
                    continue
                trial_id = str(_scalar(group.attrs.get("trial_id", "")))
                if not trial_id or trial_id in trial_groups:
                    self.issue(
                        "trial.id",
                        group.name,
                        "trial IDs must be non-empty and unique",
                    )
                else:
                    trial_groups[trial_id] = group
        if len(qids) != len(set(qids)) or any(not value for value in qids):
            self.issue("feedback.query_id", "/queries/query_id", "query IDs must be non-empty and unique")
        encounter_ids: set[str] = set()
        encounter_owner: dict[str, str] = {}
        if self.table_ready.get("encounters", False):
            listed = _strings(self.file.get("encounters/encounter_id"))
            owners = _strings(self.file.get("encounters/trial_id"))
            encounter_ids = set(listed)
            encounter_owner = dict(zip(listed, owners))
            if len(listed) != len(set(listed)) or any(not value for value in listed):
                self.issue("encounter.id", "/encounters/encounter_id", "encounter IDs must be non-empty and unique")
        statuses = _strings(queries.get("response_status"))
        q1s = _strings(queries.get("q1_response"))
        reasons = _strings(queries.get("rejection_reasons_json"))
        ratings = [
            np.asarray(queries[name][()], dtype=int)
            for name in (
                "q2_collision_risk",
                "q3_excessive_avoidance",
                "q4_abrupt_unpredictable",
                "q5_task_disruption",
                "q6_confidence",
            )
        ]
        issued_mono = np.asarray(queries.get("issued_monotonic_ns", []), dtype=np.int64)
        completed_mono = np.asarray(queries.get("completed_monotonic_ns", []), dtype=np.int64)
        issued_unix = np.asarray(queries.get("issued_unix_ns", []), dtype=np.int64)
        completed_unix = np.asarray(queries.get("completed_unix_ns", []), dtype=np.int64)
        latencies = np.asarray(queries.get("response_latency_ms", []), dtype=float)
        issued_sim = np.asarray(queries.get("issued_simulation_time_s", []), dtype=float)
        completed_sim = np.asarray(queries.get("completed_simulation_time_s", []), dtype=float)
        input_sources = _strings(queries.get("input_source"))
        for index, (status, q1, reason_json) in enumerate(zip(statuses, q1s, reasons)):
            path = f"/queries[{index}]"
            if input_sources[index] != "vr_controller":
                self.issue(
                    "feedback.input_source",
                    path,
                    "mandatory evaluation must be entered with a VR controller",
                )
            query_trial_id = query_trial_ids[index]
            trial_group = trial_groups.get(query_trial_id)
            if trial_group is None:
                self.issue(
                    "feedback.trial",
                    path,
                    "query references an unknown trial",
                )
            encounter_id = _dataset_scalar(queries["encounter_id"], index)
            if encounter_id and encounter_id not in encounter_ids:
                self.issue("feedback.encounter", path, "query references unknown encounter")
            elif encounter_id and encounter_owner.get(str(encounter_id)) != query_trial_id:
                self.issue(
                    "feedback.encounter_trial",
                    path,
                    "query encounter belongs to a different trial",
                )
            if status == "completed":
                if q1 not in Q1_RESPONSES:
                    self.issue("feedback.q1", path, "completed query requires exact Q1 response")
                if any(int(values[index]) not in range(1, 6) for values in ratings):
                    self.issue("feedback.likert", path, "Q2-Q6 must each be 1-5")
            elif status in {"timeout", "no_response"}:
                if q1 or any(int(values[index]) != 0 for values in ratings):
                    self.issue("feedback.missing_semantics", path, "missing response was converted into a label")
            else:
                self.issue("feedback.status", path, "invalid response status")
            try:
                decoded = json.loads(reason_json)
            except (TypeError, json.JSONDecodeError):
                decoded = None
            if not isinstance(decoded, list) or len(set(decoded)) != len(decoded) or any(item not in REJECTION_REASONS for item in (decoded or [])):
                self.issue("feedback.reason", path, "invalid rejection-reason list")
                decoded = []
            if q1 == "needs_modification" and not decoded:
                self.issue("feedback.reason_required", path, "needs_modification requires at least one reason")
            if q1 != "needs_modification" and decoded:
                self.issue("feedback.reason_unexpected", path, "reasons are only valid for needs_modification")
            if not (issued_mono[index] > 0 and completed_mono[index] >= issued_mono[index]):
                self.issue("clock.query", path, "query monotonic clocks are invalid")
            if not (
                issued_unix[index] > 0
                and completed_unix[index] >= issued_unix[index]
            ):
                self.issue(
                    "clock.query_unix",
                    path,
                    "query Unix clocks are invalid",
                )
            expected_latency = (completed_mono[index] - issued_mono[index]) / 1e6
            if not math.isclose(float(latencies[index]), expected_latency, rel_tol=1e-6, abs_tol=1e-3):
                self.issue("clock.query_latency", path, "response latency does not match monotonic timestamps")
            if completed_sim[index] < issued_sim[index]:
                self.issue("clock.query_sim", path, "query simulation time went backwards")
            if trial_group is not None:
                attrs = trial_group.attrs
                in_trial = (
                    _float(attrs.get("start_simulation_time_s", math.nan))
                    <= issued_sim[index]
                    <= completed_sim[index]
                    <= _float(attrs.get("end_simulation_time_s", math.nan))
                    and _int(attrs.get("start_monotonic_ns", 0))
                    <= int(issued_mono[index])
                    <= int(completed_mono[index])
                    <= _int(attrs.get("end_monotonic_ns", 0))
                    and _int(attrs.get("start_unix_ns", 0))
                    <= int(issued_unix[index])
                    <= int(completed_unix[index])
                    <= _int(attrs.get("end_unix_ns", 0))
                )
                if not in_trial:
                    self.issue(
                        "clock.query_trial_bounds",
                        path,
                        "query clocks must lie within the referenced trial",
                    )
        if self.table_ready.get("realtime_markers", False):
            markers = self.file["realtime_markers"]
            marker_ids = _strings(markers.get("marker_id"))
            if len(marker_ids) != len(set(marker_ids)) or any(
                not value for value in marker_ids
            ):
                self.issue(
                    "marker.id",
                    "/realtime_markers/marker_id",
                    "marker IDs must be non-empty and unique",
                )
            marker_trial_ids = _strings(markers.get("trial_id"))
            marker_types = _strings(markers.get("marker_type"))
            controller_hands = _strings(markers.get("controller_hand"))
            buttons = _strings(markers.get("button"))
            for index, marker_type in enumerate(marker_types):
                marker_path = f"/realtime_markers[{index}]"
                if marker_type not in MARKER_TYPES:
                    self.issue("marker.type", marker_path, "marker types must remain separate")
                marker_trial_id = marker_trial_ids[index]
                trial_group = trial_groups.get(marker_trial_id)
                if trial_group is None:
                    self.issue(
                        "marker.trial",
                        marker_path,
                        "marker references an unknown trial",
                    )
                    continue
                marker_sim = _float(markers["simulation_time_s"][index])
                marker_mono = _int(markers["monotonic_ns"][index])
                marker_unix = _int(markers["unix_ns"][index])
                marker_step = _int(markers["control_step"][index])
                attrs = trial_group.attrs
                if not (
                    _float(attrs.get("start_simulation_time_s", math.nan))
                    <= marker_sim
                    <= _float(attrs.get("end_simulation_time_s", math.nan))
                    and _int(attrs.get("start_monotonic_ns", 0))
                    <= marker_mono
                    <= _int(attrs.get("end_monotonic_ns", 0))
                    and _int(attrs.get("start_unix_ns", 0))
                    <= marker_unix
                    <= _int(attrs.get("end_unix_ns", 0))
                ):
                    self.issue(
                        "marker.clock",
                        marker_path,
                        "marker clocks must lie within the referenced trial",
                    )
                result_steps = np.asarray(
                    trial_group.get("control/result_step", []),
                    dtype=np.int64,
                )
                if marker_step not in set(int(item) for item in result_steps):
                    self.issue(
                        "marker.control_step",
                        marker_path,
                        "marker does not reference a recorded control transition",
                    )
                crossing_hand = str(
                    _scalar(attrs.get("crossing_hand", ""))
                )
                mapping = (
                    self.protocol_config.get("feedback", {})
                    .get("opposite_controller_buttons", {})
                    .get(crossing_hand, {})
                )
                expected_button = str(mapping.get(marker_type, ""))
                actual_button = (
                    f"{controller_hands[index]}.{buttons[index]}"
                )
                if not expected_button or actual_button != expected_button:
                    self.issue(
                        "marker.mapping",
                        marker_path,
                        "marker button does not match the frozen opposite-controller mapping",
                    )
        self._validate_encounters()
        if self.table_ready.get("query_answers", False):
            answers = self.file["query_answers"]
            count = self.table_lengths.get("query_answers", 0)
            if _int(self.file.attrs.get("query_answer_count", -1)) != count:
                self.issue("count.root", "/@query_answer_count", "root count does not match query answer audit")
            query_index_by_id = {
                query_id: index for index, query_id in enumerate(qids)
            }
            allowed_question_ids = {
                Q1_ID,
                "q2_collision_risk",
                "q3_excessive_avoidance",
                "q4_abrupt_unpredictable",
                "q5_task_disruption",
                "q6_confidence",
                "rejection_reasons",
            }
            answer_rows_by_query: dict[str, dict[str, Any]] = {}
            answer_query_ids = _strings(answers.get("query_id"))
            answer_trial_ids = _strings(answers.get("trial_id"))
            answer_question_ids = _strings(answers.get("question_id"))
            answer_values = _strings(answers.get("value_json"))
            for index, answer_query in enumerate(answer_query_ids):
                answer_path = f"/query_answers[{index}]"
                query_index = query_index_by_id.get(answer_query)
                if query_index is None:
                    self.issue(
                        "feedback.answer_query",
                        answer_path,
                        "answer references unknown query",
                    )
                    continue
                if answer_trial_ids[index] != query_trial_ids[query_index]:
                    self.issue(
                        "feedback.answer_trial",
                        answer_path,
                        "answer trial does not match its query",
                    )
                question_id = answer_question_ids[index]
                if question_id not in allowed_question_ids:
                    self.issue(
                        "feedback.answer_question",
                        answer_path,
                        "unknown question identifier",
                    )
                per_query = answer_rows_by_query.setdefault(answer_query, {})
                if question_id in per_query:
                    self.issue(
                        "feedback.answer_duplicate",
                        answer_path,
                        "a query contains duplicate answer events for one question",
                    )
                try:
                    per_query[question_id] = json.loads(answer_values[index])
                except (TypeError, ValueError, json.JSONDecodeError):
                    per_query[question_id] = None
                    self.issue(
                        "feedback.answer_value",
                        answer_path,
                        "answer value is not valid JSON",
                    )
                answer_sim = _float(answers["simulation_time_s"][index])
                answer_mono = _int(answers["monotonic_ns"][index])
                answer_unix = _int(answers["unix_ns"][index])
                if not (
                    issued_sim[query_index]
                    <= answer_sim
                    <= completed_sim[query_index]
                    and int(issued_mono[query_index])
                    <= answer_mono
                    <= int(completed_mono[query_index])
                    and int(issued_unix[query_index])
                    <= answer_unix
                    <= int(completed_unix[query_index])
                ):
                    self.issue(
                        "feedback.answer_clock",
                        answer_path,
                        "answer clocks must lie within its query interval",
                    )
                trial_group = trial_groups.get(answer_trial_ids[index])
                if trial_group is not None:
                    recorded_steps = set(
                        int(item)
                        for item in np.asarray(
                            trial_group.get("control/result_step", []),
                            dtype=np.int64,
                        )
                    )
                    if _int(answers["control_step"][index]) not in recorded_steps:
                        self.issue(
                            "feedback.answer_control_step",
                            answer_path,
                            "answer does not reference a recorded control transition",
                        )
            answer_columns = (
                "q2_collision_risk",
                "q3_excessive_avoidance",
                "q4_abrupt_unpredictable",
                "q5_task_disruption",
                "q6_confidence",
            )
            for query_index, query_id in enumerate(qids):
                if statuses[query_index] != "completed":
                    continue
                recorded = answer_rows_by_query.get(query_id, {})
                expected_answers: dict[str, Any] = {
                    Q1_ID: q1s[query_index],
                    **{
                        name: int(queries[name][query_index])
                        for name in answer_columns
                    },
                }
                if q1s[query_index] == "needs_modification":
                    try:
                        expected_answers["rejection_reasons"] = json.loads(
                            reasons[query_index]
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        expected_answers["rejection_reasons"] = None
                if recorded != expected_answers:
                    self.issue(
                        "feedback.answer_audit",
                        f"/queries[{query_index}]",
                        "completed query fields do not exactly match its timestamped answer events",
                    )

    def _validate_encounters(self) -> None:
        if not self.table_ready.get("encounters", False):
            return
        table = self.file["encounters"]
        count = table["encounter_id"].shape[0] if "encounter_id" in table else 0
        for index in range(count):
            path = f"/encounters[{index}]"
            encounter_id = str(_dataset_scalar(table["encounter_id"], index))
            trial_id = str(_dataset_scalar(table["trial_id"], index))
            trial_group = None
            if self.group_ready.get("trials", False):
                for candidate in self.file["trials"].values():
                    if (
                        hasattr(candidate, "keys")
                        and str(_scalar(candidate.attrs.get("trial_id", "")))
                        == trial_id
                    ):
                        trial_group = candidate
                        break
            if trial_group is None:
                self.issue(
                    "encounter.trial",
                    path,
                    "encounter references an unknown trial",
                )
                continue
            onset = _int(table["onset_step"][index])
            confirmed = _int(table["onset_confirmed_step"][index])
            offset = _int(table["offset_step_exclusive"][index])
            window_start = _int(table["window_start_step"][index])
            window_end = _int(table["window_end_step_exclusive"][index])
            if not (
                0 <= window_start <= onset
                and confirmed == onset + 2
                and confirmed < offset <= window_end
                and _int(table["risk_onset_step"][index]) == onset
            ):
                self.issue("encounter.window", path, "onset/confirmation/offset/window ordering is invalid")
            source_steps = np.asarray(
                trial_group["control/source_step"][()], dtype=int
            )
            simulation_times = np.asarray(
                trial_group["control/sim_time"][()], dtype=float
            )
            monotonic_times = np.asarray(
                trial_group["control/post_step_monotonic_ns"][()], dtype=np.int64
            )
            index_by_step = {
                int(step): row_index
                for row_index, step in enumerate(source_steps)
            }
            required_steps = (onset, onset + 1, confirmed, offset - 1)
            if any(step not in index_by_step for step in required_steps):
                self.issue(
                    "encounter.step_relation",
                    path,
                    "encounter onset/confirmation/offset must reference recorded transitions",
                )
                continue
            onset_index = index_by_step[onset]
            confirmed_index = index_by_step[confirmed]
            offset_index = index_by_step[offset - 1]
            start_mono = _int(table["onset_monotonic_ns"][index])
            end_mono = _int(table["offset_monotonic_ns"][index])
            start_sim = _float(table["onset_simulation_time_s"][index])
            end_sim = _float(table["offset_simulation_time_s"][index])
            if not (
                0 < start_mono <= end_mono
                and 0 <= start_sim <= end_sim
                and math.isclose(
                    start_sim,
                    float(simulation_times[onset_index]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    end_sim,
                    float(simulation_times[offset_index]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and start_mono == int(monotonic_times[onset_index])
                and end_mono == int(monotonic_times[offset_index])
            ):
                self.issue("encounter.clock", path, "encounter clocks are invalid")

            minimum_gap = np.asarray(
                trial_group["safety/min_surface_gap_m"][()], dtype=float
            )
            ttc = np.asarray(
                trial_group["safety/min_ttc_s"][()], dtype=float
            )
            closing = np.asarray(
                trial_group["safety/closing"][()], dtype=int
            ).astype(bool)
            dynamic_valid = np.asarray(
                trial_group["safety/dynamic_measurement_valid"][()], dtype=int
            ).astype(bool)
            intervention = np.asarray(
                trial_group["cbf/intervention_norm_radps"][()], dtype=float
            )
            cbf_active = np.asarray(
                trial_group["cbf/active"][()], dtype=int
            ).astype(bool)
            trial_states = np.asarray(
                _strings(trial_group["control/trial_state"]), dtype=object
            )
            detector_rows = np.isin(trial_states, ("crossing", "response"))
            risk = (
                (minimum_gap <= 0.13)
                | (
                    dynamic_valid
                    & closing
                    & (ttc >= 0.0)
                    & (ttc <= 0.75)
                )
                | (intervention >= 0.05)
            )
            confirmation_rows = [
                index_by_step[step] for step in (onset, onset + 1, confirmed)
            ]
            if not all(
                detector_rows[row_index] and risk[row_index]
                for row_index in confirmation_rows
            ):
                self.issue(
                    "encounter.onset_confirmation",
                    path,
                    "onset is not backed by three consecutive runtime risk frames",
                )
            row_ids = _strings(trial_group["encounter/row_encounter_id"])
            encounter_states = _strings(trial_group["encounter/state"])
            if (
                row_ids[confirmed_index] != encounter_id
                or row_ids[offset_index] != encounter_id
            ):
                self.issue(
                    "encounter.row_identity",
                    path,
                    "confirmed/offset rows do not carry the recorded encounter ID",
                )
            interval = (
                source_steps >= onset
            ) & (source_steps < offset)
            expected_minimum_gap = float(np.min(minimum_gap[interval]))
            expected_maximum_intervention = float(
                np.max(intervention[interval])
            )
            if not math.isclose(
                _float(table["minimum_surface_gap_m"][index]),
                expected_minimum_gap,
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or not math.isclose(
                _float(table["maximum_intervention_norm_rad_s"][index]),
                expected_maximum_intervention,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                self.issue(
                    "encounter.aggregate",
                    path,
                    "encounter min-gap/max-intervention do not match their raw window",
                )

            expected_window_indices = np.flatnonzero(
                (simulation_times >= max(0.0, start_sim - 1.0) - 1e-12)
                & (simulation_times <= end_sim + 2.0 + 1e-12)
            )
            expected_window_start = (
                int(source_steps[expected_window_indices[0]])
                if expected_window_indices.size
                else onset
            )
            expected_window_end = (
                int(source_steps[expected_window_indices[-1]]) + 1
                if expected_window_indices.size
                else offset
            )
            if (
                window_start != expected_window_start
                or window_end != expected_window_end
            ):
                self.issue(
                    "encounter.saved_window",
                    path,
                    "saved encounter window is not onset-1 s through offset+2 s",
                )

            natural_offset = encounter_states[offset_index] == "merge_wait"
            if natural_offset:
                clear = (
                    (minimum_gap >= 0.15)
                    & ~cbf_active
                    & dynamic_valid
                    & ((~closing) | (ttc > 1.0))
                    & (intervention <= 0.01)
                )
                clear_start_index = offset_index
                while (
                    clear_start_index > 0
                    and clear[clear_start_index - 1]
                    and detector_rows[clear_start_index - 1]
                ):
                    clear_start_index -= 1
                if (
                    not clear[offset_index]
                    or simulation_times[offset_index]
                    - simulation_times[clear_start_index]
                    < 0.5 - 1e-9
                ):
                    self.issue(
                        "encounter.offset_confirmation",
                        path,
                        "natural offset lacks 0.5 s of all-clear frames",
                    )

            merge_count = sum(
                1
                for row_index in range(1, offset_index + 1)
                if encounter_states[row_index - 1] == "merge_wait"
                and encounter_states[row_index] == "active"
                and row_ids[row_index] == encounter_id
                and simulation_times[row_index] - end_sim <= 1.0 + 1e-12
            )
            if _int(table["merged_reentry_count"][index]) != merge_count:
                self.issue(
                    "encounter.merge",
                    path,
                    "merged re-entry count does not match <=1 s row transitions",
                )
            expected_timeout = bool(
                np.any(
                    detector_rows[: offset_index + 1]
                    & (simulation_times[: offset_index + 1] - start_sim > 8.0)
                )
            )
            if _bool(table["encounter_timeout"][index]) != expected_timeout:
                self.issue(
                    "encounter.timeout",
                    path,
                    "8 s encounter timeout flag disagrees with simulation time",
                )

            cbf_start = _int(table["cbf_intervention_start_step"][index])
            cbf_confirmed = _int(table["cbf_intervention_confirmed_step"][index])
            intervention_rows = np.flatnonzero(
                detector_rows[: offset_index + 1]
                & (intervention[: offset_index + 1] >= 0.05)
            )
            expected_cbf_pair = (-1, -1)
            for row_index in intervention_rows:
                if (
                    row_index + 2 <= offset_index
                    and detector_rows[row_index : row_index + 3].all()
                    and (intervention[row_index : row_index + 3] >= 0.05).all()
                    and np.array_equal(
                        np.diff(source_steps[row_index : row_index + 3]),
                        np.ones(2, dtype=int),
                    )
                ):
                    expected_cbf_pair = (
                        int(source_steps[row_index]),
                        int(source_steps[row_index + 2]),
                    )
                    break
            if (cbf_start, cbf_confirmed) != expected_cbf_pair:
                self.issue("encounter.cbf_confirmation", path, "confirmed intervention must follow three consecutive frames")
            trial_attrs = trial_group.attrs
            if not (
                _int(trial_attrs.get("risk_onset_step", -2)) == onset
                and _int(
                    trial_attrs.get("cbf_intervention_start_step", -2)
                )
                == cbf_start
                and _int(
                    trial_attrs.get("cbf_intervention_confirmed_step", -2)
                )
                == cbf_confirmed
            ):
                self.issue(
                    "encounter.trial_summary",
                    path,
                    "trial steering timestamps do not match its encounter",
                )

    def _validate_design_counts(self) -> None:
        if not self.table_ready.get("schedule", False):
            return
        schedule = self.file["schedule"]
        practice = np.asarray(schedule.get("practice", []), dtype=int)
        anchor = np.asarray(schedule.get("anchor_repeat", []), dtype=int)
        pilot = np.asarray(schedule.get("pilot", []), dtype=int)
        speed = _strings(schedule.get("speed"))
        evaluated_indices = [index for index in range(len(speed)) if not practice[index]]
        if _bool(self.file.attrs.get("practice_only", 0)):
            self.study_eligible = False
            if evaluated_indices or not (1 <= len(practice) <= 4) or np.any(practice != 1):
                self.issue(
                    "design.practice_only",
                    "/schedule",
                    "practice-only equipment runs require one to four practice trials and no evaluated trial",
                )
            return
        is_pilot = bool(np.any(pilot == 1))
        invalid_evaluated: list[int] = []
        if self.group_ready.get("trials", False):
            for index in evaluated_indices:
                trial_name = f"trial_{index:06d}"
                if trial_name not in self.file["trials"]:
                    invalid_evaluated.append(index)
                    continue
                attrs = self.file["trials"][trial_name].attrs
                if not _bool(attrs.get("protocol_valid", 0)) or _bool(
                    attrs.get("analysis_exclude", 1)
                ):
                    invalid_evaluated.append(index)
        if invalid_evaluated:
            # Off-protocol attempts are expected study observations: retaining
            # them is required to measure adherence and timeout rates.  They
            # make the session-level quota non-eligible, but are not an HDF5
            # or provenance integrity failure.  Per-trial checks above still
            # require protocol_valid=0 and analysis_exclude=1.
            self.study_eligible = False
        if is_pilot:
            if not all(pilot[index] == 1 for index in evaluated_indices):
                self.issue("design.pilot", "/schedule/pilot", "pilot metadata must cover evaluated trials")
            if not 12 <= len(evaluated_indices) <= 16:
                self.issue("design.pilot_count", "/schedule", "pilot requires 12-16 evaluated trials")
            pilot_base = [
                index for index in evaluated_indices if not anchor[index]
            ]
            for field, categories in (
                ("task_phase", TASK_PHASES),
                ("severity", SEVERITIES),
                ("direction", DIRECTIONS),
                ("speed", SPEEDS),
            ):
                values = _strings(schedule.get(field))
                counts = [
                    sum(values[index] == category for index in pilot_base)
                    for category in categories
                ]
                if counts and max(counts) - min(counts) > 1:
                    self.issue(
                        "design.pilot_balance",
                        f"/schedule/{field}",
                        "pilot non-anchor marginal counts must differ by at most one",
                    )
        else:
            non_anchor = [index for index in evaluated_indices if not anchor[index]]
            if len(non_anchor) != 24:
                self.issue("design.factorial_count", "/schedule", "production requires 24 non-anchor evaluated trials")
            slow = sum(speed[index] == "slow" for index in non_anchor)
            fast = sum(speed[index] == "fast" for index in non_anchor)
            if (slow, fast) != (12, 12):
                self.issue("design.speed_balance", "/schedule/speed", "production factorial requires slow=12 and fast=12")
            combinations = {
                (
                    _dataset_scalar(schedule["task_phase"], index),
                    _dataset_scalar(schedule["severity"], index),
                    _dataset_scalar(schedule["direction"], index),
                )
                for index in non_anchor
            }
            if len(combinations) != 24:
                self.issue("design.factorial", "/schedule", "phase x severity x direction must be a full factorial")
        if len(practice) and int(np.sum(practice)) not in {0, 4}:
            self.issue("design.practice_count", "/schedule/practice", "practice must be disabled or contain exactly four trials")
        if not is_pilot and int(np.sum(anchor)) not in {0, 4}:
            self.issue("design.anchor_count", "/schedule/anchor_repeat", "production anchors must be disabled or contain four repeats")

    def _report(self) -> OnlineValidationReport:
        return OnlineValidationReport(
            path=str(self.path),
            valid=not self.issues,
            study_eligible=bool(not self.issues and self.study_eligible),
            issues=tuple(self.issues),
            **self.counts,
        )


def _scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset_scalar(dataset: Any, index: int) -> Any:
    return _scalar(dataset[index])


def _strings(dataset: Any) -> list[str]:
    if dataset is None:
        return []
    values = dataset.asstr()[()] if hasattr(dataset, "asstr") else dataset[()]
    return [str(_scalar(value)) for value in np.asarray(values).reshape(-1)]


def _int(value: Any) -> int:
    try:
        return int(_scalar(value))
    except (TypeError, ValueError):
        return -1


def _float(value: Any) -> float:
    try:
        return float(_scalar(value))
    except (TypeError, ValueError):
        return math.nan


def _finite_vector(value: Any, length: int) -> bool:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(array.shape == (int(length),) and np.all(np.isfinite(array)))


def _bool(value: Any) -> bool:
    return _int(value) == 1
