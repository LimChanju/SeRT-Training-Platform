"""Frozen schema and value objects for online BC+CBF explicit feedback v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


ONLINE_SCHEMA_VERSION = "bc_cbf_online_explicit_feedback_v1"
ONLINE_PROTOCOL_VERSION = "single_pick_controlled_crossing_explicit_feedback_v1"
ONLINE_ROW_SEMANTICS = "obs_t_nominal_t_cbf_t_applied_t_obs_t_plus_1_v1"
FEATURE_PROVENANCE_VERSION = "runtime_observable_vs_future_outcome_v1"

TASK_PHASES = ("reach_approach", "grasp_lift", "transport", "place_release")
SEVERITIES = ("control", "shallow", "threat")
DIRECTIONS = ("left_to_right", "right_to_left")
SPEEDS = ("slow", "fast")
Q1_ID = "reuse_robot_response_needs_modification_v1"
Q1_TEXT_KO = "같은 상황에서 방금 로봇 반응을 다시 사용한다면 수정이 필요합니까?"
Q1_RESPONSES = ("needs_modification", "acceptable_as_is", "uncertain")
LIKERT_QUESTIONS = {
    "q2_collision_risk": "방금 로봇 반응에서 손과 충돌하거나 지나치게 가까워질 위험을 얼마나 느꼈습니까?",
    "q3_excessive_avoidance": "회피의 크기나 지속시간이 필요 이상으로 과도했다고 느꼈습니까?",
    "q4_abrupt_unpredictable": "로봇 움직임이 갑작스럽거나 예측하기 어렵다고 느꼈습니까?",
    "q5_task_disruption": "방금 회피가 grasp/place 등 원래 작업 진행을 얼마나 방해했다고 느꼈습니까?",
    "q6_confidence": "방금 판단에 얼마나 확신합니까?",
}
REJECTION_REASONS = (
    "too_close_or_late",
    "excessive_retreat_or_detour",
    "abrupt_or_irregular",
    "inappropriate_avoidance_direction",
    "excessive_stop_or_deadlock",
    "task_manipulation_disruption",
    "other",
)
MARKER_TYPES = ("realtime_safety_concern", "realtime_behavior_anomaly")

RUNTIME_OBSERVABLE_FIELDS = (
    "current_gap",
    "current_ttc",
    "closing_speed",
    "current_task_phase",
    "hand_position_velocity",
    "robot_state",
    "nominal_command",
    "cbf_state_so_far",
)
FUTURE_OUTCOME_FIELDS = (
    "final_task_success",
    "final_recovery_time",
    "total_intervention_duration",
    "future_minimum_gap",
    "future_maximum_jerk",
    "eventual_object_drop",
    "post_encounter_outcome",
)


class OnlineProtocolViolation(RuntimeError):
    """A fail-closed invariant of the online protocol was violated."""


@dataclass(frozen=True)
class TrialPlan:
    trial_index: int
    trial_id: str
    query_id: str
    task_phase: str
    severity: str
    direction: str
    speed: str
    crossing_hand: str
    practice: bool = False
    anchor_repeat: bool = False
    pilot: bool = False
    analysis_exclude: bool = False
    source_trial_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RealtimeMarkerRecord:
    marker_id: str
    trial_id: str
    marker_type: str
    simulation_time_s: float
    monotonic_ns: int
    unix_ns: int
    control_step: int
    controller_hand: str
    button: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    trial_id: str
    encounter_id: str
    issued_simulation_time_s: float
    issued_monotonic_ns: int
    issued_unix_ns: int
    completed_simulation_time_s: float
    completed_monotonic_ns: int
    completed_unix_ns: int
    response_status: str
    q1_response: str = ""
    q2_collision_risk: int = 0
    q3_excessive_avoidance: int = 0
    q4_abrupt_unpredictable: int = 0
    q5_task_disruption: int = 0
    q6_confidence: int = 0
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    response_latency_ms: float = 0.0
    input_source: str = "vr_controller"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rejection_reasons_json"] = _compact_json(self.rejection_reasons)
        del result["rejection_reasons"]
        return result


@dataclass(frozen=True)
class EncounterRecordV1:
    encounter_id: str
    trial_id: str
    onset_step: int
    onset_confirmed_step: int
    offset_step_exclusive: int
    onset_simulation_time_s: float
    offset_simulation_time_s: float
    onset_monotonic_ns: int
    offset_monotonic_ns: int
    window_start_step: int
    window_end_step_exclusive: int
    minimum_surface_gap_m: float
    maximum_intervention_norm_rad_s: float
    risk_onset_step: int
    cbf_intervention_start_step: int
    cbf_intervention_confirmed_step: int
    encounter_timeout: bool
    merged_reentry_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def exact_bool(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise OnlineProtocolViolation(f"{name} must be an exact boolean")


def validate_query_record(record: QueryRecord) -> None:
    if record.response_status not in {"completed", "timeout", "no_response"}:
        raise OnlineProtocolViolation("invalid query response_status")
    if record.response_status == "completed":
        if record.q1_response not in Q1_RESPONSES:
            raise OnlineProtocolViolation("completed query requires a valid Q1 response")
        ratings = (
            record.q2_collision_risk,
            record.q3_excessive_avoidance,
            record.q4_abrupt_unpredictable,
            record.q5_task_disruption,
            record.q6_confidence,
        )
        if any(int(value) not in range(1, 6) for value in ratings):
            raise OnlineProtocolViolation("completed query requires five separate 1-5 ratings")
    else:
        if record.q1_response or any(
            int(value)
            for value in (
                record.q2_collision_risk,
                record.q3_excessive_avoidance,
                record.q4_abrupt_unpredictable,
                record.q5_task_disruption,
                record.q6_confidence,
            )
        ):
            raise OnlineProtocolViolation("missing/timeout response cannot carry inferred labels")
    reasons = tuple(str(value) for value in record.rejection_reasons)
    if len(set(reasons)) != len(reasons) or any(
        reason not in REJECTION_REASONS for reason in reasons
    ):
        raise OnlineProtocolViolation("invalid or duplicate rejection reason")
    if record.q1_response == "needs_modification" and not reasons:
        raise OnlineProtocolViolation("needs_modification requires a rejection reason")
    if record.q1_response != "needs_modification" and reasons:
        raise OnlineProtocolViolation("rejection reasons are only valid for needs_modification")


def mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _compact_json(values: Sequence[Any]) -> str:
    import json

    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
