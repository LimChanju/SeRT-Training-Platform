"""Frozen identifiers and small value objects for the v1 collection protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


COLLECTION_SCHEMA_VERSION = "bc_cbf_explicit_feedback_collection_v1"
ENCOUNTER_SCHEMA_VERSION = "bc_cbf_human_robot_encounter_v1"
EXPLICIT_FEEDBACK_SCHEMA_VERSION = "explicit_safety_acceptability_v1"
ACTION_TRACE_SCHEMA_VERSION = "bc_cbf_articulation_action_trace_v1"
CLOCK_SCHEMA_VERSION = "server_monotonic_and_unix_ns_v1"
ROW_SEMANTICS = "transition_obs_t_action_t_obs_t_plus_1_v1"

RUNTIME_CONTRACT_SCHEMA_VERSION = "bc_cbf_runtime_handoff_v1"
RUNTIME_CONTRACT_SHA256 = (
    "df1d0cc30a25b0b6394e83f5ca8403fb8c2efe11264f543f972c0c5d9dbabd71"
)
RUNTIME_HANDOFF_COMMIT = "3682056fd8888d2881290f1e1f65f3cce10270d9"
RUNTIME_BASE_COMMIT = "c4ce195f4942dbbe21e0090b6b28bd379a022d11"
POLICY_SHA256 = "fc4801dfdd7ae4e3edf8760277a1d87fd540c12bf27aeb0a06a3eefdbd5ba8c3"
POLICY_SIZE_BYTES = 366_976
POLICY_RELATIVE_PATH = "v3_chan/policies/bc_pick_place_v1_100eps.pt"

QUESTION_ID = "recent_robot_motion_safe_and_acceptable_v1"
QUESTION_TEXT_KO = "방금 로봇 동작은 안전하고 수용 가능했나요?"
QUESTION_TEXT_EN = "Was the robot's recent motion safe and acceptable?"
RESPONSE_CODES = (
    "acceptable",
    "unacceptable",
    "unsure",
)
RESPONSE_LABELS_KO = {
    "acceptable": "안전하고 수용 가능함",
    "unacceptable": "안전하지 않거나 수용하기 어려움",
    "unsure": "잘 모르겠음",
}


class ProtocolViolation(RuntimeError):
    """A collection invariant was violated and collection must fail closed."""


@dataclass(frozen=True)
class EncounterRecord:
    encounter_id: str
    episode_index: int
    start_step: int
    end_step_exclusive: int
    start_monotonic_ns: int
    end_monotonic_ns: int
    minimum_surface_gap_m: float
    involved_hand_mask: tuple[int, int]
    eligible: bool
    eligibility_reason: str
    terminal_status: str
    abort_reason: str = ""
    prompt_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    encounter_id: str
    question_id: str
    issued_monotonic_ns: int
    issued_unix_ns: int
    presentation_deadline_monotonic_ns: int
    presented_monotonic_ns: int
    deadline_monotonic_ns: int
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResponseRecord:
    response_id: str
    client_response_id: str
    prompt_id: str
    response_code: str
    source: str
    client_unix_ns: int
    received_monotonic_ns: int
    received_unix_ns: int
    accepted: bool
    rejection_reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def exact_binary(value: Any, *, name: str) -> int:
    """Return an exact 0/1 value without truthy coercion."""

    if isinstance(value, bool):
        return int(value)
    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise ProtocolViolation(f"{name} must be exactly 0 or 1") from error
    if numeric not in (0, 1) or str(value).strip() not in {"0", "1"}:
        raise ProtocolViolation(f"{name} must be exactly 0 or 1")
    return numeric


def mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
