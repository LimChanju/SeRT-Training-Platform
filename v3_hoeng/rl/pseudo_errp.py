from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


PSEUDO_ERRP_SOURCE_CODES: dict[str, int] = {
    "human_robot_collision": 1 << 0,
    "near_human": 1 << 1,
    "collision_green": 1 << 2,
    "pick_miss_recent": 1 << 3,
    "drop_throw_recent": 1 << 4,
    "gripper_camera_occluded": 1 << 5,
    "external_feedback": 1 << 6,
}
DEFAULT_PSEUDO_ERRP_SOURCES: tuple[str, ...] = (
    "human_robot_collision",
    "near_human",
    "collision_green",
    "pick_miss_recent",
    "drop_throw_recent",
    "gripper_camera_occluded",
)
DEFAULT_ERRP_LABEL_THRESHOLD = 0.5
DEFAULT_NEAR_HUMAN_SAFE_DIST_M = 0.12
DEFAULT_NEAR_HUMAN_COLLISION_DIST_M = 0.03
DEFAULT_SOURCE_SEVERITY: dict[str, float] = {
    "human_robot_collision": 1.0,
    "collision_green": 0.7,
    "pick_miss_recent": 0.45,
    "drop_throw_recent": 0.8,
}

_AUX_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "gripper_camera_occluded": (
        "gripper_camera_occluded",
        "camera_occluded",
        "occluded",
        "occlusion",
    ),
}


@dataclass(frozen=True)
class PseudoErrPResult:
    feedback: float
    uncertainty: float
    label: int
    source_code: int
    source_names: tuple[str, ...]
    flags: dict[str, bool]
    source_scores: dict[str, float]


def parse_pseudo_errp_sources(value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Parse a source list while keeping source names stable for logs/checkpoints."""

    if value is None:
        return DEFAULT_PSEUDO_ERRP_SOURCES
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"all", "*"}:
            return DEFAULT_PSEUDO_ERRP_SOURCES
        if text.lower() in {"none", "off", "false", "0"}:
            return ()
        parts = [part.strip() for part in text.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in value if str(part).strip()]

    unknown = [part for part in parts if part not in PSEUDO_ERRP_SOURCE_CODES]
    if unknown:
        known = ", ".join(PSEUDO_ERRP_SOURCE_CODES)
        raise ValueError(f"Unknown pseudo-ErrP source(s): {unknown}. Known sources: {known}")
    return tuple(dict.fromkeys(parts))


def extract_pseudo_errp_aux_flags(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Split non-observation pseudo-ErrP flags out of a human_state_fn payload.

    The returned observation payload is safe to pass into build_observation().
    Auxiliary flags keep future signals, such as gripper-camera occlusion, out of
    the policy observation dimension until we intentionally version the obs schema.
    """

    clean = dict(payload)
    aux_flags: dict[str, float] = {}
    for source_name, aliases in _AUX_SOURCE_ALIASES.items():
        for alias in aliases:
            if alias in clean:
                aux_flags[source_name] = _score01(clean.pop(alias))
                break
    return clean, aux_flags


def pseudo_errp_from_observation(
    obs: Mapping[str, np.ndarray],
    *,
    aux_flags: Mapping[str, Any] | None = None,
    enabled: bool = True,
    sources: tuple[str, ...] | list[str] | str | None = None,
    override_feedback: float | None = None,
    label_threshold: float = DEFAULT_ERRP_LABEL_THRESHOLD,
) -> PseudoErrPResult:
    selected_sources = parse_pseudo_errp_sources(sources)
    source_scores = _source_scores(obs, aux_flags=aux_flags)
    flags = {name: score > 0.0 for name, score in source_scores.items()}
    label_threshold = float(np.clip(label_threshold, 0.0, 1.0))

    if override_feedback is not None:
        feedback = float(np.clip(float(override_feedback), 0.0, 1.0))
        source_scores["external_feedback"] = feedback
        flags["external_feedback"] = feedback > 0.0
        source_names = ("external_feedback",) if feedback > 0.0 else ()
        source_code = _source_code(source_names)
        return PseudoErrPResult(
            feedback=feedback,
            uncertainty=_probability_uncertainty(feedback),
            label=int(feedback >= label_threshold),
            source_code=source_code,
            source_names=source_names,
            flags=flags,
            source_scores=source_scores,
        )

    if not enabled:
        return PseudoErrPResult(
            feedback=0.0,
            uncertainty=0.0,
            label=0,
            source_code=0,
            source_names=(),
            flags=flags,
            source_scores=source_scores,
        )

    selected_scores = {
        name: float(source_scores.get(name, 0.0))
        for name in selected_sources
    }
    active_sources = tuple(name for name, score in selected_scores.items() if score > 0.0)
    feedback = _combine_probabilities(selected_scores.values())
    return PseudoErrPResult(
        feedback=feedback,
        uncertainty=_probability_uncertainty(feedback),
        label=int(feedback >= label_threshold),
        source_code=_source_code(active_sources),
        source_names=active_sources,
        flags=flags,
        source_scores=source_scores,
    )


def _source_scores(
    obs: Mapping[str, np.ndarray],
    *,
    aux_flags: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    scores = {
        "human_robot_collision": _binary_source_score(obs, "human_robot_collision"),
        "near_human": _near_human_score(obs),
        "collision_green": _binary_source_score(obs, "collision_green"),
        "pick_miss_recent": _binary_source_score(obs, "pick_miss_recent"),
        "drop_throw_recent": _binary_source_score(obs, "drop_throw_recent"),
    }
    for name, value in dict(aux_flags or {}).items():
        if name in PSEUDO_ERRP_SOURCE_CODES:
            scores[name] = _score01(value)
    for name in DEFAULT_PSEUDO_ERRP_SOURCES:
        scores.setdefault(name, 0.0)
    return {name: float(np.clip(score, 0.0, 1.0)) for name, score in scores.items()}


def _binary_source_score(obs: Mapping[str, np.ndarray], name: str) -> float:
    if not _obs_flag(obs, name):
        return 0.0
    return float(DEFAULT_SOURCE_SEVERITY.get(name, 1.0))


def _near_human_score(obs: Mapping[str, np.ndarray]) -> float:
    near_flag = _obs_flag(obs, "near_human")
    dist_value = obs.get("min_hand_gripper_dist")
    if dist_value is None:
        return 0.35 if near_flag else 0.0

    arr = np.asarray(dist_value, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.35 if near_flag else 0.0
    dist = float(arr[0])
    if not np.isfinite(dist) or dist >= 1.0:
        return 0.35 if near_flag else 0.0

    safe_dist = DEFAULT_NEAR_HUMAN_SAFE_DIST_M
    collision_dist = DEFAULT_NEAR_HUMAN_COLLISION_DIST_M
    if dist <= collision_dist:
        score = 1.0
    elif dist >= safe_dist:
        score = 0.0
    else:
        score = (safe_dist - dist) / max(safe_dist - collision_dist, 1e-6)
    if near_flag:
        score = max(score, 0.35)
    return float(np.clip(score, 0.0, 1.0))


def _obs_flag(obs: Mapping[str, np.ndarray], name: str) -> bool:
    value = obs.get(name)
    if value is None:
        return False
    arr = np.asarray(value, dtype=float).reshape(-1)
    return bool(arr.size and float(arr[0]) > 0.5)


def _source_code(source_names: tuple[str, ...] | list[str]) -> int:
    code = 0
    for name in source_names:
        code |= int(PSEUDO_ERRP_SOURCE_CODES.get(name, 0))
    return code


def _combine_probabilities(scores) -> float:
    probability_no_errp = 1.0
    for score in scores:
        probability_no_errp *= 1.0 - float(np.clip(score, 0.0, 1.0))
    return float(np.clip(1.0 - probability_no_errp, 0.0, 1.0))


def _probability_uncertainty(probability: float) -> float:
    probability = float(np.clip(probability, 0.0, 1.0))
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(np.clip(1.0 - abs(probability - 0.5) * 2.0, 0.0, 1.0))


def _score01(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "on"}:
            return 1.0
        if text in {"false", "no", "n", "off"}:
            return 0.0
        try:
            return float(np.clip(float(text), 0.0, 1.0))
        except ValueError:
            return 0.0
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return 0.0
    try:
        return float(np.clip(float(arr[0]), 0.0, 1.0))
    except (TypeError, ValueError):
        return 1.0 if bool(arr[0]) else 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return False
    try:
        return bool(float(arr[0]) > 0.5)
    except (TypeError, ValueError):
        return bool(arr[0])
