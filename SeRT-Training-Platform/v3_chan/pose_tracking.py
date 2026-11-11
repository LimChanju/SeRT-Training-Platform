"""Isaac-independent pose provenance and source-stability helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


TRACKING_UNKNOWN = -1
TRACKING_NOT_TRACKED = 0
TRACKING_TRACKED = 1

POSE_SOURCE_IDS = {
    "missing": 0,
    "xr_physical": 1,
    "xr_raw_physical": 2,
    "xr_virtual_world": 3,
    "xr_stage_visual": 4,
    "openxr_joint": 5,
    "external_hand_tracking": 6,
    "hmd_virtual_world": 7,
    "synthetic_head_fallback": 8,
    "hmd_xr_physical": 9,
}


def pose_source_id(source_name: str) -> int:
    return int(POSE_SOURCE_IDS.get(str(source_name), 0))


def _vec_or_none(value, size: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < size or not np.all(np.isfinite(array[:size])):
        return None
    return array[:size].copy()


@dataclass(frozen=True)
class PoseSample:
    """One world-space XR pose with explicit validity and provenance."""

    position_world: np.ndarray | None = None
    orientation_wxyz: np.ndarray | None = None
    pose_valid: bool = False
    position_tracked: int = TRACKING_UNKNOWN
    tracking_status_known: bool = False
    source_name: str = "missing"
    source_path: str = ""
    acquisition_monotonic_ns: int = 0
    pose_age_ms: float = -1.0
    source_switched: bool = False

    def __post_init__(self) -> None:
        position = _vec_or_none(self.position_world, 3)
        orientation = _vec_or_none(self.orientation_wxyz, 4)
        if orientation is not None:
            norm = float(np.linalg.norm(orientation))
            orientation = orientation / norm if norm > 1e-9 else None
        valid = bool(self.pose_valid and position is not None)
        tracked = int(self.position_tracked)
        if tracked not in (TRACKING_UNKNOWN, TRACKING_NOT_TRACKED, TRACKING_TRACKED):
            tracked = TRACKING_UNKNOWN
        object.__setattr__(self, "position_world", position)
        object.__setattr__(self, "orientation_wxyz", orientation)
        object.__setattr__(self, "pose_valid", valid)
        object.__setattr__(self, "position_tracked", tracked)
        object.__setattr__(
            self,
            "tracking_status_known",
            bool(self.tracking_status_known and tracked != TRACKING_UNKNOWN),
        )
        object.__setattr__(self, "source_name", str(self.source_name or "missing"))
        object.__setattr__(self, "source_path", str(self.source_path or ""))
        object.__setattr__(self, "acquisition_monotonic_ns", int(self.acquisition_monotonic_ns))
        object.__setattr__(self, "pose_age_ms", float(self.pose_age_ms))

    @property
    def source_id(self) -> int:
        return pose_source_id(self.source_name)

    @property
    def source_key(self) -> tuple[str, str]:
        return self.source_name, self.source_path

    @classmethod
    def invalid(
        cls,
        *,
        acquisition_monotonic_ns: int = 0,
        source_name: str = "missing",
        source_path: str = "",
    ) -> "PoseSample":
        return cls(
            pose_valid=False,
            source_name=source_name,
            source_path=source_path,
            acquisition_monotonic_ns=acquisition_monotonic_ns,
        )

    def invalidated(self) -> "PoseSample":
        return replace(self, position_world=None, pose_valid=False, source_switched=False)


class PoseSourceLatch:
    """Require a source to persist before accepting a provenance transition.

    A candidate from the current source passes immediately. During a source
    change, samples are marked invalid until the new source has appeared for
    ``confirmation_frames`` consecutive updates. This prevents a finite
    difference from interpreting a one-frame source jump as hand velocity.
    """

    def __init__(self, confirmation_frames: int = 3) -> None:
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.reset()

    def reset(self) -> None:
        self._current_key: tuple[str, str] | None = None
        self._pending_key: tuple[str, str] | None = None
        self._pending_count = 0

    @property
    def current_key(self) -> tuple[str, str] | None:
        return self._current_key

    def update(self, candidate: PoseSample) -> PoseSample:
        if not candidate.pose_valid:
            self._pending_key = None
            self._pending_count = 0
            return candidate.invalidated()

        key = candidate.source_key
        if self._current_key is None:
            self._current_key = key
            return replace(candidate, source_switched=False)
        if key == self._current_key:
            self._pending_key = None
            self._pending_count = 0
            return replace(candidate, source_switched=False)

        if key == self._pending_key:
            self._pending_count += 1
        else:
            self._pending_key = key
            self._pending_count = 1
        if self._pending_count < self.confirmation_frames:
            return candidate.invalidated()

        self._current_key = key
        self._pending_key = None
        self._pending_count = 0
        return replace(candidate, source_switched=True)
