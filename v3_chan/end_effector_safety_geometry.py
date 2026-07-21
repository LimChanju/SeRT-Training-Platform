"""Shared, Isaac-independent definitions for distal Panda safety geometry."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


DISTAL_LINK_NAMES = (
    "panda_link6",
    "panda_link7",
    "panda_link8",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
)
LINK_ID_BY_NAME = {name: index + 1 for index, name in enumerate(DISTAL_LINK_NAMES)}
LINK_NAME_BY_ID = {value: key for key, value in LINK_ID_BY_NAME.items()}

MISSING_SURFACE_GAP_M = 10.0


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


@dataclass(frozen=True)
class SafetyThresholds:
    """All distances are signed surface gaps in meters."""

    hand_radius_m: float = 0.035
    collision_gap_m: float = 0.0
    near_miss_gap_m: float = 0.02
    near_gap_m: float = 0.05
    gate_full_gap_m: float = 0.05
    gate_start_gap_m: float = 0.13
    max_query_gap_m: float = 2.0
    query_tolerance_m: float = 0.00025
    query_iterations: int = 14

    @classmethod
    def from_env(cls) -> "SafetyThresholds":
        return cls(
            hand_radius_m=max(
                0.001, _env_float("HRI_HAND_PROXY_RADIUS_M", cls.hand_radius_m)
            ),
            collision_gap_m=_env_float(
                "HRI_COLLISION_SURFACE_GAP_M", cls.collision_gap_m
            ),
            near_miss_gap_m=_env_float(
                "HRI_NEAR_MISS_SURFACE_GAP_M", cls.near_miss_gap_m
            ),
            near_gap_m=_env_float(
                "HRI_NEAR_HUMAN_SURFACE_GAP_M", cls.near_gap_m
            ),
            gate_full_gap_m=_env_float(
                "HRI_DISTANCE_GATE_FULL_GAP_M", cls.gate_full_gap_m
            ),
            gate_start_gap_m=_env_float(
                "HRI_DISTANCE_GATE_START_GAP_M", cls.gate_start_gap_m
            ),
            max_query_gap_m=max(
                0.13, _env_float("HRI_GEOMETRY_MAX_QUERY_GAP_M", cls.max_query_gap_m)
            ),
            query_tolerance_m=max(
                1e-5,
                _env_float("HRI_GEOMETRY_QUERY_TOLERANCE_M", cls.query_tolerance_m),
            ),
            query_iterations=max(
                4,
                int(_env_float("HRI_GEOMETRY_QUERY_ITERATIONS", cls.query_iterations)),
            ),
        ).validated()

    def validated(self) -> "SafetyThresholds":
        if self.near_miss_gap_m < self.collision_gap_m:
            raise ValueError("near_miss_gap_m must be >= collision_gap_m")
        if self.near_gap_m < self.near_miss_gap_m:
            raise ValueError("near_gap_m must be >= near_miss_gap_m")
        if self.gate_start_gap_m <= self.gate_full_gap_m:
            raise ValueError("gate_start_gap_m must be > gate_full_gap_m")
        if self.max_query_gap_m < self.gate_start_gap_m:
            raise ValueError("max_query_gap_m must be >= gate_start_gap_m")
        return self


DEFAULT_THRESHOLDS = SafetyThresholds.from_env()


@dataclass(frozen=True)
class SafetyClassification:
    collision: bool
    near_miss: bool
    near: bool
    distance_gate: float


@dataclass(frozen=True)
class HandSafetyResult:
    hand: str
    geometry_valid: bool
    surface_gap_m: float = MISSING_SURFACE_GAP_M
    closest_link: str = ""
    closest_link_id: int = 0
    closest_collider_path: str = ""
    closest_collider_id: int = 0
    contact: bool = False
    collision: bool = False
    contact_force_n: float = 0.0
    contact_force_valid: bool = False
    penetration_m: float = 0.0
    near_miss: bool = False
    near: bool = False
    distance_gate: float = 0.0
    query_time_ms: float = 0.0
    query_count: int = 0


@dataclass(frozen=True)
class EndEffectorSafetyResult:
    left: HandSafetyResult
    right: HandSafetyResult

    @property
    def geometry_valid(self) -> bool:
        return self.left.geometry_valid or self.right.geometry_valid

    @property
    def collision(self) -> bool:
        return self.left.collision or self.right.collision

    @property
    def contact(self) -> bool:
        return self.left.contact or self.right.contact

    @property
    def near_miss(self) -> bool:
        return self.left.near_miss or self.right.near_miss

    @property
    def near(self) -> bool:
        return self.left.near or self.right.near

    @property
    def min_surface_gap_m(self) -> float:
        valid = [
            result.surface_gap_m
            for result in (self.left, self.right)
            if result.geometry_valid and math.isfinite(result.surface_gap_m)
        ]
        return min(valid) if valid else MISSING_SURFACE_GAP_M

    @property
    def distance_gate(self) -> float:
        return max(self.left.distance_gate, self.right.distance_gate)

    @property
    def closest_result(self) -> HandSafetyResult | None:
        valid = [
            result
            for result in (self.left, self.right)
            if result.geometry_valid and result.surface_gap_m < MISSING_SURFACE_GAP_M
        ]
        return min(valid, key=lambda result: result.surface_gap_m) if valid else None

    @property
    def closest_human_hand(self) -> str:
        result = self.closest_result
        return result.hand if result is not None else ""

    @property
    def closest_human_hand_id(self) -> int:
        return {"left": 1, "right": 2}.get(self.closest_human_hand, 0)

    @property
    def closest_robot_link(self) -> str:
        result = self.closest_result
        return result.closest_link if result is not None else ""

    @property
    def closest_robot_link_id(self) -> int:
        result = self.closest_result
        return result.closest_link_id if result is not None else 0

    @property
    def closest_collider_path(self) -> str:
        result = self.closest_result
        return result.closest_collider_path if result is not None else ""

    @property
    def closest_collider_id(self) -> int:
        result = self.closest_result
        return result.closest_collider_id if result is not None else 0

    @property
    def penetration_depth_m(self) -> float:
        result = self.closest_result
        return result.penetration_m if result is not None else 0.0


def distance_gate(
    surface_gap_m: float,
    thresholds: SafetyThresholds = DEFAULT_THRESHOLDS,
) -> float:
    """Return the residual-safety blend gate for a signed surface gap."""

    gap = float(surface_gap_m)
    if not math.isfinite(gap) or gap >= MISSING_SURFACE_GAP_M:
        return 0.0
    denominator = thresholds.gate_start_gap_m - thresholds.gate_full_gap_m
    value = (thresholds.gate_start_gap_m - gap) / denominator
    return float(min(1.0, max(0.0, value)))


def classify_surface_gap(
    surface_gap_m: float,
    thresholds: SafetyThresholds = DEFAULT_THRESHOLDS,
    *,
    contact: bool = False,
    geometry_valid: bool = True,
) -> SafetyClassification:
    if not geometry_valid or not math.isfinite(float(surface_gap_m)):
        return SafetyClassification(False, False, False, 0.0)
    gap = float(surface_gap_m)
    collision = bool(contact or gap <= thresholds.collision_gap_m)
    near_miss = bool(
        not collision
        and thresholds.collision_gap_m < gap <= thresholds.near_miss_gap_m
    )
    near = bool(gap <= thresholds.near_gap_m)
    return SafetyClassification(
        collision=collision,
        near_miss=near_miss,
        near=near,
        distance_gate=distance_gate(gap, thresholds),
    )
