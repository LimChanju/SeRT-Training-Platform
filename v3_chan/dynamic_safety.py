"""Isaac-independent dynamic safety features for recorded HRI trajectories."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np


DEFAULT_MISSING_SURFACE_GAP_M = 10.0


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def valid_world_position(value) -> bool:
    """Return whether a tracked world-space point can seed a velocity estimate."""

    if value is None:
        return False
    arr = np.asarray(value, dtype=float).reshape(-1)
    return bool(
        arr.size >= 3
        and np.all(np.isfinite(arr[:3]))
        and np.linalg.norm(arr[:3]) > 1e-6
    )


def _vec3_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3].copy()


def _quat_wxyz_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    quat = np.asarray(value, dtype=float).reshape(-1)
    if quat.size < 4 or not np.all(np.isfinite(quat[:4])):
        return None
    quat = quat[:4].copy()
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        return None
    return quat / norm


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def _angular_velocity_world_wxyz(
    current: np.ndarray, previous: np.ndarray, dt_s: float
) -> np.ndarray:
    if float(np.dot(current, previous)) < 0.0:
        current = -current
    previous_conjugate = previous.copy()
    previous_conjugate[1:] *= -1.0
    delta = _quat_multiply_wxyz(current, previous_conjugate)
    delta /= max(float(np.linalg.norm(delta)), 1e-12)
    if delta[0] < 0.0:
        delta = -delta
    sin_half = float(np.linalg.norm(delta[1:]))
    if sin_half <= 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(sin_half, float(np.clip(delta[0], -1.0, 1.0)))
    return delta[1:] * (angle / (sin_half * dt_s))


@dataclass(frozen=True)
class DynamicSafetyConfig:
    """Configuration for sim-time finite differences and EMA filtering."""

    ema_time_constant_s: float = 0.1
    ttc_cap_s: float = 10.0
    min_valid_closing_speed_mps: float = 0.01
    min_valid_dt_s: float = 1e-6
    max_valid_dt_s: float = 0.25
    missing_surface_gap_m: float = DEFAULT_MISSING_SURFACE_GAP_M

    @classmethod
    def from_env(cls) -> "DynamicSafetyConfig":
        return cls(
            ema_time_constant_s=max(
                1e-4,
                _env_float("HRI_DYNAMIC_EMA_TIME_CONSTANT_S", cls.ema_time_constant_s),
            ),
            ttc_cap_s=max(0.01, _env_float("HRI_DYNAMIC_TTC_CAP_S", cls.ttc_cap_s)),
            min_valid_closing_speed_mps=max(
                0.0,
                _env_float(
                    "HRI_DYNAMIC_MIN_VALID_CLOSING_SPEED_MPS",
                    cls.min_valid_closing_speed_mps,
                ),
            ),
            min_valid_dt_s=max(
                1e-9, _env_float("HRI_DYNAMIC_MIN_VALID_DT_S", cls.min_valid_dt_s)
            ),
            max_valid_dt_s=max(
                1e-4, _env_float("HRI_DYNAMIC_MAX_VALID_DT_S", cls.max_valid_dt_s)
            ),
            missing_surface_gap_m=max(
                0.1,
                _env_float(
                    "HRI_DYNAMIC_MISSING_SURFACE_GAP_M", cls.missing_surface_gap_m
                ),
            ),
        ).validated()

    def validated(self) -> "DynamicSafetyConfig":
        if self.max_valid_dt_s <= self.min_valid_dt_s:
            raise ValueError("max_valid_dt_s must be greater than min_valid_dt_s")
        return self

    def metadata(self) -> dict[str, object]:
        return {
            "dynamic_safety_schema_version": "dynamic_safety_v2_surface_point_twist",
            "dynamic_time_source": "simulation_time",
            "dynamic_hand_velocity_filter": "dt_based_ema",
            "dynamic_gap_rate_filter": "dt_based_ema",
            "dynamic_ema_time_constant_s": float(self.ema_time_constant_s),
            "dynamic_ttc_cap_s": float(self.ttc_cap_s),
            "dynamic_min_valid_closing_speed_mps": float(
                self.min_valid_closing_speed_mps
            ),
            "dynamic_min_valid_dt_s": float(self.min_valid_dt_s),
            "dynamic_max_valid_dt_s": float(self.max_valid_dt_s),
            "dynamic_invalid_value_encoding": "zero_for_velocity_rate_and_closing_cap_for_ttc",
            "dynamic_robot_velocity_method": "closest_surface_point_rigid_body_twist",
            "dynamic_robot_velocity_source": "link_origin_pose_finite_difference_plus_isaacsim_angular_velocity",
            "dynamic_closest_surface_point_source": "physx_attachment_get_closest_points",
            "dynamic_exact_closest_surface_point_velocity_available": True,
            "dynamic_angular_velocity_correction_used": True,
            "dynamic_surface_point_inside_collider_behavior": "invalid",
            "dynamic_collider_switch_reset_scope": "gap_rate_and_robot_link_pose_only",
            "dynamic_hand_velocity_preserved_across_collider_switch": True,
        }


@dataclass(frozen=True)
class HandDynamicSample:
    hand_velocity_raw_mps: np.ndarray
    hand_velocity_filtered_mps: np.ndarray
    hand_velocity_valid: bool
    closest_surface_point_world_pos: np.ndarray
    closest_robot_origin_velocity_world_mps: np.ndarray
    closest_robot_angular_velocity_world_radps: np.ndarray
    closest_robot_rotational_velocity_world_mps: np.ndarray
    closest_robot_velocity_world_mps: np.ndarray
    robot_surface_velocity_valid: bool
    relative_velocity_world_mps: np.ndarray
    surface_gap_rate_raw_mps: float
    surface_gap_rate_filtered_mps: float
    closing_speed_mps: float
    ttc_s: float
    dynamic_valid: bool
    closest_collider_switched: bool


@dataclass(frozen=True)
class DynamicSafetySample:
    left: HandDynamicSample
    right: HandDynamicSample
    min_ttc_s: float
    max_closing_speed_mps: float
    dynamic_valid: bool

    def human_payload(self) -> dict[str, object]:
        return {
            "left_hand_vel_raw_mps": self.left.hand_velocity_raw_mps,
            "right_hand_vel_raw_mps": self.right.hand_velocity_raw_mps,
            "left_hand_vel_filtered_mps": self.left.hand_velocity_filtered_mps,
            "right_hand_vel_filtered_mps": self.right.hand_velocity_filtered_mps,
            "left_hand_velocity_valid": float(self.left.hand_velocity_valid),
            "right_hand_velocity_valid": float(self.right.hand_velocity_valid),
        }

    def safety_payload(self) -> dict[str, object]:
        return {
            "left_closest_surface_point_world_pos": self.left.closest_surface_point_world_pos,
            "right_closest_surface_point_world_pos": self.right.closest_surface_point_world_pos,
            "left_closest_robot_origin_velocity_world_mps": self.left.closest_robot_origin_velocity_world_mps,
            "right_closest_robot_origin_velocity_world_mps": self.right.closest_robot_origin_velocity_world_mps,
            "left_closest_robot_angular_velocity_world_radps": self.left.closest_robot_angular_velocity_world_radps,
            "right_closest_robot_angular_velocity_world_radps": self.right.closest_robot_angular_velocity_world_radps,
            "left_closest_robot_rotational_velocity_world_mps": self.left.closest_robot_rotational_velocity_world_mps,
            "right_closest_robot_rotational_velocity_world_mps": self.right.closest_robot_rotational_velocity_world_mps,
            "left_closest_robot_velocity_world_mps": self.left.closest_robot_velocity_world_mps,
            "right_closest_robot_velocity_world_mps": self.right.closest_robot_velocity_world_mps,
            "left_robot_surface_velocity_valid": float(
                self.left.robot_surface_velocity_valid
            ),
            "right_robot_surface_velocity_valid": float(
                self.right.robot_surface_velocity_valid
            ),
            "left_relative_velocity_world_mps": self.left.relative_velocity_world_mps,
            "right_relative_velocity_world_mps": self.right.relative_velocity_world_mps,
            "left_surface_gap_rate_raw_mps": self.left.surface_gap_rate_raw_mps,
            "right_surface_gap_rate_raw_mps": self.right.surface_gap_rate_raw_mps,
            "left_surface_gap_rate_filtered_mps": self.left.surface_gap_rate_filtered_mps,
            "right_surface_gap_rate_filtered_mps": self.right.surface_gap_rate_filtered_mps,
            "left_closing_speed_mps": self.left.closing_speed_mps,
            "right_closing_speed_mps": self.right.closing_speed_mps,
            "left_ttc_s": self.left.ttc_s,
            "right_ttc_s": self.right.ttc_s,
            "left_dynamic_valid": float(self.left.dynamic_valid),
            "right_dynamic_valid": float(self.right.dynamic_valid),
            "min_ttc_s": self.min_ttc_s,
            "max_closing_speed_mps": self.max_closing_speed_mps,
            "dynamic_valid": float(self.dynamic_valid),
            "closest_collider_switched_left": float(
                self.left.closest_collider_switched
            ),
            "closest_collider_switched_right": float(
                self.right.closest_collider_switched
            ),
        }


class _HandDynamicEstimator:
    def __init__(self, config: DynamicSafetyConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._previous_time_s: float | None = None
        self._previous_hand_pos: np.ndarray | None = None
        self._previous_gap_m: float | None = None
        self._previous_robot_origin_pos: np.ndarray | None = None
        self._previous_robot_orientation_wxyz: np.ndarray | None = None
        self._previous_collider_id = 0
        self._filtered_hand_velocity: np.ndarray | None = None
        self._filtered_gap_rate_mps: float | None = None

    def update(
        self,
        *,
        sim_time_s: float,
        hand_pos,
        tracking_valid: bool,
        surface_gap_m: float,
        geometry_valid: bool,
        closest_collider_id: int,
        closest_robot_origin_pos,
        closest_robot_orientation_wxyz,
        closest_surface_point_world_pos,
        closest_robot_angular_velocity_world_radps,
    ) -> HandDynamicSample:
        pos = _vec3_or_none(hand_pos)
        sim_time_s = float(sim_time_s)
        if not bool(tracking_valid) or pos is None or not math.isfinite(sim_time_s):
            self.reset()
            return self._empty_sample()

        gap_valid = self._gap_is_valid(
            surface_gap_m, geometry_valid, closest_collider_id
        )
        robot_origin = _vec3_or_none(closest_robot_origin_pos)
        robot_orientation = _quat_wxyz_or_none(closest_robot_orientation_wxyz)
        surface_point = _vec3_or_none(closest_surface_point_world_pos)
        direct_angular_velocity = _vec3_or_none(
            closest_robot_angular_velocity_world_radps
        )
        if self._previous_time_s is None or self._previous_hand_pos is None:
            self._seed(
                sim_time_s,
                pos,
                surface_gap_m,
                gap_valid,
                closest_collider_id,
                robot_origin,
                robot_orientation,
            )
            return self._empty_sample(
                gap_m=surface_gap_m if gap_valid else None,
                closest_surface_point=surface_point,
            )

        dt_s = sim_time_s - self._previous_time_s
        if not self._valid_dt(dt_s):
            self.reset()
            self._seed(
                sim_time_s,
                pos,
                surface_gap_m,
                gap_valid,
                closest_collider_id,
                robot_origin,
                robot_orientation,
            )
            return self._empty_sample(
                gap_m=surface_gap_m if gap_valid else None,
                closest_surface_point=surface_point,
            )

        raw_hand_velocity = (pos - self._previous_hand_pos) / dt_s
        filtered_hand_velocity = self._ema_vector(
            raw_hand_velocity, self._filtered_hand_velocity, dt_s
        )
        self._previous_time_s = sim_time_s
        self._previous_hand_pos = pos
        self._filtered_hand_velocity = filtered_hand_velocity

        if not gap_valid:
            self._clear_gap_history()
            return self._empty_sample(
                raw_hand_velocity=raw_hand_velocity,
                filtered_hand_velocity=filtered_hand_velocity,
                hand_velocity_valid=True,
            )

        gap_m = float(surface_gap_m)
        collider_switched = bool(
            self._previous_collider_id > 0
            and int(closest_collider_id) != self._previous_collider_id
        )
        if collider_switched:
            self._seed_gap_history(
                gap_m, closest_collider_id, robot_origin, robot_orientation
            )
            return self._empty_sample(
                raw_hand_velocity=raw_hand_velocity,
                filtered_hand_velocity=filtered_hand_velocity,
                hand_velocity_valid=True,
                gap_m=gap_m,
                closest_surface_point=surface_point,
                closest_collider_switched=True,
            )

        if self._previous_gap_m is None:
            self._seed_gap_history(
                gap_m, closest_collider_id, robot_origin, robot_orientation
            )
            return self._empty_sample(
                raw_hand_velocity=raw_hand_velocity,
                filtered_hand_velocity=filtered_hand_velocity,
                hand_velocity_valid=True,
                gap_m=gap_m,
                closest_surface_point=surface_point,
            )

        raw_gap_rate = (gap_m - self._previous_gap_m) / dt_s
        filtered_gap_rate = self._ema_scalar(
            raw_gap_rate, self._filtered_gap_rate_mps, dt_s
        )
        self._previous_gap_m = gap_m
        self._previous_collider_id = int(closest_collider_id)
        self._filtered_gap_rate_mps = filtered_gap_rate

        origin_velocity = np.zeros(3, dtype=np.float32)
        angular_velocity = np.zeros(3, dtype=np.float32)
        rotational_velocity = np.zeros(3, dtype=np.float32)
        robot_velocity = np.zeros(3, dtype=np.float32)
        direct_angular_velocity_valid = direct_angular_velocity is not None
        fallback_angular_velocity_valid = bool(
            robot_orientation is not None
            and self._previous_robot_orientation_wxyz is not None
        )
        robot_velocity_valid = bool(
            robot_origin is not None
            and surface_point is not None
            and self._previous_robot_origin_pos is not None
            and (direct_angular_velocity_valid or fallback_angular_velocity_valid)
        )
        if robot_velocity_valid:
            origin_velocity = (
                (robot_origin - self._previous_robot_origin_pos) / dt_s
            ).astype(np.float32)
            if direct_angular_velocity_valid:
                angular_velocity = direct_angular_velocity.astype(np.float32)
            else:
                angular_velocity = _angular_velocity_world_wxyz(
                    robot_orientation,
                    self._previous_robot_orientation_wxyz,
                    dt_s,
                ).astype(np.float32)
            rotational_velocity = np.cross(
                angular_velocity, surface_point - robot_origin
            ).astype(np.float32)
            robot_velocity = (origin_velocity + rotational_velocity).astype(np.float32)
            robot_velocity_valid = bool(
                np.all(np.isfinite(origin_velocity))
                and np.all(np.isfinite(angular_velocity))
                and np.all(np.isfinite(rotational_velocity))
                and np.all(np.isfinite(robot_velocity))
            )
        self._previous_robot_origin_pos = robot_origin
        self._previous_robot_orientation_wxyz = robot_orientation

        closing_speed = max(0.0, -float(filtered_gap_rate))
        relative_velocity = np.zeros(3, dtype=np.float32)
        if robot_velocity_valid:
            relative_velocity = (filtered_hand_velocity - robot_velocity).astype(
                np.float32
            )

        dynamic_valid = bool(
            robot_velocity_valid
            and closing_speed >= self.config.min_valid_closing_speed_mps
        )
        if gap_m <= 0.0:
            ttc_s = 0.0
        elif dynamic_valid:
            ttc_s = min(self.config.ttc_cap_s, gap_m / closing_speed)
        else:
            ttc_s = self.config.ttc_cap_s

        return HandDynamicSample(
            hand_velocity_raw_mps=raw_hand_velocity.astype(np.float32),
            hand_velocity_filtered_mps=filtered_hand_velocity.astype(np.float32),
            hand_velocity_valid=True,
            closest_surface_point_world_pos=(
                surface_point.astype(np.float32)
                if surface_point is not None
                else np.zeros(3, dtype=np.float32)
            ),
            closest_robot_origin_velocity_world_mps=origin_velocity,
            closest_robot_angular_velocity_world_radps=angular_velocity,
            closest_robot_rotational_velocity_world_mps=rotational_velocity,
            closest_robot_velocity_world_mps=robot_velocity,
            robot_surface_velocity_valid=robot_velocity_valid,
            relative_velocity_world_mps=relative_velocity,
            surface_gap_rate_raw_mps=float(raw_gap_rate),
            surface_gap_rate_filtered_mps=float(filtered_gap_rate),
            closing_speed_mps=float(closing_speed),
            ttc_s=float(ttc_s),
            dynamic_valid=dynamic_valid,
            closest_collider_switched=False,
        )

    def _seed(
        self,
        sim_time_s: float,
        hand_pos: np.ndarray,
        gap_m: float,
        gap_valid: bool,
        collider_id: int,
        robot_origin: np.ndarray | None,
        robot_orientation: np.ndarray | None,
    ) -> None:
        self._previous_time_s = float(sim_time_s)
        self._previous_hand_pos = hand_pos.copy()
        if gap_valid:
            self._seed_gap_history(gap_m, collider_id, robot_origin, robot_orientation)

    def _seed_gap_history(
        self,
        gap_m: float,
        collider_id: int,
        robot_origin: np.ndarray | None,
        robot_orientation: np.ndarray | None,
    ) -> None:
        self._previous_gap_m = float(gap_m)
        self._previous_collider_id = int(collider_id)
        self._previous_robot_origin_pos = robot_origin
        self._previous_robot_orientation_wxyz = robot_orientation
        self._filtered_gap_rate_mps = None

    def _clear_gap_history(self) -> None:
        self._previous_gap_m = None
        self._previous_robot_origin_pos = None
        self._previous_robot_orientation_wxyz = None
        self._previous_collider_id = 0
        self._filtered_gap_rate_mps = None

    def _gap_is_valid(
        self, gap_m: float, geometry_valid: bool, closest_collider_id: int
    ) -> bool:
        try:
            gap_m = float(gap_m)
        except (TypeError, ValueError):
            return False
        return bool(
            geometry_valid
            and int(closest_collider_id) > 0
            and math.isfinite(gap_m)
            and gap_m < self.config.missing_surface_gap_m
        )

    def _valid_dt(self, dt_s: float) -> bool:
        return bool(
            math.isfinite(dt_s)
            and self.config.min_valid_dt_s < dt_s <= self.config.max_valid_dt_s
        )

    def _ema_vector(
        self, value: np.ndarray, previous: np.ndarray | None, dt_s: float
    ) -> np.ndarray:
        if previous is None:
            return value.astype(float, copy=True)
        alpha = 1.0 - math.exp(-dt_s / self.config.ema_time_constant_s)
        return alpha * value + (1.0 - alpha) * previous

    def _ema_scalar(self, value: float, previous: float | None, dt_s: float) -> float:
        if previous is None:
            return float(value)
        alpha = 1.0 - math.exp(-dt_s / self.config.ema_time_constant_s)
        return float(alpha * value + (1.0 - alpha) * previous)

    def _empty_sample(
        self,
        *,
        raw_hand_velocity: np.ndarray | None = None,
        filtered_hand_velocity: np.ndarray | None = None,
        hand_velocity_valid: bool = False,
        gap_m: float | None = None,
        closest_surface_point: np.ndarray | None = None,
        closest_collider_switched: bool = False,
    ) -> HandDynamicSample:
        zeros = np.zeros(3, dtype=np.float32)
        raw = (
            zeros
            if raw_hand_velocity is None
            else np.asarray(raw_hand_velocity, dtype=np.float32)
        )
        filtered = (
            zeros
            if filtered_hand_velocity is None
            else np.asarray(filtered_hand_velocity, dtype=np.float32)
        )
        ttc_s = (
            0.0 if gap_m is not None and float(gap_m) <= 0.0 else self.config.ttc_cap_s
        )
        return HandDynamicSample(
            hand_velocity_raw_mps=raw,
            hand_velocity_filtered_mps=filtered,
            hand_velocity_valid=bool(hand_velocity_valid),
            closest_surface_point_world_pos=(
                zeros.copy()
                if closest_surface_point is None
                else np.asarray(closest_surface_point, dtype=np.float32)
            ),
            closest_robot_origin_velocity_world_mps=zeros.copy(),
            closest_robot_angular_velocity_world_radps=zeros.copy(),
            closest_robot_rotational_velocity_world_mps=zeros.copy(),
            closest_robot_velocity_world_mps=zeros.copy(),
            robot_surface_velocity_valid=False,
            relative_velocity_world_mps=zeros.copy(),
            surface_gap_rate_raw_mps=0.0,
            surface_gap_rate_filtered_mps=0.0,
            closing_speed_mps=0.0,
            ttc_s=float(ttc_s),
            dynamic_valid=False,
            closest_collider_switched=bool(closest_collider_switched),
        )


class DynamicSafetyEstimator:
    """Two hand estimators driven only by simulation time and cached poses."""

    def __init__(self, config: DynamicSafetyConfig | None = None) -> None:
        self.config = (config or DynamicSafetyConfig.from_env()).validated()
        self._left = _HandDynamicEstimator(self.config)
        self._right = _HandDynamicEstimator(self.config)

    def reset(self) -> None:
        self._left.reset()
        self._right.reset()

    def metadata(self) -> dict[str, object]:
        return self.config.metadata()

    def update(
        self,
        *,
        sim_time_s: float,
        left_hand_pos,
        right_hand_pos,
        left_tracking_valid: bool,
        right_tracking_valid: bool,
        left_surface_gap_m: float,
        right_surface_gap_m: float,
        left_geometry_valid: bool,
        right_geometry_valid: bool,
        left_closest_collider_id: int,
        right_closest_collider_id: int,
        left_closest_robot_origin_pos,
        right_closest_robot_origin_pos,
        left_closest_robot_orientation_wxyz=None,
        right_closest_robot_orientation_wxyz=None,
        left_closest_surface_point_world_pos=None,
        right_closest_surface_point_world_pos=None,
        left_closest_robot_angular_velocity_world_radps=None,
        right_closest_robot_angular_velocity_world_radps=None,
    ) -> DynamicSafetySample:
        left = self._left.update(
            sim_time_s=sim_time_s,
            hand_pos=left_hand_pos,
            tracking_valid=left_tracking_valid,
            surface_gap_m=left_surface_gap_m,
            geometry_valid=left_geometry_valid,
            closest_collider_id=left_closest_collider_id,
            closest_robot_origin_pos=left_closest_robot_origin_pos,
            closest_robot_orientation_wxyz=left_closest_robot_orientation_wxyz,
            closest_surface_point_world_pos=left_closest_surface_point_world_pos,
            closest_robot_angular_velocity_world_radps=(
                left_closest_robot_angular_velocity_world_radps
            ),
        )
        right = self._right.update(
            sim_time_s=sim_time_s,
            hand_pos=right_hand_pos,
            tracking_valid=right_tracking_valid,
            surface_gap_m=right_surface_gap_m,
            geometry_valid=right_geometry_valid,
            closest_collider_id=right_closest_collider_id,
            closest_robot_origin_pos=right_closest_robot_origin_pos,
            closest_robot_orientation_wxyz=right_closest_robot_orientation_wxyz,
            closest_surface_point_world_pos=right_closest_surface_point_world_pos,
            closest_robot_angular_velocity_world_radps=(
                right_closest_robot_angular_velocity_world_radps
            ),
        )
        return DynamicSafetySample(
            left=left,
            right=right,
            min_ttc_s=float(min(left.ttc_s, right.ttc_s)),
            max_closing_speed_mps=float(
                max(left.closing_speed_mps, right.closing_speed_mps)
            ),
            dynamic_valid=bool(left.dynamic_valid or right.dynamic_valid),
        )
