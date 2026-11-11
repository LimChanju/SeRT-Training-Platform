"""Deterministic cube layouts and provenance helpers."""

from __future__ import annotations

import hashlib
import secrets

import numpy as np


def resolve_session_seed(value: str | int | None = None) -> int:
    """Return a non-negative 63-bit seed, generating one when omitted."""

    if value is None or str(value).strip() == "":
        return int(secrets.randbits(63))
    seed = int(value)
    if seed < 0:
        raise ValueError("HRI_SESSION_SEED must be non-negative")
    return seed & ((1 << 63) - 1)


def episode_seed(session_seed: int, episode_index: int) -> int:
    sequence = np.random.SeedSequence([int(session_seed), int(episode_index)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] & np.uint64((1 << 63) - 1))


def episode_rng(session_seed: int, episode_index: int) -> np.random.Generator:
    return np.random.default_rng(episode_seed(session_seed, episode_index))


def sample_cube_positions(
    *,
    rng: np.random.Generator,
    table_xy: np.ndarray,
    table_size: np.ndarray,
    cube_size: float,
    count: int,
    forbidden_xy: np.ndarray | None = None,
    x_bounds: tuple[float, float] = (0.30, 0.65),
    y_bounds: tuple[float, float] = (-0.25, 0.25),
) -> list[np.ndarray]:
    """Sample non-overlapping XY positions with an explicit RNG."""

    table_xy = np.asarray(table_xy, dtype=float).reshape(2)
    table_size = np.asarray(table_size, dtype=float).reshape(-1)
    xy_half = (table_size[:2] / 2.0) - 0.1
    min_dist = float(cube_size) * 2.6
    table_lower = table_xy - xy_half
    table_upper = table_xy + xy_half
    lower = np.maximum(table_lower, np.array([x_bounds[0], y_bounds[0]]))
    upper = np.minimum(table_upper, np.array([x_bounds[1], y_bounds[1]]))
    if np.any(lower >= upper):
        raise ValueError("Cube sampling bounds do not intersect the usable table area.")

    forbidden = (
        None
        if forbidden_xy is None
        else np.asarray(forbidden_xy, dtype=float).reshape(2)
    )
    # Restart the full layout when an early sample leaves no room for later cubes.
    # This preserves rejection-sampling behavior while avoiding rare dead ends.
    attempts_per_layout = max(1000, int(count) * 250)
    for _ in range(64):
        positions: list[np.ndarray] = []
        for _ in range(attempts_per_layout):
            candidate = rng.uniform(lower, upper)
            if forbidden is not None and np.linalg.norm(candidate - forbidden) < min_dist:
                continue
            if all(
                np.linalg.norm(candidate - position) >= min_dist
                for position in positions
            ):
                positions.append(candidate.copy())
                if len(positions) == int(count):
                    return positions
    raise RuntimeError(
        "Could not place cubes without overlap after 64 layout restarts; "
        "increase the sampling area."
    )


def scene_layout_id(
    cube_positions,
    cube_orientations_wxyz,
    target_position,
    target_orientation_wxyz,
) -> str:
    """Build a stable identifier from the exact initial scene state."""

    digest = hashlib.sha256()
    for value in (
        cube_positions,
        cube_orientations_wxyz,
        target_position,
        target_orientation_wxyz,
    ):
        array = np.asarray(value, dtype="<f8")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()[:16]


def restore_cube_poses(
    cubes,
    cube_names,
    positions_world,
    orientations_wxyz,
    *,
    set_default_state: bool = True,
) -> dict[str, int]:
    """Apply an exact recorded cube configuration by cube identity."""

    names = [_text_name(value) for value in np.asarray(cube_names).reshape(-1)]
    positions = np.asarray(positions_world, dtype=float)
    orientations = np.asarray(orientations_wxyz, dtype=float)
    if positions.shape != (len(names), 3):
        raise ValueError(
            f"Expected cube positions shape {(len(names), 3)}, got {positions.shape}"
        )
    if orientations.shape != (len(names), 4):
        raise ValueError(
            "Expected cube orientations shape "
            f"{(len(names), 4)}, got {orientations.shape}"
        )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(orientations)):
        raise ValueError("Recorded cube poses must be finite")

    runtime_by_name = {
        str(getattr(cube, "name", f"cube_{index}")): cube
        for index, cube in enumerate(cubes)
    }
    missing = [name for name in names if name not in runtime_by_name]
    if missing:
        raise ValueError(f"Recorded cube identities are absent from the scene: {missing}")

    source_to_runtime: dict[str, int] = {}
    runtime_indices = {id(cube): index for index, cube in enumerate(cubes)}
    for name, position, orientation in zip(names, positions, orientations):
        cube = runtime_by_name[name]
        if hasattr(cube, "disable_rigid_body_physics"):
            cube.disable_rigid_body_physics()
        if set_default_state and hasattr(cube, "set_default_state"):
            cube.set_default_state(
                position=position,
                orientation=orientation,
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
        cube.set_world_pose(position=position, orientation=orientation)
        if hasattr(cube, "enable_rigid_body_physics"):
            cube.enable_rigid_body_physics()
        if hasattr(cube, "set_linear_velocity"):
            cube.set_linear_velocity(np.zeros(3))
        if hasattr(cube, "set_angular_velocity"):
            cube.set_angular_velocity(np.zeros(3))
        source_to_runtime[name] = int(runtime_indices[id(cube)])
    return source_to_runtime


def resolve_active_cube_index(
    source_cube_index,
    *,
    episode_index: int,
    cube_count: int,
) -> int:
    """Use source identity when provided; otherwise preserve cyclic task reset."""

    if int(cube_count) <= 0:
        raise ValueError("cube_count must be positive")
    if source_cube_index is None:
        return int(episode_index) % int(cube_count)
    try:
        index = int(source_cube_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("source active cube index is unavailable") from exc
    if not 0 <= index < int(cube_count):
        raise ValueError(
            f"source active cube index {index} is outside [0, {int(cube_count)})"
        )
    return index


def restored_pose_errors(
    cubes,
    cube_names,
    positions_world,
    orientations_wxyz,
) -> tuple[float, float]:
    """Return maximum position and quaternion-angle errors for a restored scene."""

    names = [_text_name(value) for value in np.asarray(cube_names).reshape(-1)]
    positions = np.asarray(positions_world, dtype=float)
    orientations = np.asarray(orientations_wxyz, dtype=float)
    runtime_by_name = {
        str(getattr(cube, "name", f"cube_{index}")): cube
        for index, cube in enumerate(cubes)
    }
    position_errors = []
    orientation_errors = []
    for index, name in enumerate(names):
        if name not in runtime_by_name:
            return float("inf"), float("inf")
        actual_position, actual_orientation = runtime_by_name[name].get_world_pose()
        position_errors.append(
            float(np.linalg.norm(np.asarray(actual_position) - positions[index]))
        )
        orientation_errors.append(
            _quaternion_angle_error_rad(actual_orientation, orientations[index])
        )
    return (
        max(position_errors, default=0.0),
        max(orientation_errors, default=0.0),
    )


def _quaternion_angle_error_rad(actual, expected) -> float:
    first = np.asarray(actual, dtype=float).reshape(-1)[:4]
    second = np.asarray(expected, dtype=float).reshape(-1)[:4]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return float("inf")
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _text_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
