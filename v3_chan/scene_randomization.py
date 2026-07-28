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
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < int(count) and attempts < 2000:
        attempts += 1
        candidate = table_xy + rng.uniform(-xy_half, xy_half)
        if not (x_bounds[0] <= candidate[0] <= x_bounds[1]):
            continue
        if not (y_bounds[0] <= candidate[1] <= y_bounds[1]):
            continue
        if forbidden_xy is not None:
            forbidden = np.asarray(forbidden_xy, dtype=float).reshape(2)
            if np.linalg.norm(candidate - forbidden) < min_dist:
                continue
        if all(np.linalg.norm(candidate - position) >= min_dist for position in positions):
            positions.append(candidate.copy())
    if len(positions) < int(count):
        raise RuntimeError("Could not place cubes without overlap; increase table size.")
    return positions


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
