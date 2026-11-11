"""Isaac-independent execution guards for the pick-and-place controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspGateDecision:
    """Whether the controller may leave its pre-grasp settle phase."""

    active: bool
    release: bool
    timed_out: bool
    distance_m: float | None
    waited_steps: int
    consecutive_ready_steps: int


class GraspReadinessGate:
    """Require a stable controller-reference/cube alignment before closing.

    Isaac Sim's tutorial PickPlaceController advances phases from an internal
    call counter. This helper keeps the controller in event 2 until the real
    controller reference is close enough to the target cube, avoiding a close
    command while the arm is still catching up to a fast trajectory.
    """

    SETTLE_EVENT = 2

    def __init__(
        self,
        *,
        max_distance_m: float = 0.075,
        required_consecutive_steps: int = 3,
        max_wait_steps: int = 180,
    ) -> None:
        if float(max_distance_m) <= 0.0:
            raise ValueError("max_distance_m must be positive")
        if int(required_consecutive_steps) < 1:
            raise ValueError("required_consecutive_steps must be >= 1")
        if int(max_wait_steps) < 1:
            raise ValueError("max_wait_steps must be >= 1")
        self.max_distance_m = float(max_distance_m)
        self.required_consecutive_steps = int(required_consecutive_steps)
        self.max_wait_steps = int(max_wait_steps)
        self.reset()

    def reset(self) -> None:
        self._waited_steps = 0
        self._consecutive_ready_steps = 0

    @staticmethod
    def _distance(position_a, position_b) -> float | None:
        if position_a is None or position_b is None:
            return None
        first = np.asarray(position_a, dtype=float).reshape(-1)
        second = np.asarray(position_b, dtype=float).reshape(-1)
        if first.size < 3 or second.size < 3:
            return None
        distance = float(np.linalg.norm(first[:3] - second[:3]))
        return distance if np.isfinite(distance) else None

    def update(
        self,
        *,
        controller_event: int,
        grasp_reference_position,
        cube_position,
    ) -> GraspGateDecision:
        if int(controller_event) != self.SETTLE_EVENT:
            self.reset()
            return GraspGateDecision(
                active=False,
                release=False,
                timed_out=False,
                distance_m=None,
                waited_steps=0,
                consecutive_ready_steps=0,
            )

        self._waited_steps += 1
        distance = self._distance(grasp_reference_position, cube_position)
        ready = distance is not None and distance <= self.max_distance_m
        self._consecutive_ready_steps = (
            self._consecutive_ready_steps + 1 if ready else 0
        )
        release = self._consecutive_ready_steps >= self.required_consecutive_steps
        timed_out = self._waited_steps >= self.max_wait_steps and not release
        return GraspGateDecision(
            active=True,
            release=release,
            timed_out=timed_out,
            distance_m=distance,
            waited_steps=self._waited_steps,
            consecutive_ready_steps=self._consecutive_ready_steps,
        )


def failed_attempt_reaches_limit(
    *, current_cube_attempt: int, max_attempts_per_cube: int
) -> bool:
    """Return true when the just-failed attempt exhausts the configured cap."""

    if int(current_cube_attempt) < 1:
        raise ValueError("current_cube_attempt must be >= 1")
    if int(max_attempts_per_cube) < 1:
        raise ValueError("max_attempts_per_cube must be >= 1")
    return int(current_cube_attempt) >= int(max_attempts_per_cube)
