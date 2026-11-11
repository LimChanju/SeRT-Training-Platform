"""Deterministic human-robot encounter state machine."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import numpy as np

from .schema import EncounterRecord, ProtocolViolation


@dataclass(frozen=True)
class EncounterUpdate:
    row_encounter_id: str
    state: str
    opened: bool = False
    closed: EncounterRecord | None = None


class EncounterStateMachine:
    """Create one feedback unit for each CBF activation-band crossing.

    An encounter begins on geometry/constraint entry, not only when the CBF
    changes the action.  This keeps nominal-feasible CBF interactions in the
    denominator and avoids intervention-based selection bias.
    """

    def __init__(
        self,
        *,
        activation_gap_m: float = 0.13,
        clear_confirmation_steps: int = 6,
        refractory_s: float = 2.0,
    ) -> None:
        if not math.isfinite(activation_gap_m) or activation_gap_m <= 0.0:
            raise ValueError("activation_gap_m must be finite and positive")
        if int(clear_confirmation_steps) < 1:
            raise ValueError("clear_confirmation_steps must be positive")
        if not math.isfinite(refractory_s) or refractory_s < 0.0:
            raise ValueError("refractory_s must be finite and non-negative")
        self.activation_gap_m = float(activation_gap_m)
        self.clear_confirmation_steps = int(clear_confirmation_steps)
        self.refractory_ns = int(round(float(refractory_s) * 1e9))
        self.reset_episode(0)

    def reset_episode(self, episode_index: int) -> None:
        self.episode_index = int(episode_index)
        self.state = "armed_idle"
        self._active_id = ""
        self._start_step = -1
        self._start_ns = 0
        self._involved_mask = np.zeros(2, dtype=np.int8)
        self._minimum_gap_m = math.inf
        self._clear_count = 0
        self._last_closed_ns = -self.refractory_ns
        self._candidate_was_active = False

    @property
    def active(self) -> bool:
        return bool(self._active_id)

    @property
    def active_encounter_id(self) -> str:
        return self._active_id

    @property
    def involved_hand_mask(self) -> tuple[int, int]:
        return int(self._involved_mask[0]), int(self._involved_mask[1])

    def observe(
        self,
        *,
        step: int,
        monotonic_ns: int,
        left_gap_m: float,
        right_gap_m: float,
        geometry_valid: bool,
        constraint_count: int,
        tracking_valid_mask: np.ndarray,
    ) -> EncounterUpdate:
        step = int(step)
        now_ns = int(monotonic_ns)
        gaps = np.asarray((left_gap_m, right_gap_m), dtype=np.float64)
        finite = np.isfinite(gaps)
        inside = finite & (gaps <= self.activation_gap_m)
        candidate = bool(geometry_valid and (np.any(inside) or int(constraint_count) > 0))

        if not self.active:
            rising_edge = bool(candidate and not self._candidate_was_active)
            self._candidate_was_active = candidate
            if not candidate:
                self.state = "armed_idle"
                return EncounterUpdate("", self.state)
            if not rising_edge:
                self.state = (
                    "refractory"
                    if now_ns - self._last_closed_ns < self.refractory_ns
                    else "awaiting_clearance"
                )
                return EncounterUpdate("", self.state)
            if now_ns - self._last_closed_ns < self.refractory_ns:
                self.state = "refractory"
                return EncounterUpdate("", self.state)
            mask = _validated_tracking_mask(tracking_valid_mask)
            if not np.all(mask == 1):
                raise ProtocolViolation("eligible encounter entered with invalid required tracking")
            self._active_id = str(uuid.uuid4())
            self._start_step = step
            self._start_ns = now_ns
            self._involved_mask = inside.astype(np.int8)
            if not np.any(self._involved_mask) and int(constraint_count) > 0:
                # This should not be possible with valid CBF geometry.
                raise ProtocolViolation("CBF constraint has no activation-band hand")
            self._minimum_gap_m = float(np.min(gaps[finite]))
            self._clear_count = 0
            self.state = "encounter_active"
            return EncounterUpdate(self._active_id, self.state, opened=True)

        mask = _validated_tracking_mask(tracking_valid_mask)
        if not np.all(mask == 1):
            raise ProtocolViolation("live tracking invalid during eligible encounter")
        self._candidate_was_active = candidate
        if np.any(finite):
            self._minimum_gap_m = min(
                self._minimum_gap_m, float(np.min(gaps[finite]))
            )
        newly_involved = inside.astype(np.int8)
        self._involved_mask = np.maximum(self._involved_mask, newly_involved)
        clear = bool(
            int(constraint_count) == 0
            and np.all(~finite | (gaps > self.activation_gap_m))
        )
        self._clear_count = self._clear_count + 1 if clear else 0
        if self._clear_count < self.clear_confirmation_steps:
            self.state = "clearance_confirm" if clear else "encounter_active"
            return EncounterUpdate(self._active_id, self.state)
        record = self._close(
            end_step_exclusive=step + 1,
            end_monotonic_ns=now_ns,
            terminal_status="completed",
            abort_reason="",
        )
        return EncounterUpdate(record.encounter_id, "robot_hold", closed=record)

    def close_for_episode_end(
        self,
        *,
        end_step_exclusive: int,
        end_monotonic_ns: int,
    ) -> EncounterRecord | None:
        if not self.active:
            return None
        return self._close(
            end_step_exclusive=end_step_exclusive,
            end_monotonic_ns=end_monotonic_ns,
            terminal_status="episode_end",
            abort_reason="",
        )

    def abort(
        self,
        *,
        end_step_exclusive: int,
        end_monotonic_ns: int,
        reason: str,
    ) -> EncounterRecord | None:
        if not self.active:
            return None
        reason = str(reason).strip() or "unspecified_abort"
        return self._close(
            end_step_exclusive=end_step_exclusive,
            end_monotonic_ns=end_monotonic_ns,
            terminal_status="aborted",
            abort_reason=reason,
        )

    def _close(
        self,
        *,
        end_step_exclusive: int,
        end_monotonic_ns: int,
        terminal_status: str,
        abort_reason: str,
    ) -> EncounterRecord:
        if not self.active:
            raise RuntimeError("no active encounter")
        record = EncounterRecord(
            encounter_id=self._active_id,
            episode_index=self.episode_index,
            start_step=self._start_step,
            end_step_exclusive=int(end_step_exclusive),
            start_monotonic_ns=self._start_ns,
            end_monotonic_ns=int(end_monotonic_ns),
            minimum_surface_gap_m=float(self._minimum_gap_m),
            involved_hand_mask=self.involved_hand_mask,
            eligible=True,
            eligibility_reason="cbf_activation_band_entry",
            terminal_status=str(terminal_status),
            abort_reason=str(abort_reason),
        )
        self._last_closed_ns = int(end_monotonic_ns)
        self._active_id = ""
        self._clear_count = 0
        self.state = "armed_idle" if not abort_reason else "aborted"
        return record


def _validated_tracking_mask(value: np.ndarray) -> np.ndarray:
    try:
        raw = np.asarray(value).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ProtocolViolation(
            "encounter tracking_valid_mask must be [head,left,right]"
        ) from error
    if (
        raw.shape != (3,)
        or raw.dtype.kind not in "buif"
        or not np.all(np.isfinite(raw))
        or not np.all(np.isin(raw, (0, 1)))
    ):
        raise ProtocolViolation(
            "encounter tracking_valid_mask must be [head,left,right]"
        )
    return raw.astype(np.int8, copy=True)
