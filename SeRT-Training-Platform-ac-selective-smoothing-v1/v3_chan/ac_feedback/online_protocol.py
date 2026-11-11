"""Pure-Python protocol primitives for online explicit-feedback collection.

This module deliberately has no Isaac Sim, Torch, NumPy, HDF5, or XR imports.
It defines the runtime-observable encounter detector and planned-corridor
crossing measurements.  The frozen A/C schedule and trial types have one
authoritative home in :mod:`v3_chan.ac_feedback.study`.

Isaac-dependent runtime state is intentionally kept outside this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


CROSSING_DIRECTIONS = ("left_to_right", "right_to_left")
ACTUAL_CROSSING_NOT_OBSERVED = "not_observed"


def resolve_actual_crossing_direction(
    planned_direction: str, completed_traversal_sign: int
) -> str:
    """Map the monitor's first completed signed traversal to a world label.

    The corridor start/end order encodes the assigned direction.  A positive
    traversal therefore preserves the assignment, a negative traversal is its
    reverse, and zero means that neither gate-to-gate traversal was observed.
    """

    planned = str(planned_direction)
    if planned not in CROSSING_DIRECTIONS:
        raise ValueError(f"unknown planned crossing direction: {planned!r}")
    if isinstance(completed_traversal_sign, bool):
        raise ValueError("completed_traversal_sign must be -1, 0, or 1")
    sign = int(completed_traversal_sign)
    if sign != completed_traversal_sign or sign not in (-1, 0, 1):
        raise ValueError("completed_traversal_sign must be -1, 0, or 1")
    if sign == 0:
        return ACTUAL_CROSSING_NOT_OBSERVED
    if sign > 0:
        return planned
    return (
        "right_to_left" if planned == "left_to_right" else "left_to_right"
    )


# ---------------------------------------------------------------------------
# Encounter detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncounterDetectorConfig:
    activation_gap_m: float = 0.13
    onset_ttc_s: float = 0.75
    intervention_norm_radps: float = 0.05
    onset_confirmation_frames: int = 3
    intervention_confirmation_frames: int = 3
    clear_gap_m: float = 0.15
    clear_ttc_s: float = 1.0
    clear_norm_radps: float = 0.01
    clear_duration_s: float = 0.5
    reentry_merge_s: float = 1.0
    timeout_s: float = 8.0
    pre_window_s: float = 1.0
    post_window_s: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            self.activation_gap_m,
            self.onset_ttc_s,
            self.intervention_norm_radps,
            self.clear_gap_m,
            self.clear_ttc_s,
            self.clear_norm_radps,
            self.clear_duration_s,
            self.reentry_merge_s,
            self.timeout_s,
            self.pre_window_s,
            self.post_window_s,
        )
        if not all(math.isfinite(float(value)) and value >= 0.0 for value in positive):
            raise ValueError("encounter thresholds must be finite and non-negative")
        if self.activation_gap_m >= self.clear_gap_m:
            raise ValueError("clear_gap_m must exceed activation_gap_m")
        if self.intervention_norm_radps <= self.clear_norm_radps:
            raise ValueError("intervention norm must exceed clear norm")
        if int(self.onset_confirmation_frames) < 1:
            raise ValueError("onset_confirmation_frames must be positive")
        if int(self.intervention_confirmation_frames) < 1:
            raise ValueError("intervention_confirmation_frames must be positive")


@dataclass(frozen=True)
class EncounterRecord:
    encounter_id: str
    trial_id: str
    risk_onset_step: int
    risk_onset_simulation_time_s: float
    onset_confirmed_step: int
    onset_confirmed_simulation_time_s: float
    offset_step: int
    offset_simulation_time_s: float
    end_step_exclusive: int
    trigger_sources: tuple[str, ...]
    minimum_surface_gap_m: float
    maximum_cbf_delta_norm_radps: float
    cbf_intervention_start_step: int | None
    cbf_intervention_start_simulation_time_s: float | None
    cbf_intervention_confirmed_step: int | None
    cbf_intervention_confirmed_simulation_time_s: float | None
    reentry_merge_count: int
    timeout: bool
    terminal_status: str
    window_start_simulation_time_s: float
    window_end_simulation_time_s: float

    @property
    def duration_s(self) -> float:
        return self.offset_simulation_time_s - self.risk_onset_simulation_time_s


@dataclass(frozen=True)
class EncounterDetectorUpdate:
    state: str
    row_encounter_id: str = ""
    opened: bool = False
    onset_candidate_step: int | None = None
    onset_confirmed_step: int | None = None
    cbf_intervention_confirmed_now: bool = False
    timed_out_now: bool = False
    merged_reentry: bool = False
    closed: EncounterRecord | None = None


@dataclass
class _MutableEncounter:
    encounter_id: str
    trial_id: str
    risk_onset_step: int
    risk_onset_time_s: float
    onset_confirmed_step: int
    onset_confirmed_time_s: float
    trigger_sources: set[str]
    minimum_gap_m: float = math.inf
    maximum_norm_radps: float = 0.0
    cbf_start_step: int | None = None
    cbf_start_time_s: float | None = None
    cbf_confirmed_step: int | None = None
    cbf_confirmed_time_s: float | None = None
    reentry_count: int = 0
    timeout: bool = False
    offset_step: int | None = None
    offset_time_s: float | None = None


class EncounterDetector:
    """Detect protocol encounters without using future outcomes.

    Onset requires three consecutive frames satisfying any frozen trigger.
    Offset requires all frozen clear predicates continuously for 0.5 seconds.
    An offset is held provisionally for 1 second so a re-entry can retain the
    same encounter ID.  The record's offset remains the confirmed-clear time;
    the merge wait is not added to its observation window.
    """

    def __init__(
        self,
        config: EncounterDetectorConfig | None = None,
        *,
        trial_id: str = "trial",
    ) -> None:
        self.config = config or EncounterDetectorConfig()
        self.reset_trial(trial_id)

    def reset_trial(self, trial_id: str) -> None:
        trial_id = str(trial_id).strip()
        if not trial_id:
            raise ValueError("trial_id must be non-empty")
        self.trial_id = trial_id
        self.state = "idle"
        self._encounter_counter = 0
        self._last_step: int | None = None
        self._last_time_s: float | None = None
        self._onset_count = 0
        self._onset_start_step: int | None = None
        self._onset_start_time_s: float | None = None
        self._onset_sources: set[str] = set()
        self._onset_minimum_gap_m = math.inf
        self._onset_maximum_norm_radps = 0.0
        self._work: _MutableEncounter | None = None
        self._clear_start_time_s: float | None = None
        self._clear_start_step: int | None = None
        self._intervention_count = 0
        self._intervention_start_step: int | None = None
        self._intervention_start_time_s: float | None = None
        self._intervention_confirmed_step: int | None = None
        self._intervention_confirmed_time_s: float | None = None
        self.completed_records: list[EncounterRecord] = []

    @property
    def active(self) -> bool:
        return self._work is not None

    @property
    def active_encounter_id(self) -> str:
        return self._work.encounter_id if self._work is not None else ""

    def observe(
        self,
        *,
        step: int,
        simulation_time_s: float,
        surface_gap_m: float,
        ttc_s: float,
        closing: bool,
        cbf_delta_norm_radps: float,
        cbf_active: bool,
        dynamic_measurement_valid: bool = True,
    ) -> EncounterDetectorUpdate:
        step = int(step)
        now = float(simulation_time_s)
        self._validate_clock(step, now)
        gap = float(surface_gap_m)
        ttc = float(ttc_s)
        norm = float(cbf_delta_norm_radps)
        sources = self._onset_trigger_sources(
            gap=gap,
            ttc=ttc,
            closing=bool(closing),
            norm=norm,
            dynamic_measurement_valid=bool(dynamic_measurement_valid),
        )
        risk = bool(sources)

        closed: EncounterRecord | None = None
        if self.state == "merge_wait" and self._work is not None:
            assert self._work.offset_time_s is not None
            since_offset = now - self._work.offset_time_s
            if risk and since_offset <= self.config.reentry_merge_s + 1e-12:
                self._work.reentry_count += 1
                self._work.offset_step = None
                self._work.offset_time_s = None
                self._clear_start_step = None
                self._clear_start_time_s = None
                self.state = "active"
                self._accumulate(gap=gap, norm=norm, sources=sources)
                intervention_now = self._advance_intervention(
                    step=step, now=now, norm=norm
                )
                return EncounterDetectorUpdate(
                    state=self.state,
                    row_encounter_id=self.active_encounter_id,
                    cbf_intervention_confirmed_now=intervention_now,
                    merged_reentry=True,
                )
            if since_offset > self.config.reentry_merge_s + 1e-12:
                closed = self._finalize_work(terminal_status="completed")
                self._reset_onset_candidate()
                self._reset_intervention_candidate()
                self.state = "idle"
            else:
                # The confirmed offset stays fixed while its merge window is
                # pending.  Sliding it forward on every clear frame would make
                # a completed encounter impossible to emit.
                return EncounterDetectorUpdate(
                    state=self.state,
                    row_encounter_id=self.active_encounter_id,
                )

        intervention_now = self._advance_intervention(step=step, now=now, norm=norm)

        if self._work is None:
            if risk:
                if self._onset_count == 0:
                    self._onset_start_step = step
                    self._onset_start_time_s = now
                    self._onset_sources.clear()
                self._onset_count += 1
                self._onset_sources.update(sources)
                if math.isfinite(gap):
                    self._onset_minimum_gap_m = min(
                        self._onset_minimum_gap_m, gap
                    )
                if math.isfinite(norm):
                    self._onset_maximum_norm_radps = max(
                        self._onset_maximum_norm_radps, norm
                    )
                self.state = "onset_candidate"
                if self._onset_count >= self.config.onset_confirmation_frames:
                    self._open_encounter(
                        confirmed_step=step,
                        confirmed_time_s=now,
                        gap=gap,
                        norm=norm,
                    )
                    return EncounterDetectorUpdate(
                        state=self.state,
                        row_encounter_id=self.active_encounter_id,
                        opened=True,
                        onset_candidate_step=self._work.risk_onset_step,
                        onset_confirmed_step=step,
                        cbf_intervention_confirmed_now=intervention_now,
                        closed=closed,
                    )
            else:
                self._reset_onset_candidate()
                self._reset_intervention_candidate()
                self.state = "idle"
            return EncounterDetectorUpdate(
                state=self.state,
                onset_candidate_step=self._onset_start_step,
                cbf_intervention_confirmed_now=intervention_now,
                closed=closed,
            )

        self._accumulate(gap=gap, norm=norm, sources=sources)
        timed_out_now = False
        if (
            not self._work.timeout
            and now - self._work.risk_onset_time_s > self.config.timeout_s
        ):
            self._work.timeout = True
            timed_out_now = True

        clear = self._is_clear(
            gap=gap,
            ttc=ttc,
            closing=bool(closing),
            norm=norm,
            cbf_active=bool(cbf_active),
            dynamic_measurement_valid=bool(dynamic_measurement_valid),
        )
        if clear:
            if self._clear_start_time_s is None:
                self._clear_start_time_s = now
                self._clear_start_step = step
            if now - self._clear_start_time_s + 1e-12 >= self.config.clear_duration_s:
                self._work.offset_step = step
                self._work.offset_time_s = now
                self.state = "merge_wait"
            else:
                self.state = "clearance_confirm"
        else:
            self._clear_start_time_s = None
            self._clear_start_step = None
            self.state = "active"
        return EncounterDetectorUpdate(
            state=self.state,
            row_encounter_id=self.active_encounter_id,
            onset_confirmed_step=self._work.onset_confirmed_step,
            cbf_intervention_confirmed_now=intervention_now,
            timed_out_now=timed_out_now,
            closed=closed,
        )

    def flush(
        self,
        *,
        step: int | None = None,
        simulation_time_s: float | None = None,
        terminal_status: str = "trial_end",
    ) -> EncounterRecord | None:
        """Finalize the current encounter at trial end without inventing data."""

        if self._work is None:
            return None
        if self._work.offset_step is None:
            if step is None:
                step = self._last_step
            if simulation_time_s is None:
                simulation_time_s = self._last_time_s
            if step is None or simulation_time_s is None:
                raise ValueError("flush needs an observed or explicit terminal clock")
            self._work.offset_step = int(step)
            self._work.offset_time_s = float(simulation_time_s)
        record = self._finalize_work(terminal_status=str(terminal_status))
        self.state = "idle"
        self._reset_onset_candidate()
        self._reset_intervention_candidate()
        return record

    def _validate_clock(self, step: int, now: float) -> None:
        if step < 0 or not math.isfinite(now) or now < 0.0:
            raise ValueError("step/time must be finite and non-negative")
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("encounter observations require strictly increasing steps")
        if self._last_time_s is not None and now < self._last_time_s:
            raise ValueError("simulation_time_s must be monotonic")
        self._last_step = step
        self._last_time_s = now

    def _onset_trigger_sources(
        self,
        *,
        gap: float,
        ttc: float,
        closing: bool,
        norm: float,
        dynamic_measurement_valid: bool,
    ) -> set[str]:
        sources: set[str] = set()
        if math.isfinite(gap) and gap <= self.config.activation_gap_m:
            sources.add("activation_gap")
        if (
            dynamic_measurement_valid
            and closing
            and math.isfinite(ttc)
            and 0.0 <= ttc <= self.config.onset_ttc_s
        ):
            sources.add("closing_ttc")
        if math.isfinite(norm) and norm >= self.config.intervention_norm_radps:
            sources.add("cbf_delta_norm")
        return sources

    def _is_clear(
        self,
        *,
        gap: float,
        ttc: float,
        closing: bool,
        norm: float,
        cbf_active: bool,
        dynamic_measurement_valid: bool,
    ) -> bool:
        return bool(
            math.isfinite(gap)
            and gap >= self.config.clear_gap_m
            and not cbf_active
            and dynamic_measurement_valid
            and ((not closing) or (math.isfinite(ttc) and ttc > self.config.clear_ttc_s))
            and math.isfinite(norm)
            and norm <= self.config.clear_norm_radps
        )

    def _advance_intervention(self, *, step: int, now: float, norm: float) -> bool:
        above = bool(
            math.isfinite(norm) and norm >= self.config.intervention_norm_radps
        )
        if above:
            if self._intervention_count == 0:
                self._intervention_start_step = step
                self._intervention_start_time_s = now
            self._intervention_count += 1
        else:
            self._reset_intervention_candidate()
            return False
        confirmed_now = False
        if (
            self._intervention_count
            >= self.config.intervention_confirmation_frames
            and self._intervention_confirmed_step is None
        ):
            self._intervention_confirmed_step = step
            self._intervention_confirmed_time_s = now
            confirmed_now = True
        if self._work is not None and self._intervention_confirmed_step is not None:
            if self._work.cbf_confirmed_step is None:
                self._work.cbf_start_step = self._intervention_start_step
                self._work.cbf_start_time_s = self._intervention_start_time_s
                self._work.cbf_confirmed_step = self._intervention_confirmed_step
                self._work.cbf_confirmed_time_s = self._intervention_confirmed_time_s
        return confirmed_now

    def _open_encounter(
        self, *, confirmed_step: int, confirmed_time_s: float, gap: float, norm: float
    ) -> None:
        assert self._onset_start_step is not None
        assert self._onset_start_time_s is not None
        self._encounter_counter += 1
        encounter_id = f"{self.trial_id}:encounter-{self._encounter_counter:02d}"
        self._work = _MutableEncounter(
            encounter_id=encounter_id,
            trial_id=self.trial_id,
            risk_onset_step=self._onset_start_step,
            risk_onset_time_s=self._onset_start_time_s,
            onset_confirmed_step=confirmed_step,
            onset_confirmed_time_s=confirmed_time_s,
            trigger_sources=set(self._onset_sources),
            minimum_gap_m=self._onset_minimum_gap_m,
            maximum_norm_radps=self._onset_maximum_norm_radps,
        )
        self._accumulate(gap=gap, norm=norm, sources=self._onset_sources)
        if self._intervention_confirmed_step is not None:
            self._work.cbf_start_step = self._intervention_start_step
            self._work.cbf_start_time_s = self._intervention_start_time_s
            self._work.cbf_confirmed_step = self._intervention_confirmed_step
            self._work.cbf_confirmed_time_s = self._intervention_confirmed_time_s
        self._clear_start_step = None
        self._clear_start_time_s = None
        self.state = "active"

    def _accumulate(self, *, gap: float, norm: float, sources: Iterable[str]) -> None:
        if self._work is None:
            return
        self._work.trigger_sources.update(sources)
        if math.isfinite(gap):
            self._work.minimum_gap_m = min(self._work.minimum_gap_m, gap)
        if math.isfinite(norm):
            self._work.maximum_norm_radps = max(self._work.maximum_norm_radps, norm)

    def _finalize_work(self, *, terminal_status: str) -> EncounterRecord:
        assert self._work is not None
        assert self._work.offset_step is not None
        assert self._work.offset_time_s is not None
        work = self._work
        record = EncounterRecord(
            encounter_id=work.encounter_id,
            trial_id=work.trial_id,
            risk_onset_step=work.risk_onset_step,
            risk_onset_simulation_time_s=work.risk_onset_time_s,
            onset_confirmed_step=work.onset_confirmed_step,
            onset_confirmed_simulation_time_s=work.onset_confirmed_time_s,
            offset_step=work.offset_step,
            offset_simulation_time_s=work.offset_time_s,
            end_step_exclusive=work.offset_step + 1,
            trigger_sources=tuple(sorted(work.trigger_sources)),
            minimum_surface_gap_m=work.minimum_gap_m,
            maximum_cbf_delta_norm_radps=work.maximum_norm_radps,
            cbf_intervention_start_step=work.cbf_start_step,
            cbf_intervention_start_simulation_time_s=work.cbf_start_time_s,
            cbf_intervention_confirmed_step=work.cbf_confirmed_step,
            cbf_intervention_confirmed_simulation_time_s=work.cbf_confirmed_time_s,
            reentry_merge_count=work.reentry_count,
            timeout=work.timeout,
            terminal_status=terminal_status,
            window_start_simulation_time_s=max(
                0.0, work.risk_onset_time_s - self.config.pre_window_s
            ),
            window_end_simulation_time_s=(
                work.offset_time_s + self.config.post_window_s
            ),
        )
        self.completed_records.append(record)
        self._work = None
        self._clear_start_step = None
        self._clear_start_time_s = None
        return record

    def _reset_onset_candidate(self) -> None:
        self._onset_count = 0
        self._onset_start_step = None
        self._onset_start_time_s = None
        self._onset_sources.clear()
        self._onset_minimum_gap_m = math.inf
        self._onset_maximum_norm_radps = 0.0

    def _reset_intervention_candidate(self) -> None:
        self._intervention_count = 0
        self._intervention_start_step = None
        self._intervention_start_time_s = None
        self._intervention_confirmed_step = None
        self._intervention_confirmed_time_s = None


# ---------------------------------------------------------------------------
# Crossing measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossingMetrics:
    intended_crossing_count: int
    reverse_crossing_count: int
    first_completed_traversal_sign: int
    valid_sample_count: int
    invalid_sample_count: int
    path_length_m: float
    duration_s: float
    mean_speed_mps: float
    rms_path_deviation_m: float
    maximum_path_deviation_m: float
    path_deviation_threshold_m: float
    expected_speed_mps: float | None
    relative_speed_error: float | None
    maximum_reverse_progress_m: float
    off_protocol: bool
    off_protocol_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CrossingUpdate:
    state: str
    progress: float | None
    path_deviation_m: float | None
    crossing_started: bool = False
    intended_crossing_completed: bool = False
    reverse_crossing_completed: bool = False


class CrossingPathMonitor:
    """Measure an actual hand traversal against a planned start/end corridor."""

    def __init__(
        self,
        *,
        start_position_world: Sequence[float],
        end_position_world: Sequence[float],
        path_deviation_threshold_m: float,
        expected_speed_mps: float | None = None,
        relative_speed_tolerance: float | None = None,
        start_gate_fraction: float = 0.10,
        end_gate_fraction: float = 0.90,
        start_radius_m: float | None = None,
        end_radius_m: float | None = None,
        minimum_progress_fraction: float | None = None,
        maximum_reverse_progress_m: float | None = None,
    ) -> None:
        self.start = _point3(start_position_world, "start_position_world")
        self.end = _point3(end_position_world, "end_position_world")
        delta = tuple(b - a for a, b in zip(self.start, self.end))
        self.length = math.sqrt(sum(value * value for value in delta))
        if self.length <= 1e-9:
            raise ValueError("crossing start and end must be distinct")
        self.direction = tuple(value / self.length for value in delta)
        self.path_deviation_threshold_m = float(path_deviation_threshold_m)
        if (
            not math.isfinite(self.path_deviation_threshold_m)
            or self.path_deviation_threshold_m <= 0.0
        ):
            raise ValueError("path_deviation_threshold_m must be finite and positive")
        self.expected_speed_mps = (
            None if expected_speed_mps is None else float(expected_speed_mps)
        )
        if self.expected_speed_mps is not None and (
            not math.isfinite(self.expected_speed_mps) or self.expected_speed_mps <= 0.0
        ):
            raise ValueError("expected_speed_mps must be finite and positive")
        self.relative_speed_tolerance = (
            None
            if relative_speed_tolerance is None
            else float(relative_speed_tolerance)
        )
        if self.relative_speed_tolerance is not None and (
            not math.isfinite(self.relative_speed_tolerance)
            or self.relative_speed_tolerance < 0.0
        ):
            raise ValueError("relative_speed_tolerance must be non-negative")
        self.start_gate = float(start_gate_fraction)
        self.end_gate = float(
            end_gate_fraction
            if minimum_progress_fraction is None
            else minimum_progress_fraction
        )
        if not 0.0 <= self.start_gate < self.end_gate <= 1.0:
            raise ValueError("crossing gates must satisfy 0 <= start < end <= 1")
        self.start_radius_m = (
            self.length * self.start_gate
            if start_radius_m is None
            else float(start_radius_m)
        )
        self.end_radius_m = (
            self.length * (1.0 - self.end_gate)
            if end_radius_m is None
            else float(end_radius_m)
        )
        if (
            not math.isfinite(self.start_radius_m)
            or not math.isfinite(self.end_radius_m)
            or self.start_radius_m <= 0.0
            or self.end_radius_m <= 0.0
        ):
            raise ValueError("crossing start/end radii must be finite and positive")
        self.allowed_reverse_progress_m = (
            None
            if maximum_reverse_progress_m is None
            else float(maximum_reverse_progress_m)
        )
        if self.allowed_reverse_progress_m is not None and (
            not math.isfinite(self.allowed_reverse_progress_m)
            or self.allowed_reverse_progress_m < 0.0
        ):
            raise ValueError("maximum_reverse_progress_m must be non-negative")
        self.reset()

    def reset(self) -> None:
        self.state = "awaiting_start"
        self.intended_crossing_count = 0
        self.reverse_crossing_count = 0
        self.first_completed_traversal_sign = 0
        self.invalid_sample_count = 0
        self._forward_armed = False
        self._reverse_armed = False
        self._path_samples: list[tuple[float, tuple[float, float, float], float]] = []
        self._last_time_s: float | None = None
        self._peak_axial_progress_m = -math.inf
        self.maximum_reverse_progress_m = 0.0

    def observe(
        self,
        *,
        position_world: Sequence[float] | None,
        simulation_time_s: float,
        valid: bool = True,
    ) -> CrossingUpdate:
        now = float(simulation_time_s)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("simulation_time_s must be finite and non-negative")
        if self._last_time_s is not None and now < self._last_time_s:
            raise ValueError("crossing samples require monotonic simulation time")
        self._last_time_s = now
        if not valid or position_world is None:
            self.invalid_sample_count += 1
            return CrossingUpdate(self.state, None, None)
        point = _point3(position_world, "position_world")
        progress, deviation = self._project(point)
        at_start = (
            progress <= self.start_gate
            and math.dist(point, self.start) <= self.start_radius_m
        )
        at_end = (
            progress >= self.end_gate
            and math.dist(point, self.end) <= self.end_radius_m
        )
        started = False
        intended_completed = False
        reverse_completed = False

        if at_start:
            if not self._forward_armed:
                started = True
                if self.intended_crossing_count == 0:
                    self._path_samples = []
                    self._peak_axial_progress_m = progress * self.length
            self._forward_armed = True
            if self._reverse_armed:
                self.reverse_crossing_count += 1
                reverse_completed = True
                if self.first_completed_traversal_sign == 0:
                    self.first_completed_traversal_sign = -1
                self._reverse_armed = False
        if self._forward_armed and self.intended_crossing_count == 0:
            self._path_samples.append((now, point, deviation))
            axial_progress_m = progress * self.length
            self._peak_axial_progress_m = max(
                self._peak_axial_progress_m, axial_progress_m
            )
            self.maximum_reverse_progress_m = max(
                self.maximum_reverse_progress_m,
                self._peak_axial_progress_m - axial_progress_m,
            )
        if at_end:
            if self._forward_armed:
                self.intended_crossing_count += 1
                intended_completed = True
                if self.first_completed_traversal_sign == 0:
                    self.first_completed_traversal_sign = 1
                self._forward_armed = False
            self._reverse_armed = True

        if self.intended_crossing_count == 0:
            self.state = "crossing" if self._forward_armed else "awaiting_start"
        elif self.intended_crossing_count == 1 and self.reverse_crossing_count == 0:
            self.state = "completed"
        else:
            self.state = "extra_traversal"
        return CrossingUpdate(
            state=self.state,
            progress=progress,
            path_deviation_m=deviation,
            crossing_started=started,
            intended_crossing_completed=intended_completed,
            reverse_crossing_completed=reverse_completed,
        )

    def finalize(
        self, *, additional_off_protocol_reasons: Iterable[str] = ()
    ) -> CrossingMetrics:
        samples = self._path_samples
        path_length = 0.0
        for (_, prior, _), (_, current, _) in zip(samples, samples[1:]):
            path_length += math.dist(prior, current)
        duration = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
        mean_speed = path_length / duration if duration > 0.0 else math.nan
        deviations = [sample[2] for sample in samples]
        rms_deviation = (
            math.sqrt(sum(value * value for value in deviations) / len(deviations))
            if deviations
            else math.nan
        )
        max_deviation = max(deviations) if deviations else math.nan
        relative_speed_error: float | None = None
        if self.expected_speed_mps is not None and math.isfinite(mean_speed):
            relative_speed_error = abs(mean_speed - self.expected_speed_mps) / (
                self.expected_speed_mps
            )

        reasons = {str(reason).strip() for reason in additional_off_protocol_reasons}
        reasons.discard("")
        if self.intended_crossing_count != 1:
            reasons.add("intended_crossing_count_not_one")
        if self.reverse_crossing_count:
            reasons.add("reverse_or_extra_crossing")
        if self.invalid_sample_count:
            reasons.add("tracking_invalid_during_crossing")
        if not math.isfinite(max_deviation) or (
            max_deviation > self.path_deviation_threshold_m
        ):
            reasons.add("path_deviation_exceeded")
        if (
            self.relative_speed_tolerance is not None
            and (
                relative_speed_error is None
                or relative_speed_error > self.relative_speed_tolerance
            )
        ):
            reasons.add("speed_deviation_exceeded")
        if (
            self.allowed_reverse_progress_m is not None
            and self.maximum_reverse_progress_m
            > self.allowed_reverse_progress_m + 1e-12
        ):
            reasons.add("reverse_progress_exceeded")
        return CrossingMetrics(
            intended_crossing_count=self.intended_crossing_count,
            reverse_crossing_count=self.reverse_crossing_count,
            first_completed_traversal_sign=self.first_completed_traversal_sign,
            valid_sample_count=len(samples),
            invalid_sample_count=self.invalid_sample_count,
            path_length_m=path_length,
            duration_s=duration,
            mean_speed_mps=mean_speed,
            rms_path_deviation_m=rms_deviation,
            maximum_path_deviation_m=max_deviation,
            path_deviation_threshold_m=self.path_deviation_threshold_m,
            expected_speed_mps=self.expected_speed_mps,
            relative_speed_error=relative_speed_error,
            maximum_reverse_progress_m=self.maximum_reverse_progress_m,
            off_protocol=bool(reasons),
            off_protocol_reasons=tuple(sorted(reasons)),
        )

    def _project(
        self, point: tuple[float, float, float]
    ) -> tuple[float, float]:
        relative = tuple(value - origin for value, origin in zip(point, self.start))
        axial_m = sum(value * axis for value, axis in zip(relative, self.direction))
        closest = tuple(
            origin + axial_m * axis for origin, axis in zip(self.start, self.direction)
        )
        deviation = math.dist(point, closest)
        return axial_m / self.length, deviation


@dataclass(frozen=True)
class FreeCrossingCompletion:
    """One complete endpoint-to-endpoint traversal of the canonical corridor."""

    hand: str
    sign: int
    origin_gate: str
    destination_gate: str
    start_time_s: float
    end_time_s: float
    path_length_m: float
    duration_s: float
    mean_speed_mps: float
    rms_path_deviation_m: float
    maximum_path_deviation_m: float
    interaction_band_entered: bool


@dataclass(frozen=True)
class FreeSingleCrossingUpdate:
    """Per-sample result from :class:`FreeSingleCrossingMonitor`.

    ``completions`` can contain two records when both hands reach their
    opposite gates in the same control sample.  Their stable order is left,
    then right; both still contribute to ``total_crossing_count``.
    """

    completions: tuple[FreeCrossingCompletion, ...]
    total_crossing_count: int
    first_crossing_hand: str | None
    first_completed_traversal_sign: int
    left_state: str
    right_state: str
    left_started_now: bool
    right_started_now: bool
    left_progress: float | None
    right_progress: float | None
    left_path_deviation_m: float | None
    right_path_deviation_m: float | None


@dataclass(frozen=True)
class FreeSingleCrossingMetrics:
    """Latched first-crossing measurements and whole-window validity."""

    crossing_count: int
    first_crossing_hand: str | None
    first_completed_traversal_sign: int
    first_start_time_s: float
    first_end_time_s: float
    first_path_length_m: float
    first_duration_s: float
    first_mean_speed_mps: float
    first_rms_path_deviation_m: float
    first_maximum_path_deviation_m: float
    first_interaction_band_entered: bool
    invalid_sample_count: int
    left_invalid_sample_count: int
    right_invalid_sample_count: int
    off_protocol: bool
    off_protocol_reasons: tuple[str, ...]


class _FreeHandTraversalState:
    """Mutable per-hand state owned by ``FreeSingleCrossingMonitor``."""

    def __init__(self) -> None:
        self.origin_gate: str | None = None
        self.samples: list[
            tuple[float, tuple[float, float, float], float]
        ] = []
        self.interaction_band_entered = False
        self.invalid_sample_count = 0
        self.progress: float | None = None
        self.path_deviation_m: float | None = None

    @property
    def state(self) -> str:
        if self.origin_gate is None:
            return "awaiting_gate"
        return f"armed_{self.origin_gate.lower()}"


class FreeSingleCrossingMonitor:
    """Observe one free, bilateral crossing after an explicit START reset.

    The corridor is direction-free.  Gate A lies at ``center - tangent *
    half_span_m`` and gate B at ``center + tangent * half_span_m``.  A hand is
    in gate A when it is beyond A's axial side plane while remaining inside
    the broad corridor cylinder; gate B is defined symmetrically.  Reaching
    either gate arms that hand, and only reaching the opposite gate completes
    a traversal.  The canonical sign is +1 for A (``-tangent``) to B
    (``+tangent``), and -1 for B-to-A.  The monitor deliberately has no world
    direction labels; with the collector's world +Y tangent convention, +1 is
    the participant's right-to-left traversal.

    The interaction band is a sphere centered on the corridor center.  A
    completed traversal enters it when a valid sample, or a straight segment
    between consecutive valid samples, comes within ``interaction_radius_m``.
    This makes the check robust to a control sample stepping across the band.
    Call :meth:`reset` at START; observations before that reset then cannot
    contribute to the trial's cardinality or first-crossing metrics.
    """

    def __init__(
        self,
        *,
        center_position_world: Sequence[float],
        tangent_world: Sequence[float],
        half_span_m: float,
        gate_radius_m: float,
        interaction_radius_m: float,
    ) -> None:
        self.center = _point3(center_position_world, "center_position_world")
        tangent = _point3(tangent_world, "tangent_world")
        tangent_norm = math.sqrt(sum(value * value for value in tangent))
        if tangent_norm <= 1e-9:
            raise ValueError("tangent_world must have non-zero length")
        self.tangent = tuple(value / tangent_norm for value in tangent)
        self.half_span_m = float(half_span_m)
        self.gate_radius_m = float(gate_radius_m)
        self.interaction_radius_m = float(interaction_radius_m)
        if not math.isfinite(self.half_span_m) or self.half_span_m <= 0.0:
            raise ValueError("half_span_m must be finite and positive")
        if (
            not math.isfinite(self.gate_radius_m)
            or self.gate_radius_m <= 0.0
        ):
            raise ValueError("gate_radius_m must be finite and positive")
        if (
            not math.isfinite(self.interaction_radius_m)
            or self.interaction_radius_m <= 0.0
        ):
            raise ValueError("interaction_radius_m must be finite and positive")
        self.gate_a = tuple(
            center - self.half_span_m * axis
            for center, axis in zip(self.center, self.tangent)
        )
        self.gate_b = tuple(
            center + self.half_span_m * axis
            for center, axis in zip(self.center, self.tangent)
        )
        self.reset()

    def reset(self) -> None:
        """Clear every pre-START arm, count, sample, and validity latch."""

        self._hands = {
            "left": _FreeHandTraversalState(),
            "right": _FreeHandTraversalState(),
        }
        self.crossing_count = 0
        self._first_completion: FreeCrossingCompletion | None = None
        self._last_time_s: float | None = None

    def observe(
        self,
        *,
        left_position_world: Sequence[float] | None,
        right_position_world: Sequence[float] | None,
        left_valid: bool,
        right_valid: bool,
        simulation_time_s: float,
    ) -> FreeSingleCrossingUpdate:
        now = float(simulation_time_s)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("simulation_time_s must be finite and non-negative")
        if self._last_time_s is not None and now < self._last_time_s:
            raise ValueError("crossing samples require monotonic simulation time")
        self._last_time_s = now

        completions: list[FreeCrossingCompletion] = []
        positions = {
            "left": left_position_world,
            "right": right_position_world,
        }
        validity = {"left": bool(left_valid), "right": bool(right_valid)}
        started_now = {"left": False, "right": False}
        for hand in ("left", "right"):
            was_unarmed = self._hands[hand].origin_gate is None
            completion = self._observe_hand(
                hand=hand,
                position_world=positions[hand],
                valid=validity[hand],
                simulation_time_s=now,
            )
            started_now[hand] = bool(
                was_unarmed and self._hands[hand].origin_gate is not None
            )
            if completion is None:
                continue
            completions.append(completion)
            self.crossing_count += 1
            if self._first_completion is None:
                self._first_completion = completion

        first = self._first_completion
        left = self._hands["left"]
        right = self._hands["right"]
        return FreeSingleCrossingUpdate(
            completions=tuple(completions),
            total_crossing_count=self.crossing_count,
            first_crossing_hand=None if first is None else first.hand,
            first_completed_traversal_sign=0 if first is None else first.sign,
            left_state=left.state,
            right_state=right.state,
            left_started_now=started_now["left"],
            right_started_now=started_now["right"],
            left_progress=left.progress,
            right_progress=right.progress,
            left_path_deviation_m=left.path_deviation_m,
            right_path_deviation_m=right.path_deviation_m,
        )

    def finalize(
        self, *, additional_off_protocol_reasons: Iterable[str] = ()
    ) -> FreeSingleCrossingMetrics:
        reasons = {str(reason).strip() for reason in additional_off_protocol_reasons}
        reasons.discard("")
        if self.crossing_count == 0:
            reasons.add("crossing_not_completed")
        elif self.crossing_count > 1:
            reasons.add("multiple_crossing")

        left_invalid = self._hands["left"].invalid_sample_count
        right_invalid = self._hands["right"].invalid_sample_count
        invalid = left_invalid + right_invalid
        if invalid:
            reasons.add("tracking_invalid_during_crossing")

        first = self._first_completion
        if first is not None and not first.interaction_band_entered:
            reasons.add("interaction_band_not_entered")
        return FreeSingleCrossingMetrics(
            crossing_count=self.crossing_count,
            first_crossing_hand=None if first is None else first.hand,
            first_completed_traversal_sign=0 if first is None else first.sign,
            first_start_time_s=-1.0 if first is None else first.start_time_s,
            first_end_time_s=-1.0 if first is None else first.end_time_s,
            first_path_length_m=0.0 if first is None else first.path_length_m,
            first_duration_s=0.0 if first is None else first.duration_s,
            first_mean_speed_mps=0.0 if first is None else first.mean_speed_mps,
            first_rms_path_deviation_m=(
                0.0 if first is None else first.rms_path_deviation_m
            ),
            first_maximum_path_deviation_m=(
                0.0 if first is None else first.maximum_path_deviation_m
            ),
            first_interaction_band_entered=(
                False if first is None else first.interaction_band_entered
            ),
            invalid_sample_count=invalid,
            left_invalid_sample_count=left_invalid,
            right_invalid_sample_count=right_invalid,
            off_protocol=bool(reasons),
            off_protocol_reasons=tuple(sorted(reasons)),
        )

    def _observe_hand(
        self,
        *,
        hand: str,
        position_world: Sequence[float] | None,
        valid: bool,
        simulation_time_s: float,
    ) -> FreeCrossingCompletion | None:
        state = self._hands[hand]
        if not valid:
            state.invalid_sample_count += 1
            state.progress = None
            state.path_deviation_m = None
            return None
        if position_world is None:
            raise ValueError(f"{hand}_position_world is required when {hand}_valid is true")
        point = _point3(position_world, f"{hand}_position_world")
        progress, deviation = self._project_from_center(point)
        state.progress = progress
        state.path_deviation_m = deviation
        # Broad, invisible gates must not require the participant to find an
        # endpoint sphere.  Each gate is the intersection of a side half-space
        # and the canonical corridor cylinder.  In normalized coordinates the
        # side planes are progress=-1 (A) and progress=+1 (B).
        inside_gate_cylinder = deviation <= self.gate_radius_m
        at_a = progress <= -1.0 and inside_gate_cylinder
        at_b = progress >= 1.0 and inside_gate_cylinder
        gate = "A" if at_a else "B" if at_b else None

        if state.origin_gate is None:
            if gate is not None:
                self._begin_attempt(state, gate, point, deviation, simulation_time_s)
            return None

        self._append_attempt_sample(state, point, deviation, simulation_time_s)
        destination = "B" if state.origin_gate == "A" else "A"
        if gate == destination:
            completion = self._complete_attempt(
                hand=hand,
                state=state,
                destination_gate=destination,
            )
            # The destination is immediately the origin for a possible complete
            # return.  Mere dwell or a partial departure/return does not count.
            self._begin_attempt(
                state, destination, point, deviation, simulation_time_s
            )
            return completion
        if gate == state.origin_gate:
            # Returning to the origin is an abandoned attempt, not a crossing.
            self._begin_attempt(
                state, state.origin_gate, point, deviation, simulation_time_s
            )
        return None

    def _begin_attempt(
        self,
        state: _FreeHandTraversalState,
        origin_gate: str,
        point: tuple[float, float, float],
        deviation: float,
        simulation_time_s: float,
    ) -> None:
        state.origin_gate = origin_gate
        state.samples = [(simulation_time_s, point, deviation)]
        state.interaction_band_entered = (
            math.dist(point, self.center) <= self.interaction_radius_m
        )

    def _append_attempt_sample(
        self,
        state: _FreeHandTraversalState,
        point: tuple[float, float, float],
        deviation: float,
        simulation_time_s: float,
    ) -> None:
        if state.samples:
            prior = state.samples[-1][1]
            if (
                _point_to_segment_distance(self.center, prior, point)
                <= self.interaction_radius_m
            ):
                state.interaction_band_entered = True
        elif math.dist(point, self.center) <= self.interaction_radius_m:
            state.interaction_band_entered = True
        state.samples.append((simulation_time_s, point, deviation))

    @staticmethod
    def _complete_attempt(
        *,
        hand: str,
        state: _FreeHandTraversalState,
        destination_gate: str,
    ) -> FreeCrossingCompletion:
        samples = state.samples
        path_length = sum(
            math.dist(prior[1], current[1])
            for prior, current in zip(samples, samples[1:])
        )
        duration = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
        mean_speed = path_length / duration if duration > 0.0 else 0.0
        deviations = [sample[2] for sample in samples]
        rms_deviation = math.sqrt(
            sum(value * value for value in deviations) / len(deviations)
        )
        return FreeCrossingCompletion(
            hand=hand,
            sign=1 if state.origin_gate == "A" else -1,
            origin_gate=str(state.origin_gate),
            destination_gate=destination_gate,
            start_time_s=samples[0][0],
            end_time_s=samples[-1][0],
            path_length_m=path_length,
            duration_s=duration,
            mean_speed_mps=mean_speed,
            rms_path_deviation_m=rms_deviation,
            maximum_path_deviation_m=max(deviations),
            interaction_band_entered=state.interaction_band_entered,
        )

    def _project_from_center(
        self, point: tuple[float, float, float]
    ) -> tuple[float, float]:
        relative = tuple(value - origin for value, origin in zip(point, self.center))
        axial_m = sum(value * axis for value, axis in zip(relative, self.tangent))
        closest = tuple(
            origin + axial_m * axis for origin, axis in zip(self.center, self.tangent)
        )
        return axial_m / self.half_span_m, math.dist(point, closest)


def _point_to_segment_distance(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    delta = tuple(b - a for a, b in zip(start, end))
    length_squared = sum(value * value for value in delta)
    if length_squared <= 1e-18:
        return math.dist(point, start)
    relative = tuple(value - origin for value, origin in zip(point, start))
    fraction = sum(value * axis for value, axis in zip(relative, delta)) / length_squared
    fraction = max(0.0, min(1.0, fraction))
    closest = tuple(origin + fraction * axis for origin, axis in zip(start, delta))
    return math.dist(point, closest)


def _point3(value: Sequence[float], field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three coordinates")
    point = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"{field_name} must be finite")
    return point  # type: ignore[return-value]


__all__ = [
    "ACTUAL_CROSSING_NOT_OBSERVED",
    "CROSSING_DIRECTIONS",
    "resolve_actual_crossing_direction",
    "EncounterDetectorConfig",
    "EncounterRecord",
    "EncounterDetectorUpdate",
    "EncounterDetector",
    "CrossingMetrics",
    "CrossingUpdate",
    "CrossingPathMonitor",
    "FreeCrossingCompletion",
    "FreeSingleCrossingUpdate",
    "FreeSingleCrossingMetrics",
    "FreeSingleCrossingMonitor",
]
