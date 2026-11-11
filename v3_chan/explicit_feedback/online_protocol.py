"""Pure-Python protocol primitives for online explicit-feedback collection.

This module deliberately has no Isaac Sim, Torch, NumPy, HDF5, or XR imports.
It defines the parts of ``bc_cbf_online_explicit_feedback_v1`` that must be
deterministic and independently testable:

* participant/session-specific trial schedules;
* the online encounter detector (using only runtime-observable signals); and
* planned-corridor crossing measurements.

The older explicit-feedback schema and state machines are intentionally not
imported or modified here.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator, Mapping, Sequence


ONLINE_SCHEMA_VERSION = "bc_cbf_online_explicit_feedback_v1"

TARGET_PHASES = (
    "reach_approach",
    "grasp_lift",
    "transport",
    "place_release",
)
SEVERITIES = ("control", "shallow", "threat")
CROSSING_DIRECTIONS = ("left_to_right", "right_to_left")
CROSSING_SPEEDS = ("slow", "fast")


@dataclass(frozen=True)
class SeverityDefinition:
    """Planned protected-surface gap for one severity condition."""

    name: str
    minimum_gap_m: float
    maximum_gap_m: float
    target_gap_m: float

    def __post_init__(self) -> None:
        if self.name not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.name!r}")
        values = (self.minimum_gap_m, self.maximum_gap_m, self.target_gap_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("severity gaps must be finite")
        if not 0.0 <= self.minimum_gap_m <= self.target_gap_m <= self.maximum_gap_m:
            raise ValueError("severity gap range must contain its target")


SEVERITY_DEFINITIONS: Mapping[str, SeverityDefinition] = {
    "control": SeverityDefinition("control", 0.16, 0.20, 0.18),
    "shallow": SeverityDefinition("shallow", 0.08, 0.12, 0.10),
    "threat": SeverityDefinition("threat", 0.00, 0.04, 0.02),
}


@dataclass(frozen=True)
class TrialSpec:
    """One immutable planned trial and its mandatory query identity."""

    trial_id: str
    query_id: str
    order_index: int
    evaluated_index: int | None
    task_phase: str
    severity: str
    crossing_direction: str
    crossing_speed: str
    planned_gap_min_m: float
    planned_gap_max_m: float
    planned_gap_target_m: float
    practice: bool = False
    analysis_exclude: bool = False
    anchor_repeat: bool = False
    repeated_trial_id: str | None = None
    pilot: bool = False

    def __post_init__(self) -> None:
        if not self.trial_id or not self.query_id:
            raise ValueError("trial_id and query_id must be non-empty")
        if self.order_index < 0:
            raise ValueError("order_index must be non-negative")
        if self.task_phase not in TARGET_PHASES:
            raise ValueError(f"unknown task phase: {self.task_phase!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity!r}")
        if self.crossing_direction not in CROSSING_DIRECTIONS:
            raise ValueError(
                f"unknown crossing direction: {self.crossing_direction!r}"
            )
        if self.crossing_speed not in CROSSING_SPEEDS:
            raise ValueError(f"unknown crossing speed: {self.crossing_speed!r}")
        if self.practice != (self.evaluated_index is None):
            raise ValueError("practice trials alone must have no evaluated_index")
        if self.practice and not self.analysis_exclude:
            raise ValueError("practice trials must be excluded from analysis")
        if self.anchor_repeat != (self.repeated_trial_id is not None):
            raise ValueError(
                "anchor_repeat and repeated_trial_id must be specified together"
            )
        definition = SEVERITY_DEFINITIONS[self.severity]
        actual = (
            float(self.planned_gap_min_m),
            float(self.planned_gap_max_m),
            float(self.planned_gap_target_m),
        )
        expected = (
            definition.minimum_gap_m,
            definition.maximum_gap_m,
            definition.target_gap_m,
        )
        if actual != expected:
            raise ValueError(
                f"severity {self.severity!r} must use frozen gap values {expected}"
            )

    @property
    def is_evaluated(self) -> bool:
        return not self.practice

    @property
    def target_phase(self) -> str:
        """Protocol-facing alias used by the collector and stored schema."""

        return self.task_phase

    @property
    def condition_key(self) -> tuple[str, str, str]:
        """The three experimental factors; speed is deliberately not a factor."""

        return self.task_phase, self.severity, self.crossing_direction

    def as_dict(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "query_id": self.query_id,
            "order_index": self.order_index,
            "evaluated_index": self.evaluated_index,
            "target_phase": self.task_phase,
            "severity": self.severity,
            "crossing_direction": self.crossing_direction,
            "crossing_speed": self.crossing_speed,
            "planned_gap_min_m": self.planned_gap_min_m,
            "planned_gap_max_m": self.planned_gap_max_m,
            "planned_gap_target_m": self.planned_gap_target_m,
            "practice": self.practice,
            "analysis_exclude": self.analysis_exclude,
            "anchor_repeat": self.anchor_repeat,
            "repeated_trial_id": self.repeated_trial_id,
            "pilot": self.pilot,
        }


@dataclass(frozen=True)
class ProtocolSchedule(Sequence[TrialSpec]):
    """A sequence plus the identifiers used to derive its stable order."""

    participant_id: str
    session_id: str
    randomization_seed: str
    trials: tuple[TrialSpec, ...]
    pilot: bool
    anchors_enabled: bool

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.trials[index]

    def __iter__(self) -> Iterator[TrialSpec]:
        return iter(self.trials)

    @property
    def practice_trials(self) -> tuple[TrialSpec, ...]:
        return tuple(trial for trial in self.trials if trial.practice)

    @property
    def evaluated_trials(self) -> tuple[TrialSpec, ...]:
        return tuple(trial for trial in self.trials if trial.is_evaluated)

    @property
    def anchor_trials(self) -> tuple[TrialSpec, ...]:
        return tuple(trial for trial in self.trials if trial.anchor_repeat)


@dataclass(frozen=True)
class _Condition:
    phase: str
    severity: str
    direction: str
    speed: str

    @property
    def factor_key(self) -> tuple[str, str, str]:
        return self.phase, self.severity, self.direction


def build_participant_schedule(
    participant_id: str,
    *,
    session_id: str = "session-1",
    seed: str | int = 0,
    practice_trials: int = 4,
    include_anchor_repeats: bool | None = None,
    pilot_evaluated_trials: int | None = None,
) -> ProtocolSchedule:
    """Build a deterministic, participant-specific protocol schedule.

    A full schedule contains the exact 24-condition factorial and twelve trials
    at each non-factor crossing speed.  Four practice trials precede evaluated
    trials and are always marked ``analysis_exclude``.  When enabled, anchors
    repeat one prior evaluated condition per phase.

    ``pilot_evaluated_trials`` is the *total* evaluated count and must be in
    ``[12, 16]``.  Anchors default off for a pilot and on/off exactly as
    requested otherwise.  If pilot anchors are explicitly enabled, four of the
    requested total are repeats; the remaining trials are unique conditions.
    """

    participant_id = str(participant_id).strip()
    session_id = str(session_id).strip()
    if not participant_id:
        raise ValueError("participant_id must be non-empty")
    if not session_id:
        raise ValueError("session_id must be non-empty")
    practice_trials = int(practice_trials)
    if practice_trials < 0:
        raise ValueError("practice_trials must be non-negative")
    if pilot_evaluated_trials is not None:
        pilot_evaluated_trials = int(pilot_evaluated_trials)
        if not 12 <= pilot_evaluated_trials <= 16:
            raise ValueError("pilot_evaluated_trials must be between 12 and 16")
    pilot = pilot_evaluated_trials is not None
    anchors_enabled = (
        False if pilot and include_anchor_repeats is None
        else bool(include_anchor_repeats)
    )
    seed_text = str(seed)
    schedule_key = f"{ONLINE_SCHEMA_VERSION}|{participant_id}|{session_id}|{seed_text}"

    factorial = _factorial_conditions(schedule_key)
    if pilot:
        unique_count = int(pilot_evaluated_trials) - (4 if anchors_enabled else 0)
        if unique_count < len(TARGET_PHASES):
            raise ValueError("pilot schedule needs at least one source per phase")
        evaluated_conditions = _balanced_subset(
            factorial,
            count=unique_count,
            key=f"{schedule_key}|pilot",
        )
    else:
        evaluated_conditions = factorial

    planned: list[TrialSpec] = []
    for practice_index in range(practice_trials):
        phase = TARGET_PHASES[practice_index % len(TARGET_PHASES)]
        severity = SEVERITIES[practice_index % len(SEVERITIES)]
        direction = CROSSING_DIRECTIONS[practice_index % 2]
        speed = CROSSING_SPEEDS[(practice_index // 2) % 2]
        planned.append(
            _make_trial(
                schedule_key=schedule_key,
                token=f"practice|{practice_index}|{phase}|{severity}|{direction}|{speed}",
                order_index=len(planned),
                evaluated_index=None,
                condition=_Condition(phase, severity, direction, speed),
                practice=True,
                analysis_exclude=True,
                pilot=pilot,
            )
        )

    evaluated: list[TrialSpec] = []
    for condition_index, condition in enumerate(evaluated_conditions):
        evaluated.append(
            _make_trial(
                schedule_key=schedule_key,
                token=(
                    f"evaluated|{condition_index}|{condition.phase}|"
                    f"{condition.severity}|{condition.direction}|{condition.speed}"
                ),
                order_index=len(planned) + len(evaluated),
                evaluated_index=len(evaluated),
                condition=condition,
                practice=False,
                analysis_exclude=False,
                pilot=pilot,
            )
        )

    if anchors_enabled:
        sources = _one_anchor_source_per_phase(
            evaluated, key=f"{schedule_key}|anchor-sources"
        )
        sources = _stable_order(sources, f"{schedule_key}|anchor-order")
        for anchor_index, source in enumerate(sources):
            condition = _Condition(
                source.task_phase,
                source.severity,
                source.crossing_direction,
                source.crossing_speed,
            )
            evaluated.append(
                _make_trial(
                    schedule_key=schedule_key,
                    token=f"anchor|{anchor_index}|{source.trial_id}",
                    order_index=len(planned) + len(evaluated),
                    evaluated_index=len(evaluated),
                    condition=condition,
                    practice=False,
                    analysis_exclude=False,
                    anchor_repeat=True,
                    repeated_trial_id=source.trial_id,
                    pilot=pilot,
                )
            )

    if pilot and len(evaluated) != pilot_evaluated_trials:
        raise AssertionError("internal pilot schedule count mismatch")
    planned.extend(evaluated)
    _validate_schedule_invariants(
        planned,
        pilot=pilot,
        anchors_enabled=anchors_enabled,
    )
    return ProtocolSchedule(
        participant_id=participant_id,
        session_id=session_id,
        randomization_seed=seed_text,
        trials=tuple(planned),
        pilot=pilot,
        anchors_enabled=anchors_enabled,
    )


# Readable alias for callers that use protocol terminology rather than study
# terminology.  Keeping one implementation prevents schedule drift.
build_protocol_schedule = build_participant_schedule


def _factorial_conditions(schedule_key: str) -> list[_Condition]:
    """Return all 24 factor cells, ordered and speed-balanced per phase."""

    conditions: list[_Condition] = []
    for phase in TARGET_PHASES:
        phase_cells = [
            (severity, direction)
            for severity in SEVERITIES
            for direction in CROSSING_DIRECTIONS
        ]
        speed_order = _stable_order(
            phase_cells, f"{schedule_key}|speed-allocation|{phase}"
        )
        slow_cells = set(speed_order[: len(phase_cells) // 2])
        for severity, direction in phase_cells:
            speed = "slow" if (severity, direction) in slow_cells else "fast"
            conditions.append(_Condition(phase, severity, direction, speed))
    return _stable_order(conditions, f"{schedule_key}|condition-order")


def _balanced_subset(
    conditions: Sequence[_Condition], *, count: int, key: str
) -> list[_Condition]:
    """Choose a pilot subset whose four planned marginals differ by at most one.

    Speed is assigned to the full factorial before this function is called, so
    independently taking a balanced number from each phase is not sufficient:
    it can substantially skew severity or direction.  This deterministic
    bounded search selects unique cells while enforcing the floor/ceiling
    quota for phase, severity, direction, and speed simultaneously.
    """

    if not 0 <= count <= len(conditions):
        raise ValueError("subset count is out of range")
    ranked = _stable_order(conditions, f"{key}|rank")
    dimensions = (
        (TARGET_PHASES, lambda condition: condition.phase),
        (SEVERITIES, lambda condition: condition.severity),
        (CROSSING_DIRECTIONS, lambda condition: condition.direction),
        (CROSSING_SPEEDS, lambda condition: condition.speed),
    )
    category_keys = tuple(
        (dimension_index, category)
        for dimension_index, (categories, _getter) in enumerate(dimensions)
        for category in categories
    )
    category_index = {value: index for index, value in enumerate(category_keys)}
    lower_bounds: list[int] = []
    upper_bounds: list[int] = []
    for categories, _getter in dimensions:
        lower, extra = divmod(count, len(categories))
        upper = lower + int(extra > 0)
        lower_bounds.extend([lower] * len(categories))
        upper_bounds.extend([upper] * len(categories))

    vectors: list[tuple[int, ...]] = []
    for condition in ranked:
        vector = [0] * len(category_keys)
        for dimension_index, (_categories, getter) in enumerate(dimensions):
            vector[category_index[(dimension_index, getter(condition))]] = 1
        vectors.append(tuple(vector))

    # suffix_available[i][j] is the number of remaining condition cells that
    # can still contribute to category j.  It makes impossible branches close
    # early and keeps the search tiny for the 24-cell factorial.
    suffix_available = [[0] * len(category_keys) for _ in range(len(ranked) + 1)]
    for index in range(len(ranked) - 1, -1, -1):
        suffix_available[index] = [
            suffix_available[index + 1][category] + vectors[index][category]
            for category in range(len(category_keys))
        ]

    @lru_cache(maxsize=None)
    def search(
        index: int, remaining: int, counts: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        if remaining == 0:
            if all(
                lower <= observed <= upper
                for observed, lower, upper in zip(
                    counts, lower_bounds, upper_bounds
                )
            ):
                return ()
            return None
        if len(ranked) - index < remaining:
            return None
        if any(
            observed > upper
            or observed + suffix_available[index][category] < lower
            for category, (observed, lower, upper) in enumerate(
                zip(counts, lower_bounds, upper_bounds)
            )
        ):
            return None

        vector = vectors[index]
        included_counts = tuple(
            observed + contribution
            for observed, contribution in zip(counts, vector)
        )
        if all(
            observed <= upper
            for observed, upper in zip(included_counts, upper_bounds)
        ):
            tail = search(index + 1, remaining - 1, included_counts)
            if tail is not None:
                return (index, *tail)
        return search(index + 1, remaining, counts)

    selected_indices = search(0, count, (0,) * len(category_keys))
    if selected_indices is None:
        raise AssertionError(
            "unable to balance pilot phase/severity/direction/speed marginals"
        )
    return [ranked[index] for index in selected_indices]


def _make_trial(
    *,
    schedule_key: str,
    token: str,
    order_index: int,
    evaluated_index: int | None,
    condition: _Condition,
    practice: bool,
    analysis_exclude: bool,
    anchor_repeat: bool = False,
    repeated_trial_id: str | None = None,
    pilot: bool,
) -> TrialSpec:
    definition = SEVERITY_DEFINITIONS[condition.severity]
    trial_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{schedule_key}|trial|{token}"))
    query_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{schedule_key}|query|{token}"))
    return TrialSpec(
        trial_id=trial_id,
        query_id=query_id,
        order_index=order_index,
        evaluated_index=evaluated_index,
        task_phase=condition.phase,
        severity=condition.severity,
        crossing_direction=condition.direction,
        crossing_speed=condition.speed,
        planned_gap_min_m=definition.minimum_gap_m,
        planned_gap_max_m=definition.maximum_gap_m,
        planned_gap_target_m=definition.target_gap_m,
        practice=practice,
        analysis_exclude=analysis_exclude,
        anchor_repeat=anchor_repeat,
        repeated_trial_id=repeated_trial_id,
        pilot=pilot,
    )


def _one_anchor_source_per_phase(
    trials: Sequence[TrialSpec], *, key: str
) -> list[TrialSpec]:
    sources: list[TrialSpec] = []
    for phase in TARGET_PHASES:
        candidates = [trial for trial in trials if trial.task_phase == phase]
        if not candidates:
            raise ValueError(f"cannot create anchor: no source for phase {phase!r}")
        sources.append(_stable_order(candidates, f"{key}|{phase}")[0])
    return sources


def _stable_order(values: Iterable, key: str) -> list:  # type: ignore[type-arg]
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{key}|{value!r}".encode("utf-8")
        ).digest(),
    )


def _validate_schedule_invariants(
    trials: Sequence[TrialSpec], *, pilot: bool, anchors_enabled: bool
) -> None:
    trial_ids = [trial.trial_id for trial in trials]
    query_ids = [trial.query_id for trial in trials]
    if len(trial_ids) != len(set(trial_ids)):
        raise AssertionError("trial IDs are not unique")
    if len(query_ids) != len(set(query_ids)):
        raise AssertionError("query IDs are not unique")
    if any(trial.order_index != index for index, trial in enumerate(trials)):
        raise AssertionError("schedule order indexes are not contiguous")
    evaluated = [trial for trial in trials if trial.is_evaluated]
    if any(trial.evaluated_index != index for index, trial in enumerate(evaluated)):
        raise AssertionError("evaluated indexes are not contiguous")
    source_ids = {trial.trial_id for trial in evaluated if not trial.anchor_repeat}
    if any(
        trial.anchor_repeat and trial.repeated_trial_id not in source_ids
        for trial in evaluated
    ):
        raise AssertionError("anchor source is absent from the evaluated schedule")
    originals = [trial for trial in evaluated if not trial.anchor_repeat]
    if pilot:
        pilot_marginals = (
            ("phase", TARGET_PHASES, lambda trial: trial.task_phase),
            ("severity", SEVERITIES, lambda trial: trial.severity),
            (
                "direction",
                CROSSING_DIRECTIONS,
                lambda trial: trial.crossing_direction,
            ),
            ("speed", CROSSING_SPEEDS, lambda trial: trial.crossing_speed),
        )
        for label, categories, getter in pilot_marginals:
            counts = [sum(getter(trial) == value for trial in originals) for value in categories]
            if counts and max(counts) - min(counts) > 1:
                raise AssertionError(f"pilot {label} marginal is not balanced")
    else:
        if len(originals) != 24:
            raise AssertionError("full schedule does not contain 24 original trials")
        if len({trial.condition_key for trial in originals}) != 24:
            raise AssertionError("full schedule is not a complete 4x3x2 factorial")
        speed_counts = {
            speed: sum(trial.crossing_speed == speed for trial in originals)
            for speed in CROSSING_SPEEDS
        }
        if speed_counts != {"slow": 12, "fast": 12}:
            raise AssertionError("full schedule speed assignment is not 12/12")
    anchors = [trial for trial in evaluated if trial.anchor_repeat]
    if anchors_enabled:
        if len(anchors) != 4 or {trial.task_phase for trial in anchors} != set(
            TARGET_PHASES
        ):
            raise AssertionError("anchors must repeat exactly one trial per phase")
    elif anchors:
        raise AssertionError("anchors present when disabled")


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


def _point3(value: Sequence[float], field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three coordinates")
    point = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"{field_name} must be finite")
    return point  # type: ignore[return-value]


__all__ = [
    "ONLINE_SCHEMA_VERSION",
    "TARGET_PHASES",
    "SEVERITIES",
    "CROSSING_DIRECTIONS",
    "CROSSING_SPEEDS",
    "SEVERITY_DEFINITIONS",
    "SeverityDefinition",
    "TrialSpec",
    "ProtocolSchedule",
    "build_participant_schedule",
    "build_protocol_schedule",
    "EncounterDetectorConfig",
    "EncounterRecord",
    "EncounterDetectorUpdate",
    "EncounterDetector",
    "CrossingMetrics",
    "CrossingUpdate",
    "CrossingPathMonitor",
]
