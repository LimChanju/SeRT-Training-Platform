"""Isaac-independent speed profiles for the HRI pick-and-place controller."""

from __future__ import annotations

import json
import math


SPEED_PROFILE_ORDER = ("slow", "medium", "fast")
COUNTERBALANCED_SPEED_PROFILE_ORDERS = (
    ("slow", "medium", "fast"),
    ("medium", "fast", "slow"),
    ("fast", "slow", "medium"),
)

# Isaac Sim 4.5 Franka PickPlaceController defaults. Keeping the values here
# freezes the collection protocol instead of depending on an extension default.
BASE_EVENTS_DT = (0.008, 0.005, 1.0, 0.1, 0.05, 0.05, 0.0025, 1.0, 0.008, 0.08)

# Motion phases: pre-grasp, descend, lift, translate, place descend, retract,
# and return. Settling and gripper phases (2, 3, 7) stay unchanged.
MOTION_EVENT_INDICES = (0, 1, 4, 5, 6, 8, 9)
SPEED_PROFILE_MOTION_SCALES = {
    "slow": 1.0,
    "medium": 1.5,
    "fast": 2.0,
}


def normalize_speed_profile_order(
    profile_order: str | tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Validate and normalize one three-condition episode order."""

    if profile_order is None:
        values = SPEED_PROFILE_ORDER
    elif isinstance(profile_order, str):
        values = tuple(part.strip().lower() for part in profile_order.split(","))
    else:
        values = tuple(str(part).strip().lower() for part in profile_order)
    if len(values) != len(SPEED_PROFILE_ORDER) or set(values) != set(
        SPEED_PROFILE_ORDER
    ):
        raise ValueError(
            "speed profile order must contain slow, medium, and fast exactly once"
        )
    return values


def counterbalanced_speed_profile_order(order_index: int) -> tuple[str, ...]:
    """Return a cyclic Latin-square order for one session."""

    index = int(order_index)
    if index < 0:
        raise ValueError("order_index must be non-negative")
    return COUNTERBALANCED_SPEED_PROFILE_ORDERS[
        index % len(COUNTERBALANCED_SPEED_PROFILE_ORDERS)
    ]


def speed_profile_for_episode(
    episode_index: int,
    profile_order: str | tuple[str, ...] | list[str] | None = None,
) -> str:
    """Return the scheduled speed profile for a zero-based episode index."""

    index = int(episode_index)
    if index < 0:
        raise ValueError("episode_index must be non-negative")
    order = normalize_speed_profile_order(profile_order)
    return order[index % len(order)]


def events_dt_for_speed_profile(profile: str) -> tuple[float, ...]:
    """Build the ten FSM increments while preserving non-motion phase timing."""

    normalized = str(profile).strip().lower()
    if normalized not in SPEED_PROFILE_MOTION_SCALES:
        raise ValueError(
            f"Unknown pick-place speed profile {profile!r}; "
            f"expected one of {SPEED_PROFILE_ORDER}"
        )
    scale = SPEED_PROFILE_MOTION_SCALES[normalized]
    values = list(BASE_EVENTS_DT)
    for event_index in MOTION_EVENT_INDICES:
        values[event_index] *= scale
    return tuple(float(value) for value in values)


def nominal_profile_steps(profile: str) -> int:
    """Return nominal controller calls needed to complete all ten phases."""

    return sum(
        int(math.ceil(1.0 / value))
        for value in events_dt_for_speed_profile(profile)
    )


def speed_profile_metadata(
    profile: str,
    *,
    physics_dt_s: float = 1.0 / 60.0,
) -> dict[str, object]:
    """Return HDF5-safe metadata for one episode speed condition."""

    normalized = str(profile).strip().lower()
    events_dt = events_dt_for_speed_profile(normalized)
    nominal_steps = nominal_profile_steps(normalized)
    return {
        "controller_speed_profile": normalized,
        "controller_motion_phase_scale": float(
            SPEED_PROFILE_MOTION_SCALES[normalized]
        ),
        "controller_events_dt_json": json.dumps(events_dt),
        "controller_nominal_cycle_steps": int(nominal_steps),
        "controller_nominal_cycle_duration_s": float(
            nominal_steps * float(physics_dt_s)
        ),
    }


def speed_schedule_metadata(
    *,
    physics_dt_s: float = 1.0 / 60.0,
    profile_order: str | tuple[str, ...] | list[str] | None = None,
    counterbalance_order_index: int | None = None,
) -> dict[str, object]:
    """Return file-level metadata describing the frozen three-speed schedule."""

    order = normalize_speed_profile_order(profile_order)
    profiles = {
        profile: {
            "motion_phase_scale": SPEED_PROFILE_MOTION_SCALES[profile],
            "events_dt": events_dt_for_speed_profile(profile),
            "nominal_cycle_steps": nominal_profile_steps(profile),
            "nominal_cycle_duration_s": nominal_profile_steps(profile)
            * float(physics_dt_s),
        }
        for profile in SPEED_PROFILE_ORDER
    }
    metadata = {
        "controller_speed_schedule_version": "pick_place_speed_schedule_v2",
        "controller_speed_schedule": ",".join(order),
        "controller_speed_counterbalance_orders_json": json.dumps(
            COUNTERBALANCED_SPEED_PROFILE_ORDERS
        ),
        "controller_speed_motion_event_indices": ",".join(
            str(index) for index in MOTION_EVENT_INDICES
        ),
        "controller_speed_profiles_json": json.dumps(profiles),
    }
    if counterbalance_order_index is not None:
        metadata["controller_speed_counterbalance_order_index"] = int(
            counterbalance_order_index
        )
    return metadata
