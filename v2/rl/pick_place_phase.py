from __future__ import annotations

from typing import Sequence


PICK_PLACE_EVENT_COUNT = 10
PICK_PLACE_EVENTS_DT = (0.008, 0.005, 1.0, 0.1, 0.05, 0.05, 0.0025, 1.0, 0.008, 0.08)
PICK_PLACE_EVENT_NAMES = (
    "move_above_cube",
    "lower_to_cube",
    "settle_before_grasp",
    "close_gripper",
    "lift_cube",
    "move_to_goal_xy",
    "lower_to_goal",
    "open_gripper",
    "lift_after_release",
    "return_to_pick_xy",
)


def task_phase_from_event(event: int | None) -> str:
    if event is None:
        return "approach_cube"
    event = int(event)
    if event <= 0:
        return "approach_cube"
    if event in (1, 2, 3):
        return "grasp_cube"
    if event in (4, 5, 6):
        return "move_to_target"
    return "release_cube"


def advance_pick_place_event(
    event: int,
    t: float,
    events_dt: Sequence[float] = PICK_PLACE_EVENTS_DT,
) -> tuple[int, float]:
    event = int(event)
    if event >= len(events_dt):
        return event, 0.0
    next_t = float(t) + float(events_dt[event])
    if next_t >= 1.0:
        return event + 1, 0.0
    return event, next_t


def event_gripper_command(event: int | None, was_closed: bool) -> bool:
    if event == 3:
        return True
    if event == 7:
        return False
    return bool(was_closed)
