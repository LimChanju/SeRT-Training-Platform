from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEGACY_TRANSITION_ACTION_VERSION = "action_v0_task_space"
ACTION_VERSION = "action_v1_controller_target_delta"
CONTROLLER_TARGET_ACTION_VERSION = ACTION_VERSION
ACTION_NAMES = ("dx", "dy", "dz", "dyaw", "gripper_cmd")
ACTION_DIM = len(ACTION_NAMES)
MAX_EE_DELTA_M = 0.03
CONTROLLER_TARGET_MAX_DELTA_M = 0.75
MAX_YAW_DELTA_RAD = 0.15


@dataclass(frozen=True)
class TaskSpaceAction:
    """Normalized 5D task-space action for policy learning."""

    vector: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        if arr.shape != (ACTION_DIM,):
            raise ValueError(f"TaskSpaceAction expects shape ({ACTION_DIM},), got {arr.shape}")
        object.__setattr__(self, "vector", np.clip(arr, -1.0, 1.0))

    @property
    def ee_delta_pos_m(self) -> np.ndarray:
        return self.vector[:3] * MAX_EE_DELTA_M

    @property
    def yaw_delta_rad(self) -> float:
        return float(self.vector[3] * MAX_YAW_DELTA_RAD)

    @property
    def gripper_cmd(self) -> float:
        return float(self.vector[4])

    @property
    def gripper_should_open(self) -> bool:
        return self.gripper_cmd >= 0.0


def zero_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


def clip_action(action: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float32).reshape(ACTION_DIM), -1.0, 1.0)


def denormalize_action(action: np.ndarray) -> dict[str, np.ndarray | float | bool]:
    task_action = TaskSpaceAction(action)
    return {
        "ee_delta_pos_m": task_action.ee_delta_pos_m.astype(np.float32),
        "yaw_delta_rad": task_action.yaw_delta_rad,
        "gripper_cmd": task_action.gripper_cmd,
        "gripper_should_open": task_action.gripper_should_open,
    }


def controller_target_action_from_target(
    ee_pos_now: np.ndarray,
    target_pos: np.ndarray,
    *,
    yaw_now: float = 0.0,
    yaw_target: float = 0.0,
    gripper_cmd: float = 0.0,
) -> np.ndarray:
    """Encode the expert controller's intended EE target as a normalized 5D action."""

    ee_pos_now = np.asarray(ee_pos_now, dtype=float).reshape(3)
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    delta_pos = (target_pos - ee_pos_now) / float(CONTROLLER_TARGET_MAX_DELTA_M)
    delta_yaw = _wrap_angle(float(yaw_target) - float(yaw_now)) / float(MAX_YAW_DELTA_RAD)
    return clip_action(np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, gripper_cmd]))


def controller_target_from_action(
    ee_pos_now: np.ndarray,
    action: np.ndarray,
    *,
    action_scale: float = 1.0,
) -> np.ndarray:
    ee_pos_now = np.asarray(ee_pos_now, dtype=float).reshape(3)
    action = clip_action(action)
    return ee_pos_now + action[:3].astype(float) * float(CONTROLLER_TARGET_MAX_DELTA_M) * float(action_scale)


def task_action_from_transition(
    ee_pos_now: np.ndarray,
    ee_pos_next: np.ndarray,
    *,
    yaw_now: float = 0.0,
    yaw_next: float = 0.0,
    gripper_opening_now: float | None = None,
    gripper_opening_next: float | None = None,
) -> np.ndarray:
    """Infer normalized 5D expert task action from consecutive states."""

    ee_pos_now = np.asarray(ee_pos_now, dtype=float).reshape(3)
    ee_pos_next = np.asarray(ee_pos_next, dtype=float).reshape(3)
    delta_pos = (ee_pos_next - ee_pos_now) / float(MAX_EE_DELTA_M)
    delta_yaw = _wrap_angle(float(yaw_next) - float(yaw_now)) / float(MAX_YAW_DELTA_RAD)
    gripper_cmd = _gripper_transition_cmd(gripper_opening_now, gripper_opening_next)
    return clip_action(np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, gripper_cmd]))


def expert_joint_action_vector(action, size: int = 9) -> np.ndarray:
    """Best-effort extraction of a raw expert articulation action vector."""

    if action is None:
        return np.zeros(size, dtype=np.float32)
    for attr in ("joint_positions", "joint_velocities", "joint_efforts"):
        value = getattr(action, attr, None)
        if value is not None:
            return _fixed_array(value, size).astype(np.float32)
    try:
        return _fixed_array(action, size).astype(np.float32)
    except Exception:
        return np.zeros(size, dtype=np.float32)


def _gripper_transition_cmd(
    gripper_opening_now: float | None,
    gripper_opening_next: float | None,
) -> float:
    if gripper_opening_now is None or gripper_opening_next is None:
        return 0.0
    if float(gripper_opening_next) > float(gripper_opening_now):
        return 1.0
    if float(gripper_opening_next) < float(gripper_opening_now):
        return -1.0
    return 0.0


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def _fixed_array(value, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    n = min(size, arr.size)
    if n > 0:
        result[:n] = arr[:n]
    return result
