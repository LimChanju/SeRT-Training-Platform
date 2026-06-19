from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


OBSERVATION_VERSION = "obs_v1_state_controller_phase"
TASK_PHASES = ("approach_cube", "grasp_cube", "move_to_target", "release_cube")
CONTROLLER_EVENT_COUNT = 10
MISSING_DISTANCE_M = 10.0
DEFAULT_NEAR_HUMAN_THRESHOLD_M = 0.12


@dataclass(frozen=True)
class ObservationField:
    name: str
    shape: tuple[int, ...]
    description: str

    @property
    def dim(self) -> int:
        return int(np.prod(self.shape))


OBSERVATION_FIELDS: tuple[ObservationField, ...] = (
    ObservationField("robot_joint_pos", (7,), "Panda arm joint positions."),
    ObservationField("robot_joint_vel", (7,), "Panda arm joint velocities."),
    ObservationField("gripper_width", (1,), "Sum of the two Franka finger joint positions."),
    ObservationField("ee_pos", (3,), "End-effector world position."),
    ObservationField("ee_quat", (4,), "End-effector world orientation quaternion."),
    ObservationField("cube_pos", (3,), "Current pick cube world position."),
    ObservationField("cube_quat", (4,), "Current pick cube world orientation quaternion."),
    ObservationField("cube_lin_vel", (3,), "Current pick cube linear velocity."),
    ObservationField("cube_ang_vel", (3,), "Current pick cube angular velocity."),
    ObservationField("place_target_pos", (3,), "Desired placement target world position."),
    ObservationField("ee_to_cube", (3,), "Vector from end-effector to current pick cube."),
    ObservationField("cube_to_place_target", (3,), "Vector from current pick cube to placement target."),
    ObservationField("ee_to_place_target", (3,), "Vector from end-effector to placement target."),
    ObservationField("human_head_pos", (3,), "Human head or HMD world position."),
    ObservationField("human_left_hand_pos", (3,), "Human left hand world position."),
    ObservationField("human_right_hand_pos", (3,), "Human right hand world position."),
    ObservationField("ee_to_left_hand", (3,), "Vector from end-effector to left hand."),
    ObservationField("ee_to_right_hand", (3,), "Vector from end-effector to right hand."),
    ObservationField("min_hand_gripper_dist", (1,), "Minimum hand-to-gripper distance in meters."),
    ObservationField("human_robot_collision", (1,), "Binary flag for human/robot collision."),
    ObservationField("near_human", (1,), "Binary flag for unsafe proximity to human hand."),
    ObservationField("collision_green", (1,), "Binary flag for collision with protected green cube."),
    ObservationField("pick_miss_recent", (1,), "Binary flag for recent pick miss event."),
    ObservationField("drop_throw_recent", (1,), "Binary flag for recent drop or throw event."),
    ObservationField("has_grasped_cube", (1,), "Binary flag for current grasp estimate."),
    ObservationField("task_phase", (4,), "One-hot task phase."),
    ObservationField("controller_event", (CONTROLLER_EVENT_COUNT,), "One-hot PickPlaceController event."),
    ObservationField("controller_t", (1,), "PickPlaceController event progress in [0, 1]."),
)

OBSERVATION_DIM = sum(field.dim for field in OBSERVATION_FIELDS)
_FIELD_MAP = {field.name: field for field in OBSERVATION_FIELDS}


def observation_slices() -> dict[str, slice]:
    slices = {}
    cursor = 0
    for field in OBSERVATION_FIELDS:
        slices[field.name] = slice(cursor, cursor + field.dim)
        cursor += field.dim
    return slices


def empty_observation(dtype=np.float32) -> dict[str, np.ndarray]:
    obs = {}
    for field in OBSERVATION_FIELDS:
        obs[field.name] = np.zeros(field.shape, dtype=dtype)
    obs["min_hand_gripper_dist"][:] = MISSING_DISTANCE_M
    return obs


def flatten_observation(obs: Mapping[str, np.ndarray], dtype=np.float32) -> np.ndarray:
    validate_observation(obs)
    return np.concatenate(
        [np.asarray(obs[field.name], dtype=dtype).reshape(-1) for field in OBSERVATION_FIELDS]
    ).astype(dtype, copy=False)


def validate_observation(obs: Mapping[str, np.ndarray]) -> None:
    missing = [field.name for field in OBSERVATION_FIELDS if field.name not in obs]
    if missing:
        raise ValueError(f"Observation is missing fields: {missing}")
    for field in OBSERVATION_FIELDS:
        value = np.asarray(obs[field.name])
        if value.shape != field.shape:
            raise ValueError(
                f"Observation field '{field.name}' has shape {value.shape}, "
                f"expected {field.shape}"
            )


def build_observation(
    *,
    robot,
    cube,
    place_target,
    human_head_pos: np.ndarray | None = None,
    human_left_hand_pos: np.ndarray | None = None,
    human_right_hand_pos: np.ndarray | None = None,
    gripper_center_pos: np.ndarray | None = None,
    human_robot_collision: bool = False,
    near_human: bool | None = None,
    collision_green: bool = False,
    pick_miss_recent: bool = False,
    drop_throw_recent: bool = False,
    has_grasped_cube: bool = False,
    task_phase: str | int = "approach_cube",
    controller_event: int | None = None,
    controller_t: float = 0.0,
    near_human_threshold_m: float = DEFAULT_NEAR_HUMAN_THRESHOLD_M,
    min_hand_gripper_dist_override: float | None = None,
) -> dict[str, np.ndarray]:
    """Build the state observation from Isaac runtime objects.

    The function is intentionally free of Isaac imports so it can be imported by
    trajectory tooling and unit tests outside SimulationApp. Runtime objects only
    need to expose the Isaac-style methods used below.
    """

    obs = empty_observation()
    joint_pos = _safe_array_call(robot, "get_joint_positions", 9)
    joint_vel = _safe_array_call(robot, "get_joint_velocities", 9)
    obs["robot_joint_pos"] = _fixed_array(joint_pos[:7], 7)
    obs["robot_joint_vel"] = _fixed_array(joint_vel[:7], 7)

    gripper_joints = _safe_gripper_joint_positions(robot)
    obs["gripper_width"] = np.array([float(np.sum(gripper_joints))], dtype=np.float32)

    ee_pos, ee_quat = _safe_world_pose(getattr(robot, "end_effector", None))
    cube_pos, cube_quat = _safe_world_pose(cube)
    place_target_pos, _ = _safe_world_pose(place_target)

    obs["ee_pos"] = ee_pos.astype(np.float32)
    obs["ee_quat"] = ee_quat.astype(np.float32)
    obs["cube_pos"] = cube_pos.astype(np.float32)
    obs["cube_quat"] = cube_quat.astype(np.float32)
    obs["cube_lin_vel"] = _safe_array_call(cube, "get_linear_velocity", 3).astype(np.float32)
    obs["cube_ang_vel"] = _safe_array_call(cube, "get_angular_velocity", 3).astype(np.float32)
    obs["place_target_pos"] = place_target_pos.astype(np.float32)

    obs["ee_to_cube"] = (cube_pos - ee_pos).astype(np.float32)
    obs["cube_to_place_target"] = (place_target_pos - cube_pos).astype(np.float32)
    obs["ee_to_place_target"] = (place_target_pos - ee_pos).astype(np.float32)

    head_pos = _optional_vec3(human_head_pos)
    left_pos = _optional_vec3(human_left_hand_pos)
    right_pos = _optional_vec3(human_right_hand_pos)
    obs["human_head_pos"] = head_pos.astype(np.float32)
    obs["human_left_hand_pos"] = left_pos.astype(np.float32)
    obs["human_right_hand_pos"] = right_pos.astype(np.float32)

    obs["ee_to_left_hand"] = (
        left_pos - ee_pos if human_left_hand_pos is not None else np.zeros(3)
    ).astype(np.float32)
    obs["ee_to_right_hand"] = (
        right_pos - ee_pos if human_right_hand_pos is not None else np.zeros(3)
    ).astype(np.float32)

    gripper_pos = _optional_vec3(gripper_center_pos) if gripper_center_pos is not None else ee_pos
    hand_distances = []
    if human_left_hand_pos is not None:
        hand_distances.append(float(np.linalg.norm(left_pos - gripper_pos)))
    if human_right_hand_pos is not None:
        hand_distances.append(float(np.linalg.norm(right_pos - gripper_pos)))
    min_dist = (
        float(min_hand_gripper_dist_override)
        if min_hand_gripper_dist_override is not None
        else min(hand_distances) if hand_distances else MISSING_DISTANCE_M
    )
    obs["min_hand_gripper_dist"] = np.array([min_dist], dtype=np.float32)

    if near_human is None:
        near_human = min_dist < float(near_human_threshold_m)
    obs["human_robot_collision"] = _flag(human_robot_collision)
    obs["near_human"] = _flag(near_human)
    obs["collision_green"] = _flag(collision_green)
    obs["pick_miss_recent"] = _flag(pick_miss_recent)
    obs["drop_throw_recent"] = _flag(drop_throw_recent)
    obs["has_grasped_cube"] = _flag(has_grasped_cube)
    obs["task_phase"] = task_phase_onehot(task_phase)
    obs["controller_event"] = controller_event_onehot(controller_event)
    obs["controller_t"] = np.array([np.clip(float(controller_t), 0.0, 1.0)], dtype=np.float32)
    validate_observation(obs)
    return obs


def task_phase_onehot(phase: str | int) -> np.ndarray:
    onehot = np.zeros(len(TASK_PHASES), dtype=np.float32)
    if isinstance(phase, int):
        if 0 <= phase < len(TASK_PHASES):
            onehot[phase] = 1.0
        return onehot
    try:
        onehot[TASK_PHASES.index(str(phase).strip().lower())] = 1.0
    except ValueError:
        pass
    return onehot


def controller_event_onehot(event: int | None) -> np.ndarray:
    onehot = np.zeros(CONTROLLER_EVENT_COUNT, dtype=np.float32)
    if event is None:
        return onehot
    event_idx = int(event)
    if 0 <= event_idx < CONTROLLER_EVENT_COUNT:
        onehot[event_idx] = 1.0
    return onehot


def _flag(value: bool) -> np.ndarray:
    return np.array([1.0 if value else 0.0], dtype=np.float32)


def _optional_vec3(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=float)
    return _fixed_array(value, 3).astype(float)


def _fixed_array(value, size: int, fill: float = 0.0) -> np.ndarray:
    result = np.full(size, fill, dtype=float)
    if value is None:
        return result
    arr = np.asarray(value, dtype=float).reshape(-1)
    n = min(size, arr.size)
    if n > 0:
        result[:n] = arr[:n]
    return result


def _safe_array_call(obj, method_name: str, size: int) -> np.ndarray:
    if obj is None or not hasattr(obj, method_name):
        return np.zeros(size, dtype=float)
    try:
        return _fixed_array(getattr(obj, method_name)(), size)
    except Exception:
        return np.zeros(size, dtype=float)


def _safe_gripper_joint_positions(robot) -> np.ndarray:
    try:
        return _fixed_array(robot.gripper.get_joint_positions(), 2)
    except Exception:
        joint_pos = _safe_array_call(robot, "get_joint_positions", 9)
        return _fixed_array(joint_pos[7:9], 2)


def _safe_world_pose(obj) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(obj, (list, tuple, np.ndarray)):
        return _fixed_array(obj, 3), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if obj is None or not hasattr(obj, "get_world_pose"):
        return np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    try:
        pos, quat = obj.get_world_pose()
        quat_arr = _fixed_array(quat, 4, fill=0.0)
        if not np.any(quat_arr):
            quat_arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return _fixed_array(pos, 3), quat_arr
    except Exception:
        return np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
