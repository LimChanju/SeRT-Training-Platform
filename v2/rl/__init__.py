"""RL utilities for SeRT trajectory collection and policy training."""

from .actions import (
    ACTION_DIM,
    ACTION_NAMES,
    ACTION_VERSION,
    CONTROLLER_TARGET_ACTION_VERSION,
    CONTROLLER_TARGET_MAX_DELTA_M,
    MAX_EE_DELTA_M,
    MAX_YAW_DELTA_RAD,
    TaskSpaceAction,
    clip_action,
    controller_target_action_from_target,
    controller_target_from_action,
    denormalize_action,
    expert_joint_action_vector,
    task_action_from_transition,
    zero_action,
)
from .observations import (
    OBSERVATION_DIM,
    OBSERVATION_FIELDS,
    OBSERVATION_VERSION,
    TASK_PHASES,
    CONTROLLER_EVENT_COUNT,
    build_observation,
    controller_event_onehot,
    empty_observation,
    flatten_observation,
    observation_slices,
    validate_observation,
)
from .rewards import (
    DEFAULT_REWARD_WEIGHTS,
    LEGACY_REWARD_VERSION,
    REWARD_VERSION,
    RewardResult,
    RewardWeights,
    compute_reward,
    is_success,
    reward_component_names,
    reward_weights_dict,
)
try:
    from .pick_place_env import IsaacPickPlaceEnv, PickPlaceEnvConfig
except ModuleNotFoundError as exc:
    if exc.name not in {
        "isaacsim",
        "omni",
        "panda_robot",
        "scene_setup",
    }:
        raise

    class PickPlaceEnvConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "IsaacPickPlaceEnv requires Isaac Sim runtime modules. Create a "
                "SimulationApp first and run through launch_isaac.sh."
            ) from exc

    class IsaacPickPlaceEnv:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "IsaacPickPlaceEnv requires Isaac Sim runtime modules. Create a "
                "SimulationApp first and run through launch_isaac.sh."
            ) from exc

try:
    from .trajectory_recorder import (
        EXPERT_JOINT_ACTION_DIM,
        TRAJECTORY_SCHEMA_VERSION,
        TrajectoryRecorder,
    )
except ModuleNotFoundError as exc:
    if exc.name != "h5py":
        raise
    EXPERT_JOINT_ACTION_DIM = 9
    TRAJECTORY_SCHEMA_VERSION = "trajectory_v0_transitions"

    class TrajectoryRecorder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "h5py is required for TrajectoryRecorder. Install it in the Isaac "
                "Python environment, or run collect_expert_trajectories.py with "
                "--install-missing-deps."
            ) from exc

__all__ = [
    "ACTION_DIM",
    "ACTION_NAMES",
    "ACTION_VERSION",
    "CONTROLLER_EVENT_COUNT",
    "CONTROLLER_TARGET_ACTION_VERSION",
    "CONTROLLER_TARGET_MAX_DELTA_M",
    "DEFAULT_REWARD_WEIGHTS",
    "EXPERT_JOINT_ACTION_DIM",
    "MAX_EE_DELTA_M",
    "MAX_YAW_DELTA_RAD",
    "OBSERVATION_DIM",
    "OBSERVATION_FIELDS",
    "OBSERVATION_VERSION",
    "IsaacPickPlaceEnv",
    "LEGACY_REWARD_VERSION",
    "PickPlaceEnvConfig",
    "REWARD_VERSION",
    "RewardResult",
    "RewardWeights",
    "TASK_PHASES",
    "TRAJECTORY_SCHEMA_VERSION",
    "TaskSpaceAction",
    "TrajectoryRecorder",
    "build_observation",
    "clip_action",
    "controller_event_onehot",
    "controller_target_action_from_target",
    "controller_target_from_action",
    "compute_reward",
    "denormalize_action",
    "empty_observation",
    "expert_joint_action_vector",
    "flatten_observation",
    "is_success",
    "observation_slices",
    "reward_component_names",
    "reward_weights_dict",
    "task_action_from_transition",
    "validate_observation",
    "zero_action",
]
