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
    AUXILIARY_OBSERVATION_FIELDS,
    DYNAMIC_HRI_OBS_DIM,
    DYNAMIC_HRI_OBS_FIELD_NAMES,
    DYNAMIC_HRI_OBSERVATION_FIELDS,
    DYNAMIC_HRI_OBSERVATION_VERSION,
    HRI_OBS_DIM,
    HRI_OBS_FIELD_NAMES,
    HRI_OBSERVATION_VERSION,
    OBSERVATION_DIM,
    OBSERVATION_FIELDS,
    OBSERVATION_VERSION,
    RECORDED_OBSERVATION_FIELDS,
    TASK_PHASES,
    CONTROLLER_EVENT_COUNT,
    build_observation,
    apply_dynamic_hri_observation,
    controller_event_onehot,
    empty_observation,
    flatten_hri_observation,
    flatten_dynamic_hri_observation,
    flatten_observation,
    observation_slices,
    validate_auxiliary_observation,
    validate_observation,
)
from .pseudo_errp import (
    DEFAULT_PSEUDO_ERRP_SOURCES,
    PSEUDO_ERRP_SOURCE_CODES,
    PseudoErrPResult,
    extract_pseudo_errp_aux_flags,
    parse_pseudo_errp_sources,
    pseudo_errp_from_observation,
)
try:
    from .encounter_manifest import (
        MANIFEST_VERSION,
        SEVERITY_ORDER,
        SOURCE_CONFIGURATION_VERSION,
        EncounterBuildConfig,
        build_encounter_manifest,
        extract_episode_source_configuration,
        load_encounter_manifest,
        parse_severity_mix,
        resolve_source_restoration,
    )
    from .human_replay import (
        HumanEncounterReplay,
        HumanEncounterReplayInfo,
        HumanReplayInfo,
        HumanTrajectoryReplay,
    )
except ModuleNotFoundError as exc:
    if exc.name != "h5py":
        raise

    class HumanReplayInfo:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "h5py is required for HumanTrajectoryReplay. Install it in the active "
                "Python environment before using --human-replay-data."
            ) from exc

    class HumanTrajectoryReplay:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "h5py is required for HumanTrajectoryReplay. Install it in the active "
                "Python environment before using --human-replay-data."
            ) from exc

    class HumanEncounterReplay:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "h5py is required for HumanEncounterReplay. Install it in the active "
                "Python environment before using --encounter-manifest."
            ) from exc
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
    "AUXILIARY_OBSERVATION_FIELDS",
    "CONTROLLER_EVENT_COUNT",
    "CONTROLLER_TARGET_ACTION_VERSION",
    "CONTROLLER_TARGET_MAX_DELTA_M",
    "DEFAULT_REWARD_WEIGHTS",
    "DEFAULT_PSEUDO_ERRP_SOURCES",
    "DYNAMIC_HRI_OBS_DIM",
    "DYNAMIC_HRI_OBS_FIELD_NAMES",
    "DYNAMIC_HRI_OBSERVATION_FIELDS",
    "DYNAMIC_HRI_OBSERVATION_VERSION",
    "EXPERT_JOINT_ACTION_DIM",
    "HRI_OBS_DIM",
    "HRI_OBS_FIELD_NAMES",
    "HRI_OBSERVATION_VERSION",
    "HumanEncounterReplay",
    "HumanEncounterReplayInfo",
    "HumanReplayInfo",
    "HumanTrajectoryReplay",
    "MAX_EE_DELTA_M",
    "MAX_YAW_DELTA_RAD",
    "OBSERVATION_DIM",
    "OBSERVATION_FIELDS",
    "OBSERVATION_VERSION",
    "RECORDED_OBSERVATION_FIELDS",
    "SOURCE_CONFIGURATION_VERSION",
    "IsaacPickPlaceEnv",
    "LEGACY_REWARD_VERSION",
    "PickPlaceEnvConfig",
    "PSEUDO_ERRP_SOURCE_CODES",
    "PseudoErrPResult",
    "REWARD_VERSION",
    "RewardResult",
    "RewardWeights",
    "TASK_PHASES",
    "TRAJECTORY_SCHEMA_VERSION",
    "TaskSpaceAction",
    "TrajectoryRecorder",
    "build_observation",
    "apply_dynamic_hri_observation",
    "clip_action",
    "controller_event_onehot",
    "controller_target_action_from_target",
    "controller_target_from_action",
    "compute_reward",
    "denormalize_action",
    "empty_observation",
    "expert_joint_action_vector",
    "extract_episode_source_configuration",
    "extract_pseudo_errp_aux_flags",
    "flatten_hri_observation",
    "flatten_dynamic_hri_observation",
    "flatten_observation",
    "is_success",
    "observation_slices",
    "parse_pseudo_errp_sources",
    "pseudo_errp_from_observation",
    "reward_component_names",
    "reward_weights_dict",
    "resolve_source_restoration",
    "task_action_from_transition",
    "validate_auxiliary_observation",
    "validate_observation",
    "zero_action",
]
