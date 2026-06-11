from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REWARD_VERSION = "reward_v0_hri_errp"


@dataclass(frozen=True)
class RewardWeights:
    ee_to_cube_progress: float = 2.0
    cube_to_target_progress: float = 3.0
    grasp_bonus: float = 0.2
    success_bonus: float = 10.0
    action_penalty: float = 0.01
    near_human_penalty: float = 0.5
    human_collision_penalty: float = 5.0
    errp_penalty: float = 2.0


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: dict[str, float]


DEFAULT_REWARD_WEIGHTS = RewardWeights()


def compute_reward(
    prev_obs: dict[str, np.ndarray] | None,
    obs: dict[str, np.ndarray],
    action: np.ndarray,
    *,
    errp_feedback: float = 0.0,
    success: bool = False,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> RewardResult:
    """Compute reward v0 from consecutive observations.

    Formula:

        r_t =
          w1 * (d_ee_cube[t-1] - d_ee_cube[t])
        + w2 * (d_cube_goal[t-1] - d_cube_goal[t])
        + bg * grasp
        + bs * success
        - la * ||a_t||_2
        - ln * near_human
        - lc * human_robot_collision
        - le * errp_feedback
    """

    ee_cube_dist = _norm_field(obs, "ee_to_cube")
    cube_target_dist = _norm_field(obs, "cube_to_place_target")
    if prev_obs is None:
        ee_cube_progress = 0.0
        cube_target_progress = 0.0
    else:
        ee_cube_progress = _norm_field(prev_obs, "ee_to_cube") - ee_cube_dist
        cube_target_progress = _norm_field(prev_obs, "cube_to_place_target") - cube_target_dist

    has_grasped = _scalar_field(obs, "has_grasped_cube")
    near_human = _scalar_field(obs, "near_human")
    human_collision = _scalar_field(obs, "human_robot_collision")
    action_norm = float(np.linalg.norm(np.asarray(action, dtype=float).reshape(-1)))
    errp_feedback = float(np.clip(errp_feedback, 0.0, 1.0))

    components = {
        "ee_to_cube_progress": weights.ee_to_cube_progress * ee_cube_progress,
        "cube_to_target_progress": weights.cube_to_target_progress * cube_target_progress,
        "grasp_bonus": weights.grasp_bonus * has_grasped,
        "success_bonus": weights.success_bonus * (1.0 if success else 0.0),
        "action_penalty": -weights.action_penalty * action_norm,
        "near_human_penalty": -weights.near_human_penalty * near_human,
        "human_collision_penalty": -weights.human_collision_penalty * human_collision,
        "errp_penalty": -weights.errp_penalty * errp_feedback,
    }
    return RewardResult(total=float(sum(components.values())), components=components)


def is_success(obs: dict[str, np.ndarray], threshold_m: float = 0.06) -> bool:
    """Simple v0 success: active cube is close enough to the place target."""

    return _norm_field(obs, "cube_to_place_target") <= float(threshold_m)


def reward_component_names() -> tuple[str, ...]:
    return (
        "ee_to_cube_progress",
        "cube_to_target_progress",
        "grasp_bonus",
        "success_bonus",
        "action_penalty",
        "near_human_penalty",
        "human_collision_penalty",
        "errp_penalty",
    )


def reward_weights_dict(weights: RewardWeights = DEFAULT_REWARD_WEIGHTS) -> dict[str, float]:
    return {
        "ee_to_cube_progress": weights.ee_to_cube_progress,
        "cube_to_target_progress": weights.cube_to_target_progress,
        "grasp_bonus": weights.grasp_bonus,
        "success_bonus": weights.success_bonus,
        "action_penalty": weights.action_penalty,
        "near_human_penalty": weights.near_human_penalty,
        "human_collision_penalty": weights.human_collision_penalty,
        "errp_penalty": weights.errp_penalty,
    }


def _norm_field(obs: dict[str, np.ndarray], name: str) -> float:
    return float(np.linalg.norm(np.asarray(obs[name], dtype=float).reshape(-1)))


def _scalar_field(obs: dict[str, np.ndarray], name: str) -> float:
    return float(np.asarray(obs[name], dtype=float).reshape(-1)[0])
