from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEGACY_REWARD_VERSION = "reward_v0_hri_errp"
PLACEMENT_REWARD_VERSION = "reward_v1_placement_hri_errp"
REWARD_VERSION = "reward_v2_grasp_stability_hri_errp"


@dataclass(frozen=True)
class RewardWeights:
    ee_to_cube_progress: float = 2.0
    cube_to_target_progress: float = 1.5
    carrying_cube_to_target_progress: float = 7.0
    grasp_phase_distance_penalty: float = 0.35
    missed_grasp_transition_penalty: float = 2.0
    grasp_phase_target_dist: float = 0.058
    grasp_bonus: float = 0.03
    target_zone_bonus: float = 0.15
    success_bonus: float = 15.0
    placement_error_penalty: float = 0.08
    release_outside_target_penalty: float = 4.0
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
    success_dist: float = 0.06,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> RewardResult:
    """Compute reward v2 from consecutive observations.

    Formula:

        r_t =
          w1 * (d_ee_cube[t-1] - d_ee_cube[t])
        + w2 * (d_cube_goal[t-1] - d_cube_goal[t])
        + w3 * grasp_or_post_grasp * (d_cube_goal[t-1] - d_cube_goal[t])
        - wg * grasp_phase * no_grasp * max(0, d_ee_cube[t] - d_grasp) / d_grasp
        - wm * entered_post_grasp_without_grasp
        + bg * grasp
        + bz * target_zone
        + bs * success
        - lp * grasp_or_post_grasp * max(0, d_cube_goal[t] - d_success) / d_success
        - lr * release_outside_target
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
        prev_has_grasped = 0.0
        prev_event = -1
    else:
        ee_cube_progress = _norm_field(prev_obs, "ee_to_cube") - ee_cube_dist
        cube_target_progress = _norm_field(prev_obs, "cube_to_place_target") - cube_target_dist
        prev_has_grasped = _scalar_field(prev_obs, "has_grasped_cube")
        prev_event = _controller_event(prev_obs)

    has_grasped = _scalar_field(obs, "has_grasped_cube")
    event = _controller_event(obs)
    post_grasp_phase = float(max(event, prev_event) >= 4)
    grasp_phase = float(event in (1, 2, 3) and has_grasped <= 0.5)
    placement_active = max(has_grasped, prev_has_grasped, post_grasp_phase)
    normalized_grasp_error = max(0.0, ee_cube_dist - weights.grasp_phase_target_dist) / max(
        weights.grasp_phase_target_dist,
        1e-6,
    )
    normalized_target_error = max(0.0, cube_target_dist - float(success_dist)) / max(float(success_dist), 1e-6)
    target_zone = placement_active * float(cube_target_dist <= float(success_dist))
    entered_post_grasp_without_grasp = float(
        prev_obs is not None
        and prev_event <= 3
        and event >= 4
        and max(has_grasped, prev_has_grasped) <= 0.5
    )
    release_after_pick = (
        prev_obs is not None
        and prev_has_grasped > 0.5
        and has_grasped <= 0.5
        and max(event, prev_event) >= 7
    )
    release_outside_target = float(release_after_pick and cube_target_dist > float(success_dist))
    near_human = _scalar_field(obs, "near_human")
    human_collision = _scalar_field(obs, "human_robot_collision")
    action_norm = float(np.linalg.norm(np.asarray(action, dtype=float).reshape(-1)))
    errp_feedback = float(np.clip(errp_feedback, 0.0, 1.0))

    components = {
        "ee_to_cube_progress": weights.ee_to_cube_progress * ee_cube_progress,
        "cube_to_target_progress": weights.cube_to_target_progress * cube_target_progress,
        "carrying_cube_to_target_progress": (
            weights.carrying_cube_to_target_progress * placement_active * cube_target_progress
        ),
        "grasp_phase_distance_penalty": (
            -weights.grasp_phase_distance_penalty * grasp_phase * normalized_grasp_error
        ),
        "missed_grasp_transition_penalty": (
            -weights.missed_grasp_transition_penalty * entered_post_grasp_without_grasp
        ),
        "grasp_bonus": weights.grasp_bonus * has_grasped,
        "target_zone_bonus": weights.target_zone_bonus * target_zone,
        "success_bonus": weights.success_bonus * (1.0 if success else 0.0),
        "placement_error_penalty": -weights.placement_error_penalty * placement_active * normalized_target_error,
        "release_outside_target_penalty": -weights.release_outside_target_penalty * release_outside_target,
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
        "carrying_cube_to_target_progress",
        "grasp_phase_distance_penalty",
        "missed_grasp_transition_penalty",
        "grasp_bonus",
        "target_zone_bonus",
        "success_bonus",
        "placement_error_penalty",
        "release_outside_target_penalty",
        "action_penalty",
        "near_human_penalty",
        "human_collision_penalty",
        "errp_penalty",
    )


def reward_weights_dict(weights: RewardWeights = DEFAULT_REWARD_WEIGHTS) -> dict[str, float]:
    return {
        "ee_to_cube_progress": weights.ee_to_cube_progress,
        "cube_to_target_progress": weights.cube_to_target_progress,
        "carrying_cube_to_target_progress": weights.carrying_cube_to_target_progress,
        "grasp_phase_distance_penalty": weights.grasp_phase_distance_penalty,
        "missed_grasp_transition_penalty": weights.missed_grasp_transition_penalty,
        "grasp_phase_target_dist": weights.grasp_phase_target_dist,
        "grasp_bonus": weights.grasp_bonus,
        "target_zone_bonus": weights.target_zone_bonus,
        "success_bonus": weights.success_bonus,
        "placement_error_penalty": weights.placement_error_penalty,
        "release_outside_target_penalty": weights.release_outside_target_penalty,
        "action_penalty": weights.action_penalty,
        "near_human_penalty": weights.near_human_penalty,
        "human_collision_penalty": weights.human_collision_penalty,
        "errp_penalty": weights.errp_penalty,
    }


def _norm_field(obs: dict[str, np.ndarray], name: str) -> float:
    return float(np.linalg.norm(np.asarray(obs[name], dtype=float).reshape(-1)))


def _scalar_field(obs: dict[str, np.ndarray], name: str) -> float:
    return float(np.asarray(obs[name], dtype=float).reshape(-1)[0])


def _controller_event(obs: dict[str, np.ndarray]) -> int:
    value = obs.get("controller_event")
    if value is None:
        return -1
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or float(np.max(arr)) <= 0.0:
        return -1
    return int(np.argmax(arr))
