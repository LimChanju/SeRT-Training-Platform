from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaceValidationResult:
    success: bool
    xy_error_m: float
    z_error_m: float
    linear_speed_mps: float
    cube_lift_m: float
    grasp_observed: bool


def evaluate_place_success(
    cube_position,
    target_position,
    *,
    cube_linear_velocity=None,
    cube_lift_m: float = 0.0,
    grasp_observed: bool = False,
    xy_tolerance_m: float = 0.04,
    z_tolerance_m: float = 0.03,
    max_linear_speed_mps: float = 0.05,
    min_cube_lift_m: float = 0.05,
) -> PlaceValidationResult:
    """Validate that a released cube is stably placed at the requested target."""
    cube = np.asarray(cube_position, dtype=float).reshape(-1)
    target = np.asarray(target_position, dtype=float).reshape(-1)
    if cube.size < 3 or target.size < 3:
        raise ValueError("cube_position and target_position must each contain xyz")

    xy_error = float(np.linalg.norm(cube[:2] - target[:2]))
    z_error = float(abs(cube[2] - target[2]))
    if cube_linear_velocity is None:
        linear_speed = 0.0
    else:
        velocity = np.asarray(cube_linear_velocity, dtype=float).reshape(-1)
        if velocity.size < 3:
            raise ValueError("cube_linear_velocity must contain xyz")
        linear_speed = float(np.linalg.norm(velocity[:3]))

    success = bool(
        xy_error <= float(xy_tolerance_m)
        and z_error <= float(z_tolerance_m)
        and linear_speed <= float(max_linear_speed_mps)
        and float(cube_lift_m) >= float(min_cube_lift_m)
        and bool(grasp_observed)
    )
    return PlaceValidationResult(
        success=success,
        xy_error_m=xy_error,
        z_error_m=z_error,
        linear_speed_mps=linear_speed,
        cube_lift_m=float(cube_lift_m),
        grasp_observed=bool(grasp_observed),
    )
