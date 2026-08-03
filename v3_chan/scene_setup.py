# scene_setup.py — Isaac Sim 4.5 전용
# omni.isaac.core 네임스페이스 사용 (isaacsim.* 는 5.x 이상)
# =============================================================================

import os

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid

try:
    from v3_chan.scene_randomization import sample_cube_positions
except ImportError:
    from scene_randomization import sample_cube_positions


def create_world() -> World:
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    return world


def _sample_positions(
    table_xy: np.ndarray,
    table_size: np.ndarray,
    cube_size: float,
    count: int,
    forbidden_xy: np.ndarray = None,
    rng: np.random.Generator | None = None,
):
    x_min = float(os.environ.get("PICK_PLACE_CUBE_X_MIN", "0.30"))
    x_max = float(os.environ.get("PICK_PLACE_CUBE_X_MAX", "0.65"))
    return sample_cube_positions(
        rng=rng or np.random.default_rng(),
        table_xy=table_xy,
        table_size=table_size,
        cube_size=cube_size,
        count=count,
        forbidden_xy=forbidden_xy,
        x_bounds=(x_min, x_max),
        y_bounds=(-0.25, 0.25),
    )


def randomize_cubes(
    cubes,
    table_xy: np.ndarray,
    table_size: np.ndarray,
    cube_center_z: float,
    cube_size: float,
    forbidden_xy: np.ndarray = None,
    rng: np.random.Generator | None = None,
):
    cube_xy_positions = _sample_positions(
        table_xy,
        table_size,
        cube_size,
        len(cubes),
        forbidden_xy=forbidden_xy,
        rng=rng,
    )
    for cube, pos_xy in zip(cubes, cube_xy_positions):
        if hasattr(cube, "disable_rigid_body_physics"):
            cube.disable_rigid_body_physics()
        new_pos = np.array([pos_xy[0], pos_xy[1], cube_center_z])
        reset_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        if hasattr(cube, "set_default_state"):
            cube.set_default_state(
                position=new_pos,
                orientation=reset_orientation,
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
        cube.set_world_pose(position=new_pos, orientation=reset_orientation)
        if hasattr(cube, "enable_rigid_body_physics"):
            cube.enable_rigid_body_physics()
        if hasattr(cube, "set_linear_velocity"):
            cube.set_linear_velocity(np.zeros(3))
        if hasattr(cube, "set_angular_velocity"):
            cube.set_angular_velocity(np.zeros(3))
    return cube_xy_positions


def setup_scene(
    world: World,
    cube_count: int = 6,
    *,
    rng: np.random.Generator | None = None,
):
    """
    씬 구성
    - 바닥
    - 파란 큐브: 집을 물체 (DynamicCuboid, 물리 적용)
    - 초록 큐브: 목표 위치 마커 (VisualCuboid, 물리 없음)

        좌표계 (Panda 베이스 기준):
            x+ : 로봇 앞쪽
            y+ : 로봇 왼쪽
            z+ : 위쪽
    """
    world.scene.add_default_ground_plane()

    cube_size = 0.0515
    cube_half = cube_size / 2.0
    table_size = np.array([1.2, 0.8, 0.05])
    table_height = 0.9   # VR HMD 트래킹 사용 → 현실적인 테이블 높이로 조정
    table_center_z = table_height + (table_size[2] / 2.0)
    table_top_z = table_center_z + (table_size[2] / 2.0)
    cube_center_z = table_top_z + cube_half

    world.scene.add(
        FixedCuboid(
            prim_path="/World/table",
            name="table",
            position=np.array([0.4, 0.0, table_center_z]),
            scale=table_size,
            color=np.array([0.35, 0.3, 0.25]),
        )
    )

    table_xy = np.array([0.4, 0.0])
    table_center_z = table_height + (table_size[2] / 2.0)
    table_center = np.array([table_xy[0], table_xy[1], table_center_z])
    viewer_eye = np.array([1.1, 0.0, 1.5])
    if os.environ.get("ISAAC_DISABLE_VIEWPORT", "0").lower() not in (
        "1",
        "true",
        "yes",
    ):
        set_camera_view(eye=viewer_eye, target=table_center)
    stack_base_xy = np.array([0.6, -0.25])
    cube_xy_positions = _sample_positions(
        table_xy,
        table_size,
        cube_size,
        cube_count,
        forbidden_xy=stack_base_xy,
        rng=rng,
    )

    cubes = []
    for idx, pos_xy in enumerate(cube_xy_positions):
        is_red = idx < (cube_count // 2)
        color = np.array([1.0, 0.1, 0.1]) if is_red else np.array([0.1, 1.0, 0.1])
        cube = world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/cube_{idx}",
                name=f"cube_{idx}",
                position=np.array([pos_xy[0], pos_xy[1], cube_center_z]),
                scale=np.array([cube_size, cube_size, cube_size]),
                color=color,
            )
        )
        cubes.append(cube)

    place_target = world.scene.add(
        VisualCuboid(
            prim_path="/World/place_target",
            name="place_target",
            position=np.array([stack_base_xy[0], stack_base_xy[1], cube_center_z]),
            scale=np.array([cube_size, cube_size, cube_size]),
            color=np.array([1.0, 1.0, 0.0]),
        )
    )

    return (
        cubes,
        place_target,
        table_top_z,
        cube_size,
        table_xy,
        table_size,
        stack_base_xy,
    )
