# scene_setup.py — Isaac Sim 4.5 전용
# omni.isaac.core 네임스페이스 사용 (isaacsim.* 는 5.x 이상)
# =============================================================================

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid, VisualCylinder


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
):
    xy_half = (table_size[:2] / 2.0) - 0.1
    # Keep cubes farther apart to reduce gripper interference with neighbors
    min_dist = cube_size * 2.6
    x_min, x_max = 0.25, 0.65
    y_min, y_max = -0.25, 0.25

    positions = []
    attempts = 0
    while len(positions) < count and attempts < 2000:
        attempts += 1
        rand_xy = np.random.uniform(-xy_half, xy_half)
        candidate = table_xy + rand_xy
        if not (x_min <= candidate[0] <= x_max and y_min <= candidate[1] <= y_max):
            continue
        if forbidden_xy is not None:
            if np.linalg.norm(candidate - forbidden_xy) < min_dist:
                continue
        if all(np.linalg.norm(candidate - p) >= min_dist for p in positions):
            positions.append(candidate)
    if len(positions) < count:
        raise RuntimeError("Could not place cubes without overlap; increase table size.")
    return positions


def randomize_cubes(
    cubes,
    table_xy: np.ndarray,
    table_size: np.ndarray,
    cube_center_z: float,
    cube_size: float,
    forbidden_xy: np.ndarray = None,
):
    cube_xy_positions = _sample_positions(
        table_xy,
        table_size,
        cube_size,
        len(cubes),
        forbidden_xy=forbidden_xy,
    )
    for cube, pos_xy in zip(cubes, cube_xy_positions):
        if hasattr(cube, "disable_rigid_body_physics"):
            cube.disable_rigid_body_physics()
        new_pos = np.array([pos_xy[0], pos_xy[1], cube_center_z])
        if hasattr(cube, "set_default_state"):
            cube.set_default_state(
                position=new_pos,
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
        cube.set_world_pose(position=new_pos)
        if hasattr(cube, "enable_rigid_body_physics"):
            cube.enable_rigid_body_physics()
        if hasattr(cube, "set_linear_velocity"):
            cube.set_linear_velocity(np.zeros(3))
        if hasattr(cube, "set_angular_velocity"):
            cube.set_angular_velocity(np.zeros(3))


def setup_scene(world: World, cube_count: int = 6, show_human_proxies: bool = True):
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
    table_height = 0.4
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
    viewer_eye = np.array([1.2, 0.0, 1.2])
    set_camera_view(eye=viewer_eye, target=table_center)
    shoulder_pos = np.array([viewer_eye[0], viewer_eye[1], 1.1])
    left_hand_pos = shoulder_pos + np.array([-0.35, 0.2, -0.25])
    right_hand_pos = shoulder_pos + np.array([-0.35, -0.2, -0.25])
    left_elbow_pos = (shoulder_pos + left_hand_pos) * 0.5
    right_elbow_pos = (shoulder_pos + right_hand_pos) * 0.5
    stack_base_xy = np.array([0.6, -0.25])
    cube_xy_positions = _sample_positions(
        table_xy,
        table_size,
        cube_size,
        cube_count,
        forbidden_xy=stack_base_xy,
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

    human_proxies = [
        world.scene.add(
            VisualCylinder(
                prim_path="/World/human_left_hand",
                name="human_left_hand",
                position=left_hand_pos,
                radius=0.03,
                height=0.06,
                color=np.array([0.0, 0.8, 0.9]),
                visible=show_human_proxies,
            )
        ),
        world.scene.add(
            VisualCylinder(
                prim_path="/World/human_right_hand",
                name="human_right_hand",
                position=right_hand_pos,
                radius=0.03,
                height=0.06,
                color=np.array([0.0, 0.8, 0.9]),
                visible=show_human_proxies,
            )
        ),
        world.scene.add(
            VisualCylinder(
                prim_path="/World/human_left_arm",
                name="human_left_arm",
                position=left_elbow_pos,
                radius=0.04,
                height=0.25,
                color=np.array([0.1, 0.7, 0.8]),
                visible=show_human_proxies,
            )
        ),
        world.scene.add(
            VisualCylinder(
                prim_path="/World/human_right_arm",
                name="human_right_arm",
                position=right_elbow_pos,
                radius=0.04,
                height=0.25,
                color=np.array([0.1, 0.7, 0.8]),
                visible=show_human_proxies,
            )
        ),
    ]

    return (
        cubes,
        place_target,
        table_top_z,
        cube_size,
        table_xy,
        table_size,
        stack_base_xy,
        human_proxies,
    )
