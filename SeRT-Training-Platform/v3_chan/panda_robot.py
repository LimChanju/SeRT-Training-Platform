# =============================================================================
# panda_robot.py — Isaac Sim 4.5 전용
# omni.isaac.franka.Franka 사용
# =============================================================================

import numpy as np
from omni.isaac.franka import Franka

PANDA_PRIM_PATH = "/World/Franka"


def add_panda(world, base_z: float = None) -> Franka:
    """
    Franka Panda를 월드에 추가합니다.
    Isaac Sim 4.5의 Franka() 클래스는:
      - USD를 Nucleus에서 자동으로 불러옵니다
      - robot.gripper   : ParallelGripper (open/close 포함)
      - robot.end_effector : EE prim
      - robot.dof_names : joint 이름 리스트 (총 9개)
    """
    robot = world.scene.add(
        Franka(
            prim_path=PANDA_PRIM_PATH,
            name="panda",
        )
    )
    if base_z is not None:
        robot.set_world_pose(position=np.array([0.0, 0.0, base_z]))
    robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
    return robot


def print_robot_info(robot: Franka):
    """world.reset() 이후에 호출해야 정확한 값이 나옵니다."""
    print("=" * 60)
    print("[Panda 정보]")
    print(f"  DOF 수       : {robot.num_dof}")
    print(f"  Joint 이름   : {robot.dof_names}")
    ee_pos, ee_ori = robot.end_effector.get_world_pose()
    print(f"  EE 위치      : {np.round(ee_pos, 4)}")
    print(f"  EE 방향(quat): {np.round(ee_ori, 4)}")
    print(f"  Gripper 상태 : {robot.gripper.get_joint_positions()}")
    print("=" * 60)