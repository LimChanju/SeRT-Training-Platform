# pick_controller.py — Isaac Sim 4.5 전용
# omni.isaac.franka.controllers.PickPlaceController 사용
# =============================================================================

import numpy as np
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController


def create_pick_controller(
    robot: Franka,
    end_effector_initial_height: float = None,
) -> PickPlaceController:
    """
    PickPlaceController 생성
    내부 FSM 8단계:
      0) EE → 큐브 위 (pre-grasp)
      1) 그리퍼 열기
      2) EE → 큐브 높이 (approach)
      3) 그리퍼 닫기 (grasp)
      4) EE → 위로 들기 (lift)
      5) EE → 목표 위 이동
      6) EE → 목표 높이 (descend)
      7) 그리퍼 열기 (release)
    """
    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        end_effector_initial_height=end_effector_initial_height,
    )
    return controller


def run_pick_place(
    controller: PickPlaceController,
    robot: Franka,
    pick_position: np.ndarray,
    place_position: np.ndarray,
) -> bool:
    """
    매 physics 스텝마다 호출합니다.
    완료되면 True 반환.

    end_effector_offset:
      EE 링크 원점과 실제 grasp 중심 사이의 offset
      집기가 이상하면 이 값을 조정 (z축: 큐브 높이 절반)
    """
    current_joint_positions = robot.get_joint_positions()

    actions = controller.forward(
        picking_position=pick_position,
        placing_position=place_position,
        current_joint_positions=current_joint_positions,
        end_effector_offset=np.array([0.0, 0.005, 0.0]),
    )

    robot.apply_action(actions)
    return controller.is_done()