# main.py — Isaac Sim 4.5 전용
#
# 실행 방법:
#   isaac ~/isaac_vr_project/v2/main.py
#
# Phase 진행:
#   Phase 1 (현재): 씬 로드 + Panda 정보 확인
#   Phase 2 (다음): ENABLE_PICK_PLACE = True 로 변경
# =============================================================================

# ── 반드시 가장 먼저 임포트 ─────────────────────────────────────────────────
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

# ── SimulationApp 초기화 후에 나머지 임포트 ───────────────────────────────
import sys
import os
import csv
import json
import socket
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene_setup import create_world, randomize_cubes, setup_scene
from panda_robot import add_panda, print_robot_info
from pick_controller import create_pick_controller, run_pick_place

# =============================================================================
# 설정
# =============================================================================
ENABLE_PICK_PLACE = True  # Phase 2에서 True로 변경


# =============================================================================
# 메인
# =============================================================================
def main():
    # 1. 월드 생성
    world = create_world()

    # 2. 씬 구성
    (
        cubes,
        place_target,
        table_top_z,
        cube_size,
        table_xy,
        table_size,
        stack_base_xy,
        human_proxies,
    ) = setup_scene(world, cube_count=6, show_human_proxies=True)
    pick_targets = cubes[:3]
    green_indices = list(range(3, 6))
    green_cubes = [cubes[i] for i in green_indices if i < len(cubes)]
    cube_half = cube_size / 2.0
    cube_center_z = table_top_z + cube_half
    body_anchor = np.array([1.2, 0.0, 1.1])
    shoulder_z = 1.1
    proxy_by_name = {proxy.name: proxy for proxy in human_proxies}
    left_hand_proxy = proxy_by_name.get("human_left_hand")
    right_hand_proxy = proxy_by_name.get("human_right_hand")
    left_arm_proxy = proxy_by_name.get("human_left_arm")
    right_arm_proxy = proxy_by_name.get("human_right_arm")
    last_hand_pose = {
        "left": left_hand_proxy.get_world_pose()[0] if left_hand_proxy else None,
        "right": right_hand_proxy.get_world_pose()[0] if right_hand_proxy else None,
        "head": None,
    }
    log_path = os.path.join(os.path.dirname(__file__), "errp_markers.csv")

    # 3. Panda 추가
    panda = add_panda(world, base_z=table_top_z)

    # 4. 리셋 (물리 핸들 초기화 — 반드시 필요)
    world.reset()

    place_target.set_world_pose(position=np.array([stack_base_xy[0], stack_base_xy[1], table_top_z + cube_half]))

    # 5. 기본 정보 출력
    print_robot_info(panda)

    # 6. 컨트롤러 (Phase 2용)
    controller = None
    if ENABLE_PICK_PLACE:
        approach_height = table_top_z + 0.2
        controller = create_pick_controller(panda, end_effector_initial_height=approach_height)
        print("[Phase 2] Pick-and-Place 컨트롤러 활성화")
    else:
        print("[Phase 1] 씬 확인 모드")
        print("          GUI에서 파란 큐브와 Panda가 보이면 성공!")

    # 7. 시뮬레이션 루프
    step = 0
    task_done = False
    current_pick_idx = 0
    completed_picks = 0
    cycle_reset_requested = False
    last_event_step = {}
    miss_logged_for_pick = False
    stacked_expected = {}
    stack_failed_cubes = set()
    last_contact_step = {}
    drop_logged_cubes = set()
    last_human_contact_step = {}
    human_collision_count = 0
    max_human_collisions = 1000

    udp_host = "0.0.0.0"
    udp_port = 5555
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_host, udp_port))
    sock.setblocking(False)
    speed_threshold = 0.6
    collision_dist = cube_size * 0.9
    stack_drop_threshold = 0.03

    def _log_event(event: str, details: str = ""):
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["sim_time", "event", "details"])
            writer.writerow([sim_time, event, details])
        print(f"[ErrP] {event} | {details}")

    def _sim_time(world_obj, step_idx: int) -> float:
        if hasattr(world_obj, "current_time"):
            return float(world_obj.current_time)
        if hasattr(world_obj, "get_physics_dt"):
            return float(step_idx) * float(world_obj.get_physics_dt())
        return float(step_idx) * (1.0 / 60.0)

    while simulation_app.is_running():
        world.step(render=True)

        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset()
                if controller is not None:
                    controller.reset(end_effector_initial_height=approach_height)
                place_target.set_world_pose(
                    position=np.array([stack_base_xy[0], stack_base_xy[1], table_top_z + cube_half])
                )
                task_done = False
                step = 0
                cycle_reset_requested = False

            step += 1
            sim_time = _sim_time(world, step)

            # Update human proxies from UDP (JSON: {"left":[x,y,z],"right":[x,y,z],"head":[x,y,z]})
            while True:
                try:
                    payload, _ = sock.recvfrom(4096)
                except BlockingIOError:
                    break
                try:
                    data = json.loads(payload.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    if "left" in data:
                        last_hand_pose["left"] = np.array(data["left"], dtype=float)
                    if "right" in data:
                        last_hand_pose["right"] = np.array(data["right"], dtype=float)
                    if "head" in data:
                        last_hand_pose["head"] = np.array(data["head"], dtype=float)

            if left_hand_proxy and right_hand_proxy and left_arm_proxy and right_arm_proxy:
                head_pos = last_hand_pose.get("head")
                if head_pos is not None:
                    shoulder_pos = np.array([head_pos[0], head_pos[1], shoulder_z])
                else:
                    shoulder_pos = body_anchor

                if last_hand_pose["left"] is not None:
                    left_hand_pos = last_hand_pose["left"]
                    left_elbow_pos = (shoulder_pos + left_hand_pos) * 0.5
                    left_hand_proxy.set_world_pose(position=left_hand_pos)
                    left_arm_proxy.set_world_pose(position=left_elbow_pos)

                if last_hand_pose["right"] is not None:
                    right_hand_pos = last_hand_pose["right"]
                    right_elbow_pos = (shoulder_pos + right_hand_pos) * 0.5
                    right_hand_proxy.set_world_pose(position=right_hand_pos)
                    right_arm_proxy.set_world_pose(position=right_elbow_pos)

            # 약 2초마다 상태 출력
            if step % 120 == 0:
                cube_pos, _ = pick_targets[current_pick_idx].get_world_pose()
                stack_height = table_top_z + cube_half + (completed_picks % len(pick_targets)) * (cube_size + 0.002)
                place_pos = np.array([stack_base_xy[0], stack_base_xy[1], stack_height])
                ee_pos, _   = panda.end_effector.get_world_pose()
                gripper_pos = panda.gripper.get_joint_positions()
                print(
                    f"[Step {step:5d}] "
                    f"Cube: {np.round(cube_pos, 3)} | "
                    f"Place: {np.round(place_pos, 3)} | "
                    f"EE: {np.round(ee_pos, 3)} | "
                    f"Gripper: {np.round(gripper_pos, 4)}"
                )

            # Phase 2 동작
            if ENABLE_PICK_PLACE and controller is not None and not task_done:
                current_pick_pos, _ = pick_targets[current_pick_idx].get_world_pose()
                stack_height = table_top_z + cube_half + (completed_picks % len(pick_targets)) * (cube_size + 0.002)
                place_pos = np.array([stack_base_xy[0], stack_base_xy[1], stack_height])
                task_done = run_pick_place(
                    controller=controller,
                    robot=panda,
                    pick_position=current_pick_pos,
                    place_position=place_pos,
                )
                if task_done:
                    placed_cube = pick_targets[current_pick_idx]
                    stacked_expected[placed_cube.name] = stack_height
                    completed_picks += 1
                    current_pick_idx = (current_pick_idx + 1) % len(pick_targets)
                    controller.reset(end_effector_initial_height=approach_height)
                    task_done = False
                    miss_logged_for_pick = False
                    if completed_picks % len(pick_targets) == 0:
                        randomize_cubes(cubes, table_xy, table_size, cube_center_z, cube_size, forbidden_xy=stack_base_xy)
                        completed_picks = 0
                        current_pick_idx = 0
                        controller.reset(end_effector_initial_height=approach_height)
                        cycle_reset_requested = True
                        stacked_expected = {}
                        stack_failed_cubes = set()
                        last_contact_step = {}
                        drop_logged_cubes = set()
                        print(f"\n✅ [Step {step}] 3개 Pick-and-Place 완료! 새 배치로 재시작")

            if cycle_reset_requested:
                world.reset()
                if controller is not None:
                    controller.reset(end_effector_initial_height=approach_height)
                place_target.set_world_pose(
                    position=np.array([stack_base_xy[0], stack_base_xy[1], table_top_z + cube_half])
                )
                task_done = False
                step = 0
                cycle_reset_requested = False

            # ErrP 후보 이벤트 감지
            gripper_pos = panda.gripper.get_joint_positions()
            gripper_closed = np.all(np.array(gripper_pos) < 0.01)
            ee_pos, _ = panda.end_effector.get_world_pose()
            current_cube = pick_targets[current_pick_idx]
            cube_pos, _ = current_cube.get_world_pose()

            for cube in pick_targets:
                pos, _ = cube.get_world_pose()
                if np.linalg.norm(ee_pos - pos) < cube_size * 1.5:
                    last_contact_step[cube.name] = step

            if gripper_closed and not miss_logged_for_pick:
                if np.linalg.norm(ee_pos - cube_pos) > cube_size * 1.2:
                    if step - last_event_step.get("pick_miss", -9999) > 30:
                        _log_event("pick_miss", f"cube={current_cube.name}")
                        last_event_step["pick_miss"] = step
                        miss_logged_for_pick = True

            for cube in pick_targets:
                if hasattr(cube, "get_linear_velocity"):
                    vel = cube.get_linear_velocity()
                    speed = float(np.linalg.norm(vel))
                    if speed > speed_threshold and cube.name not in drop_logged_cubes:
                        recent_contact = step - last_contact_step.get(cube.name, -9999) <= 30
                        if recent_contact and step - last_event_step.get("drop_throw", -9999) > 10:
                            _log_event("drop_throw", f"cube={cube.name},speed={speed:.3f}")
                            last_event_step["drop_throw"] = step
                            drop_logged_cubes.add(cube.name)

            for pick_cube in pick_targets:
                pick_pos, _ = pick_cube.get_world_pose()
                for green_cube in green_cubes:
                    green_pos, _ = green_cube.get_world_pose()
                    if np.linalg.norm(pick_pos - green_pos) < collision_dist:
                        if step - last_event_step.get("collision_green", -9999) > 30:
                            _log_event("collision_green", f"pick={pick_cube.name},green={green_cube.name}")
                            last_event_step["collision_green"] = step

            for proxy in human_proxies:
                proxy_pos, _ = proxy.get_world_pose()
                if np.linalg.norm(ee_pos - proxy_pos) < 0.08:
                    key = f"human_collision:{proxy.name}"
                    if step - last_human_contact_step.get(key, -9999) > 30:
                        _log_event("human_collision", f"proxy={proxy.name}")
                        last_human_contact_step[key] = step
                        human_collision_count += 1
                        if human_collision_count >= max_human_collisions:
                            _log_event("episode_end", f"reason=human_collision_limit,count={human_collision_count}")
                            simulation_app.close()
                            return

            for cube in pick_targets:
                if cube.name in stacked_expected:
                    if cube.name in stack_failed_cubes:
                        continue
                    pos, _ = cube.get_world_pose()
                    if pos[2] < stacked_expected[cube.name] - stack_drop_threshold:
                        if step - last_event_step.get("stack_failure", -9999) > 30:
                            _log_event("stack_failure", f"cube={cube.name},z={pos[2]:.3f}")
                            last_event_step["stack_failure"] = step
                            stack_failed_cubes.add(cube.name)

    simulation_app.close()


if __name__ == "__main__":
    main()