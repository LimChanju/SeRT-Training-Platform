import sys
print("🚨 [디버그 0] 진짜 main.py 실행 시작!", flush=True)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# ==========================================================
# 🚨 [수정] 네가 직접 찾은 VR 확장팩 2개 강제 로드
# ==========================================================
from omni.isaac.core.utils.extensions import enable_extension
print("🚨 [디버그 0.5] SteamVR 및 VR Experience 모듈 강제 주입 중...", flush=True)

# 1. SteamVR 통신 코어 로드
enable_extension("omni.kit.xr.system.steamvr")
# 2. VR UI(Start VR 버튼) 및 프로필 로드
enable_extension("omni.kit.xr.profile.vr")

# 확장팩이 UI에 완전히 반영될 때까지 프레임을 밀어줌
for _ in range(5):
    simulation_app.update()

print("🚨 [디버그 1] 엔진 부팅 및 VR UI 로드 완료! 기본 모듈 로드 중...", flush=True)

import os
import time
import numpy as np
import carb
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage, is_stage_loading
import omni.kit.viewport.utility as vp_utils

# 커스텀 모듈 임포트 (경로 엇갈림 방지)
sys.path.append(os.path.expanduser("~/isaac_vr_project"))
from ssvep_manager import SSVEPManager
from scene_manager import SceneManager
from robot_manager import RobotManager
from haptic_manager import HapticManager

print("🚨 [디버그 2] 모듈 임포트 완료! USD 로딩 시도...", flush=True)
usd_path = os.path.expanduser("~/isaac_vr_project/stack_blocks_with_human.usd")
open_stage(usd_path)

print("🚨 [디버그 3] USD 로딩 대기 중 (무한루프 방지 탑재)...", flush=True)
timeout = 0
while is_stage_loading() and timeout < 500:
    simulation_app.update()
    timeout += 1

if timeout >= 500:
    print("⚠️ [경고] 로딩이 너무 오래 걸립니다. 일단 강제 진행합니다.", flush=True)
else:
    print(f"🎉 [디버그 4] {timeout} 프레임 만에 USD 로딩 완료!", flush=True)

print("🚨 [디버그 5] 월드 및 로봇 매니저 초기화...", flush=True)
world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
robot_manager = RobotManager(world)

world.reset()
world.play()

print("🚨 [디버그 6] 물리 엔진 예열 중...")
for _ in range(30):
    world.step(render=True)

print("🚨 [디버그 7] 컨트롤러 및 씬(큐브) 매니저 초기화...", flush=True)
ssvep_manager = SSVEPManager()
scene_manager = SceneManager(spawn_z=0.04026)

robot_manager.initialize_robot()
robot_manager.setup_controller()

robot_base_pos, _ = robot_manager.robot.get_world_pose()
scene_manager.set_keepout_positions([robot_base_pos], radius=0.18)
scene_manager.set_workspace_circle(center=robot_base_pos, radius=0.8)
scene_manager.set_workspace_y_range(y_min=robot_base_pos[1] + 0.05)

settings = carb.settings.get_settings()
settings.set("/app/runLoops/main/rateLimitEnabled", True)
settings.set("/app/runLoops/main/rateLimitFrequency", 60)

print("🚨 [디버그 8] 큐브 랜덤 배치 실행 중...", flush=True)
scene_manager.randomize_cubes()

# 물리 엔진이 블록을 안정화하도록 충분한 프레임을 시뮬레이션합니다.
print("🚨 [디버그 8.5] 큐브 안정화 대기 중 (물리 정착)...", flush=True)
for _ in range(120):
    world.step(render=True)

# 컨트롤러와 블록의 잔류 속도를 초기화 시도
try:
    robot_manager.reset_controller()
except Exception:
    pass

for cube in scene_manager.red_cubes + scene_manager.green_cubes:
    try:
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
    except Exception:
        # 일부 프림은 물리 속성 메서드가 없을 수 있으므로 무시
        pass

print("🎉 [디버그 9] 모든 준비 완료! 메인 루프 진입!", flush=True)

# 시뮬레이션 제어 변수
current_cube_idx = 0
stack_goal_pos = np.array([-0.3, -0.1, 0.04026])
stack_step = 0.065
episode_idx = 0
max_episodes = None  # None이면 무한 반복, 숫자를 넣으면 해당 횟수만 수행
ik_fail_count = 0
ik_fail_limit = 120
target_anchor_pos = None
target_anchor_idx = -1
cube_bounds = {
    "x": (-0.8, 0.8),
    "y": (-0.8, 0.8),
    "z": (0.0, 0.6),
}

def is_cube_lost(pos, bounds):
    return (
        pos[0] < bounds["x"][0]
        or pos[0] > bounds["x"][1]
        or pos[1] < bounds["y"][0]
        or pos[1] > bounds["y"][1]
        or pos[2] < bounds["z"][0]
        or pos[2] > bounds["z"][1]
    )

# 메인 루프
while simulation_app.is_running():
    t = time.time()
    
    # SSVEP 업데이트
    ssvep_manager.update(t)

    # 로봇 컨트롤 업데이트
    if current_cube_idx < len(scene_manager.red_cubes):
        target_cube = scene_manager.red_cubes[current_cube_idx]
        cube_pos, _ = target_cube.get_world_pose()

        robot_base_pos, _ = robot_manager.robot.get_world_pose()
        if cube_pos[1] < robot_base_pos[1] + 0.02:
            print(f"⚠️ 큐브가 로봇 뒤쪽에 있습니다. 재배치합니다: {cube_pos}")
            new_pos = scene_manager.get_safe_spawn_pos()
            scene_manager.placed_positions.append(new_pos)
            target_cube.set_world_pose(position=new_pos)
            robot_manager.reset_controller()
            ik_fail_count = 0
            target_anchor_pos = None
            target_anchor_idx = -1
            for _ in range(30):
                world.step(render=True)
            continue
        if np.linalg.norm(cube_pos[:2] - robot_base_pos[:2]) > scene_manager.workspace_radius:
            print(f"⚠️ 큐브가 로봇 작업 반경 밖에 있습니다. 재배치합니다: {cube_pos}")
            new_pos = scene_manager.get_safe_spawn_pos()
            scene_manager.placed_positions.append(new_pos)
            target_cube.set_world_pose(position=new_pos)
            robot_manager.reset_controller()
            ik_fail_count = 0
            target_cube = scene_manager.red_cubes[current_cube_idx]
            target_anchor_pos = None
            target_anchor_idx = -1
            for _ in range(30):
                world.step(render=True)
            continue
        if np.linalg.norm(cube_pos[:2] - robot_base_pos[:2]) < 0.18:
            print(f"⚠️ 큐브가 로봇 아래에 있습니다. 재배치합니다: {cube_pos}")
            new_pos = scene_manager.get_safe_spawn_pos()
            scene_manager.placed_positions.append(new_pos)
            target_cube.set_world_pose(position=new_pos)
            robot_manager.reset_controller()
            ik_fail_count = 0
            target_cube = scene_manager.red_cubes[current_cube_idx]
            target_anchor_pos = None
            target_anchor_idx = -1
            for _ in range(30):
                world.step(render=True)
            continue

        if is_cube_lost(cube_pos, cube_bounds):
            print(f"⚠️ 큐브가 범위를 벗어났습니다. 재배치합니다: {cube_pos}")
            new_pos = scene_manager.get_safe_spawn_pos()
            scene_manager.placed_positions.append(new_pos)
            target_cube.set_world_pose(position=new_pos)
            robot_manager.reset_controller()
            ik_fail_count = 0
            target_anchor_pos = None
            target_anchor_idx = -1
            for _ in range(30):
                world.step(render=True)
            continue

        if target_anchor_idx != current_cube_idx or target_anchor_pos is None:
            target_anchor_pos = cube_pos.copy()
            target_anchor_idx = current_cube_idx
        elif np.linalg.norm(cube_pos - target_anchor_pos) > 0.08:
            print(f"⚠️ 큐브 위치가 크게 변했습니다. 재계획합니다: {cube_pos}")
            robot_manager.reset_controller()
            ik_fail_count = 0
            target_anchor_pos = cube_pos.copy()
        
        actions = robot_manager.get_action(target_anchor_pos, stack_goal_pos)
        
        if actions is not None:
            robot_manager.apply_action(actions)
            ik_fail_count = 0
        else:
            ik_fail_count += 1
            if int(t * 10) % 20 == 0:
                print(f"⚠️ IK 계산 포기! (로봇이 닿을 수 없는 각도입니다) - 큐브 위치: {cube_pos}")
            if ik_fail_count >= ik_fail_limit:
                print(f"⚠️ IK 실패가 누적되었습니다. 큐브를 재배치합니다: {cube_pos}")
                new_pos = scene_manager.get_safe_spawn_pos()
                scene_manager.placed_positions.append(new_pos)
                target_cube.set_world_pose(position=new_pos)
                robot_manager.reset_controller()
                ik_fail_count = 0
                target_cube = scene_manager.red_cubes[current_cube_idx]
                target_anchor_pos = None
                target_anchor_idx = -1
                for _ in range(30):
                    world.step(render=True)
                continue

        if robot_manager.is_done():
            target_cube = scene_manager.red_cubes[current_cube_idx]
            placed_ok = robot_manager.check_cube_placed(target_cube, stack_goal_pos)
            if placed_ok:
                print(f"🎉 {current_cube_idx + 1}번째 큐브 완료!")
                robot_manager.reset_controller()
                ik_fail_count = 0
                target_anchor_pos = None
                target_anchor_idx = -1
                current_cube_idx += 1
                stack_goal_pos[2] += stack_step  # 위로 쌓기
            else:
                print(f"⚠️ {current_cube_idx + 1}번째 큐브 위치 실패! 재시도 중...")
                robot_manager.reset_controller()
                ik_fail_count = 0
                target_anchor_pos = None
                target_anchor_idx = -1

    world.step(render=True)

    # 초록 큐브가 밖으로 나갔을 때 재배치
    for green_cube in scene_manager.green_cubes:
        green_pos, _ = green_cube.get_world_pose()
        if is_cube_lost(green_pos, cube_bounds):
            print(f"⚠️ 초록 큐브가 범위를 벗어났습니다. 재배치합니다: {green_pos}")
            new_pos = scene_manager.get_safe_spawn_pos()
            scene_manager.placed_positions.append(new_pos)
            green_cube.set_world_pose(position=new_pos)
            for _ in range(10):
                world.step(render=True)

    if current_cube_idx >= len(scene_manager.red_cubes):
        print("🔁 에피소드 완료. 큐브 재배치 및 다음 에피소드 시작...")
        episode_idx += 1
        if max_episodes is not None and episode_idx >= max_episodes:
            print("✅ 지정된 에피소드 횟수 완료. 시뮬레이션을 종료합니다.")
            break
        scene_manager.randomize_cubes()
        for _ in range(120):
            world.step(render=True)
        robot_manager.reset_controller()
        ik_fail_count = 0
        target_anchor_pos = None
        target_anchor_idx = -1
        stack_goal_pos = np.array([-0.3, -0.1, 0.04026])
        stack_step = 0.065
        current_cube_idx = 0

simulation_app.close()