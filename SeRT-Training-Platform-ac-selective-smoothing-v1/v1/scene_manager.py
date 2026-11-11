import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import XFormPrim

class SceneManager:
    def __init__(self, spawn_z=0.04026):
        self.spawn_x_range = [0.0, 0.2]
        self.spawn_y_range = [-0.1, 0.3]
        self.spawn_z = spawn_z
        self.placed_positions = []
        self.keepout_positions = []
        self.keepout_radius = 0.18
        self.min_spawn_dist = 0.12
        self.workspace_center = None
        self.workspace_radius = None
        self.workspace_y_min = None
        self.workspace_y_max = None
        
        self.red_cube_paths = ["/World/Red_Cube01", "/World/Red_Cube02", "/World/Red_Cube03"]
        self.green_cube_paths = ["/World/Green_Cube_1", "/World/Green_Cube_2", "/World/Green_Cube_3"]
        
        self.red_cubes = self._load_cubes(self.red_cube_paths)
        self.green_cubes = self._load_cubes(self.green_cube_paths)

    def set_keepout_positions(self, positions, radius=0.18):
        self.keepout_positions = positions
        self.keepout_radius = radius

    def set_workspace_circle(self, center, radius):
        self.workspace_center = np.array(center)
        self.workspace_radius = radius

    def set_workspace_y_range(self, y_min=None, y_max=None):
        self.workspace_y_min = y_min
        self.workspace_y_max = y_max

    def _load_cubes(self, paths):
        cubes = []
        for path in paths:
            if get_prim_at_path(path):
                cubes.append(XFormPrim(path))
        return cubes

    def get_safe_spawn_pos(self):
        for _ in range(500):
            if self.workspace_center is not None and self.workspace_radius is not None:
                angle = np.random.uniform(0.0, 2.0 * np.pi)
                radius = np.random.uniform(0.12, self.workspace_radius)
                new_pos = np.array([
                    self.workspace_center[0] + radius * np.cos(angle),
                    self.workspace_center[1] + radius * np.sin(angle),
                    self.spawn_z
                ])
            else:
                new_pos = np.array([
                    np.random.uniform(self.spawn_x_range[0], self.spawn_x_range[1]),
                    np.random.uniform(self.spawn_y_range[0], self.spawn_y_range[1]),
                    self.spawn_z
                ])
            if all(np.linalg.norm(new_pos[:2] - p[:2]) > self.min_spawn_dist for p in self.placed_positions) and \
               all(np.linalg.norm(new_pos[:2] - p[:2]) > self.keepout_radius for p in self.keepout_positions):
                if self.workspace_y_min is not None and new_pos[1] < self.workspace_y_min:
                    continue
                if self.workspace_y_max is not None and new_pos[1] > self.workspace_y_max:
                    continue
                return new_pos
        print("⚠️ [경고] 500번 시도 후에도 겹치지 않는 위치를 찾지 못했습니다!")
        return new_pos

    def randomize_cubes(self):
        print("모든 큐브의 초기 위치를 무작위로 섞습니다...")
        self.placed_positions = []
        
        for cube in self.red_cubes:
            new_pos = self.get_safe_spawn_pos()
            self.placed_positions.append(new_pos)
            cube.set_world_pose(position=new_pos)
            
        print(f"✅ 인식된 초록 큐브 개수: {len(self.green_cubes)}개")
        for cube in self.green_cubes:
            new_pos = self.get_safe_spawn_pos()
            self.placed_positions.append(new_pos)
            cube.set_world_pose(position=new_pos)