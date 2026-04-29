import csv
import os
from typing import Dict, Iterable, Optional

import numpy as np


class EventLogger:
    def __init__(
        self,
        log_path: str,
        cube_size: float,
        speed_threshold: float,
        collision_dist: float,
        stack_drop_threshold: float,
        max_human_collisions: int,
    ) -> None:
        self._log_path = log_path
        self._cube_size = cube_size
        self._speed_threshold = speed_threshold
        self._collision_dist = collision_dist
        self._stack_drop_threshold = stack_drop_threshold
        self._max_human_collisions = max_human_collisions
        self._last_event_step: Dict[str, int] = {}
        self._last_contact_step: Dict[str, int] = {}
        self._drop_logged_cubes = set()
        self._last_human_contact_step: Dict[str, int] = {}
        self._stacked_expected: Dict[str, float] = {}
        self._stack_failed_cubes = set()
        self._miss_logged_for_pick = False
        self._human_collision_count = 0
        self._episode_started = False
        self._sim_time = 0.0
        self._step = 0

    def update_context(self, step: int, sim_time: float) -> None:
        self._step = step
        self._sim_time = sim_time

    def ensure_episode_started(self) -> None:
        if not self._episode_started:
            self.log_event("episode_start", "reason=run_start")
            self._episode_started = True

    def log_event(self, event: str, details: str = "") -> None:
        file_exists = os.path.exists(self._log_path)
        with open(self._log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["sim_time", "event", "details"])
            writer.writerow([self._sim_time, event, details])
        print(f"[ErrP] {event} | {details}")

    def reset_cycle(self) -> None:
        self._stacked_expected = {}
        self._stack_failed_cubes = set()
        self._last_contact_step = {}
        self._drop_logged_cubes = set()

    def reset_pick_miss(self) -> None:
        self._miss_logged_for_pick = False

    def record_stack_expected(self, cube_name: str, stack_height: float) -> None:
        self._stacked_expected[cube_name] = stack_height

    def update_contact(self, ee_pos: np.ndarray, pick_targets: Iterable) -> None:
        for cube in pick_targets:
            pos, _ = cube.get_world_pose()
            if np.linalg.norm(ee_pos - pos) < self._cube_size * 1.5:
                self._last_contact_step[cube.name] = self._step

    def check_pick_miss(self, gripper_closed: bool, ee_pos: np.ndarray, current_cube) -> None:
        if gripper_closed and not self._miss_logged_for_pick:
            cube_pos, _ = current_cube.get_world_pose()
            if np.linalg.norm(ee_pos - cube_pos) > self._cube_size * 1.2:
                if self._step - self._last_event_step.get("pick_miss", -9999) > 30:
                    self.log_event("pick_miss", f"cube={current_cube.name}")
                    self._last_event_step["pick_miss"] = self._step
                    self._miss_logged_for_pick = True

    def check_drop_throw(self, pick_targets: Iterable) -> None:
        for cube in pick_targets:
            if hasattr(cube, "get_linear_velocity"):
                vel = cube.get_linear_velocity()
                speed = float(np.linalg.norm(vel))
                if speed > self._speed_threshold and cube.name not in self._drop_logged_cubes:
                    recent_contact = self._step - self._last_contact_step.get(cube.name, -9999) <= 30
                    if recent_contact and self._step - self._last_event_step.get("drop_throw", -9999) > 10:
                        self.log_event("drop_throw", f"cube={cube.name},speed={speed:.3f}")
                        self._last_event_step["drop_throw"] = self._step
                        self._drop_logged_cubes.add(cube.name)

    def check_collision_green(self, pick_targets: Iterable, green_cubes: Iterable) -> None:
        for pick_cube in pick_targets:
            pick_pos, _ = pick_cube.get_world_pose()
            for green_cube in green_cubes:
                green_pos, _ = green_cube.get_world_pose()
                if np.linalg.norm(pick_pos - green_pos) < self._collision_dist:
                    if self._step - self._last_event_step.get("collision_green", -9999) > 30:
                        self.log_event("collision_green", f"pick={pick_cube.name},green={green_cube.name}")
                        self._last_event_step["collision_green"] = self._step

    def check_human_collision(self, ee_pos: np.ndarray, human_proxies: Iterable) -> bool:
        for proxy in human_proxies:
            proxy_pos, _ = proxy.get_world_pose()
            if np.linalg.norm(ee_pos - proxy_pos) < 0.08:
                key = f"human_collision:{proxy.name}"
                if self._step - self._last_human_contact_step.get(key, -9999) > 30:
                    self.log_event("human_collision", f"proxy={proxy.name}")
                    self._last_human_contact_step[key] = self._step
                    self._human_collision_count += 1
                    if self._human_collision_count >= self._max_human_collisions:
                        self.log_event(
                            "episode_end",
                            f"reason=human_collision_limit,count={self._human_collision_count}",
                        )
                        return True
        return False

    def check_stack_failure(self, pick_targets: Iterable) -> None:
        for cube in pick_targets:
            if cube.name in self._stacked_expected:
                if cube.name in self._stack_failed_cubes:
                    continue
                pos, _ = cube.get_world_pose()
                if pos[2] < self._stacked_expected[cube.name] - self._stack_drop_threshold:
                    if self._step - self._last_event_step.get("stack_failure", -9999) > 30:
                        self.log_event("stack_failure", f"cube={cube.name},z={pos[2]:.3f}")
                        self._last_event_step["stack_failure"] = self._step
                        self._stack_failed_cubes.add(cube.name)
