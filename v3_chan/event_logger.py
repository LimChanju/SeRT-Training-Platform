import csv
import os
from typing import Dict, Iterable, Optional

import numpy as np


def _format_hit_label(hit: str) -> str:
    return "->".join(part.split("/")[-1] for part in str(hit).split("->"))


class EventLogger:
    def __init__(
        self,
        log_path: str,
        cube_size: float,
        speed_threshold: float,
        collision_dist: float,
        stack_drop_threshold: float,
        max_human_collisions: int,
        sample_path: Optional[str] = None,
        sample_interval_steps: int = 1,
    ) -> None:
        self._log_path = log_path
        self._sample_path = sample_path
        self._sample_interval_steps = max(1, int(sample_interval_steps))
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
        self.start_episode("reason=run_start")

    def start_episode(self, details: str = "") -> None:
        if not self._episode_started:
            self.log_event("episode_start", details)
            self._episode_started = True

    def end_episode(self, details: str = "") -> None:
        if self._episode_started:
            self.log_event("episode_end", details)
            self._episode_started = False

    def log_event(self, event: str, details: str = "") -> None:
        file_exists = os.path.exists(self._log_path)
        with open(self._log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["sim_time", "event", "details"])
            writer.writerow([self._sim_time, event, details])
        print(f"[ErrP] {event} | {details}")

    def log_sample(
        self,
        left_hand_gripper_dist: Optional[float],
        right_hand_gripper_dist: Optional[float],
        min_hand_gripper_dist: Optional[float],
        human_robot_collision: bool,
    ) -> None:
        if self._sample_path is None:
            return
        if self._step % self._sample_interval_steps != 0:
            return

        file_exists = os.path.exists(self._sample_path)
        with open(self._sample_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(
                    [
                        "sim_time",
                        "step",
                        "left_hand_gripper_dist_m",
                        "right_hand_gripper_dist_m",
                        "min_hand_gripper_dist_m",
                        "human_robot_collision",
                    ]
                )
            writer.writerow(
                [
                    self._sim_time,
                    self._step,
                    self._format_optional_float(left_hand_gripper_dist),
                    self._format_optional_float(right_hand_gripper_dist),
                    self._format_optional_float(min_hand_gripper_dist),
                    1 if human_robot_collision else 0,
                ]
            )

    def _format_optional_float(self, value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{float(value):.6f}"

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

    def check_arm_robot_proximity(
        self,
        hand: str,
        hit_prims: list,
        surface_gap_m: float | None = None,
    ) -> None:
        """Record hand proximity to a built-in distal Panda collider."""
        if not hit_prims:
            return
        key = f"arm_robot_proximity:{hand}"
        if self._step - self._last_event_step.get(key, -9999) > 30:
            links = ",".join(_format_hit_label(p) for p in hit_prims[:3])
            gap = "" if surface_gap_m is None else f",surface_gap_m={surface_gap_m:.4f}"
            self.log_event("arm_robot_proximity", f"hand={hand},links={links}{gap}")
            self._last_event_step[key] = self._step

    def check_arm_robot_collision(
        self,
        hand: str,
        hit_prims: list,
        surface_gap_m: float | None = None,
    ) -> None:
        """Record hand overlap with a built-in distal Panda collider."""
        if not hit_prims:
            return
        key = f"arm_robot_collision:{hand}"
        if self._step - self._last_event_step.get(key, -9999) > 30:
            links = ",".join(_format_hit_label(p) for p in hit_prims[:3])
            gap = "" if surface_gap_m is None else f",surface_gap_m={surface_gap_m:.4f}"
            self.log_event("arm_robot_collision", f"hand={hand},links={links}{gap}")
            self._last_event_step[key] = self._step

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
