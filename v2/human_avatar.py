import os

import numpy as np


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


ENABLE_HUMAN_AVATAR = _env_bool("ENABLE_HUMAN_AVATAR", "0")
HUMAN_AVATAR_ROOT_PATH = os.environ.get("HUMAN_AVATAR_ROOT_PATH", "/World/HumanAvatar")
HUMAN_AVATAR_COLLISION_DIST = float(os.environ.get("HUMAN_AVATAR_COLLISION_DIST", "0.06"))
HUMAN_AVATAR_TASK = os.environ.get("HUMAN_AVATAR_TASK", "touch_green_cube").strip().lower()
HUMAN_AVATAR_TASK_GREEN_INDEX = int(os.environ.get("HUMAN_AVATAR_TASK_GREEN_INDEX", "0"))
HUMAN_AVATAR_TOUCH_DIST = float(os.environ.get("HUMAN_AVATAR_TOUCH_DIST", "0.09"))


def _dist_point_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-10:
        return float(np.linalg.norm(point - a))
    t = np.clip(float(np.dot(point - a, ab)) / denom, 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * ab)))


def _quat_from_z_axis(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    target = direction / norm
    source = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    cross = np.cross(source, target)
    quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=float)
    return quat / float(np.linalg.norm(quat))


class HumanAvatar:
    """Scene-native standing human proxy for VR/RL experiments.

    The body stands at the first tracked HMD position, while the arms follow the
    VR hand/controller poses. It is intentionally represented as Isaac prims so
    collision distance, task state, and replay/RL observations can share one
    stable scene object.
    """

    def __init__(self, table_top_z: float):
        self.enabled = ENABLE_HUMAN_AVATAR
        self._table_top_z = float(table_top_z)
        self._body_anchor_head = None
        self._prims = {}
        self._segments = {}
        self._spheres = {}
        self._last_left = None
        self._last_right = None
        self._task_marker = None
        print(
            f"[HumanAvatar] enabled={self.enabled} | "
            f"task={HUMAN_AVATAR_TASK} | "
            f"touch_dist={HUMAN_AVATAR_TOUCH_DIST:.3f}m | "
            f"collision_dist={HUMAN_AVATAR_COLLISION_DIST:.3f}m"
        )

    def setup(self, world) -> None:
        if not self.enabled:
            return
        try:
            from isaacsim.core.api.objects import VisualCapsule, VisualSphere
        except Exception:
            from omni.isaac.core.objects import VisualCapsule, VisualSphere

        base_head = np.array([1.1, 0.0, 1.5])
        skin = np.array([0.82, 0.90, 1.0])
        cloth = np.array([0.10, 0.32, 0.85])
        hand = np.array([0.94, 0.82, 0.68])
        cue = np.array([0.1, 1.0, 0.25])

        self._prims["head"] = world.scene.add(VisualSphere(
            prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/head",
            name="human_avatar_head",
            position=base_head,
            radius=0.11,
            color=skin,
        ))
        self._prims["torso"] = world.scene.add(VisualCapsule(
            prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/torso",
            name="human_avatar_torso",
            position=base_head + np.array([0.0, 0.0, -0.55]),
            radius=0.16,
            height=0.65,
            color=cloth,
        ))
        for side in ("left", "right"):
            color = np.array([0.36, 0.58, 1.0]) if side == "left" else np.array([1.0, 0.56, 0.34])
            self._prims[f"{side}_upper_arm"] = world.scene.add(VisualCapsule(
                prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/{side}_upper_arm",
                name=f"human_avatar_{side}_upper_arm",
                position=base_head + np.array([0.0, 0.0, -0.35]),
                radius=0.045,
                height=0.3,
                color=color,
            ))
            self._prims[f"{side}_forearm"] = world.scene.add(VisualCapsule(
                prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/{side}_forearm",
                name=f"human_avatar_{side}_forearm",
                position=base_head + np.array([0.0, 0.0, -0.55]),
                radius=0.04,
                height=0.3,
                color=color,
            ))
            self._prims[f"{side}_hand"] = world.scene.add(VisualSphere(
                prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/{side}_hand",
                name=f"human_avatar_{side}_hand",
                position=base_head + np.array([0.0, 0.0, -0.7]),
                radius=0.06,
                color=hand,
            ))

        self._task_marker = world.scene.add(VisualSphere(
            prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/task_touch_target",
            name="human_avatar_task_touch_target",
            position=np.array([0.0, 0.0, -100.0]),
            radius=0.035,
            color=cue,
        ))

    def update(
        self,
        head_pos: "np.ndarray | None",
        left_hand_pos: "np.ndarray | None",
        right_hand_pos: "np.ndarray | None",
    ) -> None:
        if not self.enabled or not self._prims:
            return
        if head_pos is not None and self._body_anchor_head is None:
            self._body_anchor_head = np.asarray(head_pos, dtype=float).copy()
            print(f"[HumanAvatar] standing anchor captured: head={np.round(self._body_anchor_head, 3)}")
        anchor_head = self._body_anchor_head
        if anchor_head is None:
            anchor_head = np.array([1.1, 0.0, 1.5])

        visual_head = np.asarray(head_pos, dtype=float) if head_pos is not None else anchor_head
        self._set_sphere("head", visual_head, 0.11)

        neck = anchor_head + np.array([0.0, 0.0, -0.16])
        chest = anchor_head + np.array([0.0, 0.0, -0.38])
        pelvis = anchor_head + np.array([0.0, 0.0, -0.88])
        self._set_segment("torso", pelvis, neck, 0.16)

        for side, input_pos in (("left", left_hand_pos), ("right", right_hand_pos)):
            sign = 1.0 if side == "left" else -1.0
            shoulder = chest + np.array([0.0, sign * 0.24, 0.04])
            hand_pos = self._stable_hand_pos(side, input_pos, shoulder)
            elbow = self._elbow_pos(side, shoulder, hand_pos)
            self._set_segment(f"{side}_upper_arm", shoulder, elbow, 0.045)
            self._set_segment(f"{side}_forearm", elbow, hand_pos, 0.04)
            self._set_sphere(f"{side}_hand", hand_pos, 0.06)

    def update_task(
        self,
        green_cubes: list,
        left_hand_pos: "np.ndarray | None",
        right_hand_pos: "np.ndarray | None",
    ) -> dict:
        if not self.enabled or HUMAN_AVATAR_TASK != "touch_green_cube" or not green_cubes:
            self._park_task_marker()
            return {}
        idx = HUMAN_AVATAR_TASK_GREEN_INDEX % len(green_cubes)
        cube = green_cubes[idx]
        cube_pos, _ = cube.get_world_pose()
        target_pos = np.asarray(cube_pos, dtype=float)
        if self._task_marker is not None:
            self._task_marker.set_world_pose(position=target_pos + np.array([0.0, 0.0, 0.09]))

        distances = []
        for hand_name, pos in (("left", left_hand_pos), ("right", right_hand_pos)):
            if pos is None:
                continue
            distances.append((float(np.linalg.norm(np.asarray(pos, dtype=float) - target_pos)), hand_name))
        if not distances:
            return {
                "target": cube.name,
                "target_pos": target_pos,
                "min_dist": None,
                "hand": "",
                "touched": False,
            }
        min_dist, hand_name = min(distances, key=lambda item: item[0])
        return {
            "target": cube.name,
            "target_pos": target_pos,
            "min_dist": min_dist,
            "hand": hand_name,
            "touched": min_dist <= HUMAN_AVATAR_TOUCH_DIST,
        }

    def check_gripper_collision(self, gripper_pos: "np.ndarray | None") -> list[dict]:
        if not self.enabled or gripper_pos is None:
            return []
        point = np.asarray(gripper_pos, dtype=float)
        hits = []
        for name, (a, b, radius) in self._segments.items():
            dist = max(0.0, _dist_point_segment(point, a, b) - radius)
            if dist <= HUMAN_AVATAR_COLLISION_DIST:
                hits.append({"part": name, "dist": dist})
        for name, (center, radius) in self._spheres.items():
            dist = max(0.0, float(np.linalg.norm(point - center)) - radius)
            if dist <= HUMAN_AVATAR_COLLISION_DIST:
                hits.append({"part": name, "dist": dist})
        hits.sort(key=lambda hit: hit["dist"])
        return hits

    def get_avatar_prims(self) -> list:
        return list(self._prims.values()) if self.enabled else []

    def _stable_hand_pos(
        self,
        side: str,
        input_pos: "np.ndarray | None",
        shoulder: np.ndarray,
    ) -> np.ndarray:
        attr = "_last_left" if side == "left" else "_last_right"
        if input_pos is not None:
            pos = np.asarray(input_pos, dtype=float)
            setattr(self, attr, pos)
            return pos
        last = getattr(self, attr)
        if last is not None:
            return last
        sign = 1.0 if side == "left" else -1.0
        return shoulder + np.array([-0.15, sign * 0.12, -0.35])

    def _elbow_pos(self, side: str, shoulder: np.ndarray, hand_pos: np.ndarray) -> np.ndarray:
        sign = 1.0 if side == "left" else -1.0
        midpoint = shoulder + 0.55 * (hand_pos - shoulder)
        bend = np.array([0.04, sign * 0.08, -0.10])
        return midpoint + bend

    def _set_segment(self, name: str, a: np.ndarray, b: np.ndarray, radius: float) -> None:
        prim = self._prims.get(name)
        if prim is None:
            return
        length = max(0.02, float(np.linalg.norm(b - a)))
        center = (a + b) * 0.5
        try:
            prim.set_height(length)
        except Exception:
            pass
        try:
            prim.set_world_pose(position=center, orientation=_quat_from_z_axis(b - a))
        except Exception:
            pass
        self._segments[name] = (a.copy(), b.copy(), float(radius))

    def _set_sphere(self, name: str, center: np.ndarray, radius: float) -> None:
        prim = self._prims.get(name)
        if prim is None:
            return
        try:
            prim.set_world_pose(position=center)
        except Exception:
            pass
        self._spheres[name] = (center.copy(), float(radius))

    def _park_task_marker(self) -> None:
        if self._task_marker is None:
            return
        try:
            self._task_marker.set_world_pose(position=np.array([0.0, 0.0, -100.0]))
        except Exception:
            pass
