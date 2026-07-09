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
HUMAN_AVATAR_ASSET_PATH = os.environ.get(
    "HUMAN_AVATAR_ASSET_PATH",
    "/home/railabchan/isaac-sim-4.5.0/extscache/"
    "omni.anim.retarget.core-106.5.1+106.5.0.lx64.r.cp310/"
    "data/rigs/Human/assets/human_skeleton.usd",
)
HUMAN_AVATAR_ASSET_SCALE = float(os.environ.get("HUMAN_AVATAR_ASSET_SCALE", "0.01"))
HUMAN_AVATAR_DEBUG_PROXIES = _env_bool("HUMAN_AVATAR_DEBUG_PROXIES", "0")
HUMAN_AVATAR_COLLISION_DIST = float(os.environ.get("HUMAN_AVATAR_COLLISION_DIST", "0.06"))
HUMAN_AVATAR_TASK = os.environ.get("HUMAN_AVATAR_TASK", "touch_green_cube").strip().lower()
HUMAN_AVATAR_TASK_GREEN_INDEX = int(os.environ.get("HUMAN_AVATAR_TASK_GREEN_INDEX", "0"))
HUMAN_AVATAR_TOUCH_DIST = float(os.environ.get("HUMAN_AVATAR_TOUCH_DIST", "0.09"))

JOINT_KEYWORDS = {
    "pelvis": ("pelvis", "hips", "hip"),
    "head": ("head",),
    "neck": ("neck",),
    "chest": ("chest", "spine", "spine2", "spine_2", "upperchest"),
    "left_shoulder": ("left_shoulder", "l_shoulder", "shoulder_l", "leftarm", "l_arm", "upperarm_l"),
    "left_elbow": ("left_elbow", "l_elbow", "elbow_l", "leftforearm", "l_forearm", "lowerarm_l"),
    "left_hand": ("left_hand", "l_hand", "hand_l", "leftwrist", "l_wrist", "wrist_l"),
    "right_shoulder": ("right_shoulder", "r_shoulder", "shoulder_r", "rightarm", "r_arm", "upperarm_r"),
    "right_elbow": ("right_elbow", "r_elbow", "elbow_r", "rightforearm", "r_forearm", "lowerarm_r"),
    "right_hand": ("right_hand", "r_hand", "hand_r", "rightwrist", "r_wrist", "wrist_r"),
}


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


def _as_vec3f(values):
    from pxr import Gf

    return Gf.Vec3f(float(values[0]), float(values[1]), float(values[2]))


def _as_vec3d(values):
    from pxr import Gf

    return Gf.Vec3d(float(values[0]), float(values[1]), float(values[2]))


class HumanAvatar:
    """Isaac human skeleton avatar driven by VR HMD and hand poses.

    The visible body is an Isaac USD skeleton asset. A lightweight internal
    capsule/sphere model is still maintained for stable gripper collision and
    pseudo-ErrP detection; visual debug proxies are opt-in only.
    """

    def __init__(self, table_top_z: float):
        self.enabled = ENABLE_HUMAN_AVATAR
        self._table_top_z = float(table_top_z)
        self._body_anchor_head = None
        self._proxy_prims = {}
        self._segments = {}
        self._spheres = {}
        self._last_left = None
        self._last_right = None
        self._task_marker = None

        self._stage = None
        self._root_prim = None
        self._root_translate_op = None
        self._root_scale_op = None
        self._skeleton = None
        self._skeleton_prim = None
        self._animation = None
        self._animation_uses_translations = False
        self._joints = []
        self._joint_index = {}
        self._joint_parents = []
        self._rest_local_translations = []
        self._rest_global_positions = None
        self._rest_transforms = []
        self._asset_ready = False
        self._root_world = np.zeros(3)

        print(
            f"[HumanAvatar] enabled={self.enabled} | "
            f"asset={HUMAN_AVATAR_ASSET_PATH} | "
            f"scale={HUMAN_AVATAR_ASSET_SCALE:.4f} | "
            f"debug-proxies={HUMAN_AVATAR_DEBUG_PROXIES} | "
            f"task={HUMAN_AVATAR_TASK} | "
            f"touch_dist={HUMAN_AVATAR_TOUCH_DIST:.3f}m | "
            f"collision_dist={HUMAN_AVATAR_COLLISION_DIST:.3f}m"
        )

    def setup(self, world) -> None:
        if not self.enabled:
            return
        self._setup_asset()
        if HUMAN_AVATAR_DEBUG_PROXIES:
            self._setup_debug_proxies(world)
        self._setup_task_marker(world)

    def update(
        self,
        head_pos: "np.ndarray | None",
        left_hand_pos: "np.ndarray | None",
        right_hand_pos: "np.ndarray | None",
    ) -> None:
        if not self.enabled:
            return
        if head_pos is not None and self._body_anchor_head is None:
            self._body_anchor_head = np.asarray(head_pos, dtype=float).copy()
            print(f"[HumanAvatar] standing anchor captured: head={np.round(self._body_anchor_head, 3)}")

        targets = self._compute_body_targets(head_pos, left_hand_pos, right_hand_pos)
        self._update_collision_model(targets)
        if self._asset_ready:
            self._apply_asset_pose(targets)

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
        return list(self._proxy_prims.values()) if self.enabled else []

    def _setup_asset(self) -> None:
        if not HUMAN_AVATAR_ASSET_PATH or not os.path.exists(HUMAN_AVATAR_ASSET_PATH):
            print(f"[HumanAvatar] asset path not found: {HUMAN_AVATAR_ASSET_PATH}")
            return
        try:
            import omni.usd
            from pxr import Sdf, Usd, UsdGeom, UsdSkel

            self._stage = omni.usd.get_context().get_stage()
            self._root_prim = self._stage.GetPrimAtPath(HUMAN_AVATAR_ROOT_PATH)
            if not self._root_prim.IsValid():
                self._root_prim = self._stage.DefinePrim(HUMAN_AVATAR_ROOT_PATH, "Xform")
            self._root_prim.GetReferences().AddReference(HUMAN_AVATAR_ASSET_PATH)

            xformable = UsdGeom.Xformable(self._root_prim)
            xformable.ClearXformOpOrder()
            self._root_translate_op = xformable.AddTranslateOp()
            self._root_scale_op = xformable.AddScaleOp()
            self._root_translate_op.Set(_as_vec3d([0.0, 0.0, 0.0]))
            self._root_scale_op.Set(_as_vec3f([HUMAN_AVATAR_ASSET_SCALE] * 3))

            self._skeleton_prim = self._find_skeleton_prim(Usd, UsdSkel, self._root_prim)
            if self._skeleton_prim is None:
                print(f"[HumanAvatar] skeleton prim not found under {HUMAN_AVATAR_ROOT_PATH}")
                return

            self._skeleton = UsdSkel.Skeleton(self._skeleton_prim)
            self._joints = list(self._skeleton.GetJointsAttr().Get() or [])
            if not self._joints:
                print("[HumanAvatar] skeleton has no joints")
                return
            self._rest_transforms = list(
                self._skeleton.GetRestTransformsAttr().Get()
                or self._skeleton.GetBindTransformsAttr().Get()
                or []
            )
            if len(self._rest_transforms) != len(self._joints):
                print(
                    "[HumanAvatar] rest transform count mismatch: "
                    f"joints={len(self._joints)} rest={len(self._rest_transforms)}"
                )
                return

            self._joint_parents = self._build_joint_parent_indices(self._joints)
            self._rest_local_translations = [
                np.array(transform.ExtractTranslation(), dtype=float)
                for transform in self._rest_transforms
            ]
            self._rest_global_positions = self._compute_global_positions(self._rest_local_translations)
            self._joint_index = self._match_joint_indices(self._joints)

            anim_path = Sdf.Path(f"{HUMAN_AVATAR_ROOT_PATH}/VRTrackingAnimation")
            try:
                self._animation = UsdSkel.Animation.Define(self._stage, anim_path)
            except Exception:
                self._animation = UsdSkel.Animation(self._stage.DefinePrim(anim_path, "SkelAnimation"))
            self._animation.GetJointsAttr().Set(self._joints)
            try:
                self._animation.GetTranslationsAttr().Set([
                    _as_vec3f(pos) for pos in self._rest_local_translations
                ])
                self._animation_uses_translations = True
            except Exception:
                self._animation_uses_translations = False

            self._bind_animation_source(UsdSkel, anim_path)
            self._asset_ready = True
            found = ", ".join(
                f"{key}={str(self._joints[idx]).split('/')[-1]}"
                for key, idx in sorted(self._joint_index.items())
            )
            print(f"[HumanAvatar] asset referenced: {HUMAN_AVATAR_ASSET_PATH}")
            print(f"[HumanAvatar] skeleton found: {self._skeleton_prim.GetPath()} joints={len(self._joints)}")
            print(f"[HumanAvatar] joint map: {found if found else 'none'}")
            if not self._animation_uses_translations:
                print("[HumanAvatar] animation translations unavailable; falling back to rest-transform edits")
        except Exception as exc:
            print(f"[HumanAvatar] asset setup failed: {exc}")
            self._asset_ready = False

    def _setup_debug_proxies(self, world) -> None:
        try:
            from isaacsim.core.api.objects import VisualCapsule, VisualSphere
        except Exception:
            from omni.isaac.core.objects import VisualCapsule, VisualSphere

        base_head = np.array([1.1, 0.0, 1.5])
        skin = np.array([0.82, 0.90, 1.0])
        cloth = np.array([0.10, 0.32, 0.85])
        hand = np.array([0.94, 0.82, 0.68])

        self._proxy_prims["head"] = world.scene.add(VisualSphere(
            prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/debug_head",
            name="human_avatar_debug_head",
            position=base_head,
            radius=0.11,
            color=skin,
        ))
        self._proxy_prims["torso"] = world.scene.add(VisualCapsule(
            prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/debug_torso",
            name="human_avatar_debug_torso",
            position=base_head + np.array([0.0, 0.0, -0.55]),
            radius=0.16,
            height=0.65,
            color=cloth,
        ))
        for side in ("left", "right"):
            color = np.array([0.36, 0.58, 1.0]) if side == "left" else np.array([1.0, 0.56, 0.34])
            self._proxy_prims[f"{side}_upper_arm"] = world.scene.add(VisualCapsule(
                prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/debug_{side}_upper_arm",
                name=f"human_avatar_debug_{side}_upper_arm",
                position=base_head + np.array([0.0, 0.0, -0.35]),
                radius=0.045,
                height=0.3,
                color=color,
            ))
            self._proxy_prims[f"{side}_forearm"] = world.scene.add(VisualCapsule(
                prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/debug_{side}_forearm",
                name=f"human_avatar_debug_{side}_forearm",
                position=base_head + np.array([0.0, 0.0, -0.55]),
                radius=0.04,
                height=0.3,
                color=color,
            ))
            self._proxy_prims[f"{side}_hand"] = world.scene.add(VisualSphere(
                prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/debug_{side}_hand",
                name=f"human_avatar_debug_{side}_hand",
                position=base_head + np.array([0.0, 0.0, -0.7]),
                radius=0.06,
                color=hand,
            ))

    def _setup_task_marker(self, world) -> None:
        try:
            from isaacsim.core.api.objects import VisualSphere
        except Exception:
            from omni.isaac.core.objects import VisualSphere

        self._task_marker = world.scene.add(VisualSphere(
            prim_path=f"{HUMAN_AVATAR_ROOT_PATH}/task_touch_target",
            name="human_avatar_task_touch_target",
            position=np.array([0.0, 0.0, -100.0]),
            radius=0.035,
            color=np.array([0.1, 1.0, 0.25]),
        ))

    def _find_skeleton_prim(self, Usd, UsdSkel, root_prim):
        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() == "Skeleton":
                return prim
            try:
                skel = UsdSkel.Skeleton(prim)
                if skel and skel.GetPrim().IsValid() and skel.GetJointsAttr().Get():
                    return prim
            except Exception:
                pass
        return None

    def _bind_animation_source(self, UsdSkel, anim_path) -> None:
        try:
            binding = UsdSkel.BindingAPI.Apply(self._skeleton_prim)
            binding.CreateAnimationSourceRel().SetTargets([anim_path])
            return
        except Exception:
            pass
        try:
            self._skeleton_prim.CreateRelationship("skel:animationSource").SetTargets([anim_path])
        except Exception as exc:
            print(f"[HumanAvatar] failed to bind skeleton animation source: {exc}")

    def _build_joint_parent_indices(self, joints: list) -> list[int]:
        token_to_index = {str(token): idx for idx, token in enumerate(joints)}
        parents = []
        for token in joints:
            parts = str(token).split("/")
            parent_token = "/".join(parts[:-1])
            parents.append(token_to_index.get(parent_token, -1))
        return parents

    def _compute_global_positions(self, local_positions: list[np.ndarray]) -> np.ndarray:
        positions = []
        for idx, local in enumerate(local_positions):
            parent_idx = self._joint_parents[idx] if idx < len(self._joint_parents) else -1
            if parent_idx >= 0:
                positions.append(positions[parent_idx] + local)
            else:
                positions.append(local.copy())
        return np.asarray(positions, dtype=float)

    def _match_joint_indices(self, joints: list) -> dict:
        names = [str(joint).split("/")[-1].lower() for joint in joints]
        full_names = [str(joint).lower() for joint in joints]
        matches = {}
        for key, keywords in JOINT_KEYWORDS.items():
            for idx, (name, full_name) in enumerate(zip(names, full_names)):
                if any(keyword in name or keyword in full_name for keyword in keywords):
                    matches[key] = idx
                    break
        return matches

    def _compute_body_targets(
        self,
        head_pos: "np.ndarray | None",
        left_hand_pos: "np.ndarray | None",
        right_hand_pos: "np.ndarray | None",
    ) -> dict:
        anchor_head = self._body_anchor_head
        if anchor_head is None:
            anchor_head = np.array([1.1, 0.0, 1.5])
        visual_head = np.asarray(head_pos, dtype=float) if head_pos is not None else anchor_head

        neck = anchor_head + np.array([0.0, 0.0, -0.16])
        chest = anchor_head + np.array([0.0, 0.0, -0.38])
        pelvis = anchor_head + np.array([0.0, 0.0, -0.88])
        targets = {
            "head": visual_head,
            "neck": neck,
            "chest": chest,
            "pelvis": pelvis,
        }
        for side, input_pos in (("left", left_hand_pos), ("right", right_hand_pos)):
            sign = 1.0 if side == "left" else -1.0
            shoulder = chest + np.array([0.0, sign * 0.24, 0.04])
            hand_pos = self._stable_hand_pos(side, input_pos, shoulder)
            elbow = self._elbow_pos(side, shoulder, hand_pos)
            targets[f"{side}_shoulder"] = shoulder
            targets[f"{side}_elbow"] = elbow
            targets[f"{side}_hand"] = hand_pos
        return targets

    def _update_collision_model(self, targets: dict) -> None:
        self._set_sphere("head", targets["head"], 0.11)
        self._set_segment("torso", targets["pelvis"], targets["neck"], 0.16)
        for side in ("left", "right"):
            self._set_segment(
                f"{side}_upper_arm",
                targets[f"{side}_shoulder"],
                targets[f"{side}_elbow"],
                0.045,
            )
            self._set_segment(
                f"{side}_forearm",
                targets[f"{side}_elbow"],
                targets[f"{side}_hand"],
                0.04,
            )
            self._set_sphere(f"{side}_hand", targets[f"{side}_hand"], 0.06)

    def _apply_asset_pose(self, targets: dict) -> None:
        try:
            if self._rest_global_positions is None or not self._joints:
                return
            scale = max(1e-6, HUMAN_AVATAR_ASSET_SCALE)
            head_idx = self._joint_index.get("head")
            pelvis_idx = self._joint_index.get("pelvis")
            if head_idx is not None:
                self._root_world = targets["head"] - self._rest_global_positions[head_idx] * scale
            elif pelvis_idx is not None:
                self._root_world = targets["pelvis"] - self._rest_global_positions[pelvis_idx] * scale
            else:
                self._root_world = targets["pelvis"]

            if self._root_translate_op is not None:
                self._root_translate_op.Set(_as_vec3d(self._root_world))
            if self._root_scale_op is not None:
                self._root_scale_op.Set(_as_vec3f([scale, scale, scale]))

            local_translations = [pos.copy() for pos in self._rest_local_translations]
            for key, world_target in targets.items():
                idx = self._joint_index.get(key)
                if idx is None:
                    continue
                skeleton_target = (np.asarray(world_target, dtype=float) - self._root_world) / scale
                parent_idx = self._joint_parents[idx]
                if parent_idx >= 0:
                    parent_pos = self._rest_global_positions[parent_idx]
                    parent_key = self._key_for_joint_index(parent_idx)
                    if parent_key in targets:
                        parent_pos = (np.asarray(targets[parent_key], dtype=float) - self._root_world) / scale
                    local_translations[idx] = skeleton_target - parent_pos
                else:
                    local_translations[idx] = skeleton_target

            if self._animation is not None and self._animation_uses_translations:
                self._animation.GetTranslationsAttr().Set([_as_vec3f(pos) for pos in local_translations])
            else:
                self._apply_rest_transform_fallback(local_translations)
        except Exception as exc:
            print(f"[HumanAvatar] pose update failed: {exc}")
            self._asset_ready = False

    def _apply_rest_transform_fallback(self, local_translations: list[np.ndarray]) -> None:
        if self._skeleton is None or not self._rest_transforms:
            return
        from pxr import Gf

        transforms = []
        for transform, translation in zip(self._rest_transforms, local_translations):
            mat = Gf.Matrix4d(transform)
            mat.SetTranslateOnly(_as_vec3d(translation))
            transforms.append(mat)
        self._skeleton.GetRestTransformsAttr().Set(transforms)

    def _key_for_joint_index(self, joint_idx: int) -> str:
        for key, idx in self._joint_index.items():
            if idx == joint_idx:
                return key
        return ""

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
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        self._segments[name] = (a.copy(), b.copy(), float(radius))
        prim = self._proxy_prims.get(name)
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

    def _set_sphere(self, name: str, center: np.ndarray, radius: float) -> None:
        center = np.asarray(center, dtype=float)
        self._spheres[name] = (center.copy(), float(radius))
        prim = self._proxy_prims.get(name)
        if prim is None:
            return
        try:
            prim.set_world_pose(position=center)
        except Exception:
            pass

    def _park_task_marker(self) -> None:
        if self._task_marker is None:
            return
        try:
            self._task_marker.set_world_pose(position=np.array([0.0, 0.0, -100.0]))
        except Exception:
            pass
