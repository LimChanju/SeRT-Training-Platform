# vr_grab.py — VR 컨트롤러로 블록 집기 (XRCore 기반 입력)
#
# ── 그립 감지 전략 ────────────────────────────────────────────────────────────
# Isaac Sim 4.5 는 OpenXR 을 사용하므로 OpenVR getControllerState() 는 btns=0.
# 대신 XRCore 의 입력 디바이스 객체를 통해 grip/squeeze 액션 값을 읽는다.
# 최초 실행 시 device 메서드 목록을 출력하여 올바른 API 를 확인한다.
#
# 키보드 폴백 (XR 그립 감지 실패 시):
#   G 키 = 왼쪽 컨트롤러 grab / H 키 = 오른쪽 컨트롤러 grab

import os

import numpy as np

from vr.vr_avatar import XR_SWAP_HANDS, xr_path_for_hand

GRAB_RADIUS = float(os.environ.get("GRAB_RADIUS", "0.35"))  # VR 물리-시뮬 스케일 차이 보정
GRAB_THRESHOLD = 0.03
UNGRAB_THRESHOLD = 0.06
POSE_PINCH_GRAB = os.environ.get("POSE_PINCH_GRAB", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
POSE_PINCH_MAX_DIST = float(os.environ.get("POSE_PINCH_MAX_DIST", "0.18"))
POSE_PINCH_RELEASE_DIST = float(
    os.environ.get("POSE_PINCH_RELEASE_DIST", str(POSE_PINCH_MAX_DIST * 1.25))
)
STICKY_PROXIMITY_GRAB = os.environ.get(
    "STICKY_PROXIMITY_GRAB",
    os.environ.get(
        "HANDOVER_STICKY_PROXIMITY_GRAB",
        os.environ.get("ENABLE_HANDOVER_TASK", "0"),
    ),
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
STICKY_PROXIMITY_RADIUS = float(
    os.environ.get(
        "STICKY_PROXIMITY_RADIUS",
        os.environ.get(
            "HANDOVER_STICKY_PROXIMITY_RADIUS",
            "0.18" if os.environ.get("ENABLE_HANDOVER_TASK", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ) else "0.12",
        ),
    )
)
GRAB_POSE_PRIORITY = tuple(
    pose.strip()
    for pose in os.environ.get("GRAB_POSE_PRIORITY", "pinch,grip,aim,palm,poke").split(",")
    if pose.strip()
)
DEBUG_GRAB_DISTANCE = os.environ.get("DEBUG_GRAB_DISTANCE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PHYSICAL_PROXY_CONTACT_GRAB = os.environ.get(
    "XR_PHYSICAL_HAND_CONTACT_GRAB",
    os.environ.get(
        "PHYSICAL_PROXY_CONTACT_GRAB",
        os.environ.get("ENABLE_HANDOVER_TASK", "0"),
    ),
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PHYSICAL_PROXY_CONTACT_MARGIN = float(
    os.environ.get(
        "XR_PHYSICAL_HAND_CONTACT_MARGIN",
        os.environ.get(
            "PHYSICAL_PROXY_CONTACT_MARGIN",
            "0.08" if os.environ.get("ENABLE_HANDOVER_TASK", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ) else "0.015",
        ),
    )
)
PHYSICAL_FIXED_JOINT_GRAB = os.environ.get(
    "XR_PHYSICAL_HAND_FIXED_JOINT_GRAB",
    os.environ.get("PHYSICAL_FIXED_JOINT_GRAB", os.environ.get("ENABLE_HANDOVER_TASK", "0")),
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PHYSICAL_JOINT_COLLISION = os.environ.get(
    "XR_PHYSICAL_HAND_JOINT_COLLISION",
    os.environ.get("PHYSICAL_HAND_JOINT_COLLISION", "0"),
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PHYSICAL_ALLOW_LOGICAL_FALLBACK = os.environ.get(
    "XR_PHYSICAL_HAND_ALLOW_LOGICAL_FALLBACK",
    os.environ.get("PHYSICAL_HAND_ALLOW_LOGICAL_FALLBACK", "0" if PHYSICAL_FIXED_JOINT_GRAB else "1"),
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
STICKY_PROXIMITY_INITIAL_GRAB = os.environ.get(
    "STICKY_PROXIMITY_INITIAL_GRAB",
    os.environ.get(
        "HANDOVER_STICKY_PROXIMITY_INITIAL_GRAB",
        "0" if PHYSICAL_PROXY_CONTACT_GRAB else "1",
    ),
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)


class VRGrabManager:
    def __init__(self, cubes, avatar=None, cube_size: "float | None" = None):
        self.cubes       = cubes
        self.grabbed     = {"left": None, "right": None}
        self.grab_offset = {"left": None, "right": None}
        self.grab_source = {"left": None, "right": None}
        self.grab_joint_path = {"left": None, "right": None}
        self.grab_mode = {"left": None, "right": None}
        self.grab_suppressed = {"left": set(), "right": set()}
        self._last_suppressed_grab_log = {"left": None, "right": None}
        self.prev_grip   = {"left": False, "right": False}
        self._active_cube = None
        self._active_hand = None
        self._cube_half_extent = float(cube_size) * 0.5 if cube_size else None

        self._avatar     = avatar   # VRAvatar 인스턴스 (XRCore 재사용)
        self._xr_core    = None
        self._xr_tried   = False
        self._inspected  = False    # device 메서드 목록 1회 출력 플래그
        self._gesture_inspected = False
        self._inputs_inspected = False
        self._input_gestures_inspected = False
        self._no_gesture_logged = False
        self._gesture_candidates_by_path = {}
        self._last_grip_log = {"left": None, "right": None}
        self._frame = 0
        self._pose_pinch_unavailable = set()

        # carb.input 캐시
        self._carb_iface = None
        self._carb_kb    = None
        self._carb_tried = False

        print(
            "[VRGrab] Initialized. Keyboard fallback: G=left, H=right | "
            f"hand-swap={XR_SWAP_HANDS} | "
            f"pose-pinch={POSE_PINCH_GRAB} "
            f"grab={POSE_PINCH_MAX_DIST:.3f}m release={POSE_PINCH_RELEASE_DIST:.3f}m | "
            f"sticky-proximity={STICKY_PROXIMITY_GRAB} "
            f"initial={STICKY_PROXIMITY_INITIAL_GRAB} "
            f"radius={STICKY_PROXIMITY_RADIUS:.3f}m | "
            f"grab-poses={GRAB_POSE_PRIORITY} | "
            f"physical-contact={PHYSICAL_PROXY_CONTACT_GRAB} "
            f"margin={PHYSICAL_PROXY_CONTACT_MARGIN:.3f}m | "
            f"fixed-joint={PHYSICAL_FIXED_JOINT_GRAB} "
            f"joint-collision={PHYSICAL_JOINT_COLLISION} "
            f"logical-fallback={PHYSICAL_ALLOW_LOGICAL_FALLBACK}"
        )

    def set_active_target(self, cube=None, hand: "str | None" = None) -> None:
        self._active_cube = cube
        self._active_hand = hand if hand in ("left", "right") else None

    # ── XRCore 획득 ──────────────────────────────────────────────────────────
    def _get_xr_core(self):
        """avatar._xr_core 를 우선 재사용; 없으면 자체 초기화."""
        if self._avatar is not None and self._avatar._xr_core is not None:
            return self._avatar._xr_core
        if not self._xr_tried:
            self._xr_tried = True
            try:
                from omni.kit.xr.core import XRCore
                self._xr_core = XRCore.get_singleton()
                print("[VRGrab] XRCore acquired.")
            except Exception as e:
                print(f"[VRGrab] XRCore not available: {e}")
        return self._xr_core

    # ── carb.input 획득 ──────────────────────────────────────────────────────
    def _get_carb_input(self):
        if not self._carb_tried:
            self._carb_tried = True
            try:
                import carb.input
                import omni.appwindow
                self._carb_iface = carb.input.acquire_input_interface()
                app_window = omni.appwindow.get_default_app_window()
                self._carb_kb = app_window.get_keyboard() if app_window is not None else None
            except Exception as e:
                print(f"[VRGrab] carb.input not available: {e}")
        return self._carb_iface, self._carb_kb

    # ── 그립 읽기 ─────────────────────────────────────────────────────────────
    def _read_grip(self, hand: str) -> bool:
        threshold = 0.3 if self.prev_grip[hand] else 0.7
        xr_path   = xr_path_for_hand(hand)

        xr_core = self._get_xr_core()
        if xr_core is not None:
            try:
                # Isaac XRCore can segfault if get_input_device() is called repeatedly
                # while SteamVR is changing state. Reuse the device cached by VRAvatar.
                device = None
                if self._avatar is not None:
                    get_cached = getattr(self._avatar, "get_cached_xr_device", None)
                    if get_cached is not None:
                        device = get_cached(xr_path)
                if device is None:
                    return False
                if device is not None:

                    # ── 첫 실행 시 device 메서드 목록 출력 ──────────────────
                    if not self._inspected:
                        self._inspected = True
                        methods = [m for m in dir(device) if not m.startswith("_")]
                        print(f"[VRGrab] XRInputDevice methods: {methods}")

                    gesture_candidates = self._gesture_candidates_for_device(device, xr_path)
                    for input_name, gesture in gesture_candidates:
                        try:
                            val = device.get_input_gesture_value(input_name, gesture)
                        except Exception:
                            continue
                        if val is None:
                            continue
                        try:
                            v = float(val)
                        except Exception:
                            continue
                        if v <= 0.01:
                            continue
                        active = v > (0.5 if gesture in ("click", "press", "activate") else threshold)
                        log_key = (input_name, gesture, active)
                        if log_key != self._last_grip_log.get(hand) or self._frame % 60 == 0:
                            self._last_grip_log[hand] = log_key
                            print(f"[VRGrab] {hand} input({input_name}:{gesture})={v:.3f} active={active}")
                        if active:
                            return True

            except Exception:
                pass

        # ── 키보드 폴백: G=left, H=right ────────────────────────────────────
        iface, kb = self._get_carb_input()
        if iface is not None and kb is not None:
            try:
                import carb.input
                key = (carb.input.KeyboardInput.G if hand == "left"
                       else carb.input.KeyboardInput.H)
                val = iface.get_keyboard_value(kb, key)
                if val > 0:
                    return True
            except Exception:
                pass

        return False

    def _gesture_candidates_for_device(self, device, xr_path: str) -> list[tuple[str, str]]:
        cached = self._gesture_candidates_by_path.get(xr_path)
        if cached is not None:
            return cached

        try:
            input_names = [str(n) for n in device.get_input_names()]
        except Exception:
            input_names = []
        if input_names and not self._inputs_inspected:
            self._inputs_inspected = True
            print(f"[VRGrab] XR input names: {input_names}")

        input_gestures = {}
        for input_name in input_names:
            try:
                gestures = [str(n) for n in device.get_input_gesture_names(input_name)]
            except Exception:
                gestures = []
            input_gestures[input_name] = gestures

        if input_gestures and not self._input_gestures_inspected:
            self._input_gestures_inspected = True
            interesting = {
                name: gestures
                for name, gestures in input_gestures.items()
                if gestures or name in ("trigger", "squeeze", "grip", "pinch", "select")
            }
            print(f"[VRGrab] XR input gesture map: {interesting}")
        if not any(input_gestures.values()) and not self._no_gesture_logged:
            self._no_gesture_logged = True
            print("[VRGrab] XR input gestures: none yet; bare-hand grab may need ALVR pinch/controller emulation.")

        input_priority = (
            "squeeze",
            "trigger",
            "grip",
            "select",
            "pinch",
            "a",
            "x",
        )
        gesture_priority = (
            "value",
            "click",
            "press",
            "activate",
        )
        candidates = []
        for input_name in tuple(dict.fromkeys([*input_priority, *input_names])):
            if input_names and input_name not in input_names:
                continue
            gestures = input_gestures.get(input_name, [])
            if not gestures:
                continue
            for gesture in gesture_priority:
                if gesture in gestures:
                    candidates.append((input_name, gesture))

        self._gesture_candidates_by_path[xr_path] = candidates
        return candidates

    # ── 매 프레임 호출 ─────────────────────────────────────────────────────────
    def update(self, left_pos, right_pos, pinch_points=None):
        self._frame += 1
        pinch_points = pinch_points or {}
        for hand, pos in (("left", left_pos), ("right", right_pos)):
            if (
                self._active_cube is not None
                and self._active_hand is not None
                and hand != self._active_hand
                and self.grabbed[hand] is None
            ):
                self.prev_grip[hand] = False
                continue
            candidates = self._grab_candidates(hand, pos, pinch_points.get(hand))
            if not candidates:
                if self.grabbed[hand] is not None:
                    self._release(hand)
                    self.prev_grip[hand] = False
                continue

            pinch = self._read_pinch(hand, pinch_points.get(hand))
            if pinch is not None:
                grip, grab_pos = pinch
                grab_source = "pinch_mid"
            else:
                grip = self._read_grip(hand)
                grab_source, grab_pos = self._nearest_candidate_to_cube(candidates)
                if grab_pos is None:
                    grab_source, grab_pos = candidates[0]
                if not grip:
                    pose_pinch = self._read_pose_pinch(hand, pos)
                    if pose_pinch is not None:
                        grip, grab_pos = pose_pinch
                        grab_source = "pose_pinch"
                if not grip:
                    contact = self._read_physical_proxy_contact(hand, candidates)
                    if contact is not None:
                        grip, grab_source, grab_pos = contact
                if not grip:
                    sticky = self._read_sticky_proximity(hand, candidates)
                    if sticky is not None:
                        grip, grab_source, grab_pos = sticky
                elif self._avatar is not None:
                    get_pose = getattr(self._avatar, "get_hand_pose_pos", None)
                    if get_pose is not None:
                        pinch_pos = get_pose(hand, "pinch")
                        if pinch_pos is not None:
                            grab_pos = pinch_pos
                            grab_source = "pinch"

            if grab_pos is None:
                continue

            prev = self.prev_grip[hand]

            if grip and not prev:
                grab_candidates = (
                    [(grab_source, grab_pos)]
                    if grab_source == "physical_proxy"
                    else candidates
                )
                self._try_grab(
                    hand,
                    grab_pos,
                    grab_source=grab_source,
                    candidates=grab_candidates,
                )
            elif not grip and prev:
                self._release(hand)

            if (
                grip
                and self.grabbed[hand] is not None
                and self.grab_mode.get(hand) != "fixed_joint"
            ):
                follow_pos = self._follow_position_for_hand(hand, candidates, fallback=grab_pos)
                self.grabbed[hand].set_world_pose(
                    position=follow_pos + self.grab_offset[hand]
                )

            self.prev_grip[hand] = grip

    def _read_pinch(self, hand: str, points) -> "tuple[bool, np.ndarray] | None":
        if not points:
            return None
        index_tip = points.get("index_tip")
        thumb_tip = points.get("thumb_tip")
        if index_tip is None or thumb_tip is None:
            return None
        pinch_dist = float(np.linalg.norm(index_tip - thumb_tip))
        threshold = UNGRAB_THRESHOLD if self.prev_grip[hand] else GRAB_THRESHOLD
        pinch_pos = (index_tip + thumb_tip) * 0.5
        return pinch_dist < threshold, pinch_pos

    def _read_pose_pinch(self, hand: str, palm_pos: np.ndarray) -> "tuple[bool, np.ndarray] | None":
        """Fallback for Quest/ALVR bare-hand tracking when squeeze/trigger stays 0."""
        if not POSE_PINCH_GRAB or self._avatar is None or palm_pos is None:
            return None
        get_pose = getattr(self._avatar, "get_hand_pose_pos", None)
        if get_pose is None:
            return None
        pinch_pos = get_pose(hand, "pinch")
        if pinch_pos is None:
            if hand not in self._pose_pinch_unavailable:
                self._pose_pinch_unavailable.add(hand)
                print(f"[VRGrab] {hand} pose-pinch unavailable; waiting for XR pinch pose.")
            return None
        pinch_dist = float(np.linalg.norm(pinch_pos - palm_pos))
        threshold = POSE_PINCH_RELEASE_DIST if self.prev_grip[hand] else POSE_PINCH_MAX_DIST
        active = pinch_dist < threshold
        if self._frame % 60 == 0:
            print(
                f"[VRGrab] {hand} pose-pinch dist={pinch_dist:.3f} "
                f"threshold={threshold:.3f} active={active}"
            )
        return active, pinch_pos

    def _grab_candidates(self, hand: str, hand_pos, points) -> list[tuple[str, np.ndarray]]:
        candidates: list[tuple[str, np.ndarray]] = []
        if points:
            index_tip = points.get("index_tip")
            thumb_tip = points.get("thumb_tip")
            if index_tip is not None and thumb_tip is not None:
                candidates.append(("pinch_mid", (index_tip + thumb_tip) * 0.5))
                candidates.append(("index_tip", index_tip))
                candidates.append(("thumb_tip", thumb_tip))
        if self._avatar is not None:
            get_proxy_pos = getattr(self._avatar, "get_physical_hand_proxy_pos", None)
            if get_proxy_pos is not None:
                proxy_pos = get_proxy_pos(hand)
                if proxy_pos is not None:
                    candidates.append(("physical_proxy", proxy_pos))
            get_pose = getattr(self._avatar, "get_hand_pose_pos", None)
            if get_pose is not None:
                for pose_name in GRAB_POSE_PRIORITY:
                    pose_pos = get_pose(hand, pose_name)
                    if pose_pos is not None:
                        candidates.append((pose_name, pose_pos))
        if hand_pos is not None:
            candidates.append(("hand", np.asarray(hand_pos, dtype=float)))

        unique: list[tuple[str, np.ndarray]] = []
        seen = set()
        for source, candidate in candidates:
            arr = np.asarray(candidate, dtype=float)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                continue
            key = (source, tuple(np.round(arr, 4)))
            if key in seen:
                continue
            seen.add(key)
            unique.append((source, arr))
        return unique

    def _read_sticky_proximity(self, hand: str, candidates: list[tuple[str, np.ndarray]]) -> "tuple[bool, str, np.ndarray] | None":
        if not STICKY_PROXIMITY_GRAB or not candidates:
            return None
        if self.grabbed[hand] is not None:
            source, pos = self._follow_candidate_for_hand(hand, candidates)
            return True, source, pos
        if not STICKY_PROXIMITY_INITIAL_GRAB:
            return None

        source, pos, nearest, nearest_dist = self._nearest_candidate_cube_pair(
            candidates,
            STICKY_PROXIMITY_RADIUS,
        )
        if nearest is None or nearest_dist > STICKY_PROXIMITY_RADIUS:
            self._debug_nearest(hand, candidates, max_dist=STICKY_PROXIMITY_RADIUS)
            return None
        return True, source, pos

    def _read_physical_proxy_contact(
        self,
        hand: str,
        candidates: list[tuple[str, np.ndarray]],
    ) -> "tuple[bool, str, np.ndarray] | None":
        if not PHYSICAL_PROXY_CONTACT_GRAB:
            return None
        proxy_candidates = [
            (source, pos)
            for source, pos in candidates
            if source == "physical_proxy"
        ]
        if not proxy_candidates:
            return None
        source, pos, cube, dist = self._nearest_candidate_cube_pair(
            proxy_candidates,
            PHYSICAL_PROXY_CONTACT_MARGIN,
        )
        if cube is None or dist > PHYSICAL_PROXY_CONTACT_MARGIN:
            return None
        return True, source, pos

    # ── 근접 집기 ──────────────────────────────────────────────────────────────
    def _try_grab(self, hand: str, controller_pos: np.ndarray, *, grab_source: str = "hand", candidates=None):
        if self.grabbed[hand] is not None:
            return
        candidates = candidates or [(grab_source, controller_pos)]
        source, pos, nearest, nearest_dist = self._nearest_candidate_cube_pair(
            candidates,
            GRAB_RADIUS,
        )
        if PHYSICAL_FIXED_JOINT_GRAB:
            proxy_candidates = [
                (candidate_source, candidate_pos)
                for candidate_source, candidate_pos in candidates
                if candidate_source == "physical_proxy"
            ]
            proxy_source, proxy_pos, proxy_cube, proxy_dist = self._nearest_candidate_cube_pair(
                proxy_candidates,
                PHYSICAL_PROXY_CONTACT_MARGIN,
            )
            if proxy_cube is not None:
                source, pos, nearest, nearest_dist = (
                    proxy_source,
                    proxy_pos,
                    proxy_cube,
                    proxy_dist,
                )
            elif not PHYSICAL_ALLOW_LOGICAL_FALLBACK:
                print(
                    f"[VRGrab] {hand} physical grab requires hand proxy contact "
                    f"(margin={PHYSICAL_PROXY_CONTACT_MARGIN:.3f} m)"
                )
                return

        if nearest is not None:
            if self._grab_is_suppressed(hand, nearest):
                log_key = (getattr(nearest, "name", "cube"), self._frame // 30)
                if self._last_suppressed_grab_log.get(hand) != log_key:
                    self._last_suppressed_grab_log[hand] = log_key
                    print(
                        f"[VRGrab] {hand} grab suppressed for '{nearest.name}' "
                        "during handover takeover"
                    )
                return
            cube_pos, _ = nearest.get_world_pose()
            attached = self._attach(hand, nearest, cube_pos, pos, source)
            if attached:
                print(
                    f"[VRGrab] {hand} grabbed '{nearest.name}' "
                    f"(dist={nearest_dist:.3f} m via {source})"
                )
            else:
                print(
                    f"[VRGrab] {hand} grab failed for '{nearest.name}' "
                    f"(dist={nearest_dist:.3f} m via {source})"
                )
        else:
            print(f"[VRGrab] {hand} grip pressed — no cube within {GRAB_RADIUS} m")

    def _nearest_cube(
        self,
        position: np.ndarray,
        max_dist: float = GRAB_RADIUS,
        *,
        source: str = "hand",
    ):
        nearest, nearest_dist = None, max_dist
        for cube in self._candidate_cubes():
            if (
                self._same_cube(cube, self.grabbed["left"])
                or self._same_cube(cube, self.grabbed["right"])
            ):
                continue
            dist = self._candidate_cube_distance(source, position, cube)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = cube
        return nearest, nearest_dist

    def _candidate_cubes(self):
        if self._active_cube is not None:
            return [self._active_cube]
        return self.cubes

    def _nearest_candidate_cube_pair(self, candidates, max_dist: float):
        best_source, best_pos, best_cube, best_dist = None, None, None, max_dist
        for source, pos in candidates:
            cube, dist = self._nearest_cube(pos, max_dist=best_dist, source=source)
            if cube is not None and dist < best_dist:
                best_source, best_pos, best_cube, best_dist = source, pos, cube, dist
        return best_source, best_pos, best_cube, best_dist

    def _candidate_cube_distance(self, source: str, position: np.ndarray, cube) -> float:
        position = np.asarray(position, dtype=float)
        try:
            cube_pos, _ = cube.get_world_pose()
        except Exception:
            return float("inf")
        cube_pos = np.asarray(cube_pos, dtype=float)
        if position.shape != (3,) or cube_pos.shape != (3,):
            return float("inf")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(cube_pos)):
            return float("inf")

        if source == "physical_proxy":
            half = self._cube_half_extent
            if half is None:
                half = self._infer_cube_half_extent(cube)
            radius = self._physical_proxy_radius()
            delta = np.maximum(np.abs(position - cube_pos) - half, 0.0)
            return max(0.0, float(np.linalg.norm(delta)) - radius)

        return float(np.linalg.norm(position - cube_pos))

    def _infer_cube_half_extent(self, cube) -> float:
        try:
            prim = getattr(cube, "prim", None)
            if prim is not None:
                scale_attr = prim.GetAttribute("xformOp:scale")
                scale = scale_attr.Get() if scale_attr.IsValid() else None
                if scale is not None:
                    vals = np.asarray([scale[0], scale[1], scale[2]], dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        return float(np.max(vals)) * 0.5
        except Exception:
            pass
        return 0.025

    def _physical_proxy_radius(self) -> float:
        if self._avatar is None:
            return 0.0
        getter = getattr(self._avatar, "get_physical_hand_proxy_radius", None)
        if getter is None:
            return 0.0
        try:
            return max(0.0, float(getter()))
        except Exception:
            return 0.0

    def _nearest_candidate_to_cube(self, candidates):
        source, pos, cube, _ = self._nearest_candidate_cube_pair(candidates, GRAB_RADIUS)
        if cube is None:
            return (None, None)
        return source, pos

    def _follow_candidate_for_hand(self, hand: str, candidates):
        wanted = self.grab_source.get(hand)
        if wanted is not None:
            for source, pos in candidates:
                if source == wanted:
                    return source, pos
        source, pos = self._nearest_candidate_to_cube(candidates)
        if pos is not None:
            return source, pos
        return candidates[0]

    def _follow_position_for_hand(self, hand: str, candidates, fallback: np.ndarray) -> np.ndarray:
        if candidates:
            _, pos = self._follow_candidate_for_hand(hand, candidates)
            return pos
        return np.asarray(fallback, dtype=float)

    def _debug_nearest(self, hand: str, candidates, max_dist: float) -> None:
        if not DEBUG_GRAB_DISTANCE or self._frame % 30 != 0:
            return
        best_source, best_pos, best_cube, best_dist = self._nearest_candidate_cube_pair(
            candidates,
            max_dist=GRAB_RADIUS,
        )
        cube_name = getattr(best_cube, "name", None) if best_cube is not None else None
        print(
            f"[VRGrab] {hand} nearest cube={cube_name} dist={best_dist:.3f} "
            f"source={best_source} sticky_radius={max_dist:.3f}"
        )

    # ── 부착 ───────────────────────────────────────────────────────────────────
    def _attach(self, hand: str, cube, cube_pos, controller_pos, source: str = "hand") -> bool:
        use_fixed_joint = (
            PHYSICAL_FIXED_JOINT_GRAB
            and source == "physical_proxy"
            and self._physical_proxy_path(hand) is not None
            and self._cube_prim_path(cube) is not None
        )
        if use_fixed_joint:
            self._set_cube_collision_enabled(cube, True)
            if hasattr(cube, "enable_rigid_body_physics"):
                cube.enable_rigid_body_physics()
            self._zero_cube_velocity(cube)
            joint_path = self._create_hand_cube_fixed_joint(
                hand,
                cube,
                proxy_pos=np.asarray(controller_pos, dtype=float),
                cube_pos=np.asarray(cube_pos, dtype=float),
            )
            if not joint_path:
                self.grab_joint_path[hand] = None
                self.grab_mode[hand] = None
                return False
            self.grab_joint_path[hand] = joint_path
            self.grab_mode[hand] = "fixed_joint"
        else:
            if hasattr(cube, "set_default_state"):
                cube.set_default_state(position=cube_pos)
            cube.set_world_pose(position=cube_pos)
            self._set_cube_collision_enabled(cube, False)
            if hasattr(cube, "disable_rigid_body_physics"):
                cube.disable_rigid_body_physics()
            self.grab_joint_path[hand] = None
            self.grab_mode[hand] = "logical_pose"
        self.grabbed[hand]     = cube
        self.grab_offset[hand] = cube_pos - controller_pos
        self.grab_source[hand] = source
        return True

    # ── 놓기 ───────────────────────────────────────────────────────────────────
    def _release(self, hand: str):
        cube = self.grabbed[hand]
        if cube is not None:
            self._remove_grab_joint(hand)
            self._set_cube_collision_enabled(cube, True)
            if hasattr(cube, "enable_rigid_body_physics"):
                cube.enable_rigid_body_physics()
            if hasattr(cube, "set_linear_velocity"):
                cube.set_linear_velocity(np.zeros(3))
            if hasattr(cube, "set_angular_velocity"):
                cube.set_angular_velocity(np.zeros(3))
            print(f"[VRGrab] {hand} released '{cube.name}'")
            self.grabbed[hand]     = None
            self.grab_offset[hand] = None
            self.grab_source[hand] = None
            self.grab_mode[hand] = None

    def _create_hand_cube_fixed_joint(
        self,
        hand: str,
        cube,
        *,
        proxy_pos: np.ndarray,
        cube_pos: np.ndarray,
    ) -> "str | None":
        proxy_path = self._physical_proxy_path(hand)
        cube_path = self._cube_prim_path(cube)
        if proxy_path is None or cube_path is None:
            return None
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return None
            safe_cube_name = getattr(cube, "name", "cube").replace("/", "_")
            if not stage.GetPrimAtPath("/World/HandoverJoints").IsValid():
                UsdGeom.Xform.Define(stage, "/World/HandoverJoints")
            joint_path = f"/World/HandoverJoints/{hand}_{safe_cube_name}_fixed_joint"
            if stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(proxy_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(cube_path)])
            local0 = np.asarray(cube_pos, dtype=float) - np.asarray(proxy_pos, dtype=float)
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(local0[0]), float(local0[1]), float(local0[2])))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateCollisionEnabledAttr().Set(bool(PHYSICAL_JOINT_COLLISION))
            joint.CreateBreakForceAttr().Set(1.0e20)
            joint.CreateBreakTorqueAttr().Set(1.0e20)
            print(
                f"[VRGrab] {hand} fixed joint created: {joint_path} "
                f"body0={proxy_path} body1={cube_path} offset={np.round(local0, 3)}"
            )
            return joint_path
        except Exception as exc:
            print(f"[VRGrab] failed to create fixed joint for {hand}: {exc}")
            return None

    def _remove_grab_joint(self, hand: str) -> None:
        joint_path = self.grab_joint_path.get(hand)
        self.grab_joint_path[hand] = None
        if not joint_path:
            return
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is not None and stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)
                print(f"[VRGrab] {hand} fixed joint removed: {joint_path}")
        except Exception as exc:
            print(f"[VRGrab] failed to remove fixed joint {joint_path}: {exc}")

    def _set_cube_collision_enabled(self, cube, enabled: bool) -> None:
        method = getattr(cube, "enable_collision" if enabled else "disable_collision", None)
        if callable(method):
            try:
                method()
            except Exception:
                pass

        prim = getattr(cube, "prim", None)
        if prim is None:
            return
        try:
            from pxr import Usd, UsdGeom, UsdPhysics

            geom_types = {
                "Cube",
                "Mesh",
                "Sphere",
                "Capsule",
                "Cylinder",
                "Cone",
            }
            for child in Usd.PrimRange(prim):
                if (
                    not child.HasAPI(UsdPhysics.CollisionAPI)
                    and child.GetTypeName() not in geom_types
                    and not child.IsA(UsdGeom.Gprim)
                ):
                    continue
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    collision_api = UsdPhysics.CollisionAPI(child)
                else:
                    collision_api = UsdPhysics.CollisionAPI.Apply(child)
                attr = collision_api.GetCollisionEnabledAttr()
                if not attr.IsValid():
                    attr = collision_api.CreateCollisionEnabledAttr()
                attr.Set(bool(enabled))
        except Exception:
            pass

    def _zero_cube_velocity(self, cube) -> None:
        for method_name in ("set_linear_velocity", "set_angular_velocity"):
            if hasattr(cube, method_name):
                try:
                    getattr(cube, method_name)(np.zeros(3))
                except Exception:
                    pass

    def _physical_proxy_path(self, hand: str) -> "str | None":
        if self._avatar is not None:
            getter = getattr(self._avatar, "get_physical_hand_proxy_path", None)
            if getter is not None:
                try:
                    path = getter(hand)
                    if path:
                        return str(path)
                except Exception:
                    pass
        return f"/World/Avatar/{hand}_physical_proxy"

    def _cube_prim_path(self, cube) -> "str | None":
        if cube is None:
            return None
        prim = getattr(cube, "prim", None)
        if prim is not None:
            try:
                path = prim.GetPath()
                if path:
                    return str(path)
            except Exception:
                pass
        for attr_name in ("prim_path", "_prim_path"):
            path = getattr(cube, attr_name, None)
            if path:
                return str(path)
        return None

    def _same_cube(self, lhs, rhs) -> bool:
        if lhs is None or rhs is None:
            return False
        if lhs is rhs:
            return True
        lhs_path = self._cube_prim_path(lhs)
        rhs_path = self._cube_prim_path(rhs)
        return bool(lhs_path and rhs_path and lhs_path == rhs_path)

    def held_cube(self, hand: str):
        if hand not in self.grabbed:
            return None
        return self.grabbed[hand]

    def holding_hand_for(self, cube) -> "str | None":
        for hand, grabbed_cube in self.grabbed.items():
            if self._same_cube(grabbed_cube, cube):
                return hand
        return None

    def is_holding(self, cube=None) -> bool:
        if cube is None:
            return any(grabbed_cube is not None for grabbed_cube in self.grabbed.values())
        return self.holding_hand_for(cube) is not None

    def force_release(self, hand: "str | None" = None, cube=None) -> None:
        if hand is not None:
            self._release(hand)
            self.prev_grip[hand] = False
            return
        if cube is not None:
            holding_hand = self.holding_hand_for(cube)
            if holding_hand is not None:
                self._release(holding_hand)
                self.prev_grip[holding_hand] = False
            return
        self.release_all()

    def release_all(self):
        for hand in ("left", "right"):
            self._release(hand)
            self.prev_grip[hand] = False

    def set_grab_suppressed(
        self,
        hand: "str | None" = None,
        cube=None,
        suppressed: bool = True,
    ) -> bool:
        if cube is None:
            return False
        hands = (hand,) if hand in ("left", "right") else ("left", "right")
        cube_key = id(cube)
        changed = False
        for target_hand in hands:
            suppressed_set = self.grab_suppressed[target_hand]
            if suppressed:
                if cube_key not in suppressed_set:
                    suppressed_set.add(cube_key)
                    changed = True
            elif cube_key in suppressed_set:
                suppressed_set.discard(cube_key)
                changed = True
        return changed

    def _grab_is_suppressed(self, hand: str, cube) -> bool:
        if hand not in self.grab_suppressed or cube is None:
            return False
        return id(cube) in self.grab_suppressed[hand]
