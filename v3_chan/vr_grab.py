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

from vr_avatar import XR_SWAP_HANDS, xr_path_for_hand

GRAB_RADIUS = 0.35  # 35 cm 이내 블록 집기 (VR 물리-시뮬 스케일 차이 보정)
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
STICKY_PROXIMITY_GRAB = os.environ.get("STICKY_PROXIMITY_GRAB", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
STICKY_PROXIMITY_RADIUS = float(os.environ.get("STICKY_PROXIMITY_RADIUS", "0.12"))


class VRGrabManager:
    def __init__(self, cubes, avatar=None):
        self.cubes       = cubes
        self.grabbed     = {"left": None, "right": None}
        self.grab_offset = {"left": None, "right": None}
        self.prev_grip   = {"left": False, "right": False}

        self._avatar     = avatar   # VRAvatar 인스턴스 (XRCore 재사용)
        self._xr_core    = None
        self._xr_tried   = False
        self._inspected  = False    # device 메서드 목록 1회 출력 플래그
        self._gesture_inspected = False
        self._inputs_inspected = False
        self._input_gestures_inspected = False
        self._no_gesture_logged = False
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
            f"sticky-proximity={STICKY_PROXIMITY_GRAB} radius={STICKY_PROXIMITY_RADIUS:.3f}m"
        )

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
                        "touch",
                        "press",
                        "activate",
                        "x",
                        "y",
                    )
                    for input_name in tuple(dict.fromkeys([*input_priority, *input_names])):
                        if input_names and input_name not in input_names:
                            continue
                        gestures = input_gestures.get(input_name, [])
                        if not gestures:
                            continue
                        for gesture in tuple(dict.fromkeys([*gesture_priority, *gestures])):
                            if gesture not in gestures:
                                continue
                            try:
                                val = device.get_input_gesture_value(input_name, gesture)
                                if val is not None:
                                    v = float(val)
                                    if v > 0.01:
                                        print(f"[VRGrab] {hand} input({input_name}:{gesture})={v:.3f}")
                                        if gesture in ("click", "press", "activate"):
                                            return v > 0.5
                                        return v > threshold
                            except Exception:
                                continue

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

    # ── 매 프레임 호출 ─────────────────────────────────────────────────────────
    def update(self, left_pos, right_pos, pinch_points=None):
        self._frame += 1
        pinch_points = pinch_points or {}
        for hand, pos in (("left", left_pos), ("right", right_pos)):
            if pos is None:
                if STICKY_PROXIMITY_GRAB and self.grabbed[hand] is not None:
                    self._release(hand)
                    self.prev_grip[hand] = False
                continue

            pinch = self._read_pinch(hand, pinch_points.get(hand))
            if pinch is not None:
                grip, grab_pos = pinch
            else:
                grip = self._read_grip(hand)
                grab_pos = pos
                if not grip:
                    pose_pinch = self._read_pose_pinch(hand, pos)
                    if pose_pinch is not None:
                        grip, grab_pos = pose_pinch
                if not grip:
                    sticky = self._read_sticky_proximity(hand, pos)
                    if sticky is not None:
                        grip, grab_pos = sticky
                elif self._avatar is not None:
                    get_pose = getattr(self._avatar, "get_hand_pose_pos", None)
                    if get_pose is not None:
                        pinch_pos = get_pose(hand, "pinch")
                        if pinch_pos is not None:
                            grab_pos = pinch_pos

            if grab_pos is None:
                continue

            prev = self.prev_grip[hand]

            if grip and not prev:
                self._try_grab(hand, grab_pos)
            elif not grip and prev:
                self._release(hand)

            if grip and self.grabbed[hand] is not None:
                self.grabbed[hand].set_world_pose(
                    position=grab_pos + self.grab_offset[hand]
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

    def _read_sticky_proximity(self, hand: str, hand_pos: np.ndarray) -> "tuple[bool, np.ndarray] | None":
        if not STICKY_PROXIMITY_GRAB or hand_pos is None:
            return None
        if self.grabbed[hand] is not None:
            return True, hand_pos

        nearest, nearest_dist = self._nearest_cube(hand_pos)
        if nearest is None or nearest_dist > STICKY_PROXIMITY_RADIUS:
            return None
        return True, hand_pos

    # ── 근접 집기 ──────────────────────────────────────────────────────────────
    def _try_grab(self, hand: str, controller_pos: np.ndarray):
        if self.grabbed[hand] is not None:
            return
        nearest, nearest_dist = self._nearest_cube(controller_pos, max_dist=GRAB_RADIUS)

        if nearest is not None:
            cube_pos, _ = nearest.get_world_pose()
            self._attach(hand, nearest, cube_pos, controller_pos)
            print(f"[VRGrab] {hand} grabbed '{nearest.name}' (dist={nearest_dist:.3f} m)")
        else:
            print(f"[VRGrab] {hand} grip pressed — no cube within {GRAB_RADIUS} m")

    def _nearest_cube(self, position: np.ndarray, max_dist: float = GRAB_RADIUS):
        nearest, nearest_dist = None, max_dist
        for cube in self.cubes:
            if cube is self.grabbed["left"] or cube is self.grabbed["right"]:
                continue
            cube_pos, _ = cube.get_world_pose()
            dist = float(np.linalg.norm(position - cube_pos))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = cube
        return nearest, nearest_dist

    # ── 부착 ───────────────────────────────────────────────────────────────────
    def _attach(self, hand: str, cube, cube_pos, controller_pos):
        if hasattr(cube, "set_linear_velocity"):
            cube.set_linear_velocity(np.zeros(3))
        if hasattr(cube, "set_angular_velocity"):
            cube.set_angular_velocity(np.zeros(3))
        if hasattr(cube, "set_default_state"):
            cube.set_default_state(
                position=cube_pos,
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
        cube.disable_rigid_body_physics()
        self.grabbed[hand]     = cube
        self.grab_offset[hand] = cube_pos - controller_pos

    # ── 놓기 ───────────────────────────────────────────────────────────────────
    def _release(self, hand: str):
        cube = self.grabbed[hand]
        if cube is not None:
            if hasattr(cube, "set_linear_velocity"):
                cube.set_linear_velocity(np.zeros(3))
            if hasattr(cube, "set_angular_velocity"):
                cube.set_angular_velocity(np.zeros(3))
            cube.enable_rigid_body_physics()
            print(f"[VRGrab] {hand} released '{cube.name}'")
            self.grabbed[hand]     = None
            self.grab_offset[hand] = None

    def release_all(self):
        for hand in ("left", "right"):
            self._release(hand)
