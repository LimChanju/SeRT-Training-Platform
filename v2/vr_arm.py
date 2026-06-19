# vr_arm.py — 가상 팔 (HMD→컨트롤러) 실린더 시각화 + 캘리브레이션 + 로봇 충돌 감지

import numpy as np
from pxr import Gf

ARM_RADIUS         = 0.03   # 실린더 반경 3 cm
ARM_COLLISION_DIST = 0.08   # 로봇 링크 중심까지 이 거리 미만이면 충돌
ARM_MAX_LENGTH     = 1.2    # 이 길이 초과 시 데이터 이상으로 간주, 그리지 않음
ARM_MIN_LENGTH     = 0.10   # 너무 짧으면 컨트롤러 미연결로 간주, 그리지 않음
SHOULDER_Z_DROP    = 0.25   # HMD에서 어깨까지 아래로 내리는 거리 (m)

ROBOT_LINK_PATHS = [
    "/World/Franka/panda_link1",
    "/World/Franka/panda_link2",
    "/World/Franka/panda_link3",
    "/World/Franka/panda_link4",
    "/World/Franka/panda_link5",
    "/World/Franka/panda_link6",
    "/World/Franka/panda_link7",
    "/World/Franka/panda_hand",
    "/World/Franka/panda_leftfinger",
    "/World/Franka/panda_rightfinger",
]


def _seg_dist(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """점 point에서 선분 a-b까지의 최단거리 (캡슐 충돌 감지용)."""
    ab = b - a
    sq = float(np.dot(ab, ab))
    if sq < 1e-10:
        return float(np.linalg.norm(point - a))
    t = np.clip(float(np.dot(point - a, ab)) / sq, 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * ab)))


def _quat_z_to(direction: np.ndarray) -> np.ndarray:
    """Isaac Sim Z-up 기준: Z축을 direction으로 정렬하는 쿼터니언 [x,y,z,w]."""
    v = direction / np.linalg.norm(direction)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z, v), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    cross = np.cross(z, v)
    s = np.sqrt((1.0 + dot) * 2.0)
    return np.array([*(cross / s), s * 0.5])


COLOR_LEFT  = 0xFF88AAFF   # 파란색 (왼팔)
COLOR_RIGHT = 0xFFFF8844   # 주황색 (오른팔)


class VRArm:
    """
    양손 가상 팔(어깨→컨트롤러 선분)을 DebugDraw 선분으로 시각화하고,
    Franka 링크와의 근접 충돌을 감지합니다.
    DebugDraw는 USD 프림이 아니므로 XR 내장 그랩에 잡히지 않습니다.
    """

    def __init__(self):
        self._xr_core      = None
        self._world_offset = np.zeros(3)
        self._debug_draw   = None
        self._robot_links  = None
        self.calibrated    = False

    def _init_debug_draw(self):
        try:
            from omni.debugdraw import get_debug_draw_interface
            self._debug_draw = get_debug_draw_interface()
        except Exception as e:
            print(f"[VRArm] DebugDraw not available: {e}")

    # ── XRCore ─────────────────────────────────────────────────────────────
    def _init_xr(self):
        try:
            from omni.kit.xr.core import XRCore
            self._xr_core = XRCore.get_singleton()
        except Exception as e:
            print(f"[VRArm] XRCore: {e}")

    def _raw_pos(self, xr_path: str) -> "np.ndarray | None":
        if self._xr_core is None:
            self._init_xr()
        if self._xr_core is None:
            return None
        try:
            device = self._xr_core.get_input_device(xr_path)
            if device is None:
                return None
            mat = device.get_virtual_world_pose()
            pos = np.array([mat[3][0], mat[3][1], mat[3][2]], dtype=float)
            return pos if np.all(np.isfinite(pos)) else None
        except Exception:
            return None

    # ── 보정된 위치 읽기 ───────────────────────────────────────────────────
    def get_controller_pos(self, hand: str) -> "np.ndarray | None":
        xr = "/user/hand/left" if hand == "left" else "/user/hand/right"
        raw = self._raw_pos(xr)
        return raw + self._world_offset if raw is not None else None

    def get_hmd_pos(self) -> "np.ndarray | None":
        raw = self._raw_pos("/user/head")
        return raw + self._world_offset if raw is not None else None

    def get_shoulder_pos(self) -> "np.ndarray | None":
        """HMD 위치에서 SHOULDER_Z_DROP만큼 내린 어깨 위치 (실린더 시작점)."""
        hmd = self.get_hmd_pos()
        if hmd is None:
            return None
        shoulder = hmd.copy()
        shoulder[2] -= SHOULDER_Z_DROP
        return shoulder

    # ── 캘리브레이션 ────────────────────────────────────────────────────────
    def calibrate(self, hand: str, reference_pos: np.ndarray) -> bool:
        """
        컨트롤러를 시뮬 좌표 reference_pos 위치에 갖다 댄 상태에서 호출.
        UDP 커맨드 {"calibrate":"right","ref":[x,y,z]} 로 트리거 가능.
        """
        xr = "/user/hand/left" if hand == "left" else "/user/hand/right"
        raw = self._raw_pos(xr)
        if raw is None:
            print(f"[VRArm] calibrate: {hand} controller not detected")
            return False
        self._world_offset = reference_pos - raw
        self.calibrated = True
        print(f"[VRArm] calibrated ({hand}): offset={np.round(self._world_offset, 3)}")
        return True

    # ── DebugDraw 팔 시각화 ────────────────────────────────────────────────
    def update_cylinder(self, hand: str, controller_pos: "np.ndarray | None", head_pos: "np.ndarray | None"):
        """매 프레임: 어깨→컨트롤러 선분을 DebugDraw로 그림.
        USD 프림이 아니므로 XR 내장 그랩에 절대 잡히지 않음."""
        if controller_pos is None or head_pos is None:
            return
        arm_vec = controller_pos - head_pos
        length  = float(np.linalg.norm(arm_vec))
        if length < ARM_MIN_LENGTH or length > ARM_MAX_LENGTH:
            return

        if self._debug_draw is None:
            self._init_debug_draw()
        if self._debug_draw is None:
            return

        color = COLOR_LEFT if hand == "left" else COLOR_RIGHT
        o = Gf.Vec3f(float(head_pos[0]),       float(head_pos[1]),       float(head_pos[2]))
        e = Gf.Vec3f(float(controller_pos[0]),  float(controller_pos[1]), float(controller_pos[2]))
        try:
            self._debug_draw.draw_line(o, color, e, color)
        except Exception:
            pass

    # ── 로봇 충돌 감지 ────────────────────────────────────────────────────
    def _ensure_links(self):
        if self._robot_links is not None:
            return
        import omni.usd
        from omni.isaac.core.prims import XFormPrim
        stage = omni.usd.get_context().get_stage()
        self._robot_links = [
            XFormPrim(prim_path=p)
            for p in ROBOT_LINK_PATHS
            if stage.GetPrimAtPath(p).IsValid()
        ]
        print(f"[VRArm] {len(self._robot_links)}/{len(ROBOT_LINK_PATHS)} robot links loaded.")

    def check_robot_collision(
        self,
        controller_pos: "np.ndarray | None",
        head_pos: "np.ndarray | None",
    ) -> "list[str]":
        """
        가상 팔(head→controller) 선분에서 각 로봇 링크까지의 최단거리를 계산.
        ARM_COLLISION_DIST 미만이면 충돌로 판정, 해당 prim_path 반환.
        """
        if controller_pos is None or head_pos is None:
            return []
        self._ensure_links()
        hits = []
        for link in (self._robot_links or []):
            try:
                link_pos, _ = link.get_world_pose()
            except Exception:
                continue
            if _seg_dist(link_pos, head_pos, controller_pos) < ARM_COLLISION_DIST:
                hits.append(link.prim_path)
        return hits
