import os

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux


def _env_vec3(name: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return np.array(default, dtype=float)
    try:
        parts = [float(p.strip()) for p in raw.split(",")]
        if len(parts) != 3:
            raise ValueError
        return np.array(parts, dtype=float)
    except Exception:
        print(f"[GripperCamera] Invalid {name}='{raw}', expected 'x,y,z'. Using {default}.")
        return np.array(default, dtype=float)


def _parse_resolution(value: str) -> tuple[int, int]:
    try:
        parts = [int(p.strip()) for p in value.split(",")]
        if len(parts) != 2:
            raise ValueError
        return max(1, parts[0]), max(1, parts[1])
    except Exception:
        print(f"[GripperCamera] Invalid resolution='{value}', expected 'width,height'. Using 640,480.")
        return 640, 480


MOUNT_OFFSET = _env_vec3("GRIPPER_CAMERA_MOUNT_OFFSET", (0.0, 0.0, 0.08))
DEFAULT_LOOK_DIR = _env_vec3("GRIPPER_CAMERA_DEFAULT_LOOK_DIR", (0.25, 0.0, -0.15))
UP_VECTOR = _env_vec3("GRIPPER_CAMERA_UP", (0.0, 0.0, 1.0))
FOCAL_LENGTH = float(os.environ.get("GRIPPER_CAMERA_FOCAL_LENGTH", "28.0"))
ENABLE_CAMERA_LIGHT = os.environ.get("ENABLE_GRIPPER_CAMERA_LIGHT", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
BLOCKING_RECORD_CAPTURE = os.environ.get(
    "GRIPPER_CAMERA_BLOCKING_RECORD_CAPTURE", "0"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)


class GripperCamera:
    """
    Lightweight USD camera that follows the Franka end-effector.

    This creates the camera transform only. It does not allocate a render product yet,
    so it is cheap enough to use as a basis for later occlusion/visibility checks.
    """

    def __init__(
        self,
        prim_path: str = "/World/GripperCamera",
        enabled: bool = True,
        show_viewport: bool = False,
        record_enabled: bool = False,
        record_dir: str = "",
        record_resolution: str = "640,480",
        record_interval_steps: int = 5,
    ):
        self.prim_path = prim_path
        self.enabled = enabled
        self.show_viewport = show_viewport
        self.record_enabled = record_enabled
        self.record_dir = record_dir
        self.record_resolution = _parse_resolution(record_resolution)
        self.record_interval_steps = max(1, int(record_interval_steps))
        self._camera = None
        self._xformable = None
        self._xform_op = None
        self._light = None
        self._light_xformable = None
        self._light_xform_op = None
        self._viewport_window = None
        self._rep = None
        self._render_product = None
        self._writer = None
        self._record_frame = 0
        self._update_count = 0
        self._viewport_camera_synced = False
        self.position: "np.ndarray | None" = None
        self.target: "np.ndarray | None" = None

    def setup(self) -> None:
        if not self.enabled:
            print("[GripperCamera] disabled")
            return

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        self._camera = UsdGeom.Camera.Define(stage, self.prim_path)
        self._camera.CreateFocalLengthAttr(FOCAL_LENGTH)
        self._camera.CreateHorizontalApertureAttr(20.955)
        self._camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
        self._xformable = UsdGeom.Xformable(self._camera.GetPrim())
        self._xformable.ClearXformOpOrder()
        self._xform_op = self._xformable.AddTransformOp()
        print(f"[GripperCamera] created: {self.prim_path}")
        if ENABLE_CAMERA_LIGHT:
            self._create_camera_light(stage)
        if self.show_viewport:
            self._create_viewport_window()
        if self.record_enabled:
            self._setup_recording()

    def _create_viewport_window(self) -> None:
        try:
            from omni.kit.viewport.utility import create_viewport_window

            self._viewport_window = create_viewport_window(
                "Gripper Camera",
                width=480,
                height=360,
                camera_path=Sdf.Path(self.prim_path),
            )
            self._sync_viewport_camera(force=True)
            print(f"[GripperCamera] viewport window opened: camera={self.prim_path}")
        except Exception as exc:
            self._viewport_window = None
            print(f"[GripperCamera] viewport window unavailable: {exc}")

    def _sync_viewport_camera(self, force: bool = False) -> None:
        if self._viewport_window is None:
            return
        viewport_api = getattr(self._viewport_window, "viewport_api", None)
        if viewport_api is None:
            return
        try:
            current_path = str(getattr(viewport_api, "camera_path", ""))
            if force or current_path != self.prim_path:
                viewport_api.camera_path = Sdf.Path(self.prim_path)
                self._viewport_camera_synced = True
        except Exception as exc:
            if force:
                print(f"[GripperCamera] viewport camera sync failed: {exc}")

    def _create_camera_light(self, stage) -> None:
        try:
            light_path = f"{self.prim_path}_Light"
            self._light = UsdLux.SphereLight.Define(stage, light_path)
            self._light.CreateIntensityAttr(25000.0)
            self._light.CreateRadiusAttr(0.08)
            self._light_xformable = UsdGeom.Xformable(self._light.GetPrim())
            self._light_xformable.ClearXformOpOrder()
            self._light_xform_op = self._light_xformable.AddTransformOp()
            print(f"[GripperCamera] light created: {light_path}")
        except Exception as exc:
            self._light = None
            self._light_xformable = None
            self._light_xform_op = None
            print(f"[GripperCamera] light unavailable: {exc}")

    def _setup_recording(self) -> None:
        try:
            import omni.replicator.core as rep

            os.makedirs(self.record_dir, exist_ok=True)
            self._rep = rep
            self._render_product = rep.create.render_product(
                self.prim_path,
                self.record_resolution,
            )
            self._writer = rep.WriterRegistry.get("BasicWriter")
            self._writer.initialize(output_dir=self.record_dir, rgb=True)
            self._writer.attach([self._render_product])
            try:
                rep.orchestrator.set_capture_on_play(False)
            except Exception:
                pass
            print(
                "[GripperCamera] recording enabled: "
                f"dir={self.record_dir} resolution={self.record_resolution[0]}x{self.record_resolution[1]}"
            )
            if not BLOCKING_RECORD_CAPTURE:
                print(
                    "[GripperCamera] blocking frame capture disabled "
                    "(set GRIPPER_CAMERA_BLOCKING_RECORD_CAPTURE=1 only for recording tests)"
                )
        except Exception as exc:
            self._rep = None
            self._render_product = None
            self._writer = None
            print(f"[GripperCamera] recording unavailable: {exc}")

    def update(
        self,
        ee_pos: np.ndarray,
        target_pos: "np.ndarray | None" = None,
        mount_pos: "np.ndarray | None" = None,
    ) -> None:
        if not self.enabled or self._xform_op is None:
            return

        if mount_pos is None:
            mount_pos = ee_pos
        mount_pos = np.asarray(mount_pos, dtype=float)
        camera_pos = mount_pos + MOUNT_OFFSET
        if target_pos is None:
            target_pos = camera_pos + DEFAULT_LOOK_DIR
        else:
            target_pos = np.asarray(target_pos, dtype=float)

        if np.linalg.norm(target_pos - camera_pos) < 1e-6:
            target_pos = camera_pos + DEFAULT_LOOK_DIR

        up = UP_VECTOR
        if np.linalg.norm(up) < 1e-6:
            up = np.array([0.0, 0.0, 1.0])
        up = up / np.linalg.norm(up)

        try:
            pose = Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
                Gf.Vec3d(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
                Gf.Vec3d(float(up[0]), float(up[1]), float(up[2])),
            ).GetInverse()
            self._xform_op.Set(pose)
            if self._light_xform_op is not None:
                light_pose = Gf.Matrix4d().SetTranslate(
                    Gf.Vec3d(
                        float(camera_pos[0]),
                        float(camera_pos[1]),
                        float(camera_pos[2]),
                    )
                )
                self._light_xform_op.Set(light_pose)
            self._sync_viewport_camera()
            self.position = camera_pos
            self.target = target_pos
            if BLOCKING_RECORD_CAPTURE:
                self._capture_frame()
        except Exception as exc:
            print(f"[GripperCamera] update failed: {exc}")

    def _capture_frame(self) -> None:
        if self._rep is None or self._writer is None:
            return
        self._update_count += 1
        if self._update_count % self.record_interval_steps != 0:
            return
        try:
            self._rep.orchestrator.step()
            self._record_frame += 1
            if self._record_frame == 1:
                print(f"[GripperCamera] first frame capture requested: {self.record_dir}")
        except Exception as exc:
            if self._record_frame == 0:
                print(f"[GripperCamera] frame capture failed: {exc}")

    def close(self) -> None:
        if self._writer is not None and self._render_product is not None:
            try:
                self._writer.detach()
            except Exception:
                pass
        self._writer = None
        self._render_product = None
