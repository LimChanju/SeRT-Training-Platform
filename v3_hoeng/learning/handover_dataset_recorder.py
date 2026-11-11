from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "handover_dataset_v0_npz"
DEFAULT_DATASET_DIR = "data/trajectories/handover_dataset_v0"
GRASP_METRIC_NAMES = (
    "center_dist",
    "min_dist",
    "width",
    "jaw_offset",
    "vertical_offset",
    "depth_offset",
    "head_dist",
)

_OBS_MODULE = None
_OBS_LOAD_ATTEMPTED = False
_OBS_LOAD_ERROR: str | None = None


def _load_observation_module():
    """Load rl/observations.py without importing the broken rl package."""

    global _OBS_MODULE, _OBS_LOAD_ATTEMPTED, _OBS_LOAD_ERROR
    if _OBS_LOAD_ATTEMPTED:
        return _OBS_MODULE
    _OBS_LOAD_ATTEMPTED = True
    obs_path = Path(__file__).resolve().parents[1] / "rl" / "observations.py"
    try:
        spec = importlib.util.spec_from_file_location("_handover_rl_observations", obs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {obs_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _OBS_MODULE = module
    except Exception as exc:
        _OBS_LOAD_ERROR = str(exc)
        _OBS_MODULE = None
    return _OBS_MODULE


def observation_version() -> str:
    module = _load_observation_module()
    if module is None:
        return "obs_unavailable"
    return str(getattr(module, "OBSERVATION_VERSION", "obs_unknown"))


def observation_dim() -> int:
    module = _load_observation_module()
    if module is None:
        return 0
    return int(getattr(module, "OBSERVATION_DIM", 0))


def build_flat_observation(**kwargs) -> np.ndarray:
    """Build a flattened observation using rl/observations.py when available."""

    module = _load_observation_module()
    if module is None:
        return np.zeros(0, dtype=np.float32)
    try:
        obs = module.build_observation(**kwargs)
        return module.flatten_observation(obs).astype(np.float32, copy=False)
    except Exception:
        dim = int(getattr(module, "OBSERVATION_DIM", 0))
        return np.zeros(dim, dtype=np.float32)


def observation_load_error() -> str | None:
    _load_observation_module()
    return _OBS_LOAD_ERROR


def task_phase_from_sample(sample: dict[str, Any] | None) -> str:
    if not sample:
        return "approach_cube"
    stage = str(sample.get("oriented_stage") or "").strip().lower()
    carry_stage = str(sample.get("physical_carry_stage") or "").strip().lower()
    if stage in ("orient_close", "orient_attach", "orient_verify"):
        return "grasp_cube"
    if stage == "orient_lift_place" or carry_stage in ("lift", "carry", "descend"):
        return "move_to_target"
    if carry_stage == "release":
        return "release_cube"
    return "approach_cube"


def safe_joint_positions(robot, size: int = 9) -> np.ndarray:
    try:
        values = robot.get_joint_positions()
    except Exception:
        values = None
    return _fixed_array(values, size, fill=np.nan).astype(np.float32)


def safe_vec3(value, fill: float = np.nan) -> np.ndarray:
    return _fixed_array(value, 3, fill=fill).astype(np.float32)


def _fixed_array(value, size: int, fill: float = np.nan) -> np.ndarray:
    result = np.full(size, fill, dtype=np.float32)
    if value is None:
        return result
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return result
    n = min(size, arr.size)
    if n > 0:
        result[:n] = arr[:n]
    return result


class HandoverDatasetRecorder:
    """Append-only NPZ + JSONL episode recorder for handover expert data."""

    def __init__(
        self,
        dataset_dir: str | Path = DEFAULT_DATASET_DIR,
        *,
        overwrite: bool = False,
        record_failed: bool = True,
        min_steps: int = 20,
        record_images: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        if overwrite and self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dataset_dir / "manifest.jsonl"
        self.record_failed = bool(record_failed)
        self.min_steps = max(1, int(min_steps))
        self.record_images = bool(record_images)
        self._episode_id = self._next_episode_id()
        self._active = False
        self._instruction = ""
        self._metadata: dict[str, Any] = {}
        self._buffers: dict[str, list[Any]] = {}
        self._warned_image_paths = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def instruction(self) -> str:
        return self._instruction

    def start_episode(
        self,
        *,
        instruction: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._active:
            return
        self._active = True
        self._instruction = str(instruction)
        self._metadata = dict(metadata or {})
        self._buffers = self._empty_buffers()
        print(
            "[Dataset] handover recording started "
            f"episode={self._episode_id:06d} instruction={self._instruction!r}",
            flush=True,
        )

    def add_step(
        self,
        *,
        obs_state,
        sample: dict[str, Any],
        joint_pos,
        cube_pos,
        place_pos,
        left_hand_pos,
        right_hand_pos,
        step: int,
        sim_time: float,
        image_path: str = "",
    ) -> None:
        if not self._active:
            return
        if not bool(sample.get("has_action", False)):
            return

        if self.record_images and not image_path and not self._warned_image_paths:
            print(
                "[Dataset] HANDOVER_DATASET_RECORD_IMAGES=1 but no synchronized image path "
                "is available yet; storing empty image_path values.",
                flush=True,
            )
            self._warned_image_paths = True

        metrics = sample.get("grasp_metrics") or {}
        action = sample.get("action")
        if action is None:
            action = np.zeros(8, dtype=np.float32)

        self._buffers["obs_state"].append(np.asarray(obs_state, dtype=np.float32).reshape(-1))
        self._buffers["action"].append(_fixed_array(action, 8, fill=0.0))
        self._buffers["target_ee_pos"].append(_fixed_array(sample.get("target_ee_pos"), 3))
        self._buffers["target_ee_quat"].append(_fixed_array(sample.get("target_ee_quat"), 4))
        self._buffers["joint_pos"].append(_fixed_array(joint_pos, 9))
        self._buffers["joint_action"].append(_fixed_array(sample.get("joint_action"), 9))
        self._buffers["cube_pos"].append(_fixed_array(cube_pos, 3))
        self._buffers["place_pos"].append(_fixed_array(place_pos, 3))
        self._buffers["left_hand_pos"].append(_fixed_array(left_hand_pos, 3))
        self._buffers["right_hand_pos"].append(_fixed_array(right_hand_pos, 3))
        self._buffers["phase"].append(str(sample.get("phase") or ""))
        self._buffers["oriented_stage"].append(str(sample.get("oriented_stage") or ""))
        self._buffers["physical_carry_stage"].append(
            str(sample.get("physical_carry_stage") or "")
        )
        self._buffers["active_hand"].append(str(sample.get("active_hand") or ""))
        self._buffers["active_cube"].append(str(sample.get("active_cube") or ""))
        self._buffers["robot_has_taken_over"].append(bool(sample.get("robot_has_taken_over", False)))
        self._buffers["step"].append(int(step))
        self._buffers["sim_time"].append(float(sim_time))
        self._buffers["image_path"].append(str(image_path or ""))
        for name in GRASP_METRIC_NAMES:
            self._buffers[f"grasp_metrics_{name}"].append(float(metrics.get(name, np.nan)))

    def end_episode(
        self,
        *,
        success: bool,
        failure_reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self._active:
            return None

        length = len(self._buffers.get("action", []))
        success = bool(success)
        failure_reason = "" if success else str(failure_reason or "unknown_failure")
        final_metadata = {**self._metadata, **dict(metadata or {})}

        should_save = length >= self.min_steps and (success or self.record_failed)
        if not should_save:
            print(
                "[Dataset] discarded handover episode "
                f"success={success} length={length} min_steps={self.min_steps} "
                f"reason={failure_reason or 'too_short'}",
                flush=True,
            )
            self._clear_active()
            return None

        episode_path = self.dataset_dir / f"episode_{self._episode_id:06d}.npz"
        arrays = self._arrays_for_save(success=success, failure_reason=failure_reason)
        np.savez_compressed(episode_path, **arrays)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "observation_version": observation_version(),
            "episode_id": self._episode_id,
            "episode_path": str(episode_path.relative_to(self.dataset_dir)),
            "length": length,
            "success": success,
            "failure_reason": failure_reason,
            "instruction": self._instruction,
            "active_hand": self._first_nonempty("active_hand"),
            "active_cube": self._first_nonempty("active_cube"),
            "record_images": self.record_images,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **final_metadata,
        }
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(manifest, ensure_ascii=False) + "\n")

        print(
            "[Dataset] saved handover episode "
            f"{episode_path} length={length} success={success}",
            flush=True,
        )
        self._episode_id += 1
        self._clear_active()
        return manifest

    def abort_episode(self, reason: str = "aborted") -> None:
        if not self._active:
            return
        print(f"[Dataset] aborted handover episode reason={reason}", flush=True)
        self._clear_active()

    def _empty_buffers(self) -> dict[str, list[Any]]:
        buffers = {
            "obs_state": [],
            "action": [],
            "target_ee_pos": [],
            "target_ee_quat": [],
            "joint_pos": [],
            "joint_action": [],
            "cube_pos": [],
            "place_pos": [],
            "left_hand_pos": [],
            "right_hand_pos": [],
            "phase": [],
            "oriented_stage": [],
            "physical_carry_stage": [],
            "active_hand": [],
            "active_cube": [],
            "robot_has_taken_over": [],
            "step": [],
            "sim_time": [],
            "image_path": [],
        }
        for name in GRASP_METRIC_NAMES:
            buffers[f"grasp_metrics_{name}"] = []
        return buffers

    def _arrays_for_save(self, *, success: bool, failure_reason: str) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for key, values in self._buffers.items():
            if key in (
                "phase",
                "oriented_stage",
                "physical_carry_stage",
                "active_hand",
                "active_cube",
                "image_path",
            ):
                arrays[key] = np.asarray(values, dtype=str)
            elif key == "robot_has_taken_over":
                arrays[key] = np.asarray(values, dtype=bool)
            elif key == "step":
                arrays[key] = np.asarray(values, dtype=np.int64)
            else:
                arrays[key] = np.asarray(values, dtype=np.float32)
        arrays["success"] = np.asarray(success, dtype=bool)
        arrays["failure_reason"] = np.asarray(failure_reason, dtype=str)
        arrays["instruction"] = np.asarray(self._instruction, dtype=str)
        arrays["schema_version"] = np.asarray(SCHEMA_VERSION, dtype=str)
        arrays["observation_version"] = np.asarray(observation_version(), dtype=str)
        arrays["grasp_metrics_names"] = np.asarray(GRASP_METRIC_NAMES, dtype=str)
        return arrays

    def _first_nonempty(self, key: str) -> str:
        for value in self._buffers.get(key, []):
            text = str(value)
            if text:
                return text
        value = self._metadata.get(key, "")
        return str(value or "")

    def _clear_active(self) -> None:
        self._active = False
        self._instruction = ""
        self._metadata = {}
        self._buffers = {}

    def _next_episode_id(self) -> int:
        max_id = 0
        for path in self.dataset_dir.glob("episode_*.npz"):
            try:
                max_id = max(max_id, int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return max_id + 1
