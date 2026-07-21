import importlib.util
import os
import sys
import time
from typing import Mapping

import numpy as np


_OBS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl", "observations.py")
_OBS_SPEC = importlib.util.spec_from_file_location("_v3_chan_observations", _OBS_PATH)
if _OBS_SPEC is None or _OBS_SPEC.loader is None:
    raise ImportError(f"Could not load observation schema from {_OBS_PATH}")
_obs = importlib.util.module_from_spec(_OBS_SPEC)
sys.modules[_OBS_SPEC.name] = _obs
_OBS_SPEC.loader.exec_module(_obs)

OBSERVATION_FIELDS = _obs.OBSERVATION_FIELDS
AUXILIARY_OBSERVATION_FIELDS = _obs.AUXILIARY_OBSERVATION_FIELDS
RECORDED_OBSERVATION_FIELDS = _obs.RECORDED_OBSERVATION_FIELDS
OBSERVATION_DIM = _obs.OBSERVATION_DIM
OBSERVATION_VERSION = _obs.OBSERVATION_VERSION
build_observation = _obs.build_observation
flatten_observation = _obs.flatten_observation
validate_observation = _obs.validate_observation
validate_auxiliary_observation = _obs.validate_auxiliary_observation

HRI_OBS_FIELD_NAMES = (
    "robot_joint_pos",
    "robot_joint_vel",
    "gripper_width",
    "ee_pos",
    "ee_quat",
    "cube_pos",
    "cube_quat",
    "place_target_pos",
    "ee_to_cube",
    "cube_to_place_target",
    "ee_to_place_target",
    "human_head_pos",
    "human_left_hand_pos",
    "human_right_hand_pos",
    "ee_to_left_hand",
    "ee_to_right_hand",
    "min_hand_gripper_dist",
    "min_hand_gripper_center_dist",
    "min_hand_gripper_surface_gap",
    "left_hand_end_effector_surface_gap",
    "right_hand_end_effector_surface_gap",
    "min_hand_end_effector_surface_gap",
    "human_robot_collision",
    "near_human",
    "near_miss",
    "left_hand_contact",
    "right_hand_contact",
    "distance_gate",
    "geometry_valid",
    "has_grasped_cube",
    "task_phase",
    "controller_event",
)
_RECORDED_FIELD_MAP = {field.name: field for field in RECORDED_OBSERVATION_FIELDS}
HRI_OBS_DIM = int(sum(_RECORDED_FIELD_MAP[name].dim for name in HRI_OBS_FIELD_NAMES))


def flatten_hri_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    validate_observation(obs)
    validate_auxiliary_observation(obs)
    values = [
        np.asarray(obs[name], dtype=np.float32).reshape(-1)
        for name in HRI_OBS_FIELD_NAMES
    ]
    return np.concatenate(values).astype(np.float32, copy=False)


class HRIObsRecorder:
    """Small HDF5 recorder for VR HRI sessions.

    Stores obs dictionaries and flattened obs_policy arrays that can be fed
    directly into the existing policy/training readers.
    """

    SCHEMA_VERSION = "hri_obs_v4_builtin_panda_collision_geometry"

    def __init__(
        self,
        path: str,
        *,
        overwrite: bool = False,
        sample_interval_steps: int = 1,
        compression: str | None = "gzip",
        file_metadata: Mapping | None = None,
    ) -> None:
        try:
            import h5py
        except Exception:
            h5py = None

        self._h5py = h5py
        self._use_hdf5 = h5py is not None
        self.path = os.path.abspath(path if self._use_hdf5 else f"{path}.npz")
        self.sample_interval_steps = max(1, int(sample_interval_steps))
        self.compression = compression
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._npz_payload = {}
        self._npz_episode_count = 0
        if self._use_hdf5:
            mode = "w" if overwrite else "a"
            try:
                self._file = h5py.File(self.path, mode)
            except OSError as exc:
                if overwrite:
                    raise
                root, ext = os.path.splitext(self.path)
                fallback = f"{root}_recovered_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.hdf5'}"
                print(
                    "[HRI] requested trajectory file could not be opened; "
                    f"using fallback path. original={self.path} fallback={fallback} error={exc}"
                )
                self.path = fallback
                self._file = h5py.File(self.path, "w")
            existing_schema = str(self._file.attrs.get("schema_version", ""))
            existing_episodes = self._file.get("episodes")
            if (
                not overwrite
                and existing_schema
                and existing_schema != self.SCHEMA_VERSION
                and existing_episodes is not None
                and len(existing_episodes.keys()) > 0
            ):
                original_path = self.path
                self._file.close()
                root, ext = os.path.splitext(original_path)
                self.path = (
                    f"{root}_{self.SCHEMA_VERSION}_{time.strftime('%Y%m%d_%H%M%S')}"
                    f"{ext or '.hdf5'}"
                )
                print(
                    "[HRI] existing trajectory uses a different schema; "
                    f"preserving it and writing to {self.path}. "
                    f"existing={existing_schema} new={self.SCHEMA_VERSION}",
                    flush=True,
                )
                self._file = h5py.File(self.path, "w")
            self._episodes = self._file.require_group("episodes")
            self._file.attrs["schema_version"] = self.SCHEMA_VERSION
            self._file.attrs["observation_version"] = OBSERVATION_VERSION
            self._file.attrs["observation_dim"] = int(OBSERVATION_DIM)
            self._file.attrs["hri_observation_dim"] = int(HRI_OBS_DIM)
            self._file.attrs["hri_observation_fields"] = ",".join(HRI_OBS_FIELD_NAMES)
            self._file.attrs["sample_interval_steps"] = int(self.sample_interval_steps)
            for key, value in dict(file_metadata or {}).items():
                self._file.attrs[str(key)] = value
        else:
            if overwrite and os.path.exists(self.path):
                os.remove(self.path)
            self._file = None
            self._episodes = None
            self._npz_payload.update(
                {
                    "attrs/schema_version": np.asarray(self.SCHEMA_VERSION),
                    "attrs/observation_version": np.asarray(OBSERVATION_VERSION),
                    "attrs/observation_dim": np.asarray(int(OBSERVATION_DIM), dtype=np.int32),
                    "attrs/hri_observation_dim": np.asarray(int(HRI_OBS_DIM), dtype=np.int32),
                    "attrs/hri_observation_fields": np.asarray(",".join(HRI_OBS_FIELD_NAMES)),
                    "attrs/sample_interval_steps": np.asarray(
                        int(self.sample_interval_steps), dtype=np.int32
                    ),
                }
            )
            for key, value in dict(file_metadata or {}).items():
                self._npz_payload[f"attrs/{key}"] = np.asarray(value)
        self._open = False
        self._attrs = {}
        self._buffers = {}

    @property
    def num_episodes(self) -> int:
        if self._use_hdf5:
            return len(self._episodes.keys())
        return int(self._npz_episode_count)

    @property
    def is_open(self) -> bool:
        return self._open

    def start_episode(self, metadata: Mapping | None = None) -> None:
        if self._open:
            return
        self._open = True
        self._attrs = dict(metadata or {})
        self._buffers = {
            "sim_time": [],
            "step": [],
            "obs": {field.name: [] for field in RECORDED_OBSERVATION_FIELDS},
            "obs_policy": [],
            "hri_obs_policy": [],
            "human_valid_mask": [],
            "current_pick_idx": [],
            "completed_picks": [],
            "safety": {
                "left_hand_gripper_dist_m": [],
                "right_hand_gripper_dist_m": [],
                "min_hand_gripper_dist_m": [],
                "min_hand_gripper_center_dist_m": [],
                "min_hand_gripper_surface_gap_m": [],
                "near_human": [],
                "near_miss": [],
                "human_robot_collision": [],
                "haptic_pulse_left": [],
                "haptic_pulse_right": [],
                "gripper_gap_left_m": [],
                "gripper_gap_right_m": [],
                "gripper_penetration_left_m": [],
                "gripper_penetration_right_m": [],
                "left_hand_surface_gap_m": [],
                "right_hand_surface_gap_m": [],
                "min_hand_end_effector_surface_gap_m": [],
                "left_end_effector_surface_gap_m": [],
                "right_end_effector_surface_gap_m": [],
                "end_effector_surface_gap_m": [],
                "closest_human_hand_id": [],
                "closest_robot_link_id": [],
                "closest_collider_id": [],
                "closest_link_left_id": [],
                "closest_link_right_id": [],
                "closest_collider_left_id": [],
                "closest_collider_right_id": [],
                "contact_left": [],
                "contact_right": [],
                "contact_active": [],
                "contact_force_left_n": [],
                "contact_force_right_n": [],
                "contact_force_n": [],
                "contact_force_valid_left": [],
                "contact_force_valid_right": [],
                "penetration_left_m": [],
                "penetration_right_m": [],
                "penetration_depth_m": [],
                "near_left": [],
                "near_right": [],
                "near_miss_left": [],
                "near_miss_right": [],
                "distance_gate_left": [],
                "distance_gate_right": [],
                "distance_gate": [],
                "geometry_valid_left": [],
                "geometry_valid_right": [],
                "geometry_valid": [],
                "query_time_left_ms": [],
                "query_time_right_ms": [],
                "query_count_left": [],
                "query_count_right": [],
            },
            "errp": {
                "label": [],
                "feedback": [],
                "uncertainty": [],
                "timestamp": [],
                "aligned_step": [],
            },
            "actions": {
                "robot_joint_positions": [],
                "robot_joint_velocities": [],
                "robot_joint_efforts": [],
            },
            "rewards": {
                "task": [],
                "errp": [],
                "safety": [],
                "total": [],
            },
            "task": {
                "current_pick_idx": [],
                "completed_picks": [],
                "attempt_index": [],
                "current_cube_attempt": [],
                "failed_attempts": [],
                "has_grasped_cube": [],
                "controller_event": [],
            },
        }

    def add_sample(
        self,
        *,
        step: int,
        sim_time: float,
        obs: dict[str, np.ndarray],
        current_pick_idx: int = 0,
        completed_picks: int = 0,
        attempt_index: int = 1,
        current_cube_attempt: int = 1,
        failed_attempts: int = 0,
        safety: Mapping | None = None,
        errp: Mapping | None = None,
        action=None,
        reward: Mapping | None = None,
    ) -> None:
        if not self._open:
            return
        if int(step) % self.sample_interval_steps != 0:
            return
        validate_observation(obs)
        validate_auxiliary_observation(obs)
        self._buffers["sim_time"].append(float(sim_time))
        self._buffers["step"].append(int(step))
        for field in RECORDED_OBSERVATION_FIELDS:
            self._buffers["obs"][field.name].append(
                np.asarray(obs[field.name], dtype=np.float32)
            )
        self._buffers["obs_policy"].append(flatten_observation(obs))
        self._buffers["hri_obs_policy"].append(flatten_hri_observation(obs))
        valid = np.array(
            [
                float(np.linalg.norm(obs["human_head_pos"]) > 1e-6),
                float(np.linalg.norm(obs["human_left_hand_pos"]) > 1e-6),
                float(np.linalg.norm(obs["human_right_hand_pos"]) > 1e-6),
            ],
            dtype=np.float32,
        )
        self._buffers["human_valid_mask"].append(valid)
        self._buffers["current_pick_idx"].append(int(current_pick_idx))
        self._buffers["completed_picks"].append(int(completed_picks))
        self._append_safety(obs, safety)
        self._append_errp(errp, step)
        self._append_action(action)
        self._append_reward(obs, reward)
        self._buffers["task"]["current_pick_idx"].append(int(current_pick_idx))
        self._buffers["task"]["completed_picks"].append(int(completed_picks))
        self._buffers["task"]["attempt_index"].append(int(attempt_index))
        self._buffers["task"]["current_cube_attempt"].append(int(current_cube_attempt))
        self._buffers["task"]["failed_attempts"].append(int(failed_attempts))
        self._buffers["task"]["has_grasped_cube"].append(float(np.asarray(obs["has_grasped_cube"]).reshape(-1)[0]))
        event_onehot = np.asarray(obs["controller_event"], dtype=np.float32).reshape(-1)
        event_idx = int(np.argmax(event_onehot)) if np.any(event_onehot > 0.5) else -1
        self._buffers["task"]["controller_event"].append(event_idx)

    def end_episode(self, *, success: bool = False, metadata: Mapping | None = None) -> str | None:
        if not self._open:
            return None
        if not self._buffers.get("step"):
            self._open = False
            self._attrs = {}
            self._buffers = {}
            return None

        episode_name = f"episode_{self.num_episodes:06d}"
        attrs = {**self._attrs, **dict(metadata or {})}
        attrs["success"] = bool(success)
        attrs["episode_length"] = int(len(self._buffers["step"]))
        if not self._use_hdf5:
            self._save_npz_episode(episode_name, attrs)
            self._npz_episode_count += 1
            self._flush_npz()
            self._open = False
            self._attrs = {}
            self._buffers = {}
            return f"/episodes/{episode_name}"

        group = self._episodes.create_group(episode_name)
        for key, value in attrs.items():
            group.attrs[key] = value

        self._write_dataset(group, "sim_time", self._buffers["sim_time"])
        self._write_dataset(group, "step", self._buffers["step"])
        self._write_dataset(group, "obs_policy", self._buffers["obs_policy"])
        self._write_dataset(group, "hri_obs_policy", self._buffers["hri_obs_policy"])
        self._write_dataset(group, "human_valid_mask", self._buffers["human_valid_mask"])
        self._write_dataset(group, "current_pick_idx", self._buffers["current_pick_idx"])
        self._write_dataset(group, "completed_picks", self._buffers["completed_picks"])

        obs_group = group.create_group("obs")
        for field in RECORDED_OBSERVATION_FIELDS:
            self._write_dataset(obs_group, field.name, self._buffers["obs"][field.name])

        hri_obs_group = group.create_group("hri_obs")
        for name in HRI_OBS_FIELD_NAMES:
            self._write_dataset(hri_obs_group, name, self._buffers["obs"][name])

        human_group = group.create_group("human")
        self._write_dataset(human_group, "head_pos", self._buffers["obs"]["human_head_pos"])
        self._write_dataset(human_group, "left_hand_pos", self._buffers["obs"]["human_left_hand_pos"])
        self._write_dataset(human_group, "right_hand_pos", self._buffers["obs"]["human_right_hand_pos"])

        for group_name in ("safety", "errp", "actions", "rewards", "task"):
            out_group = group.create_group(group_name)
            for name, values in self._buffers[group_name].items():
                self._write_dataset(out_group, name, values)

        self._file.flush()
        self._open = False
        self._attrs = {}
        self._buffers = {}
        return group.name

    def close(self) -> None:
        if self._open:
            self.end_episode(success=False, metadata={"reason": "recorder_closed"})
        if self._use_hdf5:
            self._file.close()
        else:
            self._flush_npz()

    def _write_dataset(self, group, name: str, values) -> None:
        arr = np.asarray(values)
        group.create_dataset(name, data=arr, compression=self.compression)

    def _save_npz_episode(self, episode_name: str, attrs: Mapping) -> None:
        base = f"episodes/{episode_name}"
        for key, value in attrs.items():
            self._npz_payload[f"{base}/attrs/{key}"] = np.asarray(value)
        self._npz_payload[f"{base}/sim_time"] = np.asarray(self._buffers["sim_time"])
        self._npz_payload[f"{base}/step"] = np.asarray(self._buffers["step"])
        self._npz_payload[f"{base}/obs_policy"] = np.asarray(self._buffers["obs_policy"])
        self._npz_payload[f"{base}/hri_obs_policy"] = np.asarray(self._buffers["hri_obs_policy"])
        self._npz_payload[f"{base}/human_valid_mask"] = np.asarray(
            self._buffers["human_valid_mask"]
        )
        self._npz_payload[f"{base}/current_pick_idx"] = np.asarray(
            self._buffers["current_pick_idx"]
        )
        self._npz_payload[f"{base}/completed_picks"] = np.asarray(
            self._buffers["completed_picks"]
        )

        for field in RECORDED_OBSERVATION_FIELDS:
            self._npz_payload[f"{base}/obs/{field.name}"] = np.asarray(
                self._buffers["obs"][field.name]
            )
        for name in HRI_OBS_FIELD_NAMES:
            self._npz_payload[f"{base}/hri_obs/{name}"] = np.asarray(
                self._buffers["obs"][name]
            )
        self._npz_payload[f"{base}/human/head_pos"] = np.asarray(
            self._buffers["obs"]["human_head_pos"]
        )
        self._npz_payload[f"{base}/human/left_hand_pos"] = np.asarray(
            self._buffers["obs"]["human_left_hand_pos"]
        )
        self._npz_payload[f"{base}/human/right_hand_pos"] = np.asarray(
            self._buffers["obs"]["human_right_hand_pos"]
        )
        for group_name in ("safety", "errp", "actions", "rewards", "task"):
            for name, values in self._buffers[group_name].items():
                self._npz_payload[f"{base}/{group_name}/{name}"] = np.asarray(values)

    def _flush_npz(self) -> None:
        self._npz_payload["attrs/episode_count"] = np.asarray(
            int(self._npz_episode_count), dtype=np.int32
        )
        np.savez_compressed(self.path, **self._npz_payload)

    def _append_safety(self, obs: dict[str, np.ndarray], safety: Mapping | None) -> None:
        safety = dict(safety or {})
        min_dist = float(np.asarray(obs["min_hand_gripper_dist"]).reshape(-1)[0])
        center_dist = float(
            np.asarray(obs["min_hand_gripper_center_dist"]).reshape(-1)[0]
        )
        surface_gap = float(
            np.asarray(obs["min_hand_gripper_surface_gap"]).reshape(-1)[0]
        )
        near_human = float(np.asarray(obs["near_human"]).reshape(-1)[0])
        near_miss = float(np.asarray(obs["near_miss"]).reshape(-1)[0])
        collision = float(np.asarray(obs["human_robot_collision"]).reshape(-1)[0])
        contact = max(
            float(np.asarray(obs["left_hand_contact"]).reshape(-1)[0]),
            float(np.asarray(obs["right_hand_contact"]).reshape(-1)[0]),
        )
        values = {
            "left_hand_gripper_dist_m": safety.get("left_hand_gripper_dist_m", np.nan),
            "right_hand_gripper_dist_m": safety.get("right_hand_gripper_dist_m", np.nan),
            "min_hand_gripper_dist_m": safety.get("min_hand_gripper_dist_m", min_dist),
            "min_hand_gripper_center_dist_m": safety.get(
                "min_hand_gripper_center_dist_m", center_dist
            ),
            "min_hand_gripper_surface_gap_m": safety.get(
                "min_hand_gripper_surface_gap_m", surface_gap
            ),
            "near_human": safety.get("near_human", near_human),
            "near_miss": safety.get("near_miss", near_miss),
            "human_robot_collision": safety.get("human_robot_collision", collision),
            "haptic_pulse_left": safety.get("haptic_pulse_left", 0.0),
            "haptic_pulse_right": safety.get("haptic_pulse_right", 0.0),
            "gripper_gap_left_m": safety.get("gripper_gap_left_m", np.nan),
            "gripper_gap_right_m": safety.get("gripper_gap_right_m", np.nan),
            "gripper_penetration_left_m": safety.get("gripper_penetration_left_m", 0.0),
            "gripper_penetration_right_m": safety.get("gripper_penetration_right_m", 0.0),
            "left_hand_surface_gap_m": safety.get(
                "left_hand_surface_gap_m",
                float(np.asarray(obs["left_hand_end_effector_surface_gap"]).reshape(-1)[0]),
            ),
            "right_hand_surface_gap_m": safety.get(
                "right_hand_surface_gap_m",
                float(np.asarray(obs["right_hand_end_effector_surface_gap"]).reshape(-1)[0]),
            ),
            "min_hand_end_effector_surface_gap_m": safety.get(
                "min_hand_end_effector_surface_gap_m",
                float(np.asarray(obs["min_hand_end_effector_surface_gap"]).reshape(-1)[0]),
            ),
            "left_end_effector_surface_gap_m": safety.get(
                "left_end_effector_surface_gap_m",
                safety.get(
                    "left_hand_surface_gap_m",
                    float(
                        np.asarray(obs["left_hand_end_effector_surface_gap"]).reshape(-1)[0]
                    ),
                ),
            ),
            "right_end_effector_surface_gap_m": safety.get(
                "right_end_effector_surface_gap_m",
                safety.get(
                    "right_hand_surface_gap_m",
                    float(
                        np.asarray(obs["right_hand_end_effector_surface_gap"]).reshape(-1)[0]
                    ),
                ),
            ),
            "end_effector_surface_gap_m": safety.get(
                "end_effector_surface_gap_m",
                float(np.asarray(obs["min_hand_end_effector_surface_gap"]).reshape(-1)[0]),
            ),
            "closest_human_hand_id": safety.get("closest_human_hand_id", 0),
            "closest_robot_link_id": safety.get("closest_robot_link_id", 0),
            "closest_collider_id": safety.get("closest_collider_id", 0),
            "closest_link_left_id": safety.get("closest_link_left_id", 0),
            "closest_link_right_id": safety.get("closest_link_right_id", 0),
            "closest_collider_left_id": safety.get("closest_collider_left_id", 0),
            "closest_collider_right_id": safety.get("closest_collider_right_id", 0),
            "contact_left": safety.get(
                "contact_left", float(np.asarray(obs["left_hand_contact"]).reshape(-1)[0])
            ),
            "contact_right": safety.get(
                "contact_right", float(np.asarray(obs["right_hand_contact"]).reshape(-1)[0])
            ),
            "contact_active": safety.get("contact_active", contact),
            "contact_force_left_n": safety.get("contact_force_left_n", 0.0),
            "contact_force_right_n": safety.get("contact_force_right_n", 0.0),
            "contact_force_n": safety.get(
                "contact_force_n",
                max(
                    float(safety.get("contact_force_left_n", 0.0)),
                    float(safety.get("contact_force_right_n", 0.0)),
                ),
            ),
            "contact_force_valid_left": safety.get("contact_force_valid_left", 0.0),
            "contact_force_valid_right": safety.get("contact_force_valid_right", 0.0),
            "penetration_left_m": safety.get("penetration_left_m", 0.0),
            "penetration_right_m": safety.get("penetration_right_m", 0.0),
            "penetration_depth_m": safety.get(
                "penetration_depth_m",
                max(
                    float(safety.get("penetration_left_m", 0.0)),
                    float(safety.get("penetration_right_m", 0.0)),
                ),
            ),
            "near_left": safety.get("near_left", 0.0),
            "near_right": safety.get("near_right", 0.0),
            "near_miss_left": safety.get("near_miss_left", 0.0),
            "near_miss_right": safety.get("near_miss_right", 0.0),
            "distance_gate_left": safety.get("distance_gate_left", 0.0),
            "distance_gate_right": safety.get("distance_gate_right", 0.0),
            "distance_gate": safety.get(
                "distance_gate", float(np.asarray(obs["distance_gate"]).reshape(-1)[0])
            ),
            "geometry_valid_left": safety.get("geometry_valid_left", 0.0),
            "geometry_valid_right": safety.get("geometry_valid_right", 0.0),
            "geometry_valid": safety.get(
                "geometry_valid", float(np.asarray(obs["geometry_valid"]).reshape(-1)[0])
            ),
            "query_time_left_ms": safety.get("query_time_left_ms", 0.0),
            "query_time_right_ms": safety.get("query_time_right_ms", 0.0),
            "query_count_left": safety.get("query_count_left", 0),
            "query_count_right": safety.get("query_count_right", 0),
        }
        for name, value in values.items():
            if name.endswith("_id") or name.startswith("query_count_"):
                self._buffers["safety"][name].append(int(value))
            else:
                self._buffers["safety"][name].append(float(value))

    def _append_errp(self, errp: Mapping | None, step: int) -> None:
        errp = dict(errp or {})
        self._buffers["errp"]["label"].append(int(errp.get("label", 0)))
        self._buffers["errp"]["feedback"].append(float(errp.get("feedback", 0.0)))
        self._buffers["errp"]["uncertainty"].append(float(errp.get("uncertainty", 0.0)))
        self._buffers["errp"]["timestamp"].append(float(errp.get("timestamp", np.nan)))
        self._buffers["errp"]["aligned_step"].append(int(errp.get("aligned_step", step)))

    def _append_action(self, action) -> None:
        for name, attr in (
            ("robot_joint_positions", "joint_positions"),
            ("robot_joint_velocities", "joint_velocities"),
            ("robot_joint_efforts", "joint_efforts"),
        ):
            value = getattr(action, attr, None) if action is not None else None
            self._buffers["actions"][name].append(_fixed_array(value, 9, fill=np.nan))

    def _append_reward(self, obs: dict[str, np.ndarray], reward: Mapping | None) -> None:
        reward = dict(reward or {})
        collision = float(np.asarray(obs["human_robot_collision"]).reshape(-1)[0])
        near_human = float(np.asarray(obs["near_human"]).reshape(-1)[0])
        errp_reward = float(reward.get("errp", 0.0))
        safety_reward = float(reward.get("safety", -(collision + 0.2 * near_human)))
        task_reward = float(reward.get("task", 0.0))
        total = float(reward.get("total", task_reward + safety_reward + errp_reward))
        self._buffers["rewards"]["task"].append(task_reward)
        self._buffers["rewards"]["errp"].append(errp_reward)
        self._buffers["rewards"]["safety"].append(safety_reward)
        self._buffers["rewards"]["total"].append(total)


def _fixed_array(value, size: int, fill: float = 0.0) -> np.ndarray:
    out = np.full(size, fill, dtype=np.float32)
    if value is None:
        return out
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    n = min(size, arr.size)
    if n:
        out[:n] = arr[:n]
    return out
