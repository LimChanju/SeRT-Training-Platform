import importlib.util
import os
import sys
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
OBSERVATION_DIM = _obs.OBSERVATION_DIM
OBSERVATION_VERSION = _obs.OBSERVATION_VERSION
build_observation = _obs.build_observation
flatten_observation = _obs.flatten_observation
validate_observation = _obs.validate_observation

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
    "human_robot_collision",
    "near_human",
    "has_grasped_cube",
    "task_phase",
    "controller_event",
)
HRI_OBS_DIM = int(sum(field.dim for field in OBSERVATION_FIELDS if field.name in HRI_OBS_FIELD_NAMES))


def flatten_hri_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    validate_observation(obs)
    values = []
    for field in OBSERVATION_FIELDS:
        if field.name in HRI_OBS_FIELD_NAMES:
            values.append(np.asarray(obs[field.name], dtype=np.float32).reshape(-1))
    return np.concatenate(values).astype(np.float32, copy=False)


class HRIObsRecorder:
    """Small HDF5 recorder for VR HRI sessions.

    Stores obs dictionaries and flattened obs_policy arrays that can be fed
    directly into the existing policy/training readers.
    """

    SCHEMA_VERSION = "hri_obs_v1_cognitive_safety"

    def __init__(
        self,
        path: str,
        *,
        overwrite: bool = False,
        sample_interval_steps: int = 1,
        compression: str | None = "gzip",
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
            self._file = h5py.File(self.path, mode)
            self._episodes = self._file.require_group("episodes")
            self._file.attrs["schema_version"] = self.SCHEMA_VERSION
            self._file.attrs["observation_version"] = OBSERVATION_VERSION
            self._file.attrs["observation_dim"] = int(OBSERVATION_DIM)
            self._file.attrs["hri_observation_dim"] = int(HRI_OBS_DIM)
            self._file.attrs["hri_observation_fields"] = ",".join(HRI_OBS_FIELD_NAMES)
            self._file.attrs["sample_interval_steps"] = int(self.sample_interval_steps)
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
            "obs": {field.name: [] for field in OBSERVATION_FIELDS},
            "obs_policy": [],
            "hri_obs_policy": [],
            "human_valid_mask": [],
            "current_pick_idx": [],
            "completed_picks": [],
            "safety": {
                "left_hand_gripper_dist_m": [],
                "right_hand_gripper_dist_m": [],
                "min_hand_gripper_dist_m": [],
                "near_human": [],
                "human_robot_collision": [],
                "haptic_pulse_left": [],
                "haptic_pulse_right": [],
                "gripper_gap_left_m": [],
                "gripper_gap_right_m": [],
                "gripper_penetration_left_m": [],
                "gripper_penetration_right_m": [],
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
        self._buffers["sim_time"].append(float(sim_time))
        self._buffers["step"].append(int(step))
        for field in OBSERVATION_FIELDS:
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
        for field in OBSERVATION_FIELDS:
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

        for field in OBSERVATION_FIELDS:
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
        near_human = float(np.asarray(obs["near_human"]).reshape(-1)[0])
        collision = float(np.asarray(obs["human_robot_collision"]).reshape(-1)[0])
        values = {
            "left_hand_gripper_dist_m": safety.get("left_hand_gripper_dist_m", np.nan),
            "right_hand_gripper_dist_m": safety.get("right_hand_gripper_dist_m", np.nan),
            "min_hand_gripper_dist_m": safety.get("min_hand_gripper_dist_m", min_dist),
            "near_human": safety.get("near_human", near_human),
            "human_robot_collision": safety.get("human_robot_collision", collision),
            "haptic_pulse_left": safety.get("haptic_pulse_left", 0.0),
            "haptic_pulse_right": safety.get("haptic_pulse_right", 0.0),
            "gripper_gap_left_m": safety.get("gripper_gap_left_m", np.nan),
            "gripper_gap_right_m": safety.get("gripper_gap_right_m", np.nan),
            "gripper_penetration_left_m": safety.get("gripper_penetration_left_m", 0.0),
            "gripper_penetration_right_m": safety.get("gripper_penetration_right_m", 0.0),
        }
        for name, value in values.items():
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
