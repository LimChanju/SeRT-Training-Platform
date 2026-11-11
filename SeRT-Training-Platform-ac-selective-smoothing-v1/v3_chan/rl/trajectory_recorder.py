from __future__ import annotations

import os
from typing import Any, Mapping

import h5py
import numpy as np

from .actions import ACTION_DIM, ACTION_VERSION, clip_action
from .observations import (
    OBSERVATION_DIM,
    OBSERVATION_FIELDS,
    OBSERVATION_VERSION,
    flatten_observation,
    validate_observation,
)
from .rewards import REWARD_VERSION, RewardResult, reward_component_names


TRAJECTORY_SCHEMA_VERSION = "trajectory_v0_transitions"
EXPERT_JOINT_ACTION_DIM = 9


class TrajectoryRecorder:
    """HDF5 transition recorder for expert demos and offline RL datasets."""

    def __init__(
        self,
        path: str,
        *,
        overwrite: bool = False,
        compression: str | None = "gzip",
    ) -> None:
        self.path = os.path.abspath(path)
        self.compression = compression
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        mode = "w" if overwrite else "a"
        self._file = h5py.File(self.path, mode)
        self._episodes = self._file.require_group("episodes")
        self._write_root_attrs()
        self._episode_open = False
        self._episode_attrs: dict[str, Any] = {}
        self._buffers: dict[str, Any] = {}

    @property
    def num_episodes(self) -> int:
        return len(self._episodes.keys())

    def start_episode(self, metadata: dict[str, Any] | None = None) -> None:
        if self._episode_open:
            raise RuntimeError("An episode is already open")
        self._episode_open = True
        self._episode_attrs = dict(metadata or {})
        self._buffers = {
            "sim_time": [],
            "obs": {field.name: [] for field in OBSERVATION_FIELDS},
            "next_obs": {field.name: [] for field in OBSERVATION_FIELDS},
            "obs_policy": [],
            "next_obs_policy": [],
            "policy_action": [],
            "expert_task_action": [],
            "expert_target_action": [],
            "expert_target_position": [],
            "expert_controller_event": [],
            "expert_controller_t": [],
            "expert_joint_action": [],
            "reward_total": [],
            "reward_components": {name: [] for name in reward_component_names()},
            "done": [],
            "errp_label": [],
            "errp_feedback": [],
            "errp_uncertainty": [],
            "errp_source_code": [],
            "errp_event_step": [],
            "eeg_replay_used": [],
            "eeg_epoch_id": [],
            "human": {
                "head_pos": [],
                "left_hand_pos": [],
                "right_hand_pos": [],
                "left_hand_vel": [],
                "right_hand_vel": [],
                "valid_mask": [],
            },
        }

    def add_transition(
        self,
        *,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        policy_action: np.ndarray,
        expert_task_action: np.ndarray,
        expert_joint_action: np.ndarray,
        reward: RewardResult | float,
        done: bool,
        expert_target_action: np.ndarray | None = None,
        expert_target_position: np.ndarray | None = None,
        expert_controller_event: int = -1,
        expert_controller_t: float = 0.0,
        errp_label: int = 0,
        errp_feedback: float = 0.0,
        errp_uncertainty: float = 0.0,
        errp_source_code: int = 0,
        errp_event_step: int = 0,
        eeg_replay_used: int = 0,
        eeg_epoch_id: str = "",
        sim_time: float | None = None,
        human_state: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._episode_open:
            raise RuntimeError("start_episode() must be called before add_transition()")
        validate_observation(obs)
        validate_observation(next_obs)
        for field in OBSERVATION_FIELDS:
            self._buffers["obs"][field.name].append(np.asarray(obs[field.name], dtype=np.float32))
            self._buffers["next_obs"][field.name].append(
                np.asarray(next_obs[field.name], dtype=np.float32)
            )
        self._buffers["sim_time"].append(0.0 if sim_time is None else float(sim_time))

        self._buffers["obs_policy"].append(flatten_observation(obs))
        self._buffers["next_obs_policy"].append(flatten_observation(next_obs))
        self._buffers["policy_action"].append(clip_action(policy_action))
        self._buffers["expert_task_action"].append(clip_action(expert_task_action))
        if expert_target_action is None:
            expert_target_action = expert_task_action
        if expert_target_position is None:
            expert_target_position = obs.get("ee_pos", np.zeros(3, dtype=np.float32))
        self._buffers["expert_target_action"].append(clip_action(expert_target_action))
        self._buffers["expert_target_position"].append(_fixed_array(expert_target_position, 3))
        self._buffers["expert_controller_event"].append(int(expert_controller_event))
        self._buffers["expert_controller_t"].append(float(expert_controller_t))
        self._buffers["expert_joint_action"].append(
            _fixed_array(expert_joint_action, EXPERT_JOINT_ACTION_DIM)
        )

        if isinstance(reward, RewardResult):
            total = reward.total
            components = reward.components
        else:
            total = float(reward)
            components = {}
        self._buffers["reward_total"].append(float(total))
        for name in reward_component_names():
            self._buffers["reward_components"][name].append(float(components.get(name, 0.0)))

        self._buffers["done"].append(1 if done else 0)
        self._buffers["errp_label"].append(int(errp_label))
        self._buffers["errp_feedback"].append(float(errp_feedback))
        self._buffers["errp_uncertainty"].append(float(errp_uncertainty))
        self._buffers["errp_source_code"].append(int(errp_source_code))
        self._buffers["errp_event_step"].append(int(errp_event_step))
        self._buffers["eeg_replay_used"].append(int(eeg_replay_used))
        self._buffers["eeg_epoch_id"].append(str(eeg_epoch_id))
        self._append_human_state(obs, next_obs, sim_time=sim_time, human_state=human_state)

    def end_episode(
        self,
        *,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if not self._episode_open:
            raise RuntimeError("No episode is currently open")
        length = len(self._buffers["done"])
        if length == 0:
            self._episode_open = False
            self._episode_attrs = {}
            self._buffers = {}
            return None

        episode_name = f"episode_{self.num_episodes:06d}"
        group = self._episodes.create_group(episode_name)
        self._set_attrs(
            group,
            {
                **self._episode_attrs,
                **dict(metadata or {}),
                "success": bool(success),
                "episode_length": int(length),
            },
        )
        self._write_obs_group(group.create_group("obs"), self._buffers["obs"])
        self._write_obs_group(group.create_group("next_obs"), self._buffers["next_obs"])
        self._write_dataset(group, "sim_time", self._buffers["sim_time"], ())
        self._write_dataset(group, "obs_policy", self._buffers["obs_policy"], (OBSERVATION_DIM,))
        self._write_dataset(
            group,
            "next_obs_policy",
            self._buffers["next_obs_policy"],
            (OBSERVATION_DIM,),
        )

        actions_group = group.create_group("actions")
        self._write_dataset(actions_group, "policy_action", self._buffers["policy_action"], (ACTION_DIM,))
        self._write_dataset(
            actions_group,
            "expert_task_action",
            self._buffers["expert_task_action"],
            (ACTION_DIM,),
        )
        self._write_dataset(
            actions_group,
            "expert_target_action",
            self._buffers["expert_target_action"],
            (ACTION_DIM,),
        )
        self._write_dataset(
            actions_group,
            "expert_target_position",
            self._buffers["expert_target_position"],
            (3,),
        )
        self._write_dataset(
            actions_group,
            "expert_controller_event",
            self._buffers["expert_controller_event"],
            (),
        )
        self._write_dataset(
            actions_group,
            "expert_controller_t",
            self._buffers["expert_controller_t"],
            (),
        )
        self._write_dataset(
            actions_group,
            "expert_joint_action",
            self._buffers["expert_joint_action"],
            (EXPERT_JOINT_ACTION_DIM,),
        )

        rewards_group = group.create_group("rewards")
        self._write_dataset(rewards_group, "total", self._buffers["reward_total"], ())
        components_group = rewards_group.create_group("components")
        for name, values in self._buffers["reward_components"].items():
            self._write_dataset(components_group, name, values, ())

        self._write_dataset(group, "dones", self._buffers["done"], ())
        errp_group = group.create_group("errp")
        self._write_dataset(errp_group, "label", self._buffers["errp_label"], ())
        self._write_dataset(errp_group, "feedback", self._buffers["errp_feedback"], ())
        self._write_dataset(errp_group, "uncertainty", self._buffers["errp_uncertainty"], ())
        self._write_dataset(errp_group, "source_code", self._buffers["errp_source_code"], ())
        self._write_dataset(errp_group, "event_step", self._buffers["errp_event_step"], ())
        self._write_dataset(errp_group, "eeg_replay_used", self._buffers["eeg_replay_used"], ())
        errp_group.create_dataset(
            "eeg_epoch_id",
            data=np.asarray(self._buffers["eeg_epoch_id"], dtype=h5py.string_dtype("utf-8")),
        )
        human_group = group.create_group("human")
        self._write_dataset(human_group, "head_pos", self._buffers["human"]["head_pos"], (3,))
        self._write_dataset(
            human_group,
            "left_hand_pos",
            self._buffers["human"]["left_hand_pos"],
            (3,),
        )
        self._write_dataset(
            human_group,
            "right_hand_pos",
            self._buffers["human"]["right_hand_pos"],
            (3,),
        )
        self._write_dataset(
            human_group,
            "left_hand_vel",
            self._buffers["human"]["left_hand_vel"],
            (3,),
        )
        self._write_dataset(
            human_group,
            "right_hand_vel",
            self._buffers["human"]["right_hand_vel"],
            (3,),
        )
        self._write_dataset(human_group, "valid_mask", self._buffers["human"]["valid_mask"], (3,))

        self._file.flush()
        self._episode_open = False
        self._episode_attrs = {}
        self._buffers = {}
        return group.name

    def close(self) -> None:
        if self._episode_open:
            raise RuntimeError("Cannot close recorder while an episode is open")
        self._file.close()

    def __enter__(self) -> "TrajectoryRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._episode_open:
            self._episode_open = False
        self._file.close()

    def _write_root_attrs(self) -> None:
        self._set_attrs(
            self._file,
            {
                "schema_version": TRAJECTORY_SCHEMA_VERSION,
                "observation_version": OBSERVATION_VERSION,
                "action_version": ACTION_VERSION,
                "reward_version": REWARD_VERSION,
                "observation_dim": OBSERVATION_DIM,
                "action_dim": ACTION_DIM,
            },
        )

    def _write_obs_group(self, group, obs_buffers: dict[str, list[np.ndarray]]) -> None:
        for field in OBSERVATION_FIELDS:
            self._write_dataset(group, field.name, obs_buffers[field.name], field.shape)

    def _append_human_state(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        *,
        sim_time: float | None,
        human_state: Mapping[str, Any] | None,
    ) -> None:
        state = dict(human_state or {})
        head_pos = _state_or_obs_vec3(state, "human_head_pos", obs)
        left_pos = _state_or_obs_vec3(state, "human_left_hand_pos", obs)
        right_pos = _state_or_obs_vec3(state, "human_right_hand_pos", obs)
        left_vel = _state_vec3(state, "human_left_hand_vel")
        right_vel = _state_vec3(state, "human_right_hand_vel")

        if left_vel is None or right_vel is None:
            dt = _state_float(state, "dt")
            if dt is None:
                next_time = _state_float(state, "next_sim_time")
                if next_time is not None and sim_time is not None:
                    dt = float(next_time) - float(sim_time)
            if dt is None or dt <= 1e-6:
                dt = 1.0
            if left_vel is None:
                left_vel = _velocity_from_obs(obs, next_obs, "human_left_hand_pos", dt)
            if right_vel is None:
                right_vel = _velocity_from_obs(obs, next_obs, "human_right_hand_pos", dt)

        valid_mask = _state_vec3(state, "human_valid_mask")
        if valid_mask is None:
            valid_mask = np.array(
                [
                    _is_valid_human_position(head_pos),
                    _is_valid_human_position(left_pos),
                    _is_valid_human_position(right_pos),
                ],
                dtype=np.float32,
            )
        else:
            valid_mask = np.clip(valid_mask, 0.0, 1.0)

        self._buffers["human"]["head_pos"].append(head_pos)
        self._buffers["human"]["left_hand_pos"].append(left_pos)
        self._buffers["human"]["right_hand_pos"].append(right_pos)
        self._buffers["human"]["left_hand_vel"].append(left_vel)
        self._buffers["human"]["right_hand_vel"].append(right_vel)
        self._buffers["human"]["valid_mask"].append(valid_mask.astype(np.float32))

    def _write_dataset(self, group, name: str, values: list, item_shape: tuple[int, ...]) -> None:
        arr = np.asarray(values, dtype=np.float32)
        if item_shape:
            arr = arr.reshape((len(values),) + tuple(item_shape))
        else:
            arr = arr.reshape((len(values),))
        group.create_dataset(name, data=arr, compression=self.compression)

    def _set_attrs(self, obj, attrs: dict[str, Any]) -> None:
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
                obj.attrs[key] = value
            else:
                obj.attrs[key] = str(value)


def _fixed_array(value, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    n = min(size, arr.size)
    if n > 0:
        result[:n] = arr[:n]
    return result


def _state_or_obs_vec3(
    state: Mapping[str, Any],
    key: str,
    obs: Mapping[str, np.ndarray],
) -> np.ndarray:
    value = _state_vec3(state, key)
    if value is not None:
        return value
    return _fixed_array(obs.get(key, np.zeros(3, dtype=np.float32)), 3)


def _state_vec3(state: Mapping[str, Any], key: str) -> np.ndarray | None:
    if key not in state or state[key] is None:
        return None
    return _fixed_array(state[key], 3)


def _state_float(state: Mapping[str, Any], key: str) -> float | None:
    if key not in state or state[key] is None:
        return None
    try:
        return float(np.asarray(state[key]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def _velocity_from_obs(
    obs: Mapping[str, np.ndarray],
    next_obs: Mapping[str, np.ndarray],
    key: str,
    dt: float,
) -> np.ndarray:
    current_pos = _fixed_array(obs.get(key, np.zeros(3, dtype=np.float32)), 3)
    next_pos = _fixed_array(next_obs.get(key, current_pos), 3)
    if not _is_valid_human_position(current_pos) or not _is_valid_human_position(next_pos):
        return np.zeros(3, dtype=np.float32)
    return ((next_pos - current_pos) / max(float(dt), 1e-6)).astype(np.float32)


def _is_valid_human_position(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return 0.0
    return 1.0 if float(np.linalg.norm(arr[:3])) > 1e-6 else 0.0
