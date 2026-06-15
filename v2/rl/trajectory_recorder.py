from __future__ import annotations

import os
from typing import Any

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
            "errp_source_code": [],
            "errp_event_step": [],
            "eeg_replay_used": [],
            "eeg_epoch_id": [],
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
        errp_source_code: int = 0,
        errp_event_step: int = 0,
        eeg_replay_used: int = 0,
        eeg_epoch_id: str = "",
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
        self._buffers["errp_source_code"].append(int(errp_source_code))
        self._buffers["errp_event_step"].append(int(errp_event_step))
        self._buffers["eeg_replay_used"].append(int(eeg_replay_used))
        self._buffers["eeg_epoch_id"].append(str(eeg_epoch_id))

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
        self._write_dataset(errp_group, "source_code", self._buffers["errp_source_code"], ())
        self._write_dataset(errp_group, "event_step", self._buffers["errp_event_step"], ())
        self._write_dataset(errp_group, "eeg_replay_used", self._buffers["eeg_replay_used"], ())
        errp_group.create_dataset(
            "eeg_epoch_id",
            data=np.asarray(self._buffers["eeg_epoch_id"], dtype=h5py.string_dtype("utf-8")),
        )

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
