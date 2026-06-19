from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import h5py
import numpy as np


HumanReplayMode = Literal["step", "loop"]
HumanReplayEpisodePolicy = Literal["cycle", "random"]


@dataclass(frozen=True)
class HumanReplayInfo:
    path: str
    episode_count: int
    mode: str
    episode_policy: str


class HumanTrajectoryReplay:
    """Replay recorded human head/hand trajectories as an Isaac human_state_fn.

    The preferred source is the recorder's `/episodes/<ep>/human` group. Older
    trajectory files can still be replayed from `/obs/human_*` fields.
    """

    def __init__(
        self,
        path: str,
        *,
        mode: HumanReplayMode = "step",
        episode_policy: HumanReplayEpisodePolicy = "cycle",
        seed: int = 0,
    ) -> None:
        self.path = os.path.abspath(path)
        self.mode = mode
        self.episode_policy = episode_policy
        self.rng = np.random.default_rng(seed)
        self._file = h5py.File(self.path, "r")
        if "episodes" not in self._file:
            raise KeyError(f"Human replay file has no 'episodes' group: {self.path}")
        self._episodes = self._file["episodes"]
        self._episode_names = tuple(sorted(self._episodes.keys()))
        if not self._episode_names:
            raise ValueError(f"Human replay file has no episodes: {self.path}")
        self._episode_name = self._episode_names[0]
        self._episode = self._load_episode(self._episode_name)
        self._cursor = 0
        self._last_state: dict[str, Any] = {}

    @property
    def info(self) -> HumanReplayInfo:
        return HumanReplayInfo(
            path=self.path,
            episode_count=len(self._episode_names),
            mode=self.mode,
            episode_policy=self.episode_policy,
        )

    @property
    def episode_name(self) -> str:
        return self._episode_name

    def reset(self, episode_index: int = 0, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.episode_policy == "random":
            idx = int(self.rng.integers(0, len(self._episode_names)))
        elif self.episode_policy == "cycle":
            idx = int(episode_index) % len(self._episode_names)
        else:
            raise ValueError(f"Unknown human replay episode policy: {self.episode_policy}")
        self._episode_name = self._episode_names[idx]
        self._episode = self._load_episode(self._episode_name)
        self._cursor = 0
        self._last_state = {}
        return self.peek()

    def peek(self) -> dict[str, Any]:
        return self._state_at(self._cursor)

    def __call__(self) -> dict[str, Any]:
        state = self._state_at(self._cursor)
        self._last_state = state
        self._cursor += 1
        return state

    def close(self) -> None:
        self._file.close()

    def _state_at(self, idx: int) -> dict[str, Any]:
        length = int(self._episode["length"])
        if length <= 0:
            return {}
        if self.mode == "loop":
            sample_idx = int(idx) % length
        elif self.mode == "step":
            sample_idx = min(max(int(idx), 0), length - 1)
        else:
            raise ValueError(f"Unknown human replay mode: {self.mode}")

        valid_mask = self._episode["valid_mask"][sample_idx]
        state: dict[str, Any] = {
            "human_left_hand_vel": self._episode["left_hand_vel"][sample_idx],
            "human_right_hand_vel": self._episode["right_hand_vel"][sample_idx],
            "human_valid_mask": valid_mask,
        }
        if valid_mask[0] > 0.5:
            state["human_head_pos"] = self._episode["head_pos"][sample_idx]
        if valid_mask[1] > 0.5:
            state["human_left_hand_pos"] = self._episode["left_hand_pos"][sample_idx]
        if valid_mask[2] > 0.5:
            state["human_right_hand_pos"] = self._episode["right_hand_pos"][sample_idx]

        if self._episode["human_robot_collision"] is not None:
            state["recorded_human_robot_collision"] = bool(
                self._episode["human_robot_collision"][sample_idx] > 0.5
            )
        if self._episode["near_human"] is not None:
            state["recorded_near_human"] = bool(self._episode["near_human"][sample_idx] > 0.5)
        if self._episode["gripper_camera_occluded"] is not None:
            state["gripper_camera_occluded"] = float(
                np.clip(self._episode["gripper_camera_occluded"][sample_idx], 0.0, 1.0)
            )
        return state

    def _load_episode(self, episode_name: str) -> dict[str, Any]:
        group = self._episodes[episode_name]
        if "human" in group:
            human = group["human"]
            head_pos = _dataset_or_zeros(human, "head_pos", (3,))
            left_hand_pos = _dataset_or_zeros(human, "left_hand_pos", (3,))
            right_hand_pos = _dataset_or_zeros(human, "right_hand_pos", (3,))
            left_hand_vel = _dataset_or_zeros(human, "left_hand_vel", (3,))
            right_hand_vel = _dataset_or_zeros(human, "right_hand_vel", (3,))
            valid_mask = _dataset_or_derived_valid_mask(
                human,
                head_pos,
                left_hand_pos,
                right_hand_pos,
            )
        else:
            obs = group["obs"]
            head_pos = _dataset_or_zeros(obs, "human_head_pos", (3,))
            left_hand_pos = _dataset_or_zeros(obs, "human_left_hand_pos", (3,))
            right_hand_pos = _dataset_or_zeros(obs, "human_right_hand_pos", (3,))
            sim_time = _dataset_or_none(group, "sim_time")
            left_hand_vel = _finite_difference(left_hand_pos, sim_time)
            right_hand_vel = _finite_difference(right_hand_pos, sim_time)
            valid_mask = _derived_valid_mask(head_pos, left_hand_pos, right_hand_pos)

        length = int(head_pos.shape[0])
        return {
            "length": length,
            "head_pos": head_pos,
            "left_hand_pos": left_hand_pos,
            "right_hand_pos": right_hand_pos,
            "left_hand_vel": _align_length(left_hand_vel, length, (3,)),
            "right_hand_vel": _align_length(right_hand_vel, length, (3,)),
            "valid_mask": _align_length(valid_mask, length, (3,)),
            "human_robot_collision": _obs_scalar_or_none(group, "human_robot_collision"),
            "near_human": _obs_scalar_or_none(group, "near_human"),
            "gripper_camera_occluded": _human_scalar_or_none(group, "gripper_camera_occluded"),
        }


def _dataset_or_zeros(group, name: str, item_shape: tuple[int, ...]) -> np.ndarray:
    if name in group:
        arr = np.asarray(group[name], dtype=np.float32)
        return arr.reshape((arr.shape[0],) + item_shape)
    length = _infer_group_length(group)
    return np.zeros((length,) + item_shape, dtype=np.float32)


def _dataset_or_none(group, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name], dtype=np.float32).reshape(-1)


def _dataset_or_derived_valid_mask(
    human_group,
    head_pos: np.ndarray,
    left_hand_pos: np.ndarray,
    right_hand_pos: np.ndarray,
) -> np.ndarray:
    if "valid_mask" in human_group:
        arr = np.asarray(human_group["valid_mask"], dtype=np.float32)
        return arr.reshape((arr.shape[0], 3))
    return _derived_valid_mask(head_pos, left_hand_pos, right_hand_pos)


def _derived_valid_mask(
    head_pos: np.ndarray,
    left_hand_pos: np.ndarray,
    right_hand_pos: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            _valid_position_series(head_pos),
            _valid_position_series(left_hand_pos),
            _valid_position_series(right_hand_pos),
        ],
        axis=1,
    ).astype(np.float32)


def _valid_position_series(values: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(values), axis=1)
    nonzero = np.linalg.norm(values, axis=1) > 1e-6
    return np.logical_and(finite, nonzero).astype(np.float32)


def _finite_difference(values: np.ndarray, sim_time: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vel = np.zeros_like(values)
    if values.shape[0] <= 1:
        return vel
    if sim_time is None or len(sim_time) != values.shape[0]:
        dt = np.ones((values.shape[0] - 1, 1), dtype=np.float32)
    else:
        dt = np.diff(sim_time).reshape(-1, 1).astype(np.float32)
        dt = np.maximum(dt, 1e-6)
    vel[1:] = (values[1:] - values[:-1]) / dt
    return vel


def _obs_scalar_or_none(group, name: str) -> np.ndarray | None:
    if "obs" not in group or name not in group["obs"]:
        return None
    return np.asarray(group["obs"][name], dtype=np.float32).reshape(-1)


def _human_scalar_or_none(group, name: str) -> np.ndarray | None:
    if "human" not in group or name not in group["human"]:
        return None
    return np.asarray(group["human"][name], dtype=np.float32).reshape(-1)


def _align_length(arr: np.ndarray, length: int, item_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == (length,) + item_shape:
        return arr
    result = np.zeros((length,) + item_shape, dtype=np.float32)
    count = min(length, arr.shape[0])
    if count > 0:
        result[:count] = arr[:count].reshape((count,) + item_shape)
    return result


def _infer_group_length(group) -> int:
    for value in group.values():
        if hasattr(value, "shape") and value.shape:
            return int(value.shape[0])
    return 0
