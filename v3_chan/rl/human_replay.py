from __future__ import annotations

import os
import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Literal

import h5py
import numpy as np

try:
    from .encounter_manifest import (
        SEVERITY_ORDER,
        load_encounter_manifest,
        parse_severity_mix,
        resolve_scenario_source,
    )
except ImportError:
    # Keep direct file imports used by the lightweight unit tests working.
    _manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "encounter_manifest.py",
    )
    _manifest_spec = importlib.util.spec_from_file_location(
        "_v3_chan_encounter_manifest",
        _manifest_path,
    )
    if _manifest_spec is None or _manifest_spec.loader is None:
        raise
    _manifest_module = importlib.util.module_from_spec(_manifest_spec)
    sys.modules[_manifest_spec.name] = _manifest_module
    _manifest_spec.loader.exec_module(_manifest_module)
    SEVERITY_ORDER = _manifest_module.SEVERITY_ORDER
    load_encounter_manifest = _manifest_module.load_encounter_manifest
    parse_severity_mix = _manifest_module.parse_severity_mix
    resolve_scenario_source = _manifest_module.resolve_scenario_source


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
            collision = bool(self._episode["human_robot_collision"][sample_idx] > 0.5)
            state["recorded_human_robot_collision"] = collision
        if self._episode["near_human"] is not None:
            near_human = bool(self._episode["near_human"][sample_idx] > 0.5)
            state["recorded_near_human"] = near_human
        if self._episode["min_hand_gripper_dist_m"] is not None:
            state["recorded_min_hand_gripper_dist_m"] = float(
                self._episode["min_hand_gripper_dist_m"][sample_idx]
            )
        if self._episode["gripper_camera_occluded"] is not None:
            state["gripper_camera_occluded"] = float(
                np.clip(self._episode["gripper_camera_occluded"][sample_idx], 0.0, 1.0)
            )
        return state

    def _load_episode(self, episode_name: str) -> dict[str, Any]:
        return _load_episode_group(self._episodes[episode_name])


@dataclass(frozen=True)
class HumanEncounterReplayInfo:
    path: str
    scenario_count: int
    episode_policy: str
    anchor_mode: str
    phase_match: bool
    event_match: bool
    severity_mix: dict[str, float]


class HumanEncounterReplay:
    """Place one recorded encounter template into each full task rollout.

    A scenario remains inactive until the task reaches its recorded phase/event.
    Its human trajectory is then replayed once. A fixed translation can align the
    recorded end-effector anchor to the current end-effector pose. Safety labels
    from the source remain metadata; the environment recomputes current geometry.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        episode_policy: HumanReplayEpisodePolicy = "random",
        severity_mix: str | dict[str, float] | None = None,
        anchor_mode: Literal["ee", "world"] = "ee",
        phase_match: bool = True,
        event_match: bool = True,
        seed: int = 0,
    ) -> None:
        self.path = os.path.abspath(os.path.expanduser(manifest_path))
        self.manifest = load_encounter_manifest(self.path)
        self.episode_policy = episode_policy
        self.severity_mix = parse_severity_mix(severity_mix)
        self.anchor_mode = anchor_mode
        self.phase_match = bool(phase_match)
        self.event_match = bool(event_match)
        self.rng = np.random.default_rng(seed)
        self._scenarios = tuple(self.manifest["scenarios"])
        self._by_severity = {
            severity: tuple(
                scenario
                for scenario in self._scenarios
                if scenario.get("target_severity") == severity
            )
            for severity in SEVERITY_ORDER
        }
        self._files: dict[str, h5py.File] = {}
        self._scenario: dict[str, Any] = {}
        self._episode: dict[str, Any] = {}
        self._cursor = 0
        self._started = False
        self._finished = False
        self._anchor_offset = np.zeros(3, dtype=np.float32)
        self._runtime_context: dict[str, Any] = {}
        self.reset(0, seed=seed)

    @property
    def info(self) -> HumanEncounterReplayInfo:
        return HumanEncounterReplayInfo(
            path=self.path,
            scenario_count=len(self._scenarios),
            episode_policy=self.episode_policy,
            anchor_mode=self.anchor_mode,
            phase_match=self.phase_match,
            event_match=self.event_match,
            severity_mix=dict(self.severity_mix),
        )

    @property
    def episode_name(self) -> str:
        return str(self._scenario.get("id", ""))

    @property
    def current_scenario(self) -> dict[str, Any]:
        return dict(self._scenario)

    def reset(self, episode_index: int = 0, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._scenario = self._select_scenario(episode_index)
        source_path = resolve_scenario_source(self._scenario, self.path)
        h5_file = self._files.get(source_path)
        if h5_file is None:
            h5_file = h5py.File(source_path, "r")
            self._files[source_path] = h5_file
        episode_name = str(self._scenario["episode_name"])
        self._episode = _load_episode_group(h5_file["episodes"][episode_name])
        self._cursor = int(self._scenario["start_step"])
        self._started = False
        self._finished = False
        self._anchor_offset = np.zeros(3, dtype=np.float32)
        self._runtime_context = {}
        return self.peek()

    def set_runtime_context(
        self,
        *,
        step: int,
        task_phase: str,
        controller_event: int,
        controller_t: int,
        ee_pos: np.ndarray | None,
    ) -> None:
        self._runtime_context = {
            "step": int(step),
            "task_phase": str(task_phase),
            "controller_event": int(controller_event),
            "controller_t": int(controller_t),
            "ee_pos": (
                None
                if ee_pos is None
                else np.asarray(ee_pos, dtype=np.float32).reshape(-1)[:3]
            ),
        }

    def peek(self) -> dict[str, Any]:
        if not self._started:
            return self._inactive_state()
        return self._state_at(self._cursor, advance=False)

    def __call__(self) -> dict[str, Any]:
        if self._finished:
            return self._inactive_state()
        if not self._started:
            if not self._phase_is_ready():
                return self._inactive_state()
            self._start_playback()
        return self._state_at(self._cursor, advance=True)

    def close(self) -> None:
        for h5_file in self._files.values():
            h5_file.close()
        self._files.clear()

    def _select_scenario(self, episode_index: int) -> dict[str, Any]:
        if self.episode_policy == "cycle":
            return dict(self._scenarios[int(episode_index) % len(self._scenarios)])
        if self.episode_policy != "random":
            raise ValueError(
                f"Unknown encounter episode policy: {self.episode_policy}"
            )
        available = [
            severity
            for severity in SEVERITY_ORDER
            if self._by_severity[severity] and self.severity_mix[severity] > 0.0
        ]
        if not available:
            available = [
                severity
                for severity in SEVERITY_ORDER
                if self._by_severity[severity]
            ]
        weights = np.asarray(
            [self.severity_mix[severity] for severity in available],
            dtype=np.float64,
        )
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights)
        weights /= weights.sum()
        severity = str(self.rng.choice(available, p=weights))
        scenarios = self._by_severity[severity]
        return dict(scenarios[int(self.rng.integers(0, len(scenarios)))])

    def _phase_is_ready(self) -> bool:
        if not self.phase_match:
            return True
        current_phase = str(self._runtime_context.get("task_phase", ""))
        expected_phase = str(
            self._scenario.get(
                "trigger_task_phase",
                self._scenario.get("task_phase", ""),
            )
        )
        if current_phase != expected_phase:
            return False
        expected_event = int(
            self._scenario.get(
                "trigger_controller_event",
                self._scenario.get("controller_event", -1),
            )
        )
        current_event = int(self._runtime_context.get("controller_event", -1))
        if self.event_match and expected_event >= 0 and current_event >= 0:
            return current_event == expected_event
        return True

    def _start_playback(self) -> None:
        self._started = True
        if self.anchor_mode == "world":
            return
        if self.anchor_mode != "ee":
            raise ValueError(f"Unknown encounter anchor mode: {self.anchor_mode}")
        source_anchor = self._scenario.get("source_anchor_ee_pos")
        current_anchor = self._runtime_context.get("ee_pos")
        if source_anchor is None or current_anchor is None:
            return
        source = np.asarray(source_anchor, dtype=np.float32).reshape(-1)
        current = np.asarray(current_anchor, dtype=np.float32).reshape(-1)
        if (
            source.size >= 3
            and current.size >= 3
            and np.all(np.isfinite(source[:3]))
            and np.all(np.isfinite(current[:3]))
        ):
            self._anchor_offset = current[:3] - source[:3]

    def _state_at(self, source_idx: int, *, advance: bool) -> dict[str, Any]:
        end_step = min(
            int(self._scenario["end_step"]),
            int(self._episode["length"]),
        )
        if source_idx >= end_step:
            self._finished = True
            return self._inactive_state()
        state = _human_state_at(self._episode, source_idx)
        for key in (
            "human_head_pos",
            "human_left_hand_pos",
            "human_right_hand_pos",
        ):
            if key in state:
                state[key] = (
                    np.asarray(state[key], dtype=np.float32) + self._anchor_offset
                )
        state.update(self._metadata_state(active=True, source_step=source_idx))
        if advance:
            self._cursor += 1
            if self._cursor >= end_step:
                self._finished = True
        return state

    def _inactive_state(self) -> dict[str, Any]:
        return self._metadata_state(active=False, source_step=-1)

    def _metadata_state(self, *, active: bool, source_step: int) -> dict[str, Any]:
        return {
            "encounter_id": str(self._scenario.get("id", "")),
            "encounter_target_severity": str(
                self._scenario.get("target_severity", "")
            ),
            "encounter_target_phase": str(
                self._scenario.get("task_phase", "")
            ),
            "encounter_target_event": int(
                self._scenario.get("controller_event", -1)
            ),
            "encounter_source_session": str(
                self._scenario.get("session_id", "")
            ),
            "encounter_source_episode": str(
                self._scenario.get("source_episode", "")
            ),
            "encounter_source_step": int(source_step),
            "encounter_active": float(active),
            "encounter_started": float(self._started),
            "encounter_finished": float(self._finished),
            "encounter_anchor_offset_m": self._anchor_offset.copy(),
        }


def _load_episode_group(group: h5py.Group) -> dict[str, Any]:
    sim_time = _dataset_or_none(group, "sim_time")
    if "human" in group:
        human = group["human"]
        head_pos = _dataset_or_zeros(human, "head_pos", (3,))
        left_hand_pos = _dataset_or_zeros(human, "left_hand_pos", (3,))
        right_hand_pos = _dataset_or_zeros(human, "right_hand_pos", (3,))
        left_hand_vel = (
            np.asarray(human["left_hand_vel"], dtype=np.float32)
            if "left_hand_vel" in human
            else _finite_difference(left_hand_pos, sim_time)
        )
        right_hand_vel = (
            np.asarray(human["right_hand_vel"], dtype=np.float32)
            if "right_hand_vel" in human
            else _finite_difference(right_hand_pos, sim_time)
        )
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
        left_hand_vel = _finite_difference(left_hand_pos, sim_time)
        right_hand_vel = _finite_difference(right_hand_pos, sim_time)
        valid_mask = _derived_valid_mask(
            head_pos,
            left_hand_pos,
            right_hand_pos,
        )

    length = int(head_pos.shape[0])
    return {
        "length": length,
        "head_pos": head_pos,
        "left_hand_pos": left_hand_pos,
        "right_hand_pos": right_hand_pos,
        "left_hand_vel": _align_length(left_hand_vel, length, (3,)),
        "right_hand_vel": _align_length(right_hand_vel, length, (3,)),
        "valid_mask": _align_length(valid_mask, length, (3,)),
        "human_robot_collision": _scalar_dataset_or_none(
            group,
            (
                "safety/contact_active",
                "safety/human_robot_collision",
                "obs/human_robot_collision",
            ),
        ),
        "near_human": _scalar_dataset_or_none(
            group,
            ("safety/near_human", "obs/near_human"),
        ),
        "min_hand_gripper_dist_m": _scalar_dataset_or_none(
            group,
            (
                "safety/min_hand_end_effector_surface_gap_m",
                "safety/end_effector_surface_gap_m",
                "obs/min_hand_end_effector_surface_gap",
                "safety/min_hand_gripper_surface_gap_m",
                "safety/min_hand_gripper_dist_m",
                "obs/min_hand_gripper_dist",
            ),
        ),
        "gripper_camera_occluded": _human_scalar_or_none(
            group,
            "gripper_camera_occluded",
        ),
    }


def _human_state_at(episode: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    valid_mask = episode["valid_mask"][sample_idx]
    state: dict[str, Any] = {
        "human_left_hand_vel": episode["left_hand_vel"][sample_idx],
        "human_right_hand_vel": episode["right_hand_vel"][sample_idx],
        "human_valid_mask": valid_mask,
    }
    if valid_mask[0] > 0.5:
        state["human_head_pos"] = episode["head_pos"][sample_idx]
    if valid_mask[1] > 0.5:
        state["human_left_hand_pos"] = episode["left_hand_pos"][sample_idx]
    if valid_mask[2] > 0.5:
        state["human_right_hand_pos"] = episode["right_hand_pos"][sample_idx]
    if episode["human_robot_collision"] is not None:
        state["recorded_human_robot_collision"] = bool(
            episode["human_robot_collision"][sample_idx] > 0.5
        )
    if episode["near_human"] is not None:
        state["recorded_near_human"] = bool(
            episode["near_human"][sample_idx] > 0.5
        )
    if episode["min_hand_gripper_dist_m"] is not None:
        state["recorded_min_hand_gripper_dist_m"] = float(
            episode["min_hand_gripper_dist_m"][sample_idx]
        )
    if episode["gripper_camera_occluded"] is not None:
        state["gripper_camera_occluded"] = float(
            np.clip(episode["gripper_camera_occluded"][sample_idx], 0.0, 1.0)
        )
    return state


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


def _scalar_dataset_or_none(group, paths: tuple[str, ...]) -> np.ndarray | None:
    for path in paths:
        if path in group:
            return np.asarray(group[path], dtype=np.float32).reshape(-1)
    return None


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
