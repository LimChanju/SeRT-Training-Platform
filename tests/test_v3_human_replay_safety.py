import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


_REPLAY_PATH = (
    Path(__file__).resolve().parents[1] / "v3_chan" / "rl" / "human_replay.py"
)
_REPLAY_SPEC = importlib.util.spec_from_file_location(
    "_v3_chan_human_replay_test", _REPLAY_PATH
)
if _REPLAY_SPEC is None or _REPLAY_SPEC.loader is None:
    raise ImportError(f"Could not load human replay module from {_REPLAY_PATH}")
_replay_module = importlib.util.module_from_spec(_REPLAY_SPEC)
sys.modules[_REPLAY_SPEC.name] = _replay_module
_REPLAY_SPEC.loader.exec_module(_replay_module)
HumanTrajectoryReplay = _replay_module.HumanTrajectoryReplay


def test_recorded_safety_labels_are_metadata_not_current_rollout_labels(tmp_path):
    path = tmp_path / "legacy_replay.hdf5"
    with h5py.File(path, "w") as data:
        episode = data.create_group("episodes/episode_000000")
        human = episode.create_group("human")
        human.create_dataset("head_pos", data=np.array([[1.0, 0.0, 1.5]]))
        human.create_dataset("left_hand_pos", data=np.array([[0.4, 0.1, 1.0]]))
        human.create_dataset("right_hand_pos", data=np.array([[0.5, -0.1, 1.0]]))
        obs = episode.create_group("obs")
        obs.create_dataset("human_robot_collision", data=np.array([[1.0]]))
        obs.create_dataset("near_human", data=np.array([[1.0]]))

    replay = HumanTrajectoryReplay(str(path))
    try:
        state = replay.peek()
    finally:
        replay.close()

    assert "human_robot_collision" not in state
    assert "near_human" not in state
    assert state["recorded_human_robot_collision"] is True
    assert state["recorded_near_human"] is True

    assert "human_left_hand_pos" in state
    assert "human_right_hand_pos" in state
