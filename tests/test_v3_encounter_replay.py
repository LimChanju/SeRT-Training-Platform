import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


_RL_DIR = Path(__file__).resolve().parents[1] / "v3_chan" / "rl"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _RL_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_manifest = _load_module(
    "_v3_chan_encounter_manifest_test",
    "encounter_manifest.py",
)
_replay = _load_module(
    "_v3_chan_human_encounter_replay_test",
    "human_replay.py",
)


def _write_source(path: Path) -> None:
    length = 220
    gap = np.full(length, 0.20, dtype=np.float32)
    contact = np.zeros(length, dtype=np.float32)
    gap[50:60] = 0.10
    gap[60:70] = 0.04
    gap[70:75] = 0.01
    gap[75] = -0.005
    contact[75] = 1.0
    gap[76:82] = 0.04
    gap[145:160] = 0.10

    cube_index = np.zeros(length, dtype=np.int32)
    cube_index[120:] = 1
    attempt = np.zeros(length, dtype=np.int32)
    controller_event = np.zeros(length, dtype=np.int32)
    controller_event[70:82] = 2
    controller_event[120:] = 4
    task_phase = np.zeros((length, 4), dtype=np.float32)
    task_phase[:120, 0] = 1.0
    task_phase[70:82, 0] = 0.0
    task_phase[70:82, 1] = 1.0
    task_phase[120:, 2] = 1.0
    ee_pos = np.stack(
        (
            np.linspace(0.3, 0.6, length),
            np.zeros(length),
            np.full(length, 0.8),
        ),
        axis=1,
    ).astype(np.float32)
    left = ee_pos + np.array([0.1, 0.0, 0.0], dtype=np.float32)
    right = ee_pos + np.array([-0.1, 0.0, 0.0], dtype=np.float32)
    head = ee_pos + np.array([0.0, -0.5, 0.5], dtype=np.float32)

    with h5py.File(path, "w") as h5_file:
        episode = h5_file.create_group("episodes/episode_000000")
        episode.attrs["session_id"] = "session_00"
        episode.create_dataset(
            "sim_time",
            data=np.arange(length, dtype=np.float32) / 60.0,
        )
        human = episode.create_group("human")
        human.create_dataset("head_pos", data=head)
        human.create_dataset("left_hand_pos", data=left)
        human.create_dataset("right_hand_pos", data=right)
        human.create_dataset(
            "valid_mask",
            data=np.ones((length, 3), dtype=np.float32),
        )
        safety = episode.create_group("safety")
        safety.create_dataset(
            "min_hand_end_effector_surface_gap_m",
            data=gap,
        )
        safety.create_dataset("contact_active", data=contact)
        task = episode.create_group("task")
        task.create_dataset("current_pick_idx", data=cube_index)
        task.create_dataset("attempt_index", data=attempt)
        task.create_dataset("controller_event", data=controller_event)
        obs = episode.create_group("obs")
        obs.create_dataset("task_phase", data=task_phase)
        obs.create_dataset("ee_pos", data=ee_pos)


def test_manifest_segments_encounters_without_crossing_cube_boundaries(tmp_path):
    source = tmp_path / "session_00.hdf5"
    output = tmp_path / "train_manifest.json"
    _write_source(source)
    config = _manifest.EncounterBuildConfig(
        onset_frames=3,
        clear_frames=5,
        margin_frames=4,
        safe_window_frames=20,
        safe_stride_frames=20,
        safe_min_frames=20,
        max_safe_per_episode=2,
    )

    manifest = _manifest.build_encounter_manifest(
        [str(source)],
        str(output),
        config=config,
    )
    assert manifest["schema_version"] == "hri_encounter_manifest_v2"
    assert manifest["phase_anchor_policy"] == "minimum_surface_gap_in_core"

    severities = {
        scenario["target_severity"] for scenario in manifest["scenarios"]
    }
    assert "collision" in severities
    assert "gate_only" in severities
    assert "safe" in severities
    for scenario in manifest["scenarios"]:
        if scenario["cube_index"] == 0:
            assert scenario["end_step"] <= 120
        elif scenario["cube_index"] == 1:
            assert scenario["start_step"] >= 120
        else:
            raise AssertionError(scenario)

    collision = next(
        scenario
        for scenario in manifest["scenarios"]
        if scenario["target_severity"] == "collision"
    )
    assert collision["start_step"] < 70
    assert collision["phase_anchor_step"] == 75
    assert collision["source_anchor_step"] == collision["start_step"]
    assert collision["task_phase"] == "grasp_cube"
    assert collision["controller_event"] == 2
    assert collision["trigger_task_phase"] == "approach_cube"
    assert collision["trigger_controller_event"] == 0


def test_encounter_replay_waits_for_phase_and_applies_fixed_ee_anchor(tmp_path):
    source = tmp_path / "session_00.hdf5"
    output = tmp_path / "eval_manifest.json"
    _write_source(source)
    _manifest.build_encounter_manifest(
        [str(source)],
        str(output),
        config=_manifest.EncounterBuildConfig(
            onset_frames=3,
            clear_frames=5,
            margin_frames=4,
            safe_window_frames=20,
            safe_stride_frames=20,
            safe_min_frames=20,
            max_safe_per_episode=1,
        ),
    )
    replay = _replay.HumanEncounterReplay(
        str(output),
        episode_policy="cycle",
        anchor_mode="ee",
        phase_match=True,
        event_match=True,
        seed=7,
    )
    try:
        scenario = replay.current_scenario
        assert scenario["target_severity"] == "collision"
        replay.set_runtime_context(
            step=0,
            task_phase="move_to_target",
            controller_event=4,
            controller_t=0,
            ee_pos=np.array([1.0, 2.0, 3.0]),
        )
        waiting = replay()
        assert waiting["encounter_active"] == 0.0
        assert "human_left_hand_pos" not in waiting

        expected_phase = scenario["trigger_task_phase"]
        expected_event = scenario["trigger_controller_event"]
        current_anchor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        replay.set_runtime_context(
            step=1,
            task_phase=expected_phase,
            controller_event=expected_event,
            controller_t=0,
            ee_pos=current_anchor,
        )
        state = replay()
        assert state["encounter_active"] == 1.0
        source_anchor = np.asarray(
            scenario["source_anchor_ee_pos"],
            dtype=np.float32,
        )
        expected_offset = current_anchor - source_anchor
        source_step = int(scenario["start_step"])
        with h5py.File(source, "r") as h5_file:
            source_left = np.asarray(
                h5_file[
                    f"episodes/{scenario['episode_name']}/human/left_hand_pos"
                ][source_step],
                dtype=np.float32,
            )
        np.testing.assert_allclose(
            state["human_left_hand_pos"],
            source_left + expected_offset,
            atol=1e-6,
        )
    finally:
        replay.close()
