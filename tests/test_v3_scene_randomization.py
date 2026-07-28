import numpy as np

from v3_chan.scene_randomization import (
    episode_rng,
    episode_seed,
    sample_cube_positions,
    scene_layout_id,
)


def _positions(session_seed: int, episode_index: int):
    return sample_cube_positions(
        rng=episode_rng(session_seed, episode_index),
        table_xy=np.array([0.4, 0.0]),
        table_size=np.array([1.2, 0.8, 0.05]),
        cube_size=0.0515,
        count=6,
        forbidden_xy=np.array([0.6, -0.25]),
    )


def test_episode_seed_and_layout_are_reproducible():
    assert episode_seed(42, 3) == episode_seed(42, 3)
    first = np.asarray(_positions(42, 3))
    second = np.asarray(_positions(42, 3))
    np.testing.assert_allclose(first, second)


def test_different_episode_indices_produce_different_layouts():
    assert not np.allclose(_positions(42, 0), _positions(42, 1))


def test_scene_layout_id_changes_with_scene_state():
    positions = np.asarray(_positions(42, 0))
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (len(positions), 1))
    first = scene_layout_id(positions, quaternions, [0.6, -0.25, 0.95], [1, 0, 0, 0])
    positions[0, 0] += 0.01
    second = scene_layout_id(positions, quaternions, [0.6, -0.25, 0.95], [1, 0, 0, 0])
    assert first != second
