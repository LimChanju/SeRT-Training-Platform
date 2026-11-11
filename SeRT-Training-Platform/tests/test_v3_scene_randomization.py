import numpy as np

from v3_chan.scene_randomization import (
    episode_rng,
    episode_seed,
    resolve_active_cube_index,
    restore_cube_poses,
    restored_pose_errors,
    sample_cube_positions,
    scene_layout_id,
)


class _FakeCube:
    def __init__(self, name: str):
        self.name = name
        self.position = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.default_state = None

    def set_default_state(self, **kwargs):
        self.default_state = kwargs

    def set_world_pose(self, *, position, orientation):
        self.position = np.asarray(position, dtype=float).copy()
        self.orientation = np.asarray(orientation, dtype=float).copy()

    def get_world_pose(self):
        return self.position.copy(), self.orientation.copy()

    def set_linear_velocity(self, value):
        pass

    def set_angular_velocity(self, value):
        pass


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


def test_all_benchmark_eval_seeds_produce_valid_layouts():
    for eval_seed in (11, 1011, 2011):
        for episode_index in range(159):
            positions = sample_cube_positions(
                rng=np.random.default_rng(eval_seed + episode_index),
                table_xy=np.array([0.4, 0.0]),
                table_size=np.array([1.2, 0.8, 0.05]),
                cube_size=0.0515,
                count=6,
                forbidden_xy=np.array([0.6, -0.25]),
            )
            assert len(positions) == 6


def test_scene_layout_id_changes_with_scene_state():
    positions = np.asarray(_positions(42, 0))
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (len(positions), 1))
    first = scene_layout_id(positions, quaternions, [0.6, -0.25, 0.95], [1, 0, 0, 0])
    positions[0, 0] += 0.01
    second = scene_layout_id(positions, quaternions, [0.6, -0.25, 0.95], [1, 0, 0, 0])
    assert first != second


def test_exact_cube_poses_are_restored_by_identity():
    cubes = [_FakeCube("cube_1"), _FakeCube("cube_0")]
    names = ["cube_0", "cube_1"]
    positions = np.asarray([[0.3, -0.1, 0.95], [0.5, 0.1, 0.95]])
    orientations = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2)

    mapping = restore_cube_poses(cubes, names, positions, orientations)
    position_error, orientation_error = restored_pose_errors(
        cubes,
        names,
        positions,
        orientations,
    )

    assert mapping == {"cube_0": 1, "cube_1": 0}
    assert position_error == 0.0
    assert orientation_error == 0.0
    np.testing.assert_allclose(cubes[1].default_state["position"], positions[0])


def test_source_cube_index_is_independent_of_screen_episode_index():
    assert resolve_active_cube_index(2, episode_index=0, cube_count=3) == 2
    assert resolve_active_cube_index(2, episode_index=101, cube_count=3) == 2
    assert resolve_active_cube_index(None, episode_index=101, cube_count=3) == 2
