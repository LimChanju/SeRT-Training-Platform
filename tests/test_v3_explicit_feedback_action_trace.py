from types import SimpleNamespace

import numpy as np
import pytest

from v3_chan.explicit_feedback.action_trace import (
    ActionTraceRecorder,
    canonicalize_joint_action,
)


PANDA_JOINTS = tuple(
    [f"panda_joint{index}" for index in range(1, 8)]
    + ["panda_finger_joint1", "panda_finger_joint2"]
)


def _action(*, positions=None, velocities=None, efforts=None, indices=None):
    return SimpleNamespace(
        joint_positions=positions,
        joint_velocities=velocities,
        joint_efforts=efforts,
        joint_indices=indices,
    )


class _Clock:
    def __init__(self):
        self.value = 100

    def __call__(self):
        self.value += 10
        return self.value


class _Robot:
    def __init__(self):
        self.dof_names = PANDA_JOINTS
        self.positions = np.linspace(0.0, 0.8, len(PANDA_JOINTS))
        self.velocities = np.zeros(len(PANDA_JOINTS), dtype=float)
        self.apply_call_count = 0

    def get_applied_action(self):
        return _action(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            efforts=None,
        )

    def apply_action(self, control_actions):
        self.apply_call_count += 1
        indices = control_actions.joint_indices
        fields = (
            ("joint_positions", self.positions),
            ("joint_velocities", self.velocities),
        )
        for field_name, target in fields:
            values = getattr(control_actions, field_name, None)
            if values is None:
                continue
            values = np.asarray(values, dtype=object).reshape(-1)
            mapped_indices = (
                np.arange(values.size, dtype=int)
                if indices is None
                else np.asarray(indices, dtype=int).reshape(-1)
            )
            for source_index, robot_index in enumerate(mapped_indices):
                value = values[source_index]
                if value is not None and not np.isnan(float(value)):
                    target[robot_index] = float(value)


class _Controller:
    def __init__(self):
        self.last_action = None

    def forward(self):
        self.last_action = _action(
            positions=np.arange(7, dtype=float) + 1.0,
            velocities=np.arange(7, dtype=float) + 0.1,
            indices=np.arange(7, dtype=int),
        )
        return self.last_action


class _MutatingCBF:
    def filter_action(self, *, arm_action):
        arm_action.joint_positions += 10.0
        arm_action.joint_velocities *= -1.0
        return arm_action, {"status": "filtered"}


def _installed_trace(robot=None):
    robot = _Robot() if robot is None else robot
    controller = _Controller()
    cbf = _MutatingCBF()
    recorder = ActionTraceRecorder(
        robot=robot,
        controller=controller,
        cbf=cbf,
        clock_ns=_Clock(),
    ).install()
    return recorder, robot, controller, cbf


def test_canonicalize_subset_uses_robot_joint_names_and_masks():
    canonical = canonicalize_joint_action(
        _action(
            positions=[1.25, -2.5],
            velocities=[None, 0.4],
            indices=[1, 7],
        ),
        PANDA_JOINTS,
    )

    assert canonical.joint_positions.shape == (9,)
    np.testing.assert_array_equal(
        canonical.joint_positions_mask,
        [False, True, False, False, False, False, False, True, False],
    )
    np.testing.assert_allclose(canonical.joint_positions[[1, 7]], [1.25, -2.5])
    np.testing.assert_array_equal(
        canonical.joint_velocities_mask,
        [False, False, False, False, False, False, False, True, False],
    )
    assert not canonical.joint_positions.flags.writeable
    assert not canonical.joint_positions_mask.flags.writeable
    np.testing.assert_array_equal(canonical.raw_joint_indices, [1, 7])
    assert not canonical.raw_joint_indices.flags.writeable


def test_canonicalize_full_sparse_gripper_action_does_not_command_arm():
    sparse_positions = [None] * 7 + [0.02, 0.02]
    canonical = canonicalize_joint_action(
        _action(positions=sparse_positions),
        PANDA_JOINTS,
    )

    np.testing.assert_array_equal(
        canonical.joint_positions_mask,
        [False, False, False, False, False, False, False, True, True],
    )
    np.testing.assert_allclose(canonical.joint_positions[-2:], [0.02, 0.02])
    assert not np.any(canonical.joint_velocities_mask)
    assert canonical.raw_joint_indices is None


def test_trace_snapshots_nominal_before_in_place_cbf_mutation():
    recorder, robot, controller, cbf = _installed_trace()
    pre_positions = robot.positions.copy()
    recorder.start_step(step_id=17)

    nominal_action = controller.forward()
    filtered_action, diagnostics = cbf.filter_action(arm_action=nominal_action)
    robot.apply_action(filtered_action)
    trace = recorder.finish_step()

    assert diagnostics == {"status": "filtered"}
    assert filtered_action is nominal_action
    assert trace.step_id == 17
    assert trace.joint_names == PANDA_JOINTS
    assert trace.pipeline_complete is True
    assert trace.arm_action_submitted is True
    assert trace.cbf_filtered is trace.filtered
    assert trace.nominal_rmpflow is trace.nominal
    np.testing.assert_allclose(trace.nominal.joint_positions[:7], np.arange(7) + 1.0)
    np.testing.assert_allclose(trace.filtered.joint_positions[:7], np.arange(7) + 11.0)
    np.testing.assert_allclose(trace.submitted.joint_positions[:7], np.arange(7) + 11.0)
    np.testing.assert_allclose(trace.pre_applied.joint_positions, pre_positions)
    np.testing.assert_allclose(
        trace.post_applied.joint_positions[:7], np.arange(7) + 11.0
    )
    np.testing.assert_allclose(
        trace.post_applied.joint_positions[-2:], pre_positions[-2:]
    )
    assert robot.apply_call_count == 1
    timestamps = list(trace.timestamps_monotonic_ns.values())
    assert trace.rmpflow_output_monotonic_ns == trace.timestamps_monotonic_ns["nominal"]
    assert trace.cbf_output_monotonic_ns == trace.timestamps_monotonic_ns["filtered"]
    assert (
        trace.action_command_monotonic_ns
        == trace.timestamps_monotonic_ns["apply_started"]
    )

    recorder.uninstall()


def test_trace_preserves_filtered_arm_but_records_submitted_gripper_arbitration():
    recorder, robot, controller, cbf = _installed_trace()
    pre_positions = robot.positions.copy()
    recorder.start_step(step_id="gripper-close")

    nominal_action = controller.forward()
    cbf.filter_action(arm_action=nominal_action)
    gripper_only = _action(
        positions=[None] * 7 + [0.0, 0.0],
        velocities=[None] * 9,
        efforts=[None] * 9,
    )
    robot.apply_action(gripper_only)
    trace = recorder.finish_step()

    assert np.all(trace.filtered.joint_positions_mask[:7])
    assert not np.any(trace.submitted.joint_positions_mask[:7])
    assert np.all(trace.submitted.joint_positions_mask[-2:])
    assert trace.arm_action_submitted is False
    np.testing.assert_allclose(
        trace.post_applied.joint_positions[:7], pre_positions[:7]
    )
    np.testing.assert_allclose(trace.post_applied.joint_positions[-2:], [0.0, 0.0])

    recorder.uninstall()


def test_arm_submission_accepts_structural_arm_and_gripper_merge():
    recorder, robot, controller, cbf = _installed_trace()
    recorder.start_step(step_id="merged")

    nominal_action = controller.forward()
    filtered_action, _ = cbf.filter_action(arm_action=nominal_action)
    merged = _action(
        positions=list(filtered_action.joint_positions) + [0.01, 0.01],
        velocities=list(filtered_action.joint_velocities) + [None, None],
        efforts=[None] * 9,
    )
    robot.apply_action(merged)
    trace = recorder.finish_step()

    assert merged is not filtered_action
    assert trace.arm_action_submitted is True
    assert np.all(trace.submitted.joint_positions_mask)
    assert np.all(trace.submitted.joint_velocities_mask[:7])
    assert not np.any(trace.submitted.joint_velocities_mask[-2:])

    recorder.uninstall()


def test_finish_step_rejects_incomplete_trace_and_keeps_it_abortable():
    recorder, _, controller, _ = _installed_trace()
    recorder.start_step(step_id=3)
    controller.forward()

    with pytest.raises(
        RuntimeError,
        match="missing stages: filtered, submitted, post_applied",
    ):
        recorder.finish_step()
    assert recorder.step_active is True

    recorder.abort_step()
    assert recorder.step_active is False
    recorder.uninstall()


def test_apply_before_cbf_capture_is_fail_closed():
    recorder, robot, controller, _ = _installed_trace()
    recorder.start_step()
    nominal_action = controller.forward()

    with pytest.raises(RuntimeError, match="before nominal and filtered"):
        robot.apply_action(nominal_action)
    assert robot.apply_call_count == 0

    recorder.abort_step()
    recorder.uninstall()


def test_uninstall_restores_original_boundaries():
    robot = _Robot()
    controller = _Controller()
    cbf = _MutatingCBF()
    original_apply_func = robot.apply_action.__func__
    original_forward_func = controller.forward.__func__
    original_filter_func = cbf.filter_action.__func__
    recorder = ActionTraceRecorder(
        robot=robot,
        controller=controller,
        cbf=cbf,
    ).install()

    recorder.uninstall()

    assert robot.apply_action.__func__ is original_apply_func
    assert controller.forward.__func__ is original_forward_func
    assert cbf.filter_action.__func__ is original_filter_func


def test_unwrappable_apply_action_aborts_install_without_partial_hooks():
    class SlottedRobot:
        __slots__ = ("dof_names", "positions", "velocities")

        def __init__(self):
            self.dof_names = ("joint_a", "joint_b")
            self.positions = np.zeros(2)
            self.velocities = np.zeros(2)

        def get_applied_action(self):
            return _action(
                positions=self.positions,
                velocities=self.velocities,
            )

        def apply_action(self, action):
            raise AssertionError("must never be reached")

    robot = SlottedRobot()
    controller = _Controller()
    cbf = _MutatingCBF()
    original_forward_func = controller.forward.__func__
    original_filter_func = cbf.filter_action.__func__
    recorder = ActionTraceRecorder(
        robot=robot,
        controller=controller,
        cbf=cbf,
    )

    with pytest.raises(
        RuntimeError, match="fail-closed wrapper for robot.apply_action"
    ):
        recorder.install()

    assert recorder.installed is False
    assert controller.forward.__func__ is original_forward_func
    assert cbf.filter_action.__func__ is original_filter_func
