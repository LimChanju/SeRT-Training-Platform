from types import SimpleNamespace

import numpy as np
import pytest

from v3_chan.explicit_feedback.action_trace import ActionTraceRecorder
from v3_chan.explicit_feedback.tracking import TrackingDropout
from v3_chan.explicit_feedback.tracking_guard import PreApplyTrackingGuard


JOINT_NAMES = ("joint_a", "joint_b", "joint_c", "joint_d")


def _action(*, positions=None, velocities=None, efforts=None, indices=None):
    return SimpleNamespace(
        joint_positions=positions,
        joint_velocities=velocities,
        joint_efforts=efforts,
        joint_indices=indices,
    )


class _Clock:
    def __init__(self):
        self.value = 1_000

    def __call__(self):
        self.value += 10
        return self.value


class _Robot:
    def __init__(self):
        self.dof_names = JOINT_NAMES
        self.measured_positions = np.array([-0.4, -0.1, 0.2, 0.5])
        self.applied_positions = np.array([0.1, 0.2, 0.3, 0.4])
        self.applied_velocities = np.array([0.01, 0.02, 0.03, 0.04])
        self.applied_efforts = np.zeros(4, dtype=float)
        self.physical_commands = []

    def get_joint_positions(self):
        return self.measured_positions.copy()

    def get_applied_action(self):
        return _action(
            positions=self.applied_positions.copy(),
            velocities=self.applied_velocities.copy(),
            efforts=self.applied_efforts.copy(),
        )

    def apply_action(self, control_actions):
        self.physical_commands.append(control_actions)
        indices = getattr(control_actions, "joint_indices", None)
        for field_name, target in (
            ("joint_positions", self.applied_positions),
            ("joint_velocities", self.applied_velocities),
            ("joint_efforts", self.applied_efforts),
        ):
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
                if value is None:
                    continue
                numeric = float(value)
                if not np.isnan(numeric):
                    target[robot_index] = numeric


class _Controller:
    def __init__(self, robot):
        self.robot = robot

    def forward(self):
        return _action(
            positions=self.robot.measured_positions + 1.0,
            velocities=np.full(len(JOINT_NAMES), 0.25),
        )


class _CBF:
    def filter_action(self, *, arm_action):
        return arm_action, {"status": "valid"}


class _Provider:
    def __init__(self, outcome):
        self.outcome = outcome
        self.observation_snapshot = (
            object() if isinstance(outcome, BaseException) else outcome
        )
        self.calls = 0

    def sample(self):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Watchdog:
    def __init__(self, *, invalid=False):
        self.invalid = invalid
        self.require_current_calls = 0
        self.require_valid_calls = 0

    def require_current(self, snapshot):
        self.require_current_calls += 1
        if self.invalid:
            raise TrackingDropout(("left:pose_invalid",), snapshot)

    def require_valid(self, snapshot):
        self.require_valid_calls += 1
        raise AssertionError("require_current must be preferred when available")


class _SamplingFailure(RuntimeError):
    pass


def _installed_stack(*, provider, watchdog, abort_reason_fn=None):
    robot = _Robot()
    controller = _Controller(robot)
    cbf = _CBF()
    trace = ActionTraceRecorder(
        robot=robot,
        controller=controller,
        cbf=cbf,
        clock_ns=_Clock(),
    ).install()
    guard = PreApplyTrackingGuard(
        robot=robot,
        provider=provider,
        watchdog=watchdog,
        trace=trace,
        abort_reason_fn=abort_reason_fn,
    ).install()
    return robot, controller, cbf, trace, guard


def _begin_filtered_action(trace, controller, cbf):
    trace.start_step(step_id=7)
    nominal = controller.forward()
    filtered, _ = cbf.filter_action(arm_action=nominal)
    return filtered


def _apply_then_advance_physics(robot, action, physics_steps):
    robot.apply_action(action)
    physics_steps.append("world.step")


def _assert_full_hold(command, measured_positions):
    positions = np.asarray(command.joint_positions, dtype=float).reshape(-1)
    velocities = np.asarray(command.joint_velocities, dtype=float).reshape(-1)
    assert positions.shape == (len(JOINT_NAMES),)
    assert velocities.shape == (len(JOINT_NAMES),)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(velocities))
    np.testing.assert_allclose(positions, measured_positions)
    np.testing.assert_allclose(velocities, 0.0)


def _assert_traced_hold(trace, measured_positions):
    result = trace.finish_step()
    assert result.pipeline_complete is True
    assert result.arm_action_submitted is False
    assert result.submitted.raw_joint_indices is None
    assert np.all(result.submitted.joint_positions_mask)
    assert np.all(result.submitted.joint_velocities_mask)
    np.testing.assert_allclose(
        result.submitted.joint_positions, measured_positions
    )
    np.testing.assert_allclose(result.submitted.joint_velocities, 0.0)
    np.testing.assert_allclose(
        result.post_applied.joint_positions, measured_positions
    )
    np.testing.assert_allclose(result.post_applied.joint_velocities, 0.0)


def test_guard_refuses_install_before_action_trace_wrapper():
    robot = _Robot()
    controller = _Controller(robot)
    trace = ActionTraceRecorder(
        robot=robot,
        controller=controller,
        cbf=_CBF(),
    )
    original_apply = robot.apply_action.__func__
    guard = PreApplyTrackingGuard(
        robot=robot,
        provider=_Provider(object()),
        watchdog=_Watchdog(),
        trace=trace,
    )

    with pytest.raises(RuntimeError, match="install ActionTraceRecorder"):
        guard.install()

    assert robot.apply_action.__func__ is original_apply


def test_valid_pre_apply_sample_passes_intended_action_through_trace_wrapper():
    snapshot = object()
    provider = _Provider(snapshot)
    watchdog = _Watchdog()
    robot, controller, cbf, trace, guard = _installed_stack(
        provider=provider,
        watchdog=watchdog,
    )
    physics_steps = []
    try:
        intended = _begin_filtered_action(trace, controller, cbf)

        _apply_then_advance_physics(robot, intended, physics_steps)
        result = trace.finish_step()

        assert provider.calls == 1
        assert watchdog.require_current_calls == 2
        assert watchdog.require_valid_calls == 0
        assert robot.physical_commands == [intended]
        assert physics_steps == ["world.step"]
        assert result.arm_action_submitted is True
        np.testing.assert_allclose(
            result.submitted.joint_positions, intended.joint_positions
        )
    finally:
        guard.uninstall()
        trace.uninstall()


def test_invalid_pre_apply_sample_submits_verified_hold_then_raises_before_physics():
    snapshot = object()
    provider = _Provider(snapshot)
    watchdog = _Watchdog(invalid=True)
    robot, controller, cbf, trace, guard = _installed_stack(
        provider=provider,
        watchdog=watchdog,
    )
    physics_steps = []
    measured_positions = robot.measured_positions.copy()
    try:
        intended = _begin_filtered_action(trace, controller, cbf)

        with pytest.raises(TrackingDropout, match="left:pose_invalid"):
            _apply_then_advance_physics(robot, intended, physics_steps)

        assert provider.calls == 1
        assert watchdog.require_current_calls == 1
        assert len(robot.physical_commands) == 1
        assert robot.physical_commands[0] is not intended
        assert physics_steps == []
        _assert_full_hold(robot.physical_commands[0], measured_positions)
        _assert_traced_hold(trace, measured_positions)
    finally:
        guard.uninstall()
        trace.uninstall()


def test_sampling_exception_submits_verified_hold_then_propagates_before_physics():
    failure = _SamplingFailure("provider transport failed")
    provider = _Provider(failure)
    watchdog = _Watchdog()
    robot, controller, cbf, trace, guard = _installed_stack(
        provider=provider,
        watchdog=watchdog,
    )
    physics_steps = []
    measured_positions = robot.measured_positions.copy()
    try:
        intended = _begin_filtered_action(trace, controller, cbf)

        with pytest.raises(_SamplingFailure, match="provider transport failed"):
            _apply_then_advance_physics(robot, intended, physics_steps)

        assert provider.calls == 1
        assert watchdog.require_current_calls == 0
        assert len(robot.physical_commands) == 1
        assert robot.physical_commands[0] is not intended
        assert physics_steps == []
        _assert_full_hold(robot.physical_commands[0], measured_positions)
        _assert_traced_hold(trace, measured_positions)
    finally:
        guard.uninstall()
        trace.uninstall()


def test_missing_control_observation_snapshot_holds_before_physics():
    provider = _Provider(object())
    provider.observation_snapshot = None
    watchdog = _Watchdog()
    robot, controller, cbf, trace, guard = _installed_stack(
        provider=provider,
        watchdog=watchdog,
    )
    physics_steps = []
    measured_positions = robot.measured_positions.copy()
    try:
        intended = _begin_filtered_action(trace, controller, cbf)

        with pytest.raises(RuntimeError, match="no tracking provenance"):
            _apply_then_advance_physics(robot, intended, physics_steps)

        assert physics_steps == []
        _assert_full_hold(robot.physical_commands[0], measured_positions)
        _assert_traced_hold(trace, measured_positions)
    finally:
        guard.uninstall()
        trace.uninstall()


def test_abort_request_submits_traced_full_hold_then_raises_before_sampling():
    provider = _Provider(object())
    watchdog = _Watchdog()
    robot, controller, cbf, trace, guard = _installed_stack(
        provider=provider,
        watchdog=watchdog,
        abort_reason_fn=lambda: "signal_2",
    )
    physics_steps = []
    measured_positions = robot.measured_positions.copy()
    try:
        intended = _begin_filtered_action(trace, controller, cbf)

        with pytest.raises(
            RuntimeError,
            match="control aborted before articulation apply: signal_2",
        ):
            _apply_then_advance_physics(robot, intended, physics_steps)

        assert provider.calls == 0
        assert watchdog.require_current_calls == 0
        assert len(robot.physical_commands) == 1
        assert robot.physical_commands[0] is not intended
        assert physics_steps == []
        _assert_full_hold(robot.physical_commands[0], measured_positions)
        _assert_traced_hold(trace, measured_positions)
    finally:
        guard.uninstall()
        trace.uninstall()
