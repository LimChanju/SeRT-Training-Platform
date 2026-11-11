from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import v3_chan.collect_bc_cbf_explicit_feedback as collector
from v3_chan.explicit_feedback.live_vr import LiveVRTrackingProvider
from v3_chan.explicit_feedback.online_scenario import CrossingCorridor
from v3_chan.explicit_feedback.schema import (
    EncounterRecord,
    PromptRecord,
    ResponseRecord,
)


def test_study_runs_require_four_practice_trials() -> None:
    parser = collector.build_parser()
    args = parser.parse_args(
        [
            "--participant-id",
            "P001",
            "--session-id",
            "S001",
            "--practice-trials",
            "3",
        ]
    )
    with pytest.raises(ValueError, match="exactly four"):
        collector._validate_args(args)


def test_one_trial_practice_smoke_is_allowed() -> None:
    parser = collector.build_parser()
    args = parser.parse_args(
        [
            "--participant-id",
            "SMOKE",
            "--session-id",
            "SMOKE01",
            "--practice-only",
            "--practice-trials",
            "1",
        ]
    )
    collector._validate_args(args)


def test_success_is_never_written_as_a_task_failure_reason() -> None:
    assert collector._normalized_task_failure_reason(
        {"task_terminal_reason": "success", "step": 20},
        strict_state={"failure_reason": ""},
        task_success=True,
        reset_after_query=False,
        max_episode_steps=100,
    ) == ""


def test_task_horizon_always_has_an_explicit_failure_reason() -> None:
    assert collector._normalized_task_failure_reason(
        {"task_terminal_reason": "", "step": 100},
        strict_state={"failure_reason": ""},
        task_success=False,
        reset_after_query=False,
        max_episode_steps=100,
    ) == "max_episode_steps"


class _StoppedSimulationApp:
    def __init__(self) -> None:
        self.is_running_calls = 0
        self.update_calls = 0

    def is_running(self) -> bool:
        self.is_running_calls += 1
        return False

    def update(self) -> None:
        self.update_calls += 1


def test_corridor_direction_matches_avatar_left_plus_y_convention() -> None:
    base = CrossingCorridor(
        start_world_m=np.asarray([0.4, -0.2, 1.0]),
        end_world_m=np.asarray([0.4, 0.2, 1.0]),
        center_world_m=np.asarray([0.4, 0.0, 1.0]),
        tangent_world=np.asarray([0.0, 1.0, 0.0]),
        outward_normal_world=np.asarray([1.0, 0.0, 0.0]),
        planned_minimum_surface_gap_m=0.1,
        closest_link="panda_hand",
        closest_collider_path="/World/Franka/panda_hand/collisions",
        calibration_iterations=1,
    )

    left_to_right = collector._oriented_corridor(base, "left_to_right")
    right_to_left = collector._oriented_corridor(base, "right_to_left")

    assert left_to_right.start_world_m[1] > left_to_right.end_world_m[1]
    assert right_to_left.start_world_m[1] < right_to_left.end_world_m[1]


def test_tracking_readiness_stops_before_sampling_when_simulation_app_stops(
    monkeypatch,
) -> None:
    provider = LiveVRTrackingProvider(watchdog=SimpleNamespace())
    sample_calls = []
    stop_calls = []
    monkeypatch.setattr(
        provider,
        "refresh_observation_state",
        lambda: sample_calls.append("sample"),
    )
    app = _StoppedSimulationApp()

    with pytest.raises(
        RuntimeError,
        match="SimulationApp stopped during tracking readiness",
    ):
        provider.wait_until_ready(
            app,
            timeout_s=1.0,
            stop_requested=lambda: stop_calls.append("checked"),
        )

    assert stop_calls == ["checked"]
    assert sample_calls == []
    assert app.is_running_calls == 1
    assert app.update_calls == 0


def test_tracking_readiness_propagates_stop_callback_before_sampling(
    monkeypatch,
) -> None:
    provider = LiveVRTrackingProvider(watchdog=SimpleNamespace())
    sample_calls = []
    monkeypatch.setattr(
        provider,
        "refresh_observation_state",
        lambda: sample_calls.append("sample"),
    )

    def stop_requested() -> None:
        raise collector.CollectionAbort("signal_15")

    app = _StoppedSimulationApp()
    with pytest.raises(collector.CollectionAbort, match="signal_15"):
        provider.wait_until_ready(
            app,
            timeout_s=1.0,
            stop_requested=stop_requested,
        )

    assert sample_calls == []
    assert app.is_running_calls == 0
    assert app.update_calls == 0


class _World:
    def __init__(self) -> None:
        self.pause_calls = 0
        self.play_calls = 0
        self.playing = True

    def pause(self) -> None:
        self.pause_calls += 1
        self.playing = False

    def play(self) -> None:
        self.play_calls += 1
        self.playing = True

    def is_playing(self) -> bool:
        return self.playing


class _Recorder:
    def __init__(self) -> None:
        self.encounters = []
        self.prompts = []
        self.responses = []

    def append_encounter(self, value) -> None:
        self.encounters.append(value)

    def append_prompt(self, value) -> None:
        self.prompts.append(value)

    def append_response(self, value) -> None:
        self.responses.append(value)


def _accepted_exchange(*, invoke_idle_callback: bool = False):
    prompt = PromptRecord(
        prompt_id="prompt-1",
        encounter_id="encounter-1",
        question_id="recent_robot_motion_safe_and_acceptable_v1",
        issued_monotonic_ns=100,
        issued_unix_ns=1_000,
        presentation_deadline_monotonic_ns=1_000_000_100,
        presented_monotonic_ns=200,
        deadline_monotonic_ns=1_000_000_200,
        status="accepted",
    )
    response = ResponseRecord(
        response_id="server-response-1",
        client_response_id="client-response-1",
        prompt_id=prompt.prompt_id,
        response_code="acceptable",
        source="browser_ui_v1",
        client_unix_ns=0,
        received_monotonic_ns=300,
        received_unix_ns=1_200,
        accepted=True,
        rejection_reason="",
    )

    class Broker:
        def publish_prompt(self, encounter_id):
            assert encounter_id == prompt.encounter_id
            return prompt

        def wait_for_response(self, prompt_id, **kwargs):
            assert prompt_id == prompt.prompt_id
            assert kwargs["poll_interval_s"] == 0.01
            if invoke_idle_callback:
                kwargs["idle_callback"]()
            return response

        def prompt_record(self, prompt_id):
            assert prompt_id == prompt.prompt_id
            return prompt

        def snapshot(self):
            return (prompt,), (response,)

    return Broker(), prompt, response


def _encounter() -> EncounterRecord:
    return EncounterRecord(
        encounter_id="encounter-1",
        episode_index=0,
        start_step=1,
        end_step_exclusive=2,
        start_monotonic_ns=10,
        end_monotonic_ns=20,
        minimum_surface_gap_m=0.04,
        involved_hand_mask=(1, 0),
        eligible=True,
        eligibility_reason="eligible",
        terminal_status="completed",
    )


def test_terminal_prompt_records_accepted_exchange_without_resuming(monkeypatch) -> None:
    hold_calls = []
    monkeypatch.setattr(
        collector,
        "apply_controlled_hold",
        lambda robot, *, joint_count: hold_calls.append((robot, joint_count)),
    )
    broker, prompt, response = _accepted_exchange()
    world = _World()
    robot = object()
    env = SimpleNamespace(robot=robot, world=world)
    provider = SimpleNamespace(
        wait_until_ready=lambda *args, **kwargs: pytest.fail(
            "terminal prompt must not wait for tracking readiness"
        ),
        refresh_observation_state=lambda: pytest.fail(
            "terminal prompt must not refresh tracking"
        ),
    )
    recorder = _Recorder()
    stop_calls = []

    result = collector._record_prompt_exchange(
        env=env,
        provider=provider,
        broker=broker,
        recorder=recorder,
        encounter=_encounter(),
        joint_count=9,
        tracking_timeout_s=1.0,
        tracking_stable_frames=2,
        clear_confirmation_frames=2,
        simulation_app=SimpleNamespace(),
        stop_requested=lambda: stop_calls.append("stop"),
        abort_reason_fn=lambda: "",
        resume_after_prompt=False,
        recorded_response_ids=set(),
    )

    assert result is None
    assert hold_calls == [(robot, 9)]
    assert world.pause_calls == 1
    assert world.play_calls == 0
    assert stop_calls == []
    assert len(recorder.encounters) == 1
    assert recorder.encounters[0].prompt_id == prompt.prompt_id
    assert recorder.prompts == [prompt]
    assert recorder.responses == [response]


def test_nonterminal_prompt_holds_until_stably_clear_then_resumes(monkeypatch) -> None:
    hold_calls = []
    monkeypatch.setattr(
        collector,
        "apply_controlled_hold",
        lambda robot, *, joint_count: hold_calls.append((robot, joint_count)),
    )
    broker, prompt, response = _accepted_exchange()
    world = _World()
    robot = object()
    snapshots = [object(), object()]
    observations = [
        (
            np.array([1.0, 2.0], dtype=np.float64),
            {
                "geometry_valid": True,
                "left_end_effector_surface_gap_m": 0.14,
                "right_end_effector_surface_gap_m": 0.15,
            },
        ),
        (
            np.array([3.0, 4.0], dtype=np.float64),
            {
                "geometry_valid": True,
                "left_end_effector_surface_gap_m": 0.16,
                "right_end_effector_surface_gap_m": 0.17,
            },
        ),
    ]
    provider = SimpleNamespace(
        observation_snapshot=None,
        watchdog=SimpleNamespace(require_current=lambda snapshot: None),
        readiness_calls=[],
    )

    def wait_until_ready(app, **kwargs):
        provider.readiness_calls.append((app, kwargs))

    provider.wait_until_ready = wait_until_ready

    def refresh_observation():
        index = 2 - len(observations)
        provider.observation_snapshot = snapshots[index]
        return observations.pop(0)

    env = SimpleNamespace(
        robot=robot,
        world=world,
        refresh_observation=refresh_observation,
    )
    app = SimpleNamespace(update_calls=0)

    def update():
        app.update_calls += 1

    app.update = update
    recorder = _Recorder()
    stop_calls = []

    result = collector._record_prompt_exchange(
        env=env,
        provider=provider,
        broker=broker,
        recorder=recorder,
        encounter=_encounter(),
        joint_count=9,
        tracking_timeout_s=1.0,
        tracking_stable_frames=3,
        clear_confirmation_frames=2,
        simulation_app=app,
        stop_requested=lambda: stop_calls.append("stop"),
        abort_reason_fn=lambda: "",
        resume_after_prompt=True,
        recorded_response_ids=set(),
    )

    obs, info, snapshot = result
    np.testing.assert_array_equal(obs, np.array([3.0, 4.0], dtype=np.float32))
    assert info["left_end_effector_surface_gap_m"] == 0.16
    assert snapshot is snapshots[1]
    # The prompt boundary and the post-prompt estimator reseed each establish a
    # hold; an additional hold is refreshed between clear-confirmation frames.
    assert len(hold_calls) >= 2
    assert all(call == (robot, 9) for call in hold_calls)
    assert world.pause_calls >= 1
    assert world.play_calls == 1
    assert app.update_calls == 1
    assert stop_calls == ["stop", "stop", "stop"]
    assert len(provider.readiness_calls) == 1
    _, readiness_kwargs = provider.readiness_calls[0]
    assert readiness_kwargs["timeout_s"] == 1.0
    assert readiness_kwargs["stable_frames"] == 3
    assert readiness_kwargs["stop_requested"] is not None
    assert recorder.prompts == [prompt]
    assert recorder.responses == [response]


def test_prompt_wait_pumps_kit_only_while_world_is_paused(monkeypatch) -> None:
    hold_calls = []
    monkeypatch.setattr(
        collector,
        "apply_controlled_hold",
        lambda robot, *, joint_count: hold_calls.append((robot, joint_count)),
    )
    broker, prompt, response = _accepted_exchange(invoke_idle_callback=True)
    world = _World()
    robot = object()
    env = SimpleNamespace(robot=robot, world=world)
    recorder = _Recorder()
    app = SimpleNamespace(update_calls=0, observed_play_states=[])

    def update() -> None:
        app.update_calls += 1
        app.observed_play_states.append(world.is_playing())

    app.update = update
    stop_calls = []

    result = collector._record_prompt_exchange(
        env=env,
        provider=SimpleNamespace(),
        broker=broker,
        recorder=recorder,
        encounter=_encounter(),
        joint_count=9,
        tracking_timeout_s=1.0,
        tracking_stable_frames=2,
        clear_confirmation_frames=2,
        simulation_app=app,
        stop_requested=lambda: stop_calls.append("stop"),
        abort_reason_fn=lambda: "",
        resume_after_prompt=False,
        recorded_response_ids=set(),
    )

    assert result is None
    assert hold_calls == [(robot, 9)]
    assert world.pause_calls == 1
    assert world.play_calls == 0
    assert app.update_calls == 1
    assert app.observed_play_states == [False]
    assert stop_calls == ["stop", "stop"]
    assert recorder.prompts == [prompt]
    assert recorder.responses == [response]


def test_response_audit_snapshot_is_drained_exactly_once() -> None:
    first = ResponseRecord(
        response_id="server-response-1",
        client_response_id="client-response-1",
        prompt_id="prompt-1",
        response_code="acceptable",
        source="browser_ui_v1",
        client_unix_ns=0,
        received_monotonic_ns=100,
        received_unix_ns=1_000,
        accepted=True,
        rejection_reason="",
    )
    second = ResponseRecord(
        response_id="server-response-2",
        client_response_id="client-response-2",
        prompt_id="unknown-prompt",
        response_code="unacceptable",
        source="browser_ui_v1",
        client_unix_ns=0,
        received_monotonic_ns=200,
        received_unix_ns=1_100,
        accepted=False,
        rejection_reason="unknown_prompt_id",
    )
    third = ResponseRecord(
        response_id="server-response-3",
        client_response_id="client-response-3",
        prompt_id="prompt-1",
        response_code="unsure",
        source="browser_ui_v1",
        client_unix_ns=0,
        received_monotonic_ns=300,
        received_unix_ns=1_200,
        accepted=False,
        rejection_reason="prompt_already_answered",
    )

    class Broker:
        responses = [first, second]

        def snapshot(self):
            return (), tuple(self.responses)

    broker = Broker()
    recorder = _Recorder()
    recorded_response_ids = {first.response_id}

    collector._append_unrecorded_responses(
        broker=broker,
        recorder=recorder,
        recorded_response_ids=recorded_response_ids,
    )
    collector._append_unrecorded_responses(
        broker=broker,
        recorder=recorder,
        recorded_response_ids=recorded_response_ids,
    )
    broker.responses.append(third)
    collector._append_unrecorded_responses(
        broker=broker,
        recorder=recorder,
        recorded_response_ids=recorded_response_ids,
    )

    assert recorder.responses == [second, third]
    assert recorded_response_ids == {
        first.response_id,
        second.response_id,
        third.response_id,
    }


class _CommitRecorder:
    def __init__(self, directory) -> None:
        self.partial_path = directory / "collection.hdf5.partial"
        self.output_path = directory / "collection.hdf5"
        self.partial_path.write_bytes(b"sealed-test-artifact")

    def commit_validated(self):
        return self.partial_path.replace(self.output_path)


def test_validated_commit_declares_an_atomic_signal_boundary(
    tmp_path,
    monkeypatch,
) -> None:
    recorder = _CommitRecorder(tmp_path)
    mask_calls = []
    monkeypatch.setattr(
        collector.signal,
        "pthread_sigmask",
        lambda operation, mask: mask_calls.append((operation, set(mask))) or set(),
    )
    monkeypatch.setattr(collector.signal, "sigpending", lambda: set())
    commit_state = {"value": False}

    result = collector._commit_validated_with_signal_barrier(
        recorder=recorder,
        abort_reason_fn=lambda: "",
        commit_state=commit_state,
    )

    assert result == recorder.output_path
    assert recorder.output_path.read_bytes() == b"sealed-test-artifact"
    assert not recorder.partial_path.exists()
    assert commit_state["value"] is True
    assert mask_calls == [
        (
            collector.signal.SIG_BLOCK,
            {collector.signal.SIGINT, collector.signal.SIGTERM},
        ),
        (collector.signal.SIG_SETMASK, set()),
    ]


def test_signal_pending_during_atomic_commit_restores_partial(
    tmp_path,
    monkeypatch,
) -> None:
    recorder = _CommitRecorder(tmp_path)
    pending = iter((set(), {collector.signal.SIGINT}))
    monkeypatch.setattr(
        collector.signal,
        "pthread_sigmask",
        lambda _operation, _mask: set(),
    )
    monkeypatch.setattr(collector.signal, "sigpending", lambda: next(pending))
    commit_state = {"value": False}

    with pytest.raises(collector.CollectionAbort, match="signal_2"):
        collector._commit_validated_with_signal_barrier(
            recorder=recorder,
            abort_reason_fn=lambda: "",
            commit_state=commit_state,
        )

    assert recorder.partial_path.read_bytes() == b"sealed-test-artifact"
    assert not recorder.output_path.exists()
    assert commit_state["value"] is False


def test_stop_reason_set_between_commit_checks_restores_partial(
    tmp_path,
    monkeypatch,
) -> None:
    recorder = _CommitRecorder(tmp_path)
    monkeypatch.setattr(
        collector.signal,
        "pthread_sigmask",
        lambda _operation, _mask: set(),
    )
    monkeypatch.setattr(collector.signal, "sigpending", lambda: set())
    reason_calls = [0]

    def abort_reason() -> str:
        reason_calls[0] += 1
        return "signal_15" if reason_calls[0] == 3 else ""

    commit_state = {"value": False}
    with pytest.raises(collector.CollectionAbort, match="signal_15"):
        collector._commit_validated_with_signal_barrier(
            recorder=recorder,
            abort_reason_fn=abort_reason,
            commit_state=commit_state,
        )

    assert reason_calls[0] == 3
    assert recorder.partial_path.read_bytes() == b"sealed-test-artifact"
    assert not recorder.output_path.exists()
    assert commit_state["value"] is False
