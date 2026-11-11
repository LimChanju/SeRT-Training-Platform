from __future__ import annotations

from dataclasses import replace

import pytest

from v3_chan.explicit_feedback.xr_feedback import (
    LIKERT_QUESTIONS_KO,
    Q1_RESPONSE_CODES,
    Q1_TEXT_KO,
    REALTIME_BEHAVIOR_ANOMALY,
    REALTIME_SAFETY_CONCERN,
    REJECTION_REASON_IDS,
    ButtonEdgeBank,
    ControllerButtonSnapshot,
    DebouncedRisingEdge,
    FeedbackClocks,
    IsaacXRButtonSource,
    QuestionnaireFSM,
    QuestionnaireButtonMapping,
    RealtimeButtonMapping,
    RealtimeMarkerInput,
    XRFeedbackController,
    XRSceneViewTextRenderer,
)


def _clocks(frame: int, *, base_ns: int = 1_000_000_000) -> FeedbackClocks:
    return FeedbackClocks(
        sim_time_s=frame * 0.02,
        monotonic_ns=base_ns + frame * 20_000_000,
        unix_ns=1_800_000_000_000_000_000 + frame * 20_000_000,
        control_step=frame,
    )


def _released() -> ControllerButtonSnapshot:
    return ControllerButtonSnapshot(
        left_connected=True,
        right_connected=True,
        left_x=False,
        left_y=False,
        right_a=False,
        right_b=False,
    )


def _pressed(key: str) -> ControllerButtonSnapshot:
    hand, button = key.split(".")
    return replace(_released(), **{f"{hand}_{button}": True})


class _ControllerDriver:
    def __init__(self, controller: XRFeedbackController, *, frame: int = 0) -> None:
        self.controller = controller
        self.frame = frame

    @property
    def clocks(self) -> FeedbackClocks:
        return _clocks(self.frame)

    def update(self, snapshot: ControllerButtonSnapshot):
        self.frame += 1
        return self.controller.update(snapshot, _clocks(self.frame))

    def release(self, count: int = 3) -> None:
        for _ in range(count):
            self.update(_released())

    def press(self, key: str):
        update = self.update(_pressed(key))
        self.release()
        return update


def test_v1_questionnaire_wording_and_codes_are_frozen() -> None:
    assert Q1_TEXT_KO == "같은 상황에서 방금 로봇 반응을 다시 사용한다면 수정이 필요합니까?"
    assert Q1_RESPONSE_CODES == (
        "needs_modification",
        "acceptable_as_is",
        "uncertain",
    )
    assert LIKERT_QUESTIONS_KO == {
        "q2": "방금 로봇 반응에서 손과 충돌하거나 지나치게 가까워질 위험을 얼마나 느꼈습니까?",
        "q3": "회피의 크기나 지속시간이 필요 이상으로 과도했다고 느꼈습니까?",
        "q4": "로봇 움직임이 갑작스럽거나 예측하기 어렵다고 느꼈습니까?",
        "q5": "방금 회피가 grasp/place 등 원래 작업 진행을 얼마나 방해했다고 느꼈습니까?",
        "q6": "방금 판단에 얼마나 확신합니까?",
    }
    assert REJECTION_REASON_IDS == (
        "too_close_or_late",
        "excessive_retreat_or_detour",
        "abrupt_or_irregular",
        "inappropriate_avoidance_direction",
        "excessive_stop_or_deadlock",
        "task_manipulation_disruption",
        "other",
    )


def test_controller_snapshot_preserves_unknown_vs_known_release() -> None:
    sample = ControllerButtonSnapshot.coerce(
        {
            "left": {"connected": True, "x": False},
            "right": {"connected": False, "a": True, "b": False},
        }
    )
    assert sample.left_x is False
    assert sample.left_y is None
    assert sample.right_connected is False
    assert sample.right_a is None
    assert sample.right_b is None


def test_debounced_edge_requires_three_known_release_frames() -> None:
    edge = DebouncedRisingEdge(release_frames=3)
    assert edge.update(True, connected=True) is False
    assert edge.update(False, connected=True) is False
    assert edge.update(False, connected=True) is False
    assert edge.armed is False
    assert edge.update(False, connected=True) is False
    assert edge.armed is True
    assert edge.update(True, connected=True) is True
    assert edge.update(True, connected=True) is False
    assert edge.update(False, connected=True) is False
    assert edge.update(False, connected=True) is False
    assert edge.update(True, connected=True) is False


@pytest.mark.parametrize("unknown", [None, False])
def test_disconnect_or_unknown_disarms_and_held_reconnect_has_no_ghost_edge(unknown) -> None:
    edge = DebouncedRisingEdge(release_frames=3)
    for _ in range(3):
        edge.update(False, connected=True)
    assert edge.armed

    if unknown is None:
        assert edge.update(None, connected=True) is False
    else:
        assert edge.update(None, connected=False) is False
    assert edge.update(True, connected=True) is False
    for _ in range(3):
        assert edge.update(False, connected=True) is False
    assert edge.update(True, connected=True) is True


def test_button_bank_reports_canonical_edges_only_once_per_press() -> None:
    bank = ButtonEdgeBank(release_frames=3)
    for _ in range(3):
        assert bank.update(_released()) == ()
    assert bank.update(_pressed("left.x")) == ("left.x",)
    assert bank.update(_pressed("left.x")) == ()


def test_default_realtime_mapping_uses_opposite_controller_and_exact_marker_clocks() -> None:
    marker_input = RealtimeMarkerInput(
        crossing_hand="left",
        id_factory=lambda: "marker-1",
    )
    for frame in range(3):
        assert marker_input.update(_released(), _clocks(frame)) == ()
    records = marker_input.update(_pressed("right.a"), _clocks(3))
    assert len(records) == 1
    record = records[0]
    assert record.marker_id == "marker-1"
    assert record.marker_type == REALTIME_SAFETY_CONCERN
    assert record.crossing_hand == "left"
    assert record.controller_hand == "right"
    assert record.button == "a"
    assert record.sim_time_s == pytest.approx(0.06)
    assert record.monotonic_ns == _clocks(3).monotonic_ns
    assert record.unix_ns == _clocks(3).unix_ns
    assert record.control_step == 3


def test_default_right_crossing_anomaly_is_left_y() -> None:
    marker_input = RealtimeMarkerInput(crossing_hand="right")
    for frame in range(3):
        marker_input.update(_released(), _clocks(frame))
    records = marker_input.update(_pressed("left.y"), _clocks(3))
    assert [record.marker_type for record in records] == [REALTIME_BEHAVIOR_ANOMALY]


def test_questionnaire_navigation_is_configurable_and_rendered_from_mapping() -> None:
    mapping = QuestionnaireButtonMapping.from_mapping(
        {
            "previous": "right.b",
            "next": "right.a",
            "select": "left.y",
            "submit": "left.x",
        }
    )
    fsm = QuestionnaireFSM(button_mapping=mapping)
    fsm.start(_clocks(0))

    assert mapping.action_for_edge("right.a") == "next"
    assert "RIGHT.B: 이전" in fsm.renderer_text
    assert "RIGHT.A: 다음" in fsm.renderer_text
    assert "LEFT.Y: 선택" in fsm.renderer_text
    assert "LEFT.X: 확인" in fsm.renderer_text


def test_realtime_mapping_can_swap_opposite_controller_buttons() -> None:
    mapping = RealtimeButtonMapping.from_mapping(
        {
            "left": {"safety": "right.b", "anomaly": "right.a"},
            "right": {"safety": "left.y", "anomaly": "left.x"},
        }
    )
    assert mapping.event_for_button("left", "right.b") == REALTIME_SAFETY_CONCERN
    assert mapping.event_for_button("right", "left.x") == REALTIME_BEHAVIOR_ANOMALY
    with pytest.raises(ValueError, match="opposite"):
        RealtimeButtonMapping(crossing_left_safety="left.x")


def test_questionnaire_acceptable_path_stores_q2_to_q6_separately() -> None:
    controller = XRFeedbackController(
        questionnaire_id_factory=lambda: "questionnaire-1",
    )
    driver = _ControllerDriver(controller)
    controller.start_questionnaire(driver.clocks, context_id="encounter-9")
    driver.release()

    # Q1: move from needs_modification to acceptable_as_is and commit.
    driver.press("left.y")
    driver.press("right.a")
    assert controller.questionnaire.state == "q2"

    # Q2..Q6: values 1, 2, 3, 4, 5.
    completed = None
    for value in range(1, 6):
        for _ in range(value - 1):
            driver.press("left.y")
        completed = driver.press("right.a")

    assert completed is not None
    result = completed.questionnaire_result
    assert result is not None
    assert result.questionnaire_id == "questionnaire-1"
    assert result.context_id == "encounter-9"
    assert result.completion_status == "completed"
    assert result.response_disposition == "answered"
    assert result.study_complete is True
    assert result.q1_response == "acceptable_as_is"
    assert (result.q2, result.q3, result.q4, result.q5, result.q6) == (1, 2, 3, 4, 5)
    assert result.rejection_reason_ids == ()
    assert result.missing_question_ids == ()
    assert len(result.answers) == 6
    assert all(answer.monotonic_ns > 0 for answer in result.answers)
    assert result.completed_control_step == completed.questionnaire_result.completed_control_step


def test_needs_modification_requires_one_or_more_rejection_reasons() -> None:
    fsm = QuestionnaireFSM(id_factory=lambda: "questionnaire-reasons")
    fsm.start(_clocks(0))
    # Commit the default Q1 (needs_modification), then default value 1 for Q2-Q6.
    fsm.handle_edges(("right.a",), _clocks(1))
    for frame in range(2, 7):
        fsm.handle_edges(("right.a",), _clocks(frame))
    assert fsm.state == "rejection_reasons"

    assert fsm.handle_edges(("right.b",), _clocks(7)) is None
    assert fsm.state == "rejection_reasons"
    assert "하나 이상" in fsm.renderer_text

    # Select reason 0, move twice, select reason 2, then submit.
    fsm.handle_edges(("right.a",), _clocks(8))
    fsm.handle_edges(("left.y",), _clocks(9))
    fsm.handle_edges(("left.y",), _clocks(10))
    fsm.handle_edges(("right.a",), _clocks(11))
    result = fsm.handle_edges(("right.b",), _clocks(12))
    assert result is not None
    assert result.completion_status == "completed"
    assert result.q1_response == "needs_modification"
    assert result.rejection_reason_ids == (
        "too_close_or_late",
        "abrupt_or_irregular",
    )
    assert result.answers[-1].question_id == "rejection_reasons"
    assert result.answers[-1].control_step == 12


def test_timeout_at_deadline_is_missing_abstain_and_does_not_choose_defaults() -> None:
    fsm = QuestionnaireFSM(timeout_s=1.0, id_factory=lambda: "timeout-id")
    start = FeedbackClocks(0.5, 5_000_000_000, 9_000_000_000, 20)
    fsm.start(start, context_id="terminal-encounter")
    deadline = FeedbackClocks(0.5, 6_000_000_000, 10_000_000_000, 20)
    result = fsm.handle_edges(("right.a",), deadline)
    assert result is not None
    assert result.completion_status == "timed_out"
    assert result.response_disposition == "missing_abstain"
    assert result.study_complete is False
    assert result.q1_response is None
    assert (result.q2, result.q3, result.q4, result.q5, result.q6) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert result.rejection_reason_ids is None
    assert result.answers == ()
    assert result.started_control_step == 20
    assert result.completed_control_step == 20
    assert result.deadline_monotonic_ns == deadline.monotonic_ns


def test_explicit_abstain_uses_missing_semantics() -> None:
    fsm = QuestionnaireFSM()
    fsm.start(_clocks(0))
    result = fsm.abstain(_clocks(1))
    assert result.completion_status == "abstained"
    assert result.response_disposition == "missing_abstain"
    assert result.q1_response is None
    assert "q1_modification_needed" in result.missing_question_ids


def test_partial_timeout_preserves_audit_answer_but_is_not_study_complete() -> None:
    fsm = QuestionnaireFSM(timeout_s=0.1)
    fsm.start(_clocks(0))
    fsm.handle_edges(("right.a",), _clocks(1))
    result = fsm.handle_edges((), _clocks(5))
    assert result is not None
    assert result.completion_status == "timed_out"
    assert result.q1_response == "needs_modification"
    assert len(result.answers) == 1
    assert result.study_complete is False
    assert result.rejection_reason_ids is None


def test_simultaneous_questionnaire_buttons_are_ignored() -> None:
    fsm = QuestionnaireFSM()
    fsm.start(_clocks(0))
    assert fsm.handle_edges(("left.y", "right.a"), _clocks(1)) is None
    assert fsm.state == "q1"
    assert "동시에 한 버튼" in fsm.renderer_text


def test_questionnaire_mode_suppresses_realtime_marker_for_same_button() -> None:
    controller = XRFeedbackController()
    controller.set_crossing_hand("left")
    driver = _ControllerDriver(controller)
    controller.start_questionnaire(driver.clocks)
    driver.release()
    update = driver.press("right.a")
    assert update.realtime_markers == ()
    assert controller.questionnaire.state == "q2"


def test_questionnaire_start_and_completion_require_rearming_release() -> None:
    controller = XRFeedbackController()
    driver = _ControllerDriver(controller)
    controller.start_questionnaire(driver.clocks)
    # Held A at presentation cannot choose the default.
    for _ in range(5):
        update = driver.update(_pressed("right.a"))
        assert update.questionnaire_state == "q1"
    driver.release()
    assert driver.update(_pressed("right.a")).questionnaire_state == "q2"


def test_controller_rejects_backwards_monotonic_clock() -> None:
    controller = XRFeedbackController()
    controller.update(_released(), _clocks(3))
    with pytest.raises(ValueError, match="backwards"):
        controller.update(_released(), _clocks(2))


class _FakeDevice:
    def __init__(self, names, values) -> None:
        self.names = tuple(names)
        self.values = dict(values)
        self.output_calls = []

    def get_input_names(self):
        return self.names

    def get_input_gesture_names(self, name):
        return ("click",)

    def get_input_gesture_value(self, name, gesture):
        return self.values[(name, gesture)]

    def set_output_value(self, *args):
        self.output_calls.append(args)
        raise AssertionError("controller output must not be used")


class _FakeXRCore:
    def __init__(self, devices) -> None:
        self.devices = devices

    def has_input_device(self, path):
        return path in self.devices

    def get_input_device(self, path):
        return self.devices.get(path)


def test_isaac_xr_source_reads_gestures_without_controller_output() -> None:
    left = _FakeDevice(
        ("x", "y"),
        {("x", "click"): 1.0, ("y", "click"): 0.0},
    )
    right = _FakeDevice(
        ("a", "b"),
        {("a", "click"): 0.0, ("b", "click"): 0.8},
    )
    source = IsaacXRButtonSource(
        xr_core=_FakeXRCore(
            {"/user/hand/left": left, "/user/hand/right": right}
        )
    )
    sample = source.read()
    assert sample.left_connected is True
    assert sample.right_connected is True
    assert (sample.left_x, sample.left_y, sample.right_a, sample.right_b) == (
        True,
        False,
        False,
        True,
    )
    assert left.output_calls == []
    assert right.output_calls == []


def test_isaac_xr_source_reports_disconnected_and_missing_inputs_as_unknown() -> None:
    left = _FakeDevice(("x",), {("x", "click"): 0.0})
    source = IsaacXRButtonSource(
        xr_core=_FakeXRCore({"/user/hand/left": left})
    )
    sample = source.read()
    assert sample.left_connected is True
    assert sample.left_x is False
    assert sample.left_y is None
    assert sample.right_connected is False
    assert sample.right_a is None


def test_isaac_xr_source_reuses_avatar_current_frame_device_cache() -> None:
    left = _FakeDevice(
        ("x", "y"),
        {("x", "click"): 0.0, ("y", "click"): 1.0},
    )
    right = _FakeDevice(
        ("a", "b"),
        {("a", "click"): 1.0, ("b", "click"): 0.0},
    )

    class Avatar:
        def get_cached_xr_device(self, path):
            return {
                "/user/hand/left": left,
                "/user/hand/right": right,
            }.get(path)

    class CoreMustNotBeQueried:
        def get_input_device(self, _path):
            raise AssertionError("avatar-backed reads must not re-query XRCore")

    sample = IsaacXRButtonSource(
        avatar=Avatar(), xr_core=CoreMustNotBeQueried()
    ).read()

    assert (sample.left_x, sample.left_y, sample.right_a, sample.right_b) == (
        False,
        True,
        True,
        False,
    )


def test_isaac_xr_source_uses_avatar_logical_hand_mapping_when_swapped() -> None:
    physical_left = _FakeDevice(
        ("a", "b"),
        {("a", "click"): 1.0, ("b", "click"): 0.0},
    )
    physical_right = _FakeDevice(
        ("x", "y"),
        {("x", "click"): 0.0, ("y", "click"): 1.0},
    )

    class SwappedAvatar:
        def __init__(self):
            self.logical_calls = []

        def get_cached_xr_device_for_logical_hand(self, hand):
            self.logical_calls.append(hand)
            return {
                "left": physical_right,
                "right": physical_left,
            }[hand]

        def get_cached_xr_device(self, _path):
            raise AssertionError("logical-hand cache API must take precedence")

    avatar = SwappedAvatar()
    sample = IsaacXRButtonSource(avatar=avatar).read()

    assert avatar.logical_calls == ["left", "right"]
    assert (sample.left_x, sample.left_y, sample.right_a, sample.right_b) == (
        False,
        True,
        True,
        False,
    )


def test_isaac_xr_source_does_not_treat_touch_as_face_button_press() -> None:
    class TouchAndClickDevice:
        def get_input_names(self):
            return ("a", "b")

        def get_input_gesture_names(self, _name):
            return ("touch", "click")

        def get_input_gesture_value(self, name, gesture):
            if gesture == "touch":
                return 1.0
            return 0.0 if name == "a" else 1.0

    source = IsaacXRButtonSource(
        xr_core=_FakeXRCore({"/user/hand/right": TouchAndClickDevice()})
    )
    sample = source.read()

    assert sample.right_a is False
    assert sample.right_b is True


class _MemoryRenderer:
    available = True

    def __init__(self) -> None:
        self.texts = []
        self.closed = False

    def update(self, text) -> None:
        self.texts.append(text)

    def close(self) -> None:
        self.closed = True


def test_renderer_receives_exact_question_text_and_close() -> None:
    renderer = _MemoryRenderer()
    controller = XRFeedbackController(renderer=renderer)
    controller.start_questionnaire(_clocks(0))
    assert Q1_TEXT_KO in controller.renderer_text
    assert Q1_TEXT_KO in renderer.texts[-1]
    controller.close()
    assert renderer.closed is True


def test_xr_scene_renderer_constructs_without_importing_kit() -> None:
    # Construction and text updates are pure; only open() performs optional Kit imports.
    renderer = XRSceneViewTextRenderer()
    renderer.update("pure-python text")
    assert renderer.available is False
    assert renderer.text == "pure-python text"
