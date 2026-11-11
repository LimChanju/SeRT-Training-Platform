from __future__ import annotations

import json
import socket
import time
from urllib.request import Request, urlopen

import pytest

from v3_chan.explicit_feedback.protocol import (
    FeedbackBroker,
    FeedbackHTTPServer,
    FeedbackTimeout,
)


class _FakeClock:
    def __init__(
        self,
        *,
        monotonic_ns: int = 1_000_000_000,
        unix_ns: int = 1_800_000_000_000_000_000,
    ) -> None:
        self.monotonic_value_ns = int(monotonic_ns)
        self.unix_value_ns = int(unix_ns)

    def monotonic_ns(self) -> int:
        return self.monotonic_value_ns

    def unix_ns(self) -> int:
        return self.unix_value_ns

    def advance_ns(self, delta_ns: int) -> None:
        self.monotonic_value_ns += int(delta_ns)
        self.unix_value_ns += int(delta_ns)


def _broker(clock: _FakeClock, *, timeout_s: float = 1.0) -> FeedbackBroker:
    return FeedbackBroker(
        prompt_timeout_s=timeout_s,
        monotonic_ns=clock.monotonic_ns,
        unix_ns=clock.unix_ns,
    )


def test_prompt_is_presented_then_response_is_accepted_on_server_clock() -> None:
    clock = _FakeClock()
    broker = _broker(clock)

    prompt = broker.publish_prompt("encounter-0001")
    assert prompt.issued_monotonic_ns == 1_000_000_000
    assert prompt.presentation_deadline_monotonic_ns == 2_000_000_000
    assert prompt.presented_monotonic_ns == 0
    assert prompt.deadline_monotonic_ns == 0

    clock.advance_ns(25_000_000)
    presented = broker.current_prompt(mark_presented=True)
    assert presented is not None
    assert presented["prompt_id"] == prompt.prompt_id
    assert presented["presented_monotonic_ns"] == 1_025_000_000
    assert presented["deadline_monotonic_ns"] == 2_025_000_000

    clock.advance_ns(40_000_000)
    response = broker.submit(
        {
            "response_id": "response-0001",
            "prompt_id": prompt.prompt_id,
            "response_code": "acceptable",
            # This client timestamp is audit-only and deliberately unrelated to
            # the server monotonic clock used for ordering and acceptance.
            "client_unix_ns": 123,
        },
        source="test-ui",
    )

    assert response.accepted is True
    assert response.client_response_id == "response-0001"
    assert response.rejection_reason == ""
    assert response.received_monotonic_ns == 1_065_000_000
    assert response.client_unix_ns == 123
    assert broker.wait_for_response(prompt.prompt_id, timeout_s=0.0) == response
    assert broker.prompt_record(prompt.prompt_id).status == "accepted"
    assert broker.current_prompt() is None


def test_delayed_presentation_still_receives_the_full_response_window() -> None:
    clock = _FakeClock()
    broker = _broker(clock, timeout_s=1.0)
    prompt = broker.publish_prompt("encounter-delayed")

    clock.advance_ns(900_000_000)
    presented = broker.mark_presented(prompt.prompt_id)
    assert presented.presentation_deadline_monotonic_ns == 2_000_000_000
    assert presented.presented_monotonic_ns == 1_900_000_000
    assert presented.deadline_monotonic_ns == 2_900_000_000

    clock.advance_ns(999_000_000)
    response = broker.submit(
        {
            "response_id": "delayed-presentation-response",
            "prompt_id": prompt.prompt_id,
            "response_code": "acceptable",
        },
        source="test-ui",
    )

    assert response.accepted is True
    assert response.received_monotonic_ns == 2_899_000_000


def test_response_before_prompt_presentation_is_rejected() -> None:
    clock = _FakeClock()
    broker = _broker(clock)
    prompt = broker.publish_prompt("encounter-0001")

    response = broker.submit(
        {
            "response_id": "response-before-present",
            "prompt_id": prompt.prompt_id,
            "response_code": "unacceptable",
        },
        source="test-ui",
    )

    assert response.accepted is False
    assert response.rejection_reason == "prompt_not_presented"
    assert broker.prompt_record(prompt.prompt_id).status == "pending"


def test_unknown_prompt_response_is_rejected_and_audited() -> None:
    clock = _FakeClock()
    broker = _broker(clock)

    response = broker.submit(
        {
            "response_id": "unknown-response",
            "prompt_id": "does-not-exist",
            "response_code": "acceptable",
        },
        source="test-ui",
    )

    assert response.accepted is False
    assert response.rejection_reason == "unknown_prompt_id"
    prompts, responses = broker.snapshot()
    assert prompts == ()
    assert responses == (response,)


def test_response_after_broker_close_is_rejected_and_audited() -> None:
    clock = _FakeClock()
    broker = _broker(clock)
    prompt = broker.publish_prompt("encounter-closed")
    broker.current_prompt(mark_presented=True)
    broker.close()

    response = broker.submit(
        {
            "response_id": "response-after-close",
            "prompt_id": prompt.prompt_id,
            "response_code": "acceptable",
        },
        source="test-ui",
    )

    assert response.accepted is False
    assert response.rejection_reason == "broker_closed"
    assert broker.prompt_record(prompt.prompt_id).status == "cancelled"
    assert broker.snapshot()[1] == (response,)


def test_duplicate_response_id_and_second_answer_are_rejected() -> None:
    clock = _FakeClock()
    broker = _broker(clock)
    prompt = broker.publish_prompt("encounter-0001")
    broker.current_prompt(mark_presented=True)
    payload = {
        "response_id": "stable-response-id",
        "prompt_id": prompt.prompt_id,
        "response_code": "acceptable",
    }

    accepted = broker.submit(payload, source="test-ui")
    duplicate = broker.submit(payload, source="test-ui")
    second_answer = broker.submit(
        {
            "response_id": "different-response-id",
            "prompt_id": prompt.prompt_id,
            "response_code": "unacceptable",
        },
        source="test-ui",
    )

    assert accepted.accepted is True
    assert duplicate.accepted is False
    assert duplicate.response_id != accepted.response_id
    assert duplicate.client_response_id == accepted.client_response_id
    assert duplicate.rejection_reason == "duplicate_response_id"
    assert second_answer.accepted is False
    assert second_answer.rejection_reason == "prompt_already_answered"
    _, responses = broker.snapshot()
    assert responses == (accepted, duplicate, second_answer)


def test_response_at_deadline_is_rejected_as_late() -> None:
    clock = _FakeClock()
    broker = _broker(clock, timeout_s=0.5)
    prompt = broker.publish_prompt("encounter-0001")
    broker.current_prompt(mark_presented=True)
    clock.advance_ns(500_000_000)

    response = broker.submit(
        {
            "response_id": "late-response",
            "prompt_id": prompt.prompt_id,
            "response_code": "unsure",
        },
        source="test-ui",
    )

    assert response.accepted is False
    assert response.rejection_reason == "prompt_expired"
    assert broker.prompt_record(prompt.prompt_id).status == "expired"


def test_response_server_clocks_are_sampled_under_broker_lock() -> None:
    holder = {}
    clock_value = [1_000_000_000]

    def monotonic_ns() -> int:
        broker = holder["broker"]
        assert broker._condition._is_owned()
        clock_value[0] += 10
        return clock_value[0]

    def unix_ns() -> int:
        broker = holder["broker"]
        assert broker._condition._is_owned()
        return 1_800_000_000_000_000_000

    broker = FeedbackBroker(
        prompt_timeout_s=1.0,
        monotonic_ns=monotonic_ns,
        unix_ns=unix_ns,
    )
    holder["broker"] = broker
    prompt = broker.publish_prompt("encounter-locked-clock")
    broker.mark_presented(prompt.prompt_id)

    response = broker.submit(
        {
            "response_id": "locked-clock-response",
            "prompt_id": prompt.prompt_id,
            "response_code": "acceptable",
        },
        source="test-ui",
    )

    assert response.accepted is True


def test_wait_timeout_uses_fake_monotonic_clock_without_sleeping() -> None:
    clock = _FakeClock()
    broker = _broker(clock, timeout_s=0.25)
    prompt = broker.publish_prompt("encounter-0001")
    broker.current_prompt(mark_presented=True)
    clock.advance_ns(250_000_000)

    with pytest.raises(FeedbackTimeout, match=prompt.prompt_id):
        broker.wait_for_response(prompt.prompt_id)

    assert broker.prompt_record(prompt.prompt_id).status == "expired"
    assert broker.current_prompt() is None


def test_wait_runs_idle_callback_and_accepts_response_from_callback() -> None:
    clock = _FakeClock()
    broker = _broker(clock)
    prompt = broker.publish_prompt("encounter-idle-callback")
    broker.current_prompt(mark_presented=True)
    callback_calls = []

    def idle_callback() -> None:
        callback_calls.append("idle")
        broker.submit(
            {
                "response_id": "response-from-idle-callback",
                "prompt_id": prompt.prompt_id,
                "response_code": "acceptable",
            },
            source="test-ui",
        )

    response = broker.wait_for_response(
        prompt.prompt_id,
        idle_callback=idle_callback,
        poll_interval_s=0.001,
    )

    assert callback_calls == ["idle"]
    assert response.accepted is True
    assert response.client_response_id == "response-from-idle-callback"


def test_loopback_http_ui_presents_and_records_one_explicit_response() -> None:
    broker = FeedbackBroker(prompt_timeout_s=2.0)
    server = FeedbackHTTPServer(broker, host="127.0.0.1", port=0)
    server.start()
    try:
        host, port = server.address
        prompt = broker.publish_prompt("encounter-http")
        with urlopen(
            f"http://{host}:{port}/api/v1/prompt", timeout=2.0
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["pending"] is True
        assert payload["prompt"]["prompt_id"] == prompt.prompt_id
        assert payload["prompt"]["presented_monotonic_ns"] == 0
        assert broker.prompt_record(prompt.prompt_id).presented_monotonic_ns == 0

        presented_body = json.dumps({"prompt_id": prompt.prompt_id}).encode(
            "utf-8"
        )
        presented_request = Request(
            f"http://{host}:{port}/api/v1/presented",
            data=presented_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(presented_request, timeout=2.0) as response:
            presented = json.loads(response.read().decode("utf-8"))
        assert presented["prompt_id"] == prompt.prompt_id
        assert presented["presented_monotonic_ns"] > 0
        assert presented["deadline_monotonic_ns"] > presented["presented_monotonic_ns"]

        body = json.dumps(
            {
                "response_id": "http-response",
                "prompt_id": prompt.prompt_id,
                "response_code": "acceptable",
                "client_unix_ns": 123,
            }
        ).encode("utf-8")
        request = Request(
            f"http://{host}:{port}/api/v1/feedback",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2.0) as response:
            accepted = json.loads(response.read().decode("utf-8"))

        assert accepted["accepted"] is True
        assert accepted["source"] == "browser_ui_v1"
        assert broker.wait_for_response(prompt.prompt_id) == broker.snapshot()[1][0]
    finally:
        server.close()
        broker.close()


def test_server_close_interrupts_a_slow_incomplete_request() -> None:
    broker = FeedbackBroker(prompt_timeout_s=2.0)
    server = FeedbackHTTPServer(broker, host="127.0.0.1", port=0)
    server.start()
    client = None
    try:
        client = socket.create_connection(server.address, timeout=1.0)
        client.sendall(
            b"POST /api/v1/feedback HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Length: 100\r\n"
        )
        deadline = time.monotonic() + 1.0
        while server._server.active_request_count() == 0:
            assert time.monotonic() < deadline
            time.sleep(0.005)

        started = time.monotonic()
        server.close()
        elapsed = time.monotonic() - started

        # serve_forever shutdown may consume one 0.5 s poll, but the handler's
        # 2 s inactivity timeout must not control collection shutdown.
        assert elapsed < 1.5
        assert server._server.active_request_count() == 0
    finally:
        if client is not None:
            client.close()
        server.close()
        broker.close()


def test_nonloopback_http_server_requires_a_long_token() -> None:
    broker = FeedbackBroker()
    try:
        with pytest.raises(ValueError, match="bearer token"):
            FeedbackHTTPServer(
                broker,
                host="0.0.0.0",
                port=0,
                bearer_token="too-short",
            )
    finally:
        broker.close()
