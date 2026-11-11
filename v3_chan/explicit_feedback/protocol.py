"""Timestamped explicit-feedback broker and a small same-origin web UI."""

from __future__ import annotations

import hmac
import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .schema import (
    QUESTION_ID,
    QUESTION_TEXT_EN,
    QUESTION_TEXT_KO,
    RESPONSE_CODES,
    RESPONSE_LABELS_KO,
    PromptRecord,
    ResponseRecord,
)


MAX_REQUEST_BYTES = 8 * 1024


class FeedbackTimeout(TimeoutError):
    """No valid explicit response arrived before the prompt deadline."""


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Track request sockets so shutdown cannot be held by a slow client."""

    daemon_threads = False
    block_on_close = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._active_lock = threading.Lock()
        self._active_requests: set[socket.socket] = set()
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address) -> None:
        with self._active_lock:
            self._active_requests.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self._active_lock:
                self._active_requests.discard(request)
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._active_lock:
                self._active_requests.discard(request)

    def close_active_requests(self) -> None:
        with self._active_lock:
            requests = tuple(self._active_requests)
        for request in requests:
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def active_request_count(self) -> int:
        with self._active_lock:
            return len(self._active_requests)

    def handle_error(self, _request, _client_address) -> None:
        # Forced shutdown and browser disconnects are expected.  Avoid logging
        # network identifiers or noisy handler tracebacks from study sessions.
        return


@dataclass
class _PromptState:
    prompt_id: str
    encounter_id: str
    issued_monotonic_ns: int
    issued_unix_ns: int
    presentation_deadline_monotonic_ns: int
    presented_monotonic_ns: int = 0
    deadline_monotonic_ns: int = 0
    status: str = "pending"

    def record(self) -> PromptRecord:
        return PromptRecord(
            prompt_id=self.prompt_id,
            encounter_id=self.encounter_id,
            question_id=QUESTION_ID,
            issued_monotonic_ns=self.issued_monotonic_ns,
            issued_unix_ns=self.issued_unix_ns,
            presentation_deadline_monotonic_ns=(
                self.presentation_deadline_monotonic_ns
            ),
            presented_monotonic_ns=self.presented_monotonic_ns,
            deadline_monotonic_ns=self.deadline_monotonic_ns,
            status=self.status,
        )


class FeedbackBroker:
    """Serialize one prompt at a time and timestamp every response server-side.

    Client timestamps are stored for audit only.  Prompt ordering, timeout, and
    response acceptance use the collector process' monotonic clock.
    """

    def __init__(
        self,
        *,
        prompt_timeout_s: float = 30.0,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not 0.1 <= float(prompt_timeout_s) <= 3600.0:
            raise ValueError("prompt_timeout_s must be in [0.1, 3600]")
        self.prompt_timeout_s = float(prompt_timeout_s)
        self._window_ns = int(round(self.prompt_timeout_s * 1e9))
        self._monotonic_ns = monotonic_ns
        self._unix_ns = unix_ns
        self._condition = threading.Condition(threading.RLock())
        self._prompts: dict[str, _PromptState] = {}
        self._prompt_order: list[str] = []
        self._responses: list[ResponseRecord] = []
        self._accepted_by_prompt: dict[str, ResponseRecord] = {}
        self._seen_client_response_ids: set[str] = set()
        self._current_prompt_id = ""
        self._closed = False

    def publish_prompt(self, encounter_id: str) -> PromptRecord:
        encounter_id = str(encounter_id).strip()
        if not encounter_id:
            raise ValueError("encounter_id must be non-empty")
        with self._condition:
            self._expire_current_locked()
            if self._closed:
                raise RuntimeError("feedback broker is closed")
            if self._current_prompt_id:
                current = self._prompts[self._current_prompt_id]
                if current.status == "pending":
                    raise RuntimeError(
                        f"prompt {current.prompt_id} is still pending"
                    )
            issued_mono = int(self._monotonic_ns())
            issued_unix = int(self._unix_ns())
            prompt_id = str(uuid.uuid4())
            state = _PromptState(
                prompt_id=prompt_id,
                encounter_id=encounter_id,
                issued_monotonic_ns=issued_mono,
                issued_unix_ns=issued_unix,
                presentation_deadline_monotonic_ns=(
                    issued_mono + self._window_ns
                ),
            )
            self._prompts[prompt_id] = state
            self._prompt_order.append(prompt_id)
            self._current_prompt_id = prompt_id
            self._condition.notify_all()
            return state.record()

    def current_prompt(self, *, mark_presented: bool = False) -> dict[str, Any] | None:
        with self._condition:
            self._expire_current_locked()
            if not self._current_prompt_id:
                return None
            state = self._prompts[self._current_prompt_id]
            if state.status != "pending":
                return None
            if mark_presented and state.presented_monotonic_ns == 0:
                self._mark_presented_locked(state)
            return {
                **state.record().as_dict(),
                "question_text_ko": QUESTION_TEXT_KO,
                "question_text_en": QUESTION_TEXT_EN,
                "response_codes": list(RESPONSE_CODES),
                "response_labels_ko": dict(RESPONSE_LABELS_KO),
            }

    def mark_presented(self, prompt_id: str) -> PromptRecord:
        """Timestamp a browser acknowledgement sent after DOM presentation."""

        with self._condition:
            state = self._prompts.get(str(prompt_id))
            if state is None:
                raise KeyError(prompt_id)
            self._expire_current_locked()
            if state.status != "pending":
                raise FeedbackTimeout(
                    f"prompt {state.prompt_id} is no longer pending"
                )
            self._mark_presented_locked(state)
            self._condition.notify_all()
            return state.record()

    def _mark_presented_locked(self, state: _PromptState) -> None:
        if state.presented_monotonic_ns > 0:
            return
        now_ns = max(state.issued_monotonic_ns, int(self._monotonic_ns()))
        if now_ns >= state.presentation_deadline_monotonic_ns:
            state.status = "expired"
            if self._current_prompt_id == state.prompt_id:
                self._current_prompt_id = ""
            raise FeedbackTimeout(
                f"prompt {state.prompt_id} expired before presentation"
            )
        state.presented_monotonic_ns = now_ns
        state.deadline_monotonic_ns = now_ns + self._window_ns

    def submit(self, payload: Mapping[str, Any], *, source: str) -> ResponseRecord:
        client_response_id = (
            str(payload.get("response_id", "")).strip() or str(uuid.uuid4())
        )
        # Every audit row receives a server ID.  A retried client ID can then be
        # retained as a rejected row without violating table-key uniqueness.
        response_id = str(uuid.uuid4())
        prompt_id = str(payload.get("prompt_id", "")).strip()
        response_code = str(payload.get("response_code", "")).strip()
        client_unix_ns = _safe_int(payload.get("client_unix_ns", 0), default=0)
        source = str(source or "unknown")[:128]

        with self._condition:
            # Timestamp under the serialization lock.  This prevents two HTTP
            # handler threads from appending records in the opposite order of
            # their clocks and ensures deadline checks use processing time,
            # rather than a timestamp captured while waiting for the broker.
            now_mono = int(self._monotonic_ns())
            now_unix = int(self._unix_ns())
            rejection = ""
            state = self._prompts.get(prompt_id)
            if self._closed:
                rejection = "broker_closed"
            elif client_response_id in self._seen_client_response_ids:
                rejection = "duplicate_response_id"
            elif state is None:
                rejection = "unknown_prompt_id"
            elif response_code not in RESPONSE_CODES:
                rejection = "invalid_response_code"
            elif state.status == "accepted":
                rejection = "prompt_already_answered"
            elif state.presented_monotonic_ns <= 0:
                if (
                    state.status == "expired"
                    or now_mono >= state.presentation_deadline_monotonic_ns
                ):
                    rejection = "prompt_expired"
                    state.status = "expired"
                else:
                    rejection = "prompt_not_presented"
            elif state.status == "expired" or now_mono >= state.deadline_monotonic_ns:
                rejection = "prompt_expired"
                state.status = "expired"
            elif now_mono < state.presented_monotonic_ns:
                rejection = "response_precedes_presentation"
            accepted = not rejection
            record = ResponseRecord(
                response_id=response_id,
                client_response_id=client_response_id,
                prompt_id=prompt_id,
                response_code=response_code,
                source=source,
                client_unix_ns=client_unix_ns,
                received_monotonic_ns=now_mono,
                received_unix_ns=now_unix,
                accepted=accepted,
                rejection_reason=rejection,
            )
            self._seen_client_response_ids.add(client_response_id)
            self._responses.append(record)
            if accepted and state is not None:
                state.status = "accepted"
                self._accepted_by_prompt[prompt_id] = record
                if self._current_prompt_id == prompt_id:
                    self._current_prompt_id = ""
            self._condition.notify_all()
            return record

    def wait_for_response(
        self,
        prompt_id: str,
        *,
        timeout_s: float | None = None,
        idle_callback: Callable[[], None] | None = None,
        poll_interval_s: float = 0.01,
    ) -> ResponseRecord:
        prompt_id = str(prompt_id)
        if idle_callback is not None and not 0.001 <= float(poll_interval_s) <= 1.0:
            raise ValueError("poll_interval_s must be in [0.001, 1.0]")
        with self._condition:
            state = self._prompts.get(prompt_id)
            if state is None:
                raise KeyError(f"unknown prompt_id {prompt_id!r}")
            caller_deadline = (
                None
                if timeout_s is None
                else int(self._monotonic_ns())
                + max(0, int(round(float(timeout_s) * 1e9)))
            )
        while True:
            run_idle_callback = False
            with self._condition:
                accepted = self._accepted_by_prompt.get(prompt_id)
                if accepted is not None:
                    return accepted
                now_ns = int(self._monotonic_ns())
                protocol_deadline = (
                    state.presentation_deadline_monotonic_ns
                    if state.presented_monotonic_ns <= 0
                    else state.deadline_monotonic_ns
                )
                effective_deadline = (
                    protocol_deadline
                    if caller_deadline is None
                    else min(protocol_deadline, caller_deadline)
                )
                if now_ns >= effective_deadline or self._closed:
                    if state.status == "pending":
                        state.status = "expired"
                    if self._current_prompt_id == prompt_id:
                        self._current_prompt_id = ""
                    self._condition.notify_all()
                    raise FeedbackTimeout(
                        f"prompt {prompt_id} expired without an accepted response"
                    )
                wait_s = (effective_deadline - now_ns) / 1e9
                if idle_callback is not None:
                    wait_s = min(wait_s, float(poll_interval_s))
                    run_idle_callback = True
                self._condition.wait(wait_s)
            if run_idle_callback:
                idle_callback()

    def prompt_record(self, prompt_id: str) -> PromptRecord:
        with self._condition:
            state = self._prompts.get(str(prompt_id))
            if state is None:
                raise KeyError(prompt_id)
            return state.record()

    def snapshot(self) -> tuple[tuple[PromptRecord, ...], tuple[ResponseRecord, ...]]:
        with self._condition:
            self._expire_current_locked()
            return (
                tuple(self._prompts[key].record() for key in self._prompt_order),
                tuple(self._responses),
            )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            if self._current_prompt_id:
                state = self._prompts[self._current_prompt_id]
                if state.status == "pending":
                    state.status = "cancelled"
            self._current_prompt_id = ""
            self._condition.notify_all()

    def _expire_current_locked(self) -> None:
        if not self._current_prompt_id:
            return
        state = self._prompts[self._current_prompt_id]
        deadline = (
            state.presentation_deadline_monotonic_ns
            if state.presented_monotonic_ns <= 0
            else state.deadline_monotonic_ns
        )
        if (
            state.status == "pending"
            and int(self._monotonic_ns()) >= deadline
        ):
            state.status = "expired"
            self._current_prompt_id = ""


class FeedbackHTTPServer:
    """Background HTTP server exposing the broker and participant UI."""

    def __init__(
        self,
        broker: FeedbackBroker,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        bearer_token: str = "",
    ) -> None:
        host = str(host).strip()
        token = str(bearer_token)
        if not host:
            raise ValueError("feedback host must be non-empty")
        if not _is_loopback_host(host) and len(token) < 16:
            raise ValueError(
                "Non-loopback feedback servers require a bearer token of at least 16 characters"
            )
        self.broker = broker
        self.host = host
        self.port = int(port)
        self.bearer_token = token
        handler = _handler_factory(broker, token)
        self._server = _BoundedThreadingHTTPServer((host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="explicit-feedback-http",
            daemon=True,
        )
        self._started = False
        self._closed = False

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("feedback HTTP server is closed")
        if self._started:
            return
        self._thread.start()
        self._started = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            self._server.shutdown()
        # A socket inactivity timeout is not an absolute request deadline: a
        # slow trickle can keep it alive indefinitely.  Force every connection
        # out of blocking header/body I/O before joining its handler thread.
        self._server.close_active_requests()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)


def _handler_factory(broker: FeedbackBroker, bearer_token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SeRTExplicitFeedback/1"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(2.0)

        def log_message(self, format: str, *args: Any) -> None:
            # Do not emit URLs or headers that could contain a participant token.
            return

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/health":
                self._json(HTTPStatus.OK, {"ok": True})
                return
            if path == "/":
                self._html(HTTPStatus.OK, _PARTICIPANT_HTML)
                return
            if path == "/api/v1/prompt":
                if not self._authorized(bearer_token):
                    return
                prompt = broker.current_prompt(mark_presented=False)
                self._json(
                    HTTPStatus.OK,
                    {"pending": prompt is not None, "prompt": prompt},
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path not in {"/api/v1/presented", "/api/v1/feedback"}:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if not self._authorized(bearer_token):
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "invalid_content_length"},
                )
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
                return
            if not isinstance(payload, dict):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "object_required"})
                return
            if path == "/api/v1/presented":
                prompt_id = str(payload.get("prompt_id", "")).strip()
                try:
                    record = broker.mark_presented(prompt_id)
                except KeyError:
                    self._json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "unknown_prompt_id"},
                    )
                    return
                except FeedbackTimeout:
                    self._json(
                        HTTPStatus.CONFLICT,
                        {"error": "prompt_expired"},
                    )
                    return
                self._json(HTTPStatus.OK, record.as_dict())
                return
            record = broker.submit(
                payload,
                # The transport type is sufficient provenance; participant IP
                # addresses are deliberately not written into study artifacts.
                source="browser_ui_v1",
            )
            status = HTTPStatus.OK if record.accepted else HTTPStatus.CONFLICT
            self._json(status, record.as_dict())

        def _authorized(self, expected_token: str) -> bool:
            if not expected_token:
                return True
            header = self.headers.get("Authorization", "")
            provided = header[7:] if header.startswith("Bearer ") else ""
            if hmac.compare_digest(provided, expected_token):
                return True
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return False

        def _headers(self, content_type: str, length: int) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'",
            )

        def _json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(
                dict(payload), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(int(status))
            self._headers("application/json; charset=utf-8", len(body))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, status: HTTPStatus, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(int(status))
            self._headers("text/html; charset=utf-8", len(body))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _is_loopback_host(host: str) -> bool:
    return host.lower() in {"127.0.0.1", "::1", "localhost"}


_PARTICIPANT_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>로봇 동작 피드백</title><style>
:root{color-scheme:dark}body{font-family:system-ui,sans-serif;background:#10151d;color:#f4f7fb;margin:0;min-height:100vh;display:grid;place-items:center}.card{width:min(680px,90vw);background:#1a2330;padding:2rem;border-radius:18px;box-shadow:0 12px 50px #0008}h1{font-size:1.6rem}.sub{color:#b8c5d6}.buttons{display:grid;gap:1rem;margin-top:2rem}button{font:inherit;font-weight:700;padding:1.1rem;border:0;border-radius:12px;cursor:pointer}.yes{background:#46c787;color:#07170f}.no{background:#ff6b72;color:#210608}.unsure{background:#e9b949;color:#1d1400}button:disabled{opacity:.45;cursor:not-allowed}#status{min-height:1.5em;margin-top:1.3rem;color:#b8c5d6}
</style></head><body><main class="card"><h1 id="question">다음 로봇 동작을 기다리는 중입니다.</h1><p class="sub">각 문항에는 한 번만 응답할 수 있습니다.</p><section class="buttons">
<button class="yes" data-code="acceptable" disabled>안전하고 수용 가능함</button>
<button class="no" data-code="unacceptable" disabled>안전하지 않거나 수용하기 어려움</button>
<button class="unsure" data-code="unsure" disabled>잘 모르겠음</button></section><p id="status"></p></main>
<script>
let promptId='', sending=false; let token=localStorage.getItem('sert-feedback-token')||'';
if(location.hash.startsWith('#token=')){token=decodeURIComponent(location.hash.slice(7));localStorage.setItem('sert-feedback-token',token);history.replaceState(null,'',location.pathname)}
const headers=()=>token?{'Authorization':'Bearer '+token}:{};
const buttons=[...document.querySelectorAll('button')]; const status=document.querySelector('#status');
async function present(p){promptId='';buttons.forEach(b=>b.disabled=true);document.querySelector('#question').textContent=p.question_text_ko;status.textContent='문항을 준비하는 중입니다.';await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));let r=await fetch('/api/v1/presented',{method:'POST',headers:{...headers(),'Content-Type':'application/json'},body:JSON.stringify({prompt_id:p.prompt_id})});if(!r.ok)throw new Error('presentation rejected');promptId=p.prompt_id;buttons.forEach(b=>b.disabled=sending);status.textContent='응답을 선택해 주세요.'}
async function poll(){try{let r=await fetch('/api/v1/prompt',{headers:headers(),cache:'no-store'});if(r.status===401){token=prompt('접속 토큰을 입력하세요')||'';if(token)localStorage.setItem('sert-feedback-token',token);setTimeout(poll,300);return}let x=await r.json();if(x.pending){if(promptId!==x.prompt.prompt_id)await present(x.prompt)}else{promptId='';buttons.forEach(b=>b.disabled=true);document.querySelector('#question').textContent='다음 로봇 동작을 기다리는 중입니다.';status.textContent=''}}catch(e){status.textContent='서버 연결을 확인해 주세요.'}setTimeout(poll,500)}
buttons.forEach(b=>b.onclick=async()=>{if(!promptId||sending)return;sending=true;buttons.forEach(x=>x.disabled=true);let rid=(crypto.randomUUID?crypto.randomUUID():(Date.now()+'-'+Math.random().toString(16).slice(2)));let body={prompt_id:promptId,response_code:b.dataset.code,response_id:rid,client_unix_ns:Date.now()*1000000};try{let r=await fetch('/api/v1/feedback',{method:'POST',headers:{...headers(),'Content-Type':'application/json'},body:JSON.stringify(body)});let x=await r.json();status.textContent=r.ok?'응답이 기록되었습니다.':('응답 거부: '+(x.rejection_reason||x.error));if(r.ok)promptId=''}catch(e){status.textContent='응답 전송에 실패했습니다. 다시 시도해 주세요.'}sending=false;if(promptId)buttons.forEach(x=>x.disabled=false)});poll();
</script></body></html>"""
