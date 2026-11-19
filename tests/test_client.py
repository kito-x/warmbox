from __future__ import annotations

import io
import time
import urllib.error

import pytest

from warmbox.agent.history import Message, canonical_json
from warmbox.errors import LLMHTTPError
from warmbox.inference.client import MockBackend, OpenAICompatBackend
from warmbox.inference.sse import iter_sse_data


def test_mock_script_ids_and_exhaustion() -> None:
    backend = MockBackend([{"tool": "read_file", "args": {"path": "a.txt"}}, {"content": "done"}])
    r1 = backend.chat([{"role": "user", "content": "go"}])
    assert r1.finish_reason == "tool_calls"
    call = r1.message.tool_calls[0]
    assert call.id == "call_0000"
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.txt"}
    assert r1.message.content is None
    assert r1.ttft_ms is None and r1.latency_ms == 0.0

    r2 = backend.chat([{"role": "user", "content": "go"}])
    assert r2.finish_reason == "stop"
    assert r2.message.content == "done"

    r3 = backend.chat([{"role": "user", "content": "go"}])
    assert r3.message.tool_calls[0].name == "finish"
    assert r3.message.tool_calls[0].arguments == {"summary": "script exhausted"}


def test_mock_usage_is_token_counts() -> None:
    msgs = [{"role": "user", "content": "hello there"}]
    r = MockBackend([{"tool": "finish", "args": {"summary": "ok"}}]).chat(msgs)
    from warmbox.inference.tokenizer import count_tokens

    assert r.usage.prompt_tokens == count_tokens(canonical_json(msgs))
    assert r.usage.cached_prompt_tokens is None


class _FakeResp:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def read(self) -> bytes:
        return b"".join(self._lines)

    def close(self) -> None:
        pass


SSE = [
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
    b"\n",
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a",'
    b'"function":{"name":"read_file","arguments":"{\\"pa"}}]}}]}\n',
    b"\n",
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    b'"function":{"arguments":"th\\": \\"a.txt\\"}"}}]}}]}\n',
    b"\n",
    b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
    b"\n",
    b'data: {"usage":{"prompt_tokens":100,"completion_tokens":7,'
    b'"prompt_tokens_details":{"cached_tokens":64}}}\n',
    b"\n",
    b"data: [DONE]\n",
    b"\n",
]


def test_sse_iteration_joins_multi_line_data() -> None:
    frames = [b"data: a\n", b"data: b\n", b"\n", b": comment\n", b"data: [DONE]\n"]
    assert list(iter_sse_data(frames)) == ["a\nb"]


def test_sse_tool_call_accumulation() -> None:
    backend = OpenAICompatBackend("http://x/v1", "k", "m", stream=True)
    out = backend._read_stream(_FakeResp(SSE), time.perf_counter())
    assert len(out.message.tool_calls) == 1
    call = out.message.tool_calls[0]
    assert call.id == "call_a"
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.txt"}
    assert call.parse_error is None
    assert out.finish_reason == "tool_calls"
    assert out.usage.prompt_tokens == 100
    assert out.usage.completion_tokens == 7
    assert out.usage.cached_prompt_tokens == 64
    assert out.ttft_ms is not None


def test_retry_on_429(monkeypatch) -> None:
    calls = {"n": 0}

    body = {
        "choices": [
            {"message": {"content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"slow down")
            )
        return _FakeResp([__import__("json").dumps(body).encode()])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    backend = OpenAICompatBackend("http://x/v1", "", "m", stream=False, max_retries=3)
    out = backend.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 3
    assert out.message.content == "hi"
    assert out.usage.prompt_tokens == 10


def test_http_error_propagates(monkeypatch) -> None:
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"no"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OpenAICompatBackend("http://x/v1", "", "m", stream=False, max_retries=1)
    with pytest.raises(LLMHTTPError):
        backend.chat([{"role": "user", "content": "hi"}])


def test_message_openai_shape() -> None:
    msg = Message("assistant", None, tool_calls=())
    assert msg.to_openai() == {"role": "assistant", "content": None}
    tool = Message("tool", "ok", tool_call_id="call_1")
    assert tool.to_openai() == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
