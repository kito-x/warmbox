from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..agent.history import Message, ToolCall, canonical_json
from ..config import Settings
from ..errors import LLMHTTPError, LLMProtocolError, LLMTimeoutError
from .sse import iter_sse_data
from .tokenizer import count_tokens

log = logging.getLogger("warmbox.llm")

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int | None = None   # usage.prompt_tokens_details.cached_tokens


@dataclass(frozen=True, slots=True)
class ChatResponse:
    message: Message                          # role="assistant"
    usage: Usage
    ttft_ms: float | None                     # None for simulated backends
    latency_ms: float
    finish_reason: str                        # "stop" | "tool_calls" | "length" | ...


class LLMClient(Protocol):
    model_id: str
    is_simulated: bool

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse: ...


MockStep = dict[str, Any]   # {"tool": name, "args": {...}}  or  {"content": "text"}


class MockBackend:
    """Scripted, deterministic, no network. Used by the demo, tests and the mock benchmark."""

    is_simulated = True

    def __init__(self, script: Sequence[MockStep] = (), model_id: str = "mock-1") -> None:
        self.script: list[MockStep] = list(script)
        self.model_id = model_id
        self._n = 0

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        if self.script:
            step = self.script.pop(0)
        else:
            step = {"tool": "finish", "args": {"summary": "script exhausted"}}
        if isinstance(step, str):  # tolerate a bare string step
            step = {"content": step}

        prompt_tokens = count_tokens(canonical_json(list(messages)))
        completion_tokens = count_tokens(canonical_json(step))

        if "tool" in step:
            args = step.get("args") or {}
            call = ToolCall(
                id=f"call_{self._n:04d}",
                name=str(step["tool"]),
                arguments=dict(args),
                raw_arguments=canonical_json(args),
            )
            self._n += 1
            msg = Message(role="assistant", content=None, tool_calls=(call,))
            finish = "tool_calls"
        else:
            msg = Message(role="assistant", content=step.get("content") or "")
            finish = "stop"

        return ChatResponse(
            message=msg,
            usage=Usage(prompt_tokens, completion_tokens, None),
            ttft_ms=None,
            latency_ms=0.0,
            finish_reason=finish,
        )


class OpenAICompatBackend:
    """Any OpenAI-compatible /chat/completions endpoint, stdlib urllib + SSE."""

    is_simulated = False

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        stream: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.stream = stream

    # ------------------------------------------------------------------ helpers

    def _request(self, body: dict[str, Any]) -> urllib.request.Request:
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self.stream else "application/json",
            "User-Agent": "warmbox/0.1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST"
        )

    def _open(self, body: dict[str, Any]):
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return urllib.request.urlopen(self._request(body), timeout=self.timeout_s)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:  # pragma: no cover
                    pass
                if exc.code in RETRY_STATUSES and attempt < self.max_retries - 1:
                    log.warning("HTTP %s from %s; retrying", exc.code, self.base_url)
                    last = LLMHTTPError(exc.code, detail)
                    time.sleep(2**attempt)
                    continue
                raise LLMHTTPError(exc.code, detail) from None
            except TimeoutError as exc:
                if attempt < self.max_retries - 1:
                    last = LLMTimeoutError(str(exc))
                    time.sleep(2**attempt)
                    continue
                raise LLMTimeoutError(str(exc)) from None
            except urllib.error.URLError as exc:
                if attempt < self.max_retries - 1:
                    last = LLMTimeoutError(str(exc))
                    time.sleep(2**attempt)
                    continue
                raise LLMTimeoutError(str(exc)) from None
        raise last or LLMTimeoutError("exhausted retries")

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": self.stream,
        }
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = list(tools)
        t0 = time.perf_counter()
        resp = self._open(body)
        try:
            if self.stream:
                out = self._read_stream(resp, t0)
            else:
                out = self._read_json(resp, t0)
        finally:
            resp.close()
        return out

    def _read_json(self, resp, t0: float) -> ChatResponse:
        raw = resp.read()
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMProtocolError(f"bad JSON response: {exc}") from None
        latency = (time.perf_counter() - t0) * 1000
        choice = (doc.get("choices") or [{}])[0]
        msg_doc = choice.get("message") or {}
        calls: list[ToolCall] = []
        for i, tc in enumerate(msg_doc.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
                err = None
            except json.JSONDecodeError as exc:
                args, err = {}, f"bad tool arguments JSON: {exc}"
            calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{i:04d}",
                    name=fn.get("name") or "",
                    arguments=args,
                    raw_arguments=raw_args,
                    parse_error=err,
                )
            )
        msg = Message(
            role="assistant",
            content=msg_doc.get("content"),
            tool_calls=tuple(calls),
        )
        return ChatResponse(
            message=msg,
            usage=_usage_from(doc.get("usage")),
            ttft_ms=None,
            latency_ms=latency,
            finish_reason=choice.get("finish_reason") or ("tool_calls" if calls else "stop"),
        )

    def _read_stream(self, resp, t0: float) -> ChatResponse:
        content_parts: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        usage_doc: dict[str, Any] | None = None
        finish = "stop"
        ttft: float | None = None

        for payload in iter_sse_data(resp):
            try:
                doc = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise LLMProtocolError(f"bad SSE payload: {exc}") from None
            if doc.get("usage"):
                usage_doc = doc["usage"]
            for choice in doc.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    content_parts.append(text)
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index", 0))
                    slot = slots.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]

        calls: list[ToolCall] = []
        for idx in sorted(slots):
            slot = slots[idx]
            raw = slot["args"] or "{}"
            try:
                args = json.loads(raw)
                err = None
            except json.JSONDecodeError as exc:
                args, err = {}, f"bad tool arguments JSON: {exc}"
            calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{idx:04d}",
                    name=slot["name"],
                    arguments=args,
                    raw_arguments=raw,
                    parse_error=err,
                )
            )
        content = "".join(content_parts) or None
        msg = Message(role="assistant", content=content, tool_calls=tuple(calls))
        return ChatResponse(
            message=msg,
            usage=_usage_from(usage_doc, fallback_prompt=_estimate_prompt(content_parts)),
            ttft_ms=ttft,
            latency_ms=(time.perf_counter() - t0) * 1000,
            finish_reason=finish or ("tool_calls" if calls else "stop"),
        )


def _estimate_prompt(parts: list[str]) -> int:
    return sum(count_tokens(p) for p in parts)


def _usage_from(doc: dict[str, Any] | None, *, fallback_prompt: int = 0) -> Usage:
    if not doc:
        return Usage(fallback_prompt, 0, None)
    details = doc.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    return Usage(
        prompt_tokens=int(doc.get("prompt_tokens") or fallback_prompt),
        completion_tokens=int(doc.get("completion_tokens") or 0),
        cached_prompt_tokens=int(cached) if cached is not None else None,
    )


def make_client(
    backend: str,
    *,
    settings: Settings,
    script: Sequence[MockStep] | None = None,
) -> LLMClient:
    if backend == "mock":
        return MockBackend(script or (), model_id=settings.model)
    if backend == "openai":
        return OpenAICompatBackend(settings.base_url, settings.api_key, settings.model)
    raise ValueError(f"unknown backend: {backend!r}")
