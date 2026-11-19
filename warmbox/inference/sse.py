from __future__ import annotations

from typing import Iterable, Iterator


def iter_sse_data(lines: Iterable[bytes]) -> Iterator[str]:
    """Yield payloads of Server-Sent Events.

    `data:` lines of one event are joined with newlines; a blank line terminates the event;
    lines starting with ':' are comments; the stream stops at a '[DONE]' payload."""
    buf: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line:
            if buf:
                payload = "\n".join(buf)
                buf = []
                if payload.strip() == "[DONE]":
                    return
                if payload:
                    yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            buf.append(value)
    if buf:
        payload = "\n".join(buf)
        if payload.strip() != "[DONE]" and payload:
            yield payload
