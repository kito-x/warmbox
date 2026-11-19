from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

from ..cache.hashing import chunk_hash

log = logging.getLogger("warmbox.sandbox.watcher")

WriteSource = Literal["tool", "rescan", "create", "delete"]


@dataclass(frozen=True, slots=True)
class TrackedFile:
    path: str                  # workspace-relative POSIX
    version: int               # 1 on first track/create, +1 per write
    content_hash: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class WriteEvent:
    path: str
    old_content: str           # "" if created
    new_content: str           # "" if deleted
    old_version: int | None
    new_version: int
    source: WriteSource
    ts: float


Listener = Callable[[WriteEvent], Any]


class FSWatcher:
    """The single choke point for writes: atomic write, version bump, synchronous notify.

    Also detects out-of-band changes to tracked files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._tracked: dict[str, TrackedFile] = {}
        self._last_content: dict[str, str] = {}
        self._last_result: dict[str, Any] = {}
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def track(self, path: str, content: str) -> TrackedFile:
        existing = self._tracked.get(path)
        if existing is not None:
            return existing
        abs_p = self.root / path
        st = abs_p.stat()
        tf = TrackedFile(path, 1, chunk_hash(content), st.st_size, st.st_mtime_ns)
        self._tracked[path] = tf
        self._last_content[path] = content
        return tf

    def intercept_write(self, path: str, new_content: str) -> WriteEvent:
        abs_p = self.root / path
        prev = self._tracked.get(path)
        old = ""
        if abs_p.exists():
            try:
                old = abs_p.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                old = ""
        abs_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = abs_p.with_name(f".{abs_p.name}.warmbox.tmp")
        tmp.write_text(new_content, "utf-8")
        os.replace(tmp, abs_p)                       # atomic
        st = abs_p.stat()
        new_v = (prev.version + 1) if prev else 1
        src: WriteSource = "tool" if (prev or old) else "create"
        self._tracked[path] = TrackedFile(
            path, new_v, chunk_hash(new_content), st.st_size, st.st_mtime_ns
        )
        ev = WriteEvent(path, old, new_content, prev.version if prev else None, new_v, src, time.time())
        self._notify(ev)                             # synchronous, in subscribe order
        return ev

    def rescan(self) -> list[WriteEvent]:
        events: list[WriteEvent] = []
        for path, tf in sorted(self._tracked.items()):
            abs_p = self.root / path
            if not abs_p.exists():
                ev = WriteEvent(
                    path,
                    self._last_content.get(path, ""),
                    "",
                    tf.version,
                    tf.version + 1,
                    "delete",
                    time.time(),
                )
                del self._tracked[path]
            else:
                st = abs_p.stat()
                if (st.st_mtime_ns, st.st_size) == (tf.mtime_ns, tf.size):
                    continue
                new = abs_p.read_text("utf-8", errors="replace")
                if chunk_hash(new) == tf.content_hash:      # touched, not changed
                    self._tracked[path] = replace(tf, mtime_ns=st.st_mtime_ns, size=st.st_size)
                    continue
                ev = WriteEvent(
                    path,
                    self._last_content.get(path, ""),
                    new,
                    tf.version,
                    tf.version + 1,
                    "rescan",
                    time.time(),
                )
                self._tracked[path] = TrackedFile(
                    path, tf.version + 1, chunk_hash(new), st.st_size, st.st_mtime_ns
                )
            self._notify(ev)
            events.append(ev)
        return events

    def version(self, path: str) -> TrackedFile | None:
        return self._tracked.get(path)

    def tracked_paths(self) -> list[str]:
        return sorted(self._tracked)

    def _notify(self, ev: WriteEvent) -> None:
        self._last_content[ev.path] = ev.new_content
        for cb in self._listeners:
            try:
                self._last_result[ev.path] = cb(ev)
            except Exception:  # a listener must never break a write
                log.exception("listener failed for %s", ev.path)
