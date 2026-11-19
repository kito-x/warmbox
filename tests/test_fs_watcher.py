from __future__ import annotations

import os
from pathlib import Path

from warmbox.cache.block_cache import KVCache
from warmbox.sandbox.fs_watcher import FSWatcher, WriteEvent


def _watcher(tmp_path: Path, content: str = "one\n") -> tuple[FSWatcher, Path]:
    file = tmp_path / "a.txt"
    file.write_text(content, encoding="utf-8")
    w = FSWatcher(tmp_path)
    w.track("a.txt", content)
    return w, file


def test_intercept_bumps_and_notifies_before_return(tmp_path) -> None:
    w, file = _watcher(tmp_path)
    seen: list[str] = []

    def listener(ev):
        seen.append(file.read_text(encoding="utf-8"))   # write already landed
        return {"ok": True}

    w.subscribe(listener)
    ev = w.intercept_write("a.txt", "two\n")
    assert seen == ["two\n"]
    assert ev.old_version == 1
    assert ev.new_version == 2
    assert ev.old_content == "one\n"
    assert ev.new_content == "two\n"
    assert ev.source == "tool"
    assert w._last_result["a.txt"] == {"ok": True}
    assert w.version("a.txt").version == 2


def test_create_event(tmp_path) -> None:
    w = FSWatcher(tmp_path)
    ev = w.intercept_write("new.txt", "hello\n")
    assert ev.source == "create"
    assert ev.old_version is None and ev.new_version == 1
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\n"


def test_rescan_detects_out_of_band(tmp_path) -> None:
    w, file = _watcher(tmp_path)
    file.write_text("replaced content\n", encoding="utf-8")   # bypasses intercept_write
    events = w.rescan()
    assert len(events) == 1
    ev = events[0]
    assert ev.source == "rescan"
    assert ev.old_version == 1 and ev.new_version == 2
    assert ev.old_content == "one\n"
    assert ev.new_content == "replaced content\n"
    assert w.version("a.txt").version == 2


def test_rescan_touch_no_event(tmp_path) -> None:
    w, file = _watcher(tmp_path)
    os.utime(file, (1_700_000_000, 1_700_000_000))
    assert w.rescan() == []
    assert w.version("a.txt").version == 1


def test_delete_event(tmp_path) -> None:
    w, file = _watcher(tmp_path)
    cache = KVCache(agent_id="t", namespace="ns")
    cache.get_blocks("a.txt", "one\n")
    assert cache.version_of("a.txt") is not None

    def on_write(ev: WriteEvent) -> int:
        return cache.drop_file(ev.path)

    w.subscribe(on_write)
    file.unlink()
    events = w.rescan()
    assert len(events) == 1
    assert events[0].source == "delete"
    assert events[0].new_content == ""
    assert events[0].new_version == 2
    assert w.version("a.txt") is None
    assert cache.version_of("a.txt") is None


def test_track_is_idempotent(tmp_path) -> None:
    w, _ = _watcher(tmp_path)
    first = w.track("a.txt", "one\n")
    second = w.track("a.txt", "one\n")
    assert first == second
    assert first.version == 1
    assert w.tracked_paths() == ["a.txt"]
