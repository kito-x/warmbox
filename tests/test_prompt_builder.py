from __future__ import annotations

import pytest

from warmbox.agent.history import History, Message
from warmbox.config import HEADER_KEY_PREFIX, HISTORY_KEY, STABLE_KEY
from warmbox.errors import PromptOrderError
from warmbox.inference.prompt_builder import (
    FileSegment,
    Segment,
    StablePrefix,
    assert_ordering,
    build_prompt,
    render_file_header,
    render_working_set,
)


def _stable() -> StablePrefix:
    return StablePrefix(system="SYS", tools=(), repo_map="repo_map\n")


def _file(path: str, content: str, version: int = 1) -> FileSegment:
    return FileSegment(path=path, content=content, version=version)


def test_ordering() -> None:
    hist = History()
    hist.append(Message("user", "do the thing"))
    prompt = build_prompt(_stable(), [_file("a.py", "x = 1\n")], hist.messages)
    kinds = [s.kind for s in prompt.segments]
    assert kinds == ["stable", "history", "file_header", "file"]
    assert prompt.segments[0].key == STABLE_KEY
    assert prompt.segments[1].key == HISTORY_KEY
    assert prompt.segments[2].key == HEADER_KEY_PREFIX + "a.py"
    assert prompt.messages[-1]["role"] == "user"
    assert prompt.messages[-1]["content"].startswith("# WORKING SET")
    # 4 segments: stable, history, header, file
    assert len(prompt.segments) == 4
    assert prompt.messages[0]["role"] == "system"
    assert "# REPO MAP" in prompt.messages[0]["content"]


def test_stable_prefix_byte_identical() -> None:
    hist = History()
    texts = []
    for i in range(5):
        hist.append(Message("user", f"turn {i}"))
        prompt = build_prompt(_stable(), [], hist.messages)
        texts.append(prompt.segments[0].text)
    assert len(set(texts)) == 1


def test_history_append_only() -> None:
    hist = History()
    hist.append(Message("user", "a"))
    assert not hasattr(hist, "__setitem__")
    assert not hasattr(hist, "pop")
    assert not hasattr(hist, "insert")
    assert not hasattr(hist, "clear")
    assert isinstance(hist.messages, tuple)
    with pytest.raises(TypeError):
        hist[0] = Message("user", "b")

    before = hist.render()
    hist.append(Message("assistant", "b"))
    after = hist.render()
    assert after.startswith(before)
    assert len(hist) == 2
    assert isinstance(hist[0:2], tuple)


def test_header_has_no_version() -> None:
    assert render_file_header("a.py") == render_file_header("a.py")
    assert "v1" not in render_file_header("a.py")
    assert render_file_header("a.py") == "=== FILE: a.py ===\n"


def test_render_working_set_normalises_trailing_newline() -> None:
    text = render_working_set([_file("a.py", "x = 1")])
    assert text == "=== FILE: a.py ===\nx = 1\n"


def test_assert_ordering_rejects() -> None:
    good = [
        Segment("stable", STABLE_KEY, "s", None, None),
        Segment("history", HISTORY_KEY, "h", STABLE_KEY, None),
        Segment("file_header", HEADER_KEY_PREFIX + "a.py", "hd", STABLE_KEY, None),
        Segment("file", "a.py", "body", HEADER_KEY_PREFIX + "a.py", 1),
    ]
    assert_ordering(good)

    # file before history
    bad = [good[0], good[2], good[3], good[1]]
    with pytest.raises(PromptOrderError):
        assert_ordering(bad)

    # missing stable
    with pytest.raises(PromptOrderError):
        assert_ordering(good[1:])

    # duplicate path
    dup = good + [good[2], good[3]]
    with pytest.raises(PromptOrderError):
        assert_ordering(dup)

    # unterminated pair
    with pytest.raises(PromptOrderError):
        assert_ordering(good + [good[2]])

    # header parented by history
    bad_parent = [
        good[0],
        good[1],
        Segment("file_header", HEADER_KEY_PREFIX + "a.py", "hd", HISTORY_KEY, None),
        good[3],
    ]
    with pytest.raises(PromptOrderError):
        assert_ordering(bad_parent)


def test_duplicate_path_in_build_prompt() -> None:
    with pytest.raises(PromptOrderError):
        build_prompt(_stable(), [_file("a.py", "1"), _file("a.py", "2")], [])
