from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from ..config import HEADER_KEY_PREFIX, HISTORY_KEY, STABLE_KEY
from ..errors import PromptOrderError
from ..agent.history import Message, canonical_json

WORKING_SET_INTRO = (
    "# WORKING SET (live files, refreshed every turn; not part of history)\n"
)


@dataclass(frozen=True, slots=True)
class StablePrefix:
    system: str
    tools: tuple[dict[str, Any], ...]          # OpenAI tool schemas
    repo_map: str

    def render(self) -> str:
        return (
            f"<|system|>\n{self.system}\n<|tools|>\n{canonical_json(list(self.tools))}\n"
            f"<|repo_map|>\n{self.repo_map}\n"
        )

    def hash(self) -> str:
        from ..cache.hashing import chunk_hash

        return chunk_hash(self.render())


@dataclass(frozen=True, slots=True)
class FileSegment:
    path: str
    content: str
    version: int


SegmentKind = Literal["stable", "history", "file_header", "file"]


@dataclass(frozen=True, slots=True)
class Segment:
    kind: SegmentKind
    key: str                 # STABLE_KEY | HISTORY_KEY | HEADER_KEY_PREFIX+path | path
    text: str
    parent_key: str | None   # None = namespace root
    version: int | None      # FileSegment.version for kind=="file", else None


@dataclass(frozen=True, slots=True)
class BuiltPrompt:
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    segments: tuple[Segment, ...]
    total_tokens: int        # sum(count_tokens(s.text) for s in segments)


def render_file_header(path: str) -> str:
    """Header text. Never contains the file version: headers are parents (I2)."""
    return f"=== FILE: {path} ===\n"


def render_working_set(files: Sequence[FileSegment]) -> str:
    return "".join(
        render_file_header(f.path) + f.content + ("" if f.content.endswith("\n") else "\n")
        for f in files
    )


def assert_ordering(segments: Sequence[Segment]) -> None:
    if not segments:
        raise PromptOrderError("empty segment list")
    if segments[0].kind != "stable":
        raise PromptOrderError(f"segment 0 is {segments[0].kind}, expected stable")
    if sum(1 for s in segments if s.kind == "stable") != 1:
        raise PromptOrderError("expected exactly one stable segment")
    if len(segments) < 2 or segments[1].kind != "history":
        raise PromptOrderError("segment 1 is not history")
    if sum(1 for s in segments if s.kind == "history") != 1:
        raise PromptOrderError("expected exactly one history segment")
    body = list(segments[2:])
    if len(body) % 2:
        raise PromptOrderError(
            f"file segments must be header/body pairs, got {len(body)} trailing segments"
        )
    for i in range(0, len(body), 2):
        hdr, body_seg = body[i], body[i + 1]
        if hdr.kind != "file_header":
            raise PromptOrderError(f"index {i + 2} is {hdr.kind}, expected file_header")
        if body_seg.kind != "file":
            raise PromptOrderError(f"index {i + 3} is {body_seg.kind}, expected file")
        if hdr.parent_key != STABLE_KEY:
            raise PromptOrderError(f"file header {hdr.key} must be parented by {STABLE_KEY}")
        if body_seg.parent_key != hdr.key:
            raise PromptOrderError(
                f"file {body_seg.key} must be parented by its header {hdr.key}"
            )
        if not hdr.key.startswith(HEADER_KEY_PREFIX):
            raise PromptOrderError(f"file header key {hdr.key} must start with {HEADER_KEY_PREFIX}")
        if hdr.key != HEADER_KEY_PREFIX + body_seg.key:
            raise PromptOrderError(f"header key {hdr.key} does not match file {body_seg.key}")
    paths = [s.key for s in body if s.kind == "file"]
    if len(set(paths)) != len(paths):
        raise PromptOrderError("duplicate file segment")


def build_prompt(
    stable_prefix: StablePrefix,
    file_blocks: Sequence[FileSegment],
    history: Sequence[Message],
) -> BuiltPrompt:
    from ..inference.tokenizer import count_tokens

    hist_text = "".join(m.render() for m in history)
    segs: list[Segment] = [
        Segment("stable", STABLE_KEY, stable_prefix.render(), None, None),
        Segment("history", HISTORY_KEY, hist_text, STABLE_KEY, None),
    ]
    seen: set[str] = set()
    for f in file_blocks:  # caller order == WorkingSet pin order
        if f.path in seen:
            raise PromptOrderError(f"duplicate file {f.path}")
        seen.add(f.path)
        hk = HEADER_KEY_PREFIX + f.path
        segs.append(Segment("file_header", hk, render_file_header(f.path), STABLE_KEY, None))
        segs.append(Segment("file", f.path, f.content, hk, f.version))
    assert_ordering(segs)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": f"{stable_prefix.system}\n\n# REPO MAP\n{stable_prefix.repo_map}",
        }
    ]
    messages += [m.to_openai() for m in history]
    if file_blocks:
        messages.append(
            {"role": "user", "content": WORKING_SET_INTRO + render_working_set(file_blocks)}
        )
    return BuiltPrompt(
        messages=tuple(messages),
        tools=stable_prefix.tools,
        segments=tuple(segs),
        total_tokens=sum(count_tokens(s.text) for s in segs),
    )
