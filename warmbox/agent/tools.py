from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import MAX_TOOL_OUTPUT_CHARS
from ..errors import SandboxError
from ..inference.tokenizer import count_tokens
from ..sandbox.runtime import Sandbox
from .history import ToolCall
from .working_set import WorkingSet

NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    _schema(
        "read_file",
        "Read a UTF-8 text file from the workspace and pin it into the WORKING SET.",
        {"path": {"type": "string", "description": "Workspace-relative POSIX path."}},
        ["path"],
    ),
    _schema(
        "write_file",
        "Create or replace an entire file with new content.",
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        ["path", "content"],
    ),
    _schema(
        "edit_file",
        "Replace one exact, unique substring of an existing file.",
        {
            "path": {"type": "string"},
            "old_str": {"type": "string", "description": "Must occur exactly once."},
            "new_str": {"type": "string"},
        },
        ["path", "old_str", "new_str"],
    ),
    _schema(
        "list_dir",
        "List directory entries; directories end with '/'.",
        {"path": {"type": "string"}},
        [],
    ),
    _schema(
        "search",
        "Regex search across workspace files.",
        {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        ["pattern"],
    ),
    _schema(
        "run",
        "Run an allowlisted command inside the sandbox (no shell, no network).",
        {
            "argv": {"type": "array", "items": {"type": "string"}},
            "timeout_s": {"type": "number"},
        },
        ["argv"],
    ),
    _schema(
        "finish",
        "Declare the task complete with a short summary.",
        {"summary": {"type": "string"}},
        ["summary"],
    ),
    _schema(
        "unpin",
        "Remove a file from the WORKING SET.",
        {"path": {"type": "string"}},
        ["path"],
    ),
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    content: str               # truncated to MAX_TOOL_OUTPUT_CHARS
    is_error: bool
    written_paths: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    finished: bool = False


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    dropped = len(text) - MAX_TOOL_OUTPUT_CHARS
    return text[:MAX_TOOL_OUTPUT_CHARS] + f"\n…[truncated {dropped} chars]"


class ToolExecutor:
    """Executes tool calls. Never raises for sandbox/IO/argument errors: they become error results."""

    def __init__(self, sandbox: Sandbox, working_set: WorkingSet) -> None:
        self.sandbox = sandbox
        self.working_set = working_set
        # DESIGN-GAP: the design document does not say who tells the working set the current
        # turn. AgentLoop sets this before executing each batch of calls.
        self.turn = 0

    # ------------------------------------------------------------------ entry

    def execute(self, call: ToolCall) -> ToolResult:
        if call.parse_error:
            return self._error(call, call.parse_error)
        handler = getattr(self, f"_tool_{call.name}", None)
        if handler is None:
            return self._error(call, f"unknown tool: {call.name!r}")
        try:
            content, written, read, finished = handler(call.arguments or {})
        except (SandboxError, FileNotFoundError, ValueError, KeyError, TypeError) as exc:
            return self._error(call, f"ERROR {type(exc).__name__}: {exc}")
        return ToolResult(
            call_id=call.id,
            name=call.name,
            content=_truncate(content),
            is_error=False,
            written_paths=tuple(written),
            read_paths=tuple(read),
            finished=finished,
        )

    def _error(self, call: ToolCall, message: str) -> ToolResult:
        text = message if message.startswith("ERROR ") else f"ERROR {message}"
        return ToolResult(call_id=call.id, name=call.name, content=_truncate(text), is_error=True)

    # ------------------------------------------------------------------ tools

    def _tool_read_file(self, args: dict[str, Any]) -> tuple[str, tuple, tuple, bool]:
        rel = self.sandbox.normalize(str(args["path"]))
        content = self.sandbox.read_file(rel)
        ntok = count_tokens(content)
        tf = self.sandbox.version_of(rel)
        ver = tf.version if tf else 1
        self.working_set.pin(rel, ntok, self.turn)
        return (
            f"OK read {rel} v{ver} ({ntok} tokens). Content is pinned in the WORKING SET.",
            (),
            (rel,),
            False,
        )

    def _write_report(self, rel: str, new_version: int, content: str) -> str:
        ntok = count_tokens(content)
        inv = self.sandbox.last_invalidation(rel)
        first = getattr(inv, "first_divergent_block", None) if inv is not None else None
        where = first if first is not None else "?"
        return f"OK wrote {rel} v{new_version} ({ntok} tokens, first changed block {where})."

    def _tool_write_file(self, args: dict[str, Any]) -> tuple[str, tuple, tuple, bool]:
        rel = self.sandbox.normalize(str(args["path"]))
        content = str(args["content"])
        ev = self.sandbox.write_file(rel, content)
        return self._write_report(ev.path, ev.new_version, content), (ev.path,), (), False

    def _tool_edit_file(self, args: dict[str, Any]) -> tuple[str, tuple, tuple, bool]:
        rel = self.sandbox.normalize(str(args["path"]))
        ev = self.sandbox.edit_file(rel, str(args["old_str"]), str(args["new_str"]))
        return self._write_report(ev.path, ev.new_version, ev.new_content), (ev.path,), (), False

    def _tool_list_dir(self, args: dict[str, Any]) -> tuple[str, tuple, tuple, bool]:
        path = str(args.get("path") or ".")
        entries = self.sandbox.list_dir(path)
        return "\n".join(entries), (), (), False

    def _tool_search(self, args: dict[str, Any]) -> tuple[str, tuple, tuple, bool]:
        path = str(args.get("path") or ".")
        hits = self.sandbox.search(str(args["pattern"]), path)
        body = "\n".join(f"{h.path}:{h.line}: {h.text}" for h in hits)
        tail = f"({len(hits)} matches)"
        return (f"{body}\n{tail}" if body else tail), (), (), False

    def _tool_run(self, args: dict[str, Any]) -> tuple[str, tuple, tuple, bool]:
        argv = args["argv"]
        if not isinstance(argv, list) or not argv:
            raise ValueError("argv must be a non-empty array of strings")
        timeout = args.get("timeout_s")
        res = self.sandbox.run_command(argv, float(timeout) if timeout is not None else None)
        header = f"exit={res.exit_code} time={res.duration_ms:.0f}ms{' TIMEOUT' if res.timed_out else ''}"
        text = f"{header}\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
        return text, (), (), False

    def _tool_finish(self, args: dict[str, Any]) -> tuple[str, tuple, tuple, bool]:
        return "FINISHED", (), (), True

    def _tool_unpin(self, args: dict[str, Any]) -> tuple[str, tuple, tuple, bool]:
        rel = self.sandbox.normalize(str(args["path"]))
        self.working_set.unpin(rel)
        return f"OK unpinned {rel}", (), (), False
