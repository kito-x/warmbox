from __future__ import annotations

import difflib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from ..cache.hashing import chunk_hash
from ..config import (
    HOME_DIR,
    MAX_CMD_TIMEOUT_S,
    MAX_FILE_BYTES,
    SEARCH_MAX_MATCHES,
)
from ..errors import (
    EditMatchError,
    FileTooLargeError,
    NotTextFileError,
    SandboxError,
    SandboxViolation,
)
from .backends import CommandResult, LocalProcessBackend
from .fs_watcher import FSWatcher, Listener, WriteEvent
from .policy import SandboxPolicy, build_env, check_argv

COPY_IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", ".warmbox", ".infercode", "node_modules", "*.pyc", ".pytest_cache"
)
SKIP_DIRS = {".git", "__pycache__", ".tmp", ".warmbox", "node_modules", ".pytest_cache"}


@dataclass(frozen=True, slots=True)
class SearchHit:
    path: str
    line: int                  # 1-based
    text: str                  # line stripped of trailing newline, max 200 chars


class Sandbox:
    """A per-agent isolated copy of the repo with jailed file ops and command execution."""

    def __init__(
        self,
        source_repo: Path,
        agent_id: str,
        *,
        backend: Any = None,
        policy: SandboxPolicy | None = None,
        workspaces_root: Path | None = None,
    ) -> None:
        self.source_repo = Path(source_repo).resolve()
        self.agent_id = agent_id
        self.backend = backend if backend is not None else LocalProcessBackend()
        self.policy = policy or SandboxPolicy()
        root = Path(workspaces_root) if workspaces_root else Path(HOME_DIR) / "workspaces"
        root.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(os.path.realpath(root / agent_id))
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        shutil.copytree(self.source_repo, self.workspace, ignore=COPY_IGNORE, symlinks=True)
        self._orig = Path(str(self.workspace) + ".orig")
        if self._orig.exists():
            shutil.rmtree(self._orig)
        shutil.copytree(self.workspace, self._orig, symlinks=True)
        (self.workspace / ".tmp").mkdir(parents=True, exist_ok=True)
        self.watcher = FSWatcher(self.workspace)

    # ------------------------------------------------------------------ paths

    def normalize(self, path: str) -> str:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise SandboxViolation("bad path")
        p = PurePosixPath(path)
        if p.is_absolute():
            try:
                p = PurePosixPath(Path(path).resolve().relative_to(self.workspace))
            except ValueError:
                raise SandboxViolation(f"absolute path outside workspace: {path}") from None
        if any(part == ".." for part in p.parts):
            raise SandboxViolation(f"parent traversal: {path}")
        real = os.path.realpath(self.workspace / p)
        if os.path.commonpath([real, str(self.workspace)]) != str(self.workspace):
            raise SandboxViolation(f"escapes workspace via symlink: {path}")
        rel = PurePosixPath(os.path.relpath(real, self.workspace)).as_posix()
        return "." if rel == "." else rel

    def _abs(self, rel: str) -> Path:
        return self.workspace / rel

    # ------------------------------------------------------------------ file ops

    def read_file(self, path: str) -> str:
        rel = self.normalize(path)
        abs_p = self._abs(rel)
        if not abs_p.exists() or not abs_p.is_file():
            raise FileNotFoundError(rel)
        if abs_p.stat().st_size > MAX_FILE_BYTES:
            raise FileTooLargeError(f"{rel} is larger than {MAX_FILE_BYTES} bytes")
        raw = abs_p.read_bytes()
        if b"\x00" in raw[:8192]:
            raise NotTextFileError(f"{rel} is binary (NUL byte)")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NotTextFileError(f"{rel} is not valid UTF-8: {exc}") from None
        self.watcher.track(rel, text)
        return text

    def write_file(self, path: str, content: str) -> WriteEvent:
        rel = self.normalize(path)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise SandboxViolation(f"content larger than {MAX_FILE_BYTES} bytes: {rel}")
        return self.watcher.intercept_write(rel, content)

    def edit_file(self, path: str, old_str: str, new_str: str) -> WriteEvent:
        rel = self.normalize(path)
        text = self.read_file(rel)
        count = text.count(old_str)
        if count != 1:
            raise EditMatchError(f"old_str occurs {count} times in {rel} (expected exactly 1)")
        return self.write_file(rel, text.replace(old_str, new_str, 1))

    def list_dir(self, path: str = ".") -> list[str]:
        rel = self.normalize(path)
        base = self._abs(rel)
        if not base.is_dir():
            raise FileNotFoundError(rel)
        out: list[str] = []
        for entry in sorted(base.iterdir(), key=lambda p: p.name):
            if entry.name in {".tmp", "__pycache__"}:
                continue
            out.append(entry.name + "/" if entry.is_dir() else entry.name)
        return out

    def search(
        self, pattern: str, path: str = ".", max_matches: int = SEARCH_MAX_MATCHES
    ) -> list[SearchHit]:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise SandboxError(f"bad regex: {exc}") from exc
        rel = self.normalize(path)
        base = self._abs(rel)
        hits: list[SearchHit] = []
        for file in sorted(base.rglob("*") if base.is_dir() else [base]):
            if not file.is_file():
                continue
            parts = file.relative_to(self.workspace).parts
            if any(part in SKIP_DIRS for part in parts):
                continue
            try:
                text = file.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(
                        SearchHit(
                            PurePosixPath(*parts).as_posix(), lineno, line.rstrip("\n")[:200]
                        )
                    )
                    if len(hits) >= max_matches:
                        return hits
        return hits

    # ------------------------------------------------------------------ commands

    def run_command(self, argv: Sequence[str], timeout_s: float | None = None) -> CommandResult:
        return run_in_workspace(
            self.workspace,
            argv,
            policy=self.policy,
            backend=self.backend,
            timeout_s=timeout_s,
        )

    # ------------------------------------------------------------------ wiring

    def sync(self) -> list[WriteEvent]:
        return self.watcher.rescan()

    def version_of(self, path: str):
        return self.watcher.version(path)

    def subscribe(self, listener: Listener) -> None:
        self.watcher.subscribe(listener)

    def last_invalidation(self, path: str) -> Any | None:
        return self.watcher._last_result.get(path)

    # ------------------------------------------------------------------ snapshot

    def snapshot_files(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for rel in self._walk_text_files(self.workspace):
            try:
                text = (self.workspace / rel).read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            out.append((rel, chunk_hash(text)))
        return out

    def diff(self) -> str:
        chunks: list[str] = []
        orig = set(self._walk_text_files(self._orig))
        now = set(self._walk_text_files(self.workspace))
        for rel in sorted(orig | now):
            a = self._read_text(self._orig / rel)
            b = self._read_text(self.workspace / rel)
            if a == b:
                continue
            chunks.append(
                "".join(
                    difflib.unified_diff(
                        a.splitlines(keepends=True),
                        b.splitlines(keepends=True),
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                    )
                )
            )
        return "".join(chunks)

    def close(self, *, keep: bool = False) -> None:
        if keep:
            return
        shutil.rmtree(self.workspace, ignore_errors=True)
        shutil.rmtree(self._orig, ignore_errors=True)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _walk_text_files(root: Path) -> list[str]:
        out: list[str] = []
        if not root.exists():
            return out
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(root).parts
            if any(part in SKIP_DIRS for part in rel_parts):
                continue
            try:
                with path.open("rb") as fh:
                    head = fh.read(8192)
            except OSError:
                continue
            if b"\x00" in head:
                continue
            out.append(PurePosixPath(*rel_parts).as_posix())
        return out

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            return ""


def run_in_workspace(
    workspace: Path,
    argv: Sequence[str],
    *,
    policy: SandboxPolicy | None = None,
    backend: Any = None,
    timeout_s: float | None = None,
    base_env: dict[str, str] | None = None,
) -> CommandResult:
    """Run a policy-checked command inside an existing workspace directory.

    Used by Sandbox.run_command and, after the agent exits, by the benchmark's check step."""
    workspace = Path(workspace)
    policy = policy or SandboxPolicy()
    backend = backend if backend is not None else LocalProcessBackend()
    argv = [str(a) for a in argv]
    check_argv(argv, workspace, policy)
    timeout = timeout_s if timeout_s is not None else policy.default_timeout_s
    timeout = min(float(timeout), MAX_CMD_TIMEOUT_S)
    env = build_env(workspace, policy, base_env if base_env is not None else os.environ)
    return backend.run(
        argv, workspace=workspace, env=env, timeout_s=timeout, policy=policy
    )
