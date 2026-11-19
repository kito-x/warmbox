from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..config import (
    DEFAULT_CMD_TIMEOUT_S,
    MAX_CMD_OUTPUT_BYTES,
    MAX_CMD_TIMEOUT_S,
)
from ..errors import SandboxViolation

DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "find",
        "diff",
        "echo",
        "true",
        "false",
        "sort",
        "uniq",
        "git",
    }
)
GIT_ALLOWED_SUBCOMMANDS: frozenset[str] = frozenset(
    {"status", "diff", "log", "show", "ls-files"}
)
FIND_FORBIDDEN_ARGS: frozenset[str] = frozenset(
    {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"}
)

SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_PASSWD")


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    allowlist: frozenset[str] = DEFAULT_ALLOWLIST
    network: bool = False
    cpu_seconds: int = 60
    memory_bytes: int = 1 << 30            # 1 GiB (Docker enforces; RLIMIT_AS best-effort on Linux)
    max_processes: int = 256
    max_file_write_bytes: int = 64 << 20   # RLIMIT_FSIZE
    max_output_bytes: int = MAX_CMD_OUTPUT_BYTES
    default_timeout_s: float = DEFAULT_CMD_TIMEOUT_S
    max_timeout_s: float = MAX_CMD_TIMEOUT_S
    env_passthrough: frozenset[str] = frozenset({"PATH", "LANG", "LC_ALL", "TERM"})


def _inside(real: str, workspace: str) -> bool:
    try:
        return os.path.commonpath([real, workspace]) == workspace
    except ValueError:
        return False


def check_argv(argv: Sequence[str], workspace: Path, policy: SandboxPolicy) -> None:
    """Raise SandboxViolation unless the argv is a policy-clean command for this workspace."""
    if not argv:
        raise SandboxViolation("empty argv")
    for arg in argv:
        if not isinstance(arg, str):
            raise SandboxViolation(f"argument is not a string: {arg!r}")
        if "\x00" in arg:
            raise SandboxViolation(f"NUL byte in argument: {arg[:40]!r}")

    prog = argv[0]
    if "/" in prog or os.sep in prog:
        raise SandboxViolation(f"program must be a bare name, not a path: {prog!r}")
    if prog not in policy.allowlist:
        raise SandboxViolation(f"command not in allowlist: {prog!r}")

    if prog == "git":
        if len(argv) < 2 or argv[1] not in GIT_ALLOWED_SUBCOMMANDS:
            raise SandboxViolation(
                f"git subcommand not allowed: {argv[1:2] or ['<none>']}"
            )
    if prog == "find":
        for arg in argv[1:]:
            if arg in FIND_FORBIDDEN_ARGS:
                raise SandboxViolation(f"find argument not allowed: {arg!r}")
    if prog in ("python", "python3"):
        rest = list(argv[1:])
        if len(rest) >= 2 and rest[0] == "-m" and rest[1] == "pip":
            raise SandboxViolation("package installation is not allowed")
        if rest and rest[0].startswith("-m") and rest[0][2:] == "pip":
            raise SandboxViolation("package installation is not allowed")

    ws = os.path.realpath(workspace)
    for arg in argv[1:]:
        parts = arg.replace(os.sep, "/").split("/")
        looks_like_path = os.path.isabs(arg) or ".." in parts
        if not looks_like_path:
            continue
        candidate = arg if os.path.isabs(arg) else os.path.join(ws, arg)
        if not _inside(os.path.realpath(candidate), ws):
            raise SandboxViolation(f"argument escapes the workspace: {arg!r}")


def build_env(
    workspace: Path, policy: SandboxPolicy, base_env: Mapping[str, str]
) -> dict[str, str]:
    """A minimal environment: pass-through keys only, no secrets, HOME and TMPDIR inside."""
    env: dict[str, str] = {
        k: base_env[k] for k in policy.env_passthrough if k in base_env
    }
    ws = str(workspace)
    env["HOME"] = ws
    env["TMPDIR"] = os.path.join(ws, ".tmp")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    if not policy.network:
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env[key] = "http://127.0.0.1:9"
        env["no_proxy"] = ""
    for key in list(env):
        if key.upper().endswith(SECRET_SUFFIXES) and key not in policy.env_passthrough:
            del env[key]
    return env
