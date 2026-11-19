from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ..errors import SandboxError
from .policy import SandboxPolicy

log = logging.getLogger("warmbox.sandbox")


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int             # -1 if timed out / killed
    stdout: str                # utf-8 errors="replace", truncated to max_output_bytes
    stderr: str
    duration_ms: float
    timed_out: bool
    truncated: bool


class SandboxBackend(Protocol):
    name: str

    def run(
        self,
        argv: Sequence[str],
        *,
        workspace: Path,
        env: Mapping[str, str],
        timeout_s: float,
        policy: SandboxPolicy,
    ) -> CommandResult: ...


def _limits(policy: SandboxPolicy):
    def apply() -> None:
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_CPU, (policy.cpu_seconds, policy.cpu_seconds))
        except (ValueError, OSError):  # pragma: no cover
            pass
        try:
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (policy.max_file_write_bytes, policy.max_file_write_bytes),
            )
        except (ValueError, OSError):  # pragma: no cover
            pass
        if sys.platform.startswith("linux"):
            try:
                resource.setrlimit(
                    resource.RLIMIT_AS, (policy.memory_bytes, policy.memory_bytes)
                )
            except (ValueError, OSError):  # pragma: no cover
                pass

    return apply


def _decode(raw: bytes, limit: int) -> tuple[str, bool]:
    truncated = len(raw) > limit
    if truncated:
        raw = raw[:limit]
    return raw.decode("utf-8", errors="replace"), truncated


class LocalProcessBackend:
    """MVP default. Filesystem jail + argv allowlist are real; network isolation is not."""

    name = "local"
    _warned = False

    def run(
        self,
        argv: Sequence[str],
        *,
        workspace: Path,
        env: Mapping[str, str],
        timeout_s: float,
        policy: SandboxPolicy,
    ) -> CommandResult:
        if not LocalProcessBackend._warned:
            LocalProcessBackend._warned = True
            log.warning(
                "local backend: network isolation not enforced; use --sandbox docker"
            )
        t0 = time.perf_counter()
        prog = argv[0]
        if prog in ("python", "python3"):
            exe = sys.executable
        else:
            exe = shutil.which(prog, path=env.get("PATH", ""))
            if exe is None:
                return CommandResult(tuple(argv), 127, "", "command not found", 0.0, False, False)

        proc = subprocess.Popen(
            [exe, *argv[1:]],
            cwd=str(workspace),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_limits(policy),
        )
        timed_out = False
        try:
            out_raw, err_raw = proc.communicate(timeout=timeout_s)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                pass
            out_raw, err_raw = proc.communicate()
            exit_code = -1
        duration_ms = (time.perf_counter() - t0) * 1000
        limit = policy.max_output_bytes
        stdout, t1 = _decode(out_raw or b"", limit)
        stderr, t2 = _decode(err_raw or b"", limit)
        return CommandResult(
            tuple(argv), exit_code, stdout, stderr, duration_ms, timed_out, t1 or t2
        )


class DockerBackend:
    """Enforced isolation: no network, read-only rootfs, dropped capabilities."""

    name = "docker"

    def __init__(self, image: str = "python:3.12-slim") -> None:
        self.image = image
        self._n = 0

    def run(
        self,
        argv: Sequence[str],
        *,
        workspace: Path,
        env: Mapping[str, str],
        timeout_s: float,
        policy: SandboxPolicy,
    ) -> CommandResult:
        self._n += 1
        name = f"warmbox-{Path(workspace).name}-{self._n}"
        cmd = ["docker", "run", "--rm", "--name", name]
        if not policy.network:
            cmd += ["--network", "none"]
        cmd += [
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(policy.max_processes),
            "--memory",
            "1g",
            "--cpus",
            "1",
            "--user",
            "65534:65534",
            "-v",
            f"{workspace}:/workspace:rw",
            "-w",
            "/workspace",
            "-e",
            "HOME=/workspace",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONHASHSEED=0",
            self.image,
            *argv,
        ]
        t0 = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout_s,
                env=dict(env),
            )
            out_raw, err_raw, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            try:
                subprocess.run(
                    ["docker", "kill", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
            except Exception:  # pragma: no cover
                pass
            out_raw, err_raw, exit_code = stdout, stderr, -1
        duration_ms = (time.perf_counter() - t0) * 1000
        limit = policy.max_output_bytes
        stdout, t1 = _decode(out_raw or b"", limit)
        stderr, t2 = _decode(err_raw or b"", limit)
        return CommandResult(
            tuple(argv), exit_code, stdout, stderr, duration_ms, timed_out, t1 or t2
        )


class FirecrackerBackend:
    """Phase 4."""

    name = "firecracker"

    def __init__(self, kernel: Path, rootfs: Path) -> None:
        self.kernel = kernel
        self.rootfs = rootfs

    def run(self, *a, **kw) -> CommandResult:
        raise NotImplementedError("Phase 4: Firecracker backend")


def make_backend(name: str) -> SandboxBackend:
    if name == "local":
        return LocalProcessBackend()
    if name == "docker":
        return DockerBackend()
    if name == "firecracker":
        raise SandboxError("Firecracker backend is a Phase 4 stub")
    raise SandboxError(f"unknown sandbox backend: {name!r}")
