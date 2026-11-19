from __future__ import annotations


class WarmboxError(Exception):
    """Base class for every error raised by warmbox."""


class CacheError(WarmboxError):
    pass


class VersionConflictError(CacheError):
    """new_version <= current version."""


class CacheCorruptError(CacheError):
    """Persisted cache unreadable or mismatched."""


class PromptOrderError(WarmboxError):
    """Segment ordering invariant violated."""


class SandboxError(WarmboxError):
    pass


class SandboxViolation(SandboxError):
    """Path escape, disallowed command, policy breach."""


class NotTextFileError(SandboxError):
    """NUL byte in first 8 KiB or invalid UTF-8."""


class FileTooLargeError(SandboxError):
    """Larger than MAX_FILE_BYTES."""


class EditMatchError(SandboxError):
    """edit_file old_str count != 1."""


class LLMError(WarmboxError):
    pass


class LLMHTTPError(LLMError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status = status
        self.body = body


class LLMTimeoutError(LLMError):
    pass


class LLMProtocolError(LLMError):
    """Malformed SSE / JSON."""


class BenchError(WarmboxError):
    pass


# Compatibility aliases: the design document calls the base error InferCodeError.
InferCodeError = WarmboxError
