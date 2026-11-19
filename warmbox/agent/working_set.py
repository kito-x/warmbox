from __future__ import annotations

from dataclasses import dataclass

from ..config import WORKING_SET_MAX_FILES, WORKING_SET_MAX_TOKENS


@dataclass(slots=True)
class _Entry:
    path: str
    tokens: int
    pin_index: int
    last_turn: int


class WorkingSet:
    """The live file set rendered at the end of every prompt.

    Keeps pin order (which never reorders) plus a last-access turn used only for eviction."""

    def __init__(
        self,
        max_files: int = WORKING_SET_MAX_FILES,
        max_tokens: int = WORKING_SET_MAX_TOKENS,
    ) -> None:
        if max_files < 1:
            raise ValueError("max_files must be >= 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._max_files = max_files
        self._max_tokens = max_tokens
        self._entries: dict[str, _Entry] = {}
        self._next_pin = 0

    # ------------------------------------------------------------------ public

    def pin(self, path: str, num_tokens: int, turn: int) -> list[str]:
        entry = self._entries.get(path)
        if entry is None:
            self._entries[path] = _Entry(path, num_tokens, self._next_pin, turn)
            self._next_pin += 1
        else:
            entry.tokens = num_tokens
            entry.last_turn = turn
        return self._evict()

    def touch(self, path: str, num_tokens: int, turn: int) -> list[str]:
        return self.pin(path, num_tokens, turn)

    def unpin(self, path: str) -> bool:
        return self._entries.pop(path, None) is not None

    def paths(self) -> list[str]:
        return [e.path for e in sorted(self._entries.values(), key=lambda e: e.pin_index)]

    def __contains__(self, path: str) -> bool:
        return path in self._entries

    def total_tokens(self) -> int:
        return sum(e.tokens for e in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ internals

    def _evict(self) -> list[str]:
        evicted: list[str] = []
        while len(self._entries) > 1 and (
            len(self._entries) > self._max_files or self.total_tokens() > self._max_tokens
        ):
            victim = min(
                self._entries.values(), key=lambda e: (e.last_turn, e.pin_index)
            )
            del self._entries[victim.path]
            evicted.append(victim.path)
        return evicted
