from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import BLOCK_SIZE
from ..inference.cost import LatencyModel
from ..inference.tokenizer import count_tokens
from .block_cache import KVCache
from .models import ParentRef

if TYPE_CHECKING:  # pragma: no cover
    from ..inference.prompt_builder import BuiltPrompt


@dataclass(frozen=True, slots=True)
class SegmentReport:
    key: str                 # STABLE_KEY | HISTORY_KEY | "<hdr>:path" | path
    kind: str                # "stable" | "history" | "file_header" | "file"
    tokens: int
    reused_tokens: int
    l2_reused_tokens: int
    recompute_tokens: int
    hit_blocks: int
    miss_blocks: int
    first_miss_block: int | None


@dataclass(frozen=True, slots=True)
class PrefillReport:
    prefill_tokens: int
    reused_tokens: int
    l2_reused_tokens: int
    recompute_tokens: int
    hit_blocks: int
    miss_blocks: int
    edit_recompute_tokens: int           # recompute tokens of edited "file" segments
    segments: tuple[SegmentReport, ...]
    sim_ttft_ms: float

    @property
    def reuse_ratio(self) -> float:
        return (self.reused_tokens / self.prefill_tokens) if self.prefill_tokens else 0.0


class CacheManager:
    """Turns a BuiltPrompt into per-segment get_blocks calls with the correct parent wiring.

    cache=None gives baseline A, where everything is recomputed."""

    def __init__(self, cache: KVCache | None, *, latency: LatencyModel | None = None) -> None:
        self._cache = cache
        self._latency = latency or LatencyModel()

    @property
    def cache(self) -> KVCache | None:
        return self._cache

    def prefill(
        self, prompt: "BuiltPrompt", *, edited_paths: frozenset[str] = frozenset()
    ) -> PrefillReport:
        reports: list[SegmentReport] = []
        if self._cache is None:
            for s in prompt.segments:
                n = count_tokens(s.text)
                reports.append(
                    SegmentReport(
                        key=s.key,
                        kind=s.kind,
                        tokens=n,
                        reused_tokens=0,
                        l2_reused_tokens=0,
                        recompute_tokens=n,
                        hit_blocks=0,
                        miss_blocks=math.ceil(n / BLOCK_SIZE) if n else 0,
                        first_miss_block=0 if n else None,
                    )
                )
        else:
            tails: dict[str | None, ParentRef] = {
                None: ParentRef.root(self._cache.namespace)
            }
            for s in prompt.segments:
                r = self._cache.get_blocks(
                    s.key,
                    s.text,
                    parent=tails[s.parent_key],
                    version=s.version if s.kind == "file" else None,
                )
                tails[s.key] = r.tail
                reports.append(
                    SegmentReport(
                        key=s.key,
                        kind=s.kind,
                        tokens=r.total_tokens,
                        reused_tokens=r.reused_tokens,
                        l2_reused_tokens=r.l2_reused_tokens,
                        recompute_tokens=r.recompute_tokens,
                        hit_blocks=r.hit_blocks,
                        miss_blocks=r.computed_blocks,
                        first_miss_block=r.first_miss_block,
                    )
                )

        prefill = sum(r.tokens for r in reports)
        reused = sum(r.reused_tokens for r in reports)
        l2 = sum(r.l2_reused_tokens for r in reports)
        recompute = sum(r.recompute_tokens for r in reports)
        hits = sum(r.hit_blocks for r in reports)
        misses = sum(r.miss_blocks for r in reports)
        edit_rc = sum(
            r.recompute_tokens for r in reports if r.kind == "file" and r.key in edited_paths
        )
        ttft = self._latency.ttft_ms(recompute_tokens=recompute, reused_tokens=reused)
        return PrefillReport(
            prefill_tokens=prefill,
            reused_tokens=reused,
            l2_reused_tokens=l2,
            recompute_tokens=recompute,
            hit_blocks=hits,
            miss_blocks=misses,
            edit_recompute_tokens=edit_rc,
            segments=tuple(reports),
            sim_ttft_ms=ttft,
        )
