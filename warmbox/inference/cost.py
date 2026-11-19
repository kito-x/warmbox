from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..config import (
    CACHED_MS_PER_TOKEN,
    LATENCY_OVERHEAD_MS,
    PREFILL_MS_PER_TOKEN,
    PRICE_CACHED_PER_MTOK,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
    Settings,
)


@dataclass(frozen=True, slots=True)
class CostModel:
    input_per_mtok: float = PRICE_INPUT_PER_MTOK
    cached_per_mtok: float = PRICE_CACHED_PER_MTOK
    output_per_mtok: float = PRICE_OUTPUT_PER_MTOK

    @classmethod
    def from_settings(cls, s: Settings) -> "CostModel":
        return cls(s.price_input, s.price_cached, s.price_output)

    def turn_cost(self, *, uncached_input: int, cached_input: int, output: int) -> float:
        return (
            uncached_input * self.input_per_mtok
            + cached_input * self.cached_per_mtok
            + output * self.output_per_mtok
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class LatencyModel:                       # SIMULATED TTFT
    overhead_ms: float = LATENCY_OVERHEAD_MS
    prefill_ms_per_token: float = PREFILL_MS_PER_TOKEN
    cached_ms_per_token: float = CACHED_MS_PER_TOKEN

    def ttft_ms(self, *, recompute_tokens: int, reused_tokens: int) -> float:
        return (
            self.overhead_ms
            + recompute_tokens * self.prefill_ms_per_token
            + reused_tokens * self.cached_ms_per_token
        )


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(values)
    if not s:
        return 0.0
    k = max(1, math.ceil(p / 100 * len(s)))
    return float(s[k - 1])
