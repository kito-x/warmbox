from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class TruncationPlan:
    first_divergent: int
    keep: range            # range(0, d)            -> re-key to new version
    drop: range            # range(d, len(old))     -> delete from L1
    compute: range         # range(d, len(new))     -> will miss on next get_blocks


def find_first_divergent_chunk(old_hashes: Sequence[str], new_hashes: Sequence[str]) -> int:
    """Smallest i with old[i] != new[i]; if one is a prefix of the other, min(len(old), len(new));
    if identical, len(old). Never raises."""
    for i in range(min(len(old_hashes), len(new_hashes))):
        if old_hashes[i] != new_hashes[i]:
            return i
    return min(len(old_hashes), len(new_hashes))


def plan_truncation(old_hashes: Sequence[str], new_hashes: Sequence[str]) -> TruncationPlan:
    d = find_first_divergent_chunk(old_hashes, new_hashes)
    return TruncationPlan(
        first_divergent=d,
        keep=range(0, d),
        drop=range(d, len(old_hashes)),
        compute=range(d, len(new_hashes)),
    )


def tokens_in(chunk_tokens: Sequence[int], r: range) -> int:
    return sum(chunk_tokens[i] for i in r)
