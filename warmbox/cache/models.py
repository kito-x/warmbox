from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .hashing import root_chain


class BlockStatus(StrEnum):
    HIT_L1 = "hit_l1"
    HIT_L2 = "hit_l2"
    COMPUTED = "computed"


@dataclass(frozen=True, slots=True)
class BlockKey:
    file_path: str          # workspace-relative POSIX path, or pseudo-file key
    file_version: int       # >= 1
    block_index: int        # >= 0
    content_hash: str       # 32 lowercase hex

    def __str__(self) -> str:
        return f"{self.file_path}@v{self.file_version}#{self.block_index}:{self.content_hash}"


@dataclass(frozen=True, slots=True)
class KVHandle:
    """SIMULATED stand-in for a paged KV block."""

    kv_id: str                           # == chain hash of the block
    num_tokens: int
    producer: str                        # agent_id that computed it
    simulated: bool = True


@dataclass(frozen=True, slots=True)
class CachedBlock:
    key: BlockKey
    kv: KVHandle
    num_tokens: int
    chain_hash: str
    status: BlockStatus                  # status *for the lookup that returned it*
    debug_text: str | None = None        # chunk text; stored only when KVCache(debug=True)


@dataclass(frozen=True, slots=True)
class ParentRef:
    namespace: str
    path: tuple[str, ...]                # content hashes from namespace root (for L2 walk)
    chain: str                           # chain hash of the last block before this segment

    @classmethod
    def root(cls, namespace: str) -> "ParentRef":
        return cls(namespace, (), root_chain(namespace))


@dataclass(frozen=True, slots=True)
class FileVersion:
    path: str
    version: int
    content_hash: str                    # chunk_hash(full content)
    chunk_hashes: tuple[str, ...]
    chunk_tokens: tuple[int, ...]        # token count per block
    num_tokens: int
    updated_at: float                    # time.time()


@dataclass(frozen=True, slots=True)
class BlockResult:
    path: str
    version: int
    blocks: tuple[CachedBlock, ...]
    tail: ParentRef                      # parent ref for a segment that follows this one
    total_tokens: int
    reused_tokens: int
    l2_reused_tokens: int
    recompute_tokens: int
    hit_blocks: int
    l1_hit_blocks: int
    l2_hit_blocks: int
    computed_blocks: int
    first_miss_block: int | None         # index of first COMPUTED block, None if all hit


@dataclass(frozen=True, slots=True)
class InvalidationResult:
    path: str
    old_version: int | None              # None if path was unknown
    new_version: int
    first_divergent_block: int           # len(old) if no divergence
    blocks_kept: int
    blocks_dropped: int
    tokens_kept: int
    tokens_dropped: int
    new_block_count: int


@dataclass(slots=True)                   # mutable
class CacheStats:
    lookups: int = 0
    hit_blocks: int = 0
    miss_blocks: int = 0
    l1_hit_blocks: int = 0
    l2_hit_blocks: int = 0
    prefill_tokens: int = 0
    reused_tokens: int = 0
    l2_reused_tokens: int = 0
    recompute_tokens: int = 0
    invalidations: int = 0
    invalidated_blocks: int = 0
    invalidated_tokens: int = 0
    kept_blocks_on_write: int = 0
    evicted_blocks: int = 0
    implicit_updates: int = 0            # pseudo-file content changed (e.g. history append)
    bypass_writes: int = 0               # real file changed without on_file_write
    chain_mismatches: int = 0            # key matched but chain did not -> dropped
    old_content_mismatches: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_blocks + self.miss_blocks
        return (self.hit_blocks / total) if total else 0.0

    @property
    def tokens_saved(self) -> int:
        return self.reused_tokens

    @property
    def reuse_ratio(self) -> float:
        return (self.reused_tokens / self.prefill_tokens) if self.prefill_tokens else 0.0

    def merge(self, other: "CacheStats") -> "CacheStats":
        out = CacheStats()
        for name in self.to_dict().keys():
            if name in ("hit_rate", "tokens_saved", "reuse_ratio"):
                continue
            setattr(out, name, getattr(self, name) + getattr(other, name))
        return out

    def to_dict(self) -> dict[str, int | float]:
        return {
            "lookups": self.lookups,
            "hit_blocks": self.hit_blocks,
            "miss_blocks": self.miss_blocks,
            "l1_hit_blocks": self.l1_hit_blocks,
            "l2_hit_blocks": self.l2_hit_blocks,
            "prefill_tokens": self.prefill_tokens,
            "reused_tokens": self.reused_tokens,
            "l2_reused_tokens": self.l2_reused_tokens,
            "recompute_tokens": self.recompute_tokens,
            "invalidations": self.invalidations,
            "invalidated_blocks": self.invalidated_blocks,
            "invalidated_tokens": self.invalidated_tokens,
            "kept_blocks_on_write": self.kept_blocks_on_write,
            "evicted_blocks": self.evicted_blocks,
            "implicit_updates": self.implicit_updates,
            "bypass_writes": self.bypass_writes,
            "chain_mismatches": self.chain_mismatches,
            "old_content_mismatches": self.old_content_mismatches,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "reuse_ratio": self.reuse_ratio,
        }
