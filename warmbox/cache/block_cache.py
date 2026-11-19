from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import (
    BLOCK_SIZE,
    CACHE_FORMAT,
    CACHE_FORMAT_VERSION,
    L1_MAX_BLOCKS,
    STABLE_KEY,
)
from ..errors import CacheCorruptError, CacheError, VersionConflictError
from ..inference.tokenizer import chunk_tokens, tokenize
from .hashing import HASH_ALGO, chain_hash, chunk_hash
from .invalidator import plan_truncation, tokens_in
from .models import (
    BlockKey,
    BlockResult,
    BlockStatus,
    CachedBlock,
    CacheStats,
    FileVersion,
    InvalidationResult,
    KVHandle,
    ParentRef,
)

if TYPE_CHECKING:  # pragma: no cover
    from .radix_tree import RadixTree

log = logging.getLogger("warmbox.cache")


def is_pseudo(path: str) -> bool:
    """A pseudo-file is a segment that is not a workspace file: <stable>, <history>, <hdr>:path."""
    return path.startswith("<")


class KVCache:
    """L1: the per-agent block index, file-version registry, invalidation and simulated KV compute."""

    def __init__(
        self,
        *,
        agent_id: str = "agent-0",
        namespace: str = "default",
        max_blocks: int = L1_MAX_BLOCKS,
        block_size: int = BLOCK_SIZE,
        shared: "RadixTree | None" = None,
        debug: bool = False,
    ) -> None:
        if max_blocks < 1:
            raise ValueError("max_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.agent_id = agent_id
        self.namespace = namespace
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.shared = shared
        self.debug = debug
        self._pinned: dict[BlockKey, CachedBlock] = {}
        self._lru: OrderedDict[BlockKey, CachedBlock] = OrderedDict()
        self._by_path: dict[str, set[BlockKey]] = {}
        self._versions: dict[str, FileVersion] = {}
        self._stats = CacheStats()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ chunking

    @staticmethod
    def chunk_hash(text: str) -> str:
        return chunk_hash(text)

    def chunk(self, content: str) -> list[tuple[str, int]]:
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        return chunk_tokens(tokenize(content), self.block_size)

    # ------------------------------------------------------------------ lookup

    def get_blocks(
        self,
        file_path: str,
        content: str,
        *,
        parent: ParentRef | None = None,
        version: int | None = None,
    ) -> BlockResult:
        with self._lock:
            parent = parent or ParentRef.root(self.namespace)
            chunks = self.chunk(content)
            hashes = [chunk_hash(t) for t, _ in chunks]
            ntoks = [n for _, n in chunks]

            # 1. version registry / implicit-update detection (I5, I6)
            fv = self._versions.get(file_path)
            if fv is None:
                ver = version if version is not None else 1
                self._register(file_path, ver, content, hashes, ntoks)
            elif version is not None and version != fv.version:
                if version < fv.version:
                    raise VersionConflictError(f"{file_path}: v{version} < v{fv.version}")
                self._invalidate(file_path, fv, content, hashes, ntoks, new_version=version)
                self._stats.bypass_writes += 1
            elif tuple(hashes) != fv.chunk_hashes:
                self._invalidate(
                    file_path, fv, content, hashes, ntoks, new_version=fv.version + 1
                )
                if is_pseudo(file_path):
                    self._stats.implicit_updates += 1
                else:
                    self._stats.bypass_writes += 1
            ver = self._versions[file_path].version

            # 2. block walk with chain verification (I1)
            chain = parent.chain
            out: list[CachedBlock] = []
            l2_match = None
            for i, (text, n) in enumerate(chunks):
                h = hashes[i]
                chain = chain_hash(chain, h)
                key = BlockKey(file_path, ver, i, h)
                blk = self._get_l1(key)
                if blk is not None and blk.chain_hash == chain:
                    out.append(replace(blk, status=BlockStatus.HIT_L1))
                    continue
                if blk is not None:  # same key, different parent context
                    self._remove_l1(key)
                    self._stats.chain_mismatches += 1

                handle: KVHandle | None = None
                status = BlockStatus.COMPUTED
                if self.shared is not None:
                    if l2_match is None:
                        l2_match = self.shared.longest_prefix_match(
                            self.namespace, parent.path + tuple(hashes)
                        )
                    j = len(parent.path) + i
                    if j < l2_match.matched and l2_match.handles[j].kv_id == chain:  # I7
                        handle, status = l2_match.handles[j], BlockStatus.HIT_L2
                if handle is None:
                    handle = self.compute_kv(text, chain, n)
                blk = CachedBlock(key, handle, n, chain, status, text if self.debug else None)
                self._put_l1(key, blk)
                out.append(blk)

            # 3. publish to L2 (idempotent: existing prefix is kept)
            if self.shared is not None and out:
                full_path = parent.path + tuple(hashes)
                self.shared.insert(self.namespace, full_path, [b.kv for b in out])
                if file_path == STABLE_KEY:
                    self.shared.pin(self.namespace, full_path)

            # 4. stats + result
            tail = ParentRef(self.namespace, parent.path + tuple(hashes), chain)
            res = _summarize(file_path, ver, out, tail)
            self._stats.lookups += 1
            self._stats.hit_blocks += res.hit_blocks
            self._stats.miss_blocks += res.computed_blocks
            self._stats.l1_hit_blocks += res.l1_hit_blocks
            self._stats.l2_hit_blocks += res.l2_hit_blocks
            self._stats.prefill_tokens += res.total_tokens
            self._stats.reused_tokens += res.reused_tokens
            self._stats.l2_reused_tokens += res.l2_reused_tokens
            self._stats.recompute_tokens += res.recompute_tokens
            return res

    def compute_kv(self, chunk_text: str, chain: str, num_tokens: int) -> KVHandle:
        """SIMULATED prefill of one block. Returns a handle whose kv_id is the chain hash.

        Phase 4 replaces the body with a real engine call. Does NOT touch stats."""
        return KVHandle(kv_id=chain, num_tokens=num_tokens, producer=self.agent_id)

    # ------------------------------------------------------------------ writes

    def on_file_write(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        *,
        new_version: int | None = None,
    ) -> InvalidationResult:
        with self._lock:
            new_chunks = self.chunk(new_content)
            new_hashes = [chunk_hash(t) for t, _ in new_chunks]
            new_ntoks = [n for _, n in new_chunks]
            fv = self._versions.get(file_path)
            if fv is None:  # never cached: nothing to invalidate
                v = new_version or 1
                self._register(file_path, v, new_content, new_hashes, new_ntoks)
                return InvalidationResult(
                    file_path, None, v, 0, 0, 0, 0, 0, len(new_hashes)
                )
            if new_version is not None and new_version <= fv.version:
                raise VersionConflictError(f"{file_path}: v{new_version} <= v{fv.version}")
            if chunk_hash(old_content) != fv.content_hash:  # registry wins
                self._stats.old_content_mismatches += 1
            return self._invalidate(
                file_path,
                fv,
                new_content,
                new_hashes,
                new_ntoks,
                new_version=new_version or fv.version + 1,
            )

    def _invalidate(
        self,
        file_path: str,
        fv: FileVersion,
        new_content: str,
        new_hashes: list[str],
        new_ntoks: list[int],
        *,
        new_version: int,
    ) -> InvalidationResult:
        plan = plan_truncation(fv.chunk_hashes, new_hashes)
        # 1. re-key surviving prefix blocks to the new version (chain hashes unchanged)
        for i in plan.keep:
            old_key = BlockKey(file_path, fv.version, i, fv.chunk_hashes[i])
            blk = self._pop_l1(old_key)
            if blk is not None:
                new_key = BlockKey(file_path, new_version, i, fv.chunk_hashes[i])
                self._put_l1(new_key, replace(blk, key=new_key))
        # 2. drop everything else for this path (divergent tail + stale-version stragglers)
        for key in list(self._by_path.get(file_path, ())):
            if key.file_version != new_version:
                self._remove_l1(key)
        # 3. registry + stats
        self._versions[file_path] = FileVersion(
            file_path,
            new_version,
            chunk_hash(new_content),
            tuple(new_hashes),
            tuple(new_ntoks),
            sum(new_ntoks),
            time.time(),
        )
        kept_tok = tokens_in(fv.chunk_tokens, plan.keep)
        drop_tok = tokens_in(fv.chunk_tokens, plan.drop)
        self._stats.invalidations += 1
        self._stats.invalidated_blocks += len(plan.drop)
        self._stats.invalidated_tokens += drop_tok
        self._stats.kept_blocks_on_write += len(plan.keep)
        return InvalidationResult(
            file_path,
            fv.version,
            new_version,
            plan.first_divergent,
            len(plan.keep),
            len(plan.drop),
            kept_tok,
            drop_tok,
            len(new_hashes),
        )

    def _register(
        self,
        file_path: str,
        ver: int,
        content: str,
        hashes: list[str],
        ntoks: list[int],
    ) -> None:
        self._versions[file_path] = FileVersion(
            file_path,
            ver,
            chunk_hash(content),
            tuple(hashes),
            tuple(ntoks),
            sum(ntoks),
            time.time(),
        )

    # ------------------------------------------------------------------ L1 internals

    def _get_l1(self, key: BlockKey) -> CachedBlock | None:
        blk = self._pinned.get(key)
        if blk is not None:
            return blk
        blk = self._lru.get(key)
        if blk is not None:
            self._lru.move_to_end(key)
        return blk

    def _put_l1(self, key: BlockKey, blk: CachedBlock) -> None:
        if key.file_path == STABLE_KEY:
            self._pinned[key] = blk
        else:
            while len(self) >= self.max_blocks:
                try:
                    old_key, _ = self._lru.popitem(last=False)
                except KeyError:
                    raise CacheError("L1 full of pinned blocks") from None
                self._by_path.get(old_key.file_path, set()).discard(old_key)
                self._stats.evicted_blocks += 1
            self._lru[key] = blk
        self._by_path.setdefault(key.file_path, set()).add(key)

    def _remove_l1(self, key: BlockKey) -> None:
        if key in self._pinned:
            del self._pinned[key]
        elif key in self._lru:
            del self._lru[key]
        keys = self._by_path.get(key.file_path)
        if keys is not None:
            keys.discard(key)
            if not keys:
                self._by_path.pop(key.file_path, None)

    def _pop_l1(self, key: BlockKey) -> CachedBlock | None:
        blk = self._pinned.get(key)
        if blk is None:
            blk = self._lru.get(key)
        if blk is not None:
            self._remove_l1(key)
        return blk

    # ------------------------------------------------------------------ public utils

    def drop_file(self, file_path: str) -> int:
        with self._lock:
            keys = list(self._by_path.get(file_path, ()))
            for key in keys:
                self._remove_l1(key)
            self._versions.pop(file_path, None)
            return len(keys)

    def evict(self, n: int = 1) -> int:
        with self._lock:
            freed = 0
            while freed < n and self._lru:
                old_key, _ = self._lru.popitem(last=False)
                self._by_path.get(old_key.file_path, set()).discard(old_key)
                self._stats.evicted_blocks += 1
                freed += 1
            return freed

    def version_of(self, file_path: str) -> FileVersion | None:
        with self._lock:
            return self._versions.get(file_path)

    def stats(self) -> CacheStats:
        with self._lock:
            return replace(self._stats)

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = CacheStats()

    def __len__(self) -> int:
        return len(self._pinned) + len(self._lru)

    def __contains__(self, key: BlockKey) -> bool:
        return key in self._pinned or key in self._lru

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        with self._lock:
            blocks: list[list[object]] = []
            for store in (self._pinned, self._lru):
                for key, blk in store.items():
                    blocks.append(
                        [
                            key.file_path,
                            key.file_version,
                            key.block_index,
                            key.content_hash,
                            blk.chain_hash,
                            blk.num_tokens,
                            blk.kv.producer,
                        ]
                    )
            files = {
                p: {
                    "version": fv.version,
                    "content_hash": fv.content_hash,
                    "chunk_hashes": list(fv.chunk_hashes),
                    "chunk_tokens": list(fv.chunk_tokens),
                    "num_tokens": fv.num_tokens,
                    "updated_at": fv.updated_at,
                }
                for p, fv in self._versions.items()
            }
            doc = {
                "format": CACHE_FORMAT,
                "format_version": CACHE_FORMAT_VERSION,
                "hash_algo": HASH_ALGO,
                "block_size": self.block_size,
                "namespace": self.namespace,
                "agent_id": self.agent_id,
                "files": files,
                "blocks": blocks,
            }
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_text(json.dumps(doc), "utf-8")
            os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path, **kwargs) -> "KVCache":
        path = Path(path)
        try:
            doc = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise CacheCorruptError(f"cannot read cache {path}: {exc}") from exc
        if not isinstance(doc, dict) or doc.get("format") != CACHE_FORMAT:
            raise CacheCorruptError(f"{path}: not a {CACHE_FORMAT} file")
        if doc.get("format_version") != CACHE_FORMAT_VERSION:
            raise CacheCorruptError(f"{path}: unsupported format_version")
        if doc.get("hash_algo") != HASH_ALGO:
            raise CacheCorruptError(
                f"{path}: hash_algo {doc.get('hash_algo')!r} != {HASH_ALGO!r}"
            )
        block_size = doc.get("block_size")
        if not isinstance(block_size, int) or block_size != kwargs.get("block_size", BLOCK_SIZE):
            raise CacheCorruptError(f"{path}: block_size {block_size!r} mismatch")
        cache = cls(
            agent_id=doc.get("agent_id", "agent-0"),
            namespace=doc.get("namespace", "default"),
            block_size=block_size,
            **{k: v for k, v in kwargs.items() if k not in ("agent_id", "namespace", "block_size")},
        )
        try:
            for p, fv in doc.get("files", {}).items():
                cache._versions[p] = FileVersion(
                    p,
                    int(fv["version"]),
                    str(fv["content_hash"]),
                    tuple(fv["chunk_hashes"]),
                    tuple(int(n) for n in fv["chunk_tokens"]),
                    int(fv["num_tokens"]),
                    float(fv["updated_at"]),
                )
            for row in doc.get("blocks", []):
                fp, ver, idx, chash, chain, ntok, producer = row
                key = BlockKey(str(fp), int(ver), int(idx), str(chash))
                handle = KVHandle(kv_id=str(chain), num_tokens=int(ntok), producer=str(producer))
                blk = CachedBlock(key, handle, int(ntok), str(chain), BlockStatus.HIT_L1)
                if key.file_path == STABLE_KEY:
                    cache._pinned[key] = blk
                else:
                    cache._lru[key] = blk
                cache._by_path.setdefault(key.file_path, set()).add(key)
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheCorruptError(f"{path}: malformed payload: {exc}") from exc
        return cache


def _summarize(
    path: str, version: int, out: list[CachedBlock], tail: ParentRef
) -> BlockResult:
    total = sum(b.num_tokens for b in out)
    l1 = sum(1 for b in out if b.status == BlockStatus.HIT_L1)
    l2 = sum(1 for b in out if b.status == BlockStatus.HIT_L2)
    hit = l1 + l2
    reused = sum(b.num_tokens for b in out if b.status != BlockStatus.COMPUTED)
    l2_reused = sum(b.num_tokens for b in out if b.status == BlockStatus.HIT_L2)
    first_miss = next(
        (i for i, b in enumerate(out) if b.status == BlockStatus.COMPUTED), None
    )
    return BlockResult(
        path=path,
        version=version,
        blocks=tuple(out),
        tail=tail,
        total_tokens=total,
        reused_tokens=reused,
        l2_reused_tokens=l2_reused,
        recompute_tokens=total - reused,
        hit_blocks=hit,
        l1_hit_blocks=l1,
        l2_hit_blocks=l2,
        computed_blocks=len(out) - hit,
        first_miss_block=first_miss,
    )
