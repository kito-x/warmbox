from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from typing import Sequence

from ..config import L2_MAX_BLOCKS
from .models import KVHandle

log = logging.getLogger("warmbox.cache.radix")


@dataclass(eq=False, slots=True)                 # mutable, identity-hashed
class RadixNode:
    key: tuple[str, ...]                         # edge label: content hashes
    handles: list[KVHandle]                      # len(handles) == len(key)
    parent: "RadixNode | None"
    children: dict[str, "RadixNode"] = field(default_factory=dict)
    ref_count: int = 0
    pinned: bool = False
    last_access: int = 0                         # logical clock, NOT wall time (determinism)
    hit_count: int = 0

    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:  # pragma: no cover - debug only
        head = self.key[0][:8] if self.key else "root"
        return f"RadixNode({head}, n={len(self.key)}, leaf={self.is_leaf()})"


@dataclass(frozen=True, slots=True)
class RadixMatch:
    namespace: str
    matched: int                                 # number of hashes matched
    handles: tuple[KVHandle, ...]                # len == matched
    nodes: tuple[RadixNode, ...]                 # nodes touched along the path


@dataclass(slots=True)
class RadixStats:
    lookups: int = 0
    matched_blocks: int = 0
    inserted_blocks: int = 0
    evicted_blocks: int = 0
    num_blocks: int = 0
    num_nodes: int = 0
    namespaces: int = 0


def _common_prefix_len(key: Sequence[str], hashes: Sequence[str], offset: int) -> int:
    limit = min(len(key), len(hashes) - offset)
    i = 0
    while i < limit and key[i] == hashes[offset + i]:
        i += 1
    return i


class RadixTree:
    """L2: process-global radix tree of content-hash sequences per namespace (SGLang RadixAttention).

    Immutable content: a write by one agent never invalidates L2. Entries only age out through
    leaf-first LRU eviction."""

    def __init__(self, max_blocks: int = L2_MAX_BLOCKS) -> None:
        if max_blocks < 1:
            raise ValueError("max_blocks must be >= 1")
        self.max_blocks = max_blocks
        self._roots: dict[str, RadixNode] = {}
        self._stats = RadixStats()
        self._tick = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ insert

    def insert(self, namespace: str, hashes: Sequence[str], handles: Sequence[KVHandle]) -> int:
        """Insert a hash path. `handles` may cover only the path suffix, in which case the
        leading hashes must already exist in the tree (their handles are reused -- first
        writer wins); this is what a segment with a non-empty parent prefix does."""
        if len(handles) > len(hashes):
            raise ValueError(
                f"len(hashes)={len(hashes)} < len(handles)={len(handles)}"
            )
        with self._lock:
            if len(handles) < len(hashes):
                have = len(hashes) - len(handles)
                prior = self._match_locked(namespace, hashes[:have])
                if prior.matched < have:
                    # The parent prefix is not in this namespace (e.g. a synthetic parent
                    # context). L2 is an optimisation: skip the insert rather than fail.
                    log.debug(
                        "radix insert skipped: %d leading hashes absent from %r",
                        have - prior.matched,
                        namespace,
                    )
                    return 0
                handles = list(prior.handles[:have]) + list(handles)
            self._tick += 1
            node = self._roots.get(namespace)
            if node is None:
                node = RadixNode(key=(), handles=[], parent=None, pinned=True)
                self._roots[namespace] = node
            i = 0
            new = 0
            while i < len(hashes):
                child = node.children.get(hashes[i])
                if child is None:
                    leaf = RadixNode(
                        tuple(hashes[i:]),
                        list(handles[i:]),
                        parent=node,
                        last_access=self._tick,
                    )
                    node.children[hashes[i]] = leaf
                    new = len(hashes) - i
                    break
                m = _common_prefix_len(child.key, hashes, i)
                if m < len(child.key):
                    child = self._split(child, m)
                child.last_access = self._tick
                node, i = child, i + m
            self._stats.num_blocks += new
            self._stats.inserted_blocks += new
            if self._stats.num_blocks > self.max_blocks:
                self._evict_locked(self._stats.num_blocks - self.max_blocks)
            return new

    def _split(self, node: RadixNode, m: int) -> RadixNode:
        parent = node.parent
        mid = RadixNode(
            node.key[:m],
            node.handles[:m],
            parent=parent,
            ref_count=node.ref_count,
            pinned=node.pinned,
            last_access=node.last_access,
        )
        node.key = node.key[m:]
        node.handles = node.handles[m:]
        node.parent = mid
        mid.children[node.key[0]] = node
        if parent is not None:
            parent.children[mid.key[0]] = mid
        return mid

    # ------------------------------------------------------------------ lookup

    def longest_prefix_match(self, namespace: str, hashes: Sequence[str]) -> RadixMatch:
        with self._lock:
            self._tick += 1
            self._stats.lookups += 1
            match = self._match_locked(namespace, hashes)
            self._stats.matched_blocks += match.matched
            return match

    def _match_locked(self, namespace: str, hashes: Sequence[str]) -> RadixMatch:
        node = self._roots.get(namespace)
        if node is None:
            return RadixMatch(namespace, 0, (), ())
        i = 0
        handles: list[KVHandle] = []
        nodes: list[RadixNode] = []
        while i < len(hashes):
            child = node.children.get(hashes[i])
            if child is None:
                break
            m = _common_prefix_len(child.key, hashes, i)
            handles.extend(child.handles[:m])
            nodes.append(child)
            child.last_access = self._tick
            child.hit_count += 1
            i += m
            if m < len(child.key):
                break
            node = child
        return RadixMatch(namespace, i, tuple(handles), tuple(nodes))

    def acquire(self, match: RadixMatch) -> None:
        with self._lock:
            for node in match.nodes:
                node.ref_count += 1

    def release(self, match: RadixMatch) -> None:
        with self._lock:
            for node in match.nodes:
                node.ref_count = max(0, node.ref_count - 1)

    def pin(self, namespace: str, hashes: Sequence[str]) -> int:
        with self._lock:
            node = self._roots.get(namespace)
            if node is None:
                return 0
            i = 0
            pinned = 0
            while i < len(hashes):
                child = node.children.get(hashes[i])
                if child is None:
                    break
                m = _common_prefix_len(child.key, hashes, i)
                child.pinned = True
                pinned += m
                i += m
                if m < len(child.key):
                    break
                node = child
            return pinned

    # ------------------------------------------------------------------ eviction

    def evict(self, n_blocks: int) -> int:
        with self._lock:
            return self._evict_locked(n_blocks)

    def _evict_locked(self, n: int) -> int:
        if n <= 0:
            return 0
        heap = [
            (lf.last_access, id(lf), lf)
            for lf in self._leaves()
            if lf.ref_count == 0 and not lf.pinned
        ]
        heapify(heap)
        freed = 0
        while freed < n and heap:
            _, _, lf = heappop(heap)
            if not lf.key:  # stale heap entry
                continue
            k = min(len(lf.key), n - freed)
            first = lf.key[0]
            lf.key = lf.key[: len(lf.key) - k]
            lf.handles = lf.handles[: len(lf.handles) - k]
            freed += k
            if not lf.key:  # edge empty -> remove node
                p = lf.parent
                if p is not None:
                    p.children.pop(first, None)
                    if (
                        p.parent is not None
                        and p.is_leaf()
                        and p.ref_count == 0
                        and not p.pinned
                    ):
                        heappush(heap, (p.last_access, id(p), p))
        self._stats.num_blocks -= freed
        self._stats.evicted_blocks += freed
        return freed

    def _leaves(self) -> list[RadixNode]:
        out: list[RadixNode] = []
        stack: list[RadixNode] = list(self._roots.values())
        while stack:
            node = stack.pop()
            if node.is_leaf():
                out.append(node)
            else:
                stack.extend(node.children.values())
        return out

    # ------------------------------------------------------------------ introspection

    def num_blocks(self) -> int:
        with self._lock:
            return self._stats.num_blocks

    def stats(self) -> RadixStats:
        with self._lock:
            nodes = 0
            stack: list[RadixNode] = list(self._roots.values())
            while stack:
                node = stack.pop()
                nodes += 1
                stack.extend(node.children.values())
            return RadixStats(
                lookups=self._stats.lookups,
                matched_blocks=self._stats.matched_blocks,
                inserted_blocks=self._stats.inserted_blocks,
                evicted_blocks=self._stats.evicted_blocks,
                num_blocks=self._stats.num_blocks,
                num_nodes=nodes,
                namespaces=len(self._roots),
            )

    def dump(self, namespace: str, max_depth: int = 6) -> str:
        with self._lock:
            root = self._roots.get(namespace)
            if root is None:
                return f"{namespace}: <empty>"
            lines: list[str] = [f"{namespace}: {self._stats.num_blocks} blocks"]
            stack: list[tuple[RadixNode, int]] = [(root, 0)]
            while stack:
                node, depth = stack.pop()
                if depth > max_depth:
                    continue
                label = " ".join(h[:8] for h in node.key) if node.key else "<root>"
                flags = []
                if node.pinned:
                    flags.append("pin")
                if node.ref_count:
                    flags.append(f"ref={node.ref_count}")
                if node.hit_count:
                    flags.append(f"hit={node.hit_count}")
                lines.append(
                    f"{'  ' * depth}{label} [{len(node.key)}]{' ' + ' '.join(flags) if flags else ''}"
                )
                for child in node.children.values():
                    stack.append((child, depth + 1))
            return "\n".join(lines)
