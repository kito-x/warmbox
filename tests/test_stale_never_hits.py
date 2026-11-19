from __future__ import annotations

import random

from warmbox.cache.block_cache import KVCache
from warmbox.cache.hashing import chain_hash, chunk_hash
from warmbox.cache.models import BlockStatus, ParentRef
from warmbox.cache.radix_tree import RadixTree

BASE = "".join(f"line {i:04d}\n" for i in range(320))


def _check(cache: KVCache, content: str, parent: ParentRef, result) -> None:
    """I1: any non-COMPUTED block must be provably derived from the current prefix."""
    chunks = cache.chunk(content)
    assert len(result.blocks) == len(chunks)
    hashes = [chunk_hash(t) for t, _ in chunks]
    chain = parent.chain
    chains = []
    for h in hashes:
        chain = chain_hash(chain, h)
        chains.append(chain)
    for blk in result.blocks:
        i = blk.key.block_index
        if blk.status is not BlockStatus.COMPUTED:
            assert blk.chain_hash == chains[i]
            assert blk.debug_text == chunks[i][0]
            assert blk.kv.kv_id == blk.chain_hash


def _mutate(rng: random.Random, content: str) -> str:
    lines = content.splitlines(keepends=True)
    r = rng.random()
    if r < 0.55 and lines:
        idx = rng.randrange(len(lines))
        lines[idx] = f"LINE {idx:04d}\n"
        return "".join(lines)
    if r < 0.8:
        return content + f"line {rng.randrange(4000):04d}\n"
    if len(lines) > 12:
        return "".join(lines[: -rng.randint(1, 5)])
    return content + "tail\n"


def _rand_hash(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(32))


def test_property_stale_never_hits() -> None:
    rng = random.Random(1234)
    cache = KVCache(agent_id="p", namespace="ns", debug=True, max_blocks=40)
    content = BASE
    parent = ParentRef.root("ns")

    res = cache.get_blocks("f.txt", content, parent=parent)
    _check(cache, content, parent, res)

    for _ in range(500):
        op = rng.choice(["announce", "bypass", "parent", "evict", "read", "noop"])
        if op == "announce":
            new = _mutate(rng, content)
            cache.on_file_write("f.txt", content, new)
            content = new
        elif op == "noop":
            cache.on_file_write("f.txt", content, content)
        elif op == "bypass":
            content = _mutate(rng, content)
        elif op == "parent":
            parent = ParentRef("ns", (_rand_hash(rng),), _rand_hash(rng))
        elif op == "evict":
            cache.evict(rng.randint(1, 5))
        res = cache.get_blocks("f.txt", content, parent=parent)
        _check(cache, content, parent, res)

    s = cache.stats()
    assert s.chain_mismatches >= 0
    assert s.prefill_tokens == s.reused_tokens + s.recompute_tokens


def test_shared_stale_never_hits() -> None:
    rng = random.Random(4321)
    tree = RadixTree(max_blocks=60)
    caches = [
        KVCache(agent_id=f"a{i}", namespace="ns", debug=True, max_blocks=40, shared=tree)
        for i in range(2)
    ]
    state = [
        {"content": BASE, "parent": ParentRef.root("ns")},
        {"content": BASE, "parent": ParentRef.root("ns")},
    ]

    for idx, cache in enumerate(caches):
        st = state[idx]
        res = cache.get_blocks("f.txt", st["content"], parent=st["parent"])
        _check(cache, st["content"], st["parent"], res)

    for _ in range(300):
        ci = rng.randrange(2)
        cache = caches[ci]
        st = state[ci]
        op = rng.choice(["announce", "bypass", "parent", "evict", "read", "noop"])
        if op == "announce":
            new = _mutate(rng, st["content"])
            cache.on_file_write("f.txt", st["content"], new)
            st["content"] = new
        elif op == "noop":
            cache.on_file_write("f.txt", st["content"], st["content"])
        elif op == "bypass":
            st["content"] = _mutate(rng, st["content"])
        elif op == "parent":
            st["parent"] = ParentRef("ns", (_rand_hash(rng),), _rand_hash(rng))
        elif op == "evict":
            cache.evict(rng.randint(1, 4))
        res = cache.get_blocks("f.txt", st["content"], parent=st["parent"])
        _check(cache, st["content"], st["parent"], res)

    assert tree.num_blocks() <= 60
    assert any(c.stats().l2_hit_blocks > 0 for c in caches)
