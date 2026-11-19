from __future__ import annotations

import random

import pytest

from warmbox.cache.block_cache import KVCache
from warmbox.cache.models import BlockStatus, ParentRef
from warmbox.errors import CacheError

ONE_BLOCK = "hello\n"  # a single block


def _statuses(res):
    return [b.status for b in res.blocks]


def test_lru_eviction() -> None:
    cache = KVCache(agent_id="t", namespace="ns", max_blocks=8)
    for i in range(12):
        cache.get_blocks(f"f{i}.txt", f"content {i}\n")
    assert len(cache) == 8
    assert cache.stats().evicted_blocks == 4
    remaining = {k.file_path for k in cache._lru}
    assert remaining == {f"f{i}.txt" for i in range(4, 12)}

    # a hit refreshes recency
    cache.get_blocks("f4.txt", f"content {4}\n")
    cache.get_blocks("f12.txt", "content 12\n")
    remaining = {k.file_path for k in cache._lru}
    assert "f4.txt" in remaining
    assert "f5.txt" not in remaining


def test_stable_pinned(lines320: str) -> None:
    cache = KVCache(agent_id="t", namespace="ns", max_blocks=12)
    cache.get_blocks("<stable>", lines320)
    assert len(cache._pinned) == 10
    cache.get_blocks("a.txt", lines320)
    assert len(cache) <= 12
    assert len(cache._pinned) == 10
    # every pinned key is still present
    for key in list(cache._pinned):
        assert key in cache


def test_pinned_only_overflow_raises(lines320: str) -> None:
    # Pinned (stable) blocks always fit; a non-stable insert with nothing evictable raises.
    cache = KVCache(agent_id="t", namespace="ns", max_blocks=5)
    cache.get_blocks("<stable>", lines320)
    assert len(cache) == 10
    with pytest.raises(CacheError):
        cache.get_blocks("a.txt", "hello\n")


def test_bypass_write_detected(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    new = lines320.replace("line 0165\n", "LINE 0165\n")
    res = cache.get_blocks("a.txt", new)  # no on_file_write: write bypassed the interceptor
    assert cache.stats().bypass_writes == 1
    assert _statuses(res) == [BlockStatus.HIT_L1] * 5 + [BlockStatus.COMPUTED] * 5
    assert res.recompute_tokens == 640


def test_parent_change_misses(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    other = ParentRef("ns", ("deadbeef",), "0" * 32)
    res = cache.get_blocks("a.txt", lines320, parent=other)
    assert _statuses(res) == [BlockStatus.COMPUTED] * 10
    assert cache.stats().chain_mismatches == 10
    # and the same parent now hits
    again = cache.get_blocks("a.txt", lines320, parent=other)
    assert again.hit_blocks == 10


def test_stats_identity(lines320: str) -> None:
    rng = random.Random(11)
    cache = KVCache(agent_id="t", namespace="ns", max_blocks=64)
    text = lines320
    version = 1
    for _ in range(60):
        op = rng.choice(["read", "edit", "append", "truncate"])
        if op == "read":
            cache.get_blocks("a.txt", text, version=version if version > 1 else None)
        elif op == "edit":
            idx = rng.randrange(len(text.splitlines()))
            parts = text.splitlines(keepends=True)
            parts[idx] = f"LINE {idx:04d}\n"
            new = "".join(parts)
            version += 1
            cache.on_file_write("a.txt", text, new, new_version=version)
            text = new
        elif op == "append":
            version += 1
            new = text + f"line {rng.randrange(1000):04d}\n"
            cache.on_file_write("a.txt", text, new, new_version=version)
            text = new
        else:
            if len(text.splitlines()) > 40:
                version += 1
                new = "".join(text.splitlines(keepends=True)[:-5])
                cache.on_file_write("a.txt", text, new, new_version=version)
                text = new
    s = cache.stats()
    assert s.prefill_tokens == s.reused_tokens + s.recompute_tokens
    assert s.tokens_saved == s.reused_tokens
    assert 0.0 <= s.reuse_ratio <= 1.0
    assert s.lookups > 0


def test_drop_file(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    assert cache.version_of("a.txt") is not None
    assert cache.drop_file("a.txt") == 10
    assert cache.version_of("a.txt") is None
    assert len(cache) == 0


def test_save_load_roundtrip(tmp_path, lines320: str) -> None:
    cache = KVCache(agent_id="agent-0", namespace="ns", debug=False)
    cache.get_blocks("a.txt", lines320)
    cache.get_blocks("<stable>", "system prompt\n")
    path = tmp_path / "cache.json"
    cache.save(path)

    loaded = KVCache.load(path)
    assert loaded.namespace == "ns"
    assert len(loaded) == len(cache)
    res = loaded.get_blocks("a.txt", lines320)
    assert res.hit_blocks == 10


def test_load_corrupt(tmp_path) -> None:
    from warmbox.errors import CacheCorruptError

    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CacheCorruptError):
        KVCache.load(path)

    import json

    from warmbox.cache.hashing import HASH_ALGO

    doc = {
        "format": "warmbox-kvcache",
        "format_version": 1,
        "hash_algo": "nope-128" if HASH_ALGO != "nope-128" else "other",
        "block_size": 128,
        "namespace": "ns",
        "agent_id": "a",
        "files": {},
        "blocks": [],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(CacheCorruptError):
        KVCache.load(path)
