from __future__ import annotations

from warmbox.cache.block_cache import KVCache
from warmbox.cache.hashing import chunk_hash
from warmbox.cache.models import BlockStatus, KVHandle
from warmbox.cache.radix_tree import RadixTree
from warmbox.inference.tokenizer import chunk_tokens, tokenize


def _hashes(text: str) -> list[str]:
    return [chunk_hash(t) for t, _ in chunk_tokens(tokenize(text))]


def _statuses(res):
    return [b.status for b in res.blocks]


def test_second_agent_hits_l2(lines320: str) -> None:
    tree = RadixTree()
    a = KVCache(agent_id="a0", namespace="ns", shared=tree)
    b = KVCache(agent_id="a1", namespace="ns", shared=tree)

    ra = a.get_blocks("<stable>", lines320)
    assert ra.hit_blocks == 0 and ra.l2_hit_blocks == 0

    rb = b.get_blocks("<stable>", lines320)
    assert _statuses(rb) == [BlockStatus.HIT_L2] * 10
    assert rb.l2_reused_tokens == 1280
    assert b.stats().l2_reused_tokens == 1280

    # the second lookup now also hits from local L1
    rb2 = b.get_blocks("<stable>", lines320)
    assert rb2.hit_blocks == 10 and rb2.l1_hit_blocks == 10


def test_edit_does_not_affect_other_agent(lines320: str) -> None:
    tree = RadixTree()
    a = KVCache(agent_id="a0", namespace="ns", shared=tree)
    b = KVCache(agent_id="a1", namespace="ns", shared=tree)

    a.get_blocks("f.py", lines320)
    b.get_blocks("f.py", lines320)
    assert b.stats().l2_hit_blocks == 10

    before = tree.num_blocks()
    new = lines320.replace("line 0165\n", "LINE 0165\n")
    a.on_file_write("f.py", lines320, new, new_version=2)

    # B is unaffected: its content still hashes to the original branch
    rb = b.get_blocks("f.py", lines320)
    assert rb.hit_blocks == 10

    # A recomputes only the divergent tail, and L2 gains exactly those 5 blocks
    ra = a.get_blocks("f.py", new)
    assert _statuses(ra) == [BlockStatus.HIT_L1] * 5 + [BlockStatus.COMPUTED] * 5
    assert tree.num_blocks() == before + 5


def test_l2_handle_verification(lines320: str) -> None:
    tree = RadixTree()
    hs = _hashes(lines320)
    bad = [KVHandle(kv_id="corrupt", num_tokens=128, producer="evil") for _ in hs]
    tree.insert("ns", hs, bad)

    cache = KVCache(agent_id="a", namespace="ns", shared=tree)
    res = cache.get_blocks("f.py", lines320)
    assert res.l2_hit_blocks == 0
    assert _statuses(res) == [BlockStatus.COMPUTED] * 10
    assert tree.num_blocks() == 10  # first writer wins, original branch untouched


def test_cross_agent_turn1_reuse(lines320: str) -> None:
    """Followers hit L2 for the stable prefix on turn 1 (the G-SHARE mechanism)."""
    tree = RadixTree()
    caches = [KVCache(agent_id=f"a{i}", namespace="ns", shared=tree) for i in range(4)]
    caches[0].get_blocks("<stable>", lines320)
    for c in caches[1:]:
        res = c.get_blocks("<stable>", lines320)
        assert res.l2_reused_tokens == 1280


def test_namespaced_sharing(lines320: str) -> None:
    tree = RadixTree()
    a = KVCache(agent_id="a0", namespace="ns1", shared=tree)
    b = KVCache(agent_id="a1", namespace="ns2", shared=tree)
    a.get_blocks("<stable>", lines320)
    rb = b.get_blocks("<stable>", lines320)
    assert rb.l2_hit_blocks == 0
    assert rb.computed_blocks == 10


def test_parent_wiring_through_tree(lines320: str) -> None:
    """A file body's L2 path is S-hashes ++ header-hashes ++ body-hashes."""
    tree = RadixTree()
    a = KVCache(agent_id="a0", namespace="ns", shared=tree)
    b = KVCache(agent_id="a1", namespace="ns", shared=tree)
    for cache in (a, b):
        s = cache.get_blocks("<stable>", "system\n")
        hdr = cache.get_blocks("<hdr>:x.py", "=== FILE: x.py ===\n", parent=s.tail)
        cache.get_blocks("x.py", lines320, parent=hdr.tail, version=1)
    # b's whole path (stable, header, body) was already in L2
    assert b.stats().recompute_tokens == 0
    assert b.stats().l2_hit_blocks == b.stats().hit_blocks
    assert b.stats().prefill_tokens > 1280
