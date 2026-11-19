from __future__ import annotations

from warmbox.cache.models import KVHandle
from warmbox.cache.radix_tree import RadixTree


def _handles(hashes, producer: str = "a") -> list[KVHandle]:
    return [KVHandle(kv_id=h, num_tokens=128, producer=producer) for h in hashes]


def _h(n: int, prefix: str = "h") -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(n)]


def test_insert_and_match() -> None:
    tree = RadixTree()
    ns = "mock-1|abc"
    a, b, c, d, x = _h(5)
    assert tree.insert(ns, [a, b, c], _handles([a, b, c])) == 3
    assert tree.num_blocks() == 3

    m = tree.longest_prefix_match(ns, [a, b, c, d])
    assert m.matched == 3
    assert [h.kv_id for h in m.handles] == [a, b, c]

    m2 = tree.longest_prefix_match(ns, [a, x])
    assert m2.matched == 1

    assert tree.longest_prefix_match("other|ns", [a, b, c]).matched == 0


def test_split() -> None:
    tree = RadixTree()
    ns = "ns"
    a, b, c, d = _h(4)
    tree.insert(ns, [a, b, c], _handles([a, b, c]))
    tree.insert(ns, [a, b, d], _handles([a, b, d]))

    assert tree.num_blocks() == 4
    root = tree._roots[ns]
    assert len(root.children) == 1
    mid = next(iter(root.children.values()))
    assert mid.key == (a, b)
    assert len(mid.children) == 2
    assert {c for c in mid.children} == {c, d}
    # the shared prefix still matches both branches
    assert tree.longest_prefix_match(ns, [a, b, c]).matched == 3
    assert tree.longest_prefix_match(ns, [a, b, d]).matched == 3


def test_first_writer_wins() -> None:
    tree = RadixTree()
    ns = "ns"
    h = _h(2)
    assert tree.insert(ns, h, _handles(h, "first")) == 2
    handle = KVHandle(kv_id="zzz", num_tokens=1, producer="second")
    assert tree.insert(ns, h, [handle, handle]) == 0
    m = tree.longest_prefix_match(ns, h)
    assert [x.producer for x in m.handles] == ["first", "first"]


def test_insert_length_mismatch() -> None:
    import pytest

    tree = RadixTree()
    h = _h(1)
    with pytest.raises(ValueError):
        tree.insert("ns", h, _handles(h) + _handles(h))
    # a suffix insert whose prefix is absent is skipped, not fatal
    assert tree.insert("ns", _h(2), _handles(_h(1))) == 0
    assert tree.num_blocks() == 0


def test_evict_leaf_lru() -> None:
    tree = RadixTree(max_blocks=5)
    ns = "ns"
    for i in range(5):
        h = _h(1, prefix=f"p{i}_")
        tree.insert(ns, h, _handles(h))
    assert tree.num_blocks() == 5

    # ref-counted and pinned nodes are never evicted
    m = tree.longest_prefix_match(ns, _h(1, prefix="p0_"))
    tree.acquire(m)
    assert tree.evict(10) == 4          # everything except the pinned root and p0
    assert tree.num_blocks() == 1
    assert tree.longest_prefix_match(ns, _h(1, prefix="p0_")).matched == 1
    tree.release(m)

    # now it can go too
    assert tree.evict(10) == 1
    assert tree.num_blocks() == 0


def test_pin_blocks_eviction() -> None:
    tree = RadixTree(max_blocks=4)
    ns = "ns"
    h4 = _h(4)
    tree.insert(ns, h4, _handles(h4))
    assert tree.pin(ns, h4) == 4
    assert tree.evict(4) == 0
    assert tree.num_blocks() == 4


def test_evict_trims_leaf_tail_keeps_prefix() -> None:
    tree = RadixTree(max_blocks=10)
    ns = "ns"
    h = _h(6)
    tree.insert(ns, h, _handles(h))
    assert tree.evict(3) == 3
    assert tree.num_blocks() == 3
    assert tree.longest_prefix_match(ns, h).matched == 3


def test_namespace_isolation() -> None:
    tree = RadixTree()
    h = _h(3)
    tree.insert("ns1", h, _handles(h))
    assert tree.longest_prefix_match("ns1", h).matched == 3
    assert tree.longest_prefix_match("ns2", h).matched == 0
    assert tree.stats().namespaces == 1


def test_stats_and_dump() -> None:
    tree = RadixTree()
    h = _h(3)
    tree.insert("ns", h, _handles(h))
    tree.longest_prefix_match("ns", h)
    s = tree.stats()
    assert s.num_blocks == 3 and s.lookups == 1 and s.matched_blocks == 3
    dump = tree.dump("ns")
    assert "3 blocks" in dump
    assert tree.dump("absent") == "absent: <empty>"
