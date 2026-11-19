from __future__ import annotations

import pytest

from warmbox.cache.hashing import chunk_hash
from warmbox.cache.invalidator import find_first_divergent_chunk, plan_truncation
from warmbox.cache.models import BlockStatus
from warmbox.errors import VersionConflictError
from warmbox.inference.tokenizer import chunk_tokens, tokenize


def _hashes(text: str) -> list[str]:
    return [chunk_hash(t) for t, _ in chunk_tokens(tokenize(text))]


def _replace_line(text: str, idx: int, new: str) -> str:
    parts = text.splitlines(keepends=True)
    parts[idx] = new
    return "".join(parts)


def _statuses(result) -> list[BlockStatus]:
    return [b.status for b in result.blocks]


def test_edit_middle(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    new = _replace_line(lines320, 165, "LINE 0165\n")
    inv = cache.on_file_write("a.txt", lines320, new, new_version=2)
    assert inv.first_divergent_block == 5
    assert (inv.blocks_kept, inv.tokens_kept) == (5, 640)
    assert (inv.blocks_dropped, inv.tokens_dropped) == (5, 640)
    assert inv.new_block_count == 10

    res = cache.get_blocks("a.txt", new)
    assert _statuses(res) == [BlockStatus.HIT_L1] * 5 + [BlockStatus.COMPUTED] * 5
    assert res.recompute_tokens == 640
    assert res.reused_tokens == 640
    assert res.first_miss_block == 5


def test_edit_start(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    new = _replace_line(lines320, 0, "LINE 0000\n")
    inv = cache.on_file_write("a.txt", lines320, new, new_version=2)
    assert inv.first_divergent_block == 0
    assert (inv.blocks_kept, inv.blocks_dropped) == (0, 10)
    assert cache.get_blocks("a.txt", new).recompute_tokens == 1280


def test_edit_end(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    new = _replace_line(lines320, 319, "LINE 0319\n")
    inv = cache.on_file_write("a.txt", lines320, new, new_version=2)
    assert inv.first_divergent_block == 9
    assert (inv.blocks_kept, inv.blocks_dropped) == (9, 1)
    res = cache.get_blocks("a.txt", new)
    assert res.recompute_tokens == 128
    assert res.hit_blocks == 9


def test_append(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    new = lines320 + "line 0320\n"
    inv = cache.on_file_write("a.txt", lines320, new, new_version=2)
    assert inv.first_divergent_block == 10
    assert (inv.blocks_kept, inv.tokens_kept) == (10, 1280)
    assert (inv.blocks_dropped, inv.tokens_dropped) == (0, 0)
    assert inv.new_block_count == 11
    res = cache.get_blocks("a.txt", new)
    assert res.recompute_tokens == 4
    assert res.total_tokens == 1284


def test_insert_middle(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    parts = lines320.splitlines(keepends=True)
    parts.insert(166, "line 9999\n")
    new = "".join(parts)
    inv = cache.on_file_write("a.txt", lines320, new, new_version=2)
    assert inv.first_divergent_block == 5
    assert cache.get_blocks("a.txt", new).recompute_tokens == 1284 - 640


def test_truncate(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    new = "".join(lines320.splitlines(keepends=True)[:288])
    inv = cache.on_file_write("a.txt", lines320, new, new_version=2)
    assert inv.first_divergent_block == 9
    assert (inv.blocks_kept, inv.blocks_dropped) == (9, 1)
    assert cache.get_blocks("a.txt", new).recompute_tokens == 0


def test_noop_write(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    inv = cache.on_file_write("a.txt", lines320, lines320, new_version=2)
    assert inv.first_divergent_block == 10
    assert (inv.blocks_kept, inv.blocks_dropped) == (10, 0)
    assert inv.new_version == 2
    assert cache.get_blocks("a.txt", lines320).recompute_tokens == 0


def test_find_first_divergent_table() -> None:
    def hs(n: int, prefix: str = "h") -> list[str]:
        return [f"{prefix}{i}" for i in range(n)]

    # edit middle (block 5 of 10)
    assert find_first_divergent_chunk(hs(5) + hs(5, "x"), hs(5) + hs(5, "y")) == 5
    # edit start
    assert find_first_divergent_chunk(hs(10) , ["z"] + hs(10)[1:]) == 0
    # edit end (block 9)
    old = hs(9) + ["last"]
    assert find_first_divergent_chunk(old, hs(9) + ["LAST"]) == 9
    # append, old last block full
    assert find_first_divergent_chunk(hs(10), hs(10) + ["new"]) == 10
    # append, old last block partial
    assert find_first_divergent_chunk(hs(9) + ["p"], hs(9) + ["pq"]) == 9
    # truncate to 9 blocks
    assert find_first_divergent_chunk(hs(10), hs(9)) == 9
    # no-op
    assert find_first_divergent_chunk(hs(10), hs(10)) == 10
    # prefix relationships
    assert find_first_divergent_chunk([], []) == 0
    assert find_first_divergent_chunk([], ["a"]) == 0
    assert find_first_divergent_chunk(["a"], []) == 0


def test_plan_truncation_ranges() -> None:
    plan = plan_truncation(["a", "b", "c", "d"], ["a", "b", "X", "d", "e"])
    assert plan.first_divergent == 2
    assert list(plan.keep) == [0, 1]
    assert list(plan.drop) == [2, 3]
    assert list(plan.compute) == [2, 3, 4]


def test_version_conflict(lines320: str, cache) -> None:
    cache.get_blocks("a.txt", lines320)
    with pytest.raises(VersionConflictError):
        cache.on_file_write("a.txt", lines320, lines320, new_version=1)


def test_block_key_format(cache) -> None:
    from warmbox.cache.models import BlockKey

    key = BlockKey("src/app.py", 3, 5, "9f86d081884c7d659a2feaa0c55ad015")
    assert str(key) == "src/app.py@v3#5:9f86d081884c7d659a2feaa0c55ad015"
