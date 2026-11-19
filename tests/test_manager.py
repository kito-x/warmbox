from __future__ import annotations

from warmbox.agent.history import History, Message
from warmbox.cache.block_cache import KVCache
from warmbox.cache.manager import CacheManager
from warmbox.inference.prompt_builder import FileSegment, StablePrefix, build_prompt

STABLE = StablePrefix(system="SYS", tools=(), repo_map="m\n")


def test_prefill_no_cache(lines320: str) -> None:
    hist = History()
    hist.append(Message("user", "goal"))
    prompt = build_prompt(STABLE, [FileSegment("a.txt", lines320, 1)], hist.messages)
    rep = CacheManager(None).prefill(prompt)
    assert rep.reused_tokens == 0
    assert rep.recompute_tokens == rep.prefill_tokens == prompt.total_tokens
    assert rep.hit_blocks == 0
    assert rep.reuse_ratio == 0.0
    assert all(s.first_miss_block == 0 for s in rep.segments if s.tokens)


def test_prefill_reuse_and_ttft(lines320: str) -> None:
    cache = KVCache(agent_id="t", namespace="ns")
    mgr = CacheManager(cache)
    hist = History()
    hist.append(Message("user", "goal"))
    prompt = build_prompt(STABLE, [FileSegment("a.txt", lines320, 1)], hist.messages)

    cold = mgr.prefill(prompt)
    assert cold.reused_tokens == 0
    warm = mgr.prefill(prompt)
    assert warm.reused_tokens == warm.prefill_tokens
    assert warm.recompute_tokens == 0
    assert warm.reuse_ratio == 1.0
    assert warm.sim_ttft_ms < cold.sim_ttft_ms


def test_file_parent_is_stable_not_history(lines320: str) -> None:
    cache = KVCache(agent_id="t", namespace="ns")
    mgr = CacheManager(cache)
    hist = History()
    hist.append(Message("user", "goal"))
    files = [FileSegment("a.txt", lines320, 1)]

    mgr.prefill(build_prompt(STABLE, files, hist.messages))
    hist.append(Message("assistant", "done"))
    rep = mgr.prefill(build_prompt(STABLE, files, hist.messages))

    file_seg = next(s for s in rep.segments if s.kind == "file")
    hdr_seg = next(s for s in rep.segments if s.kind == "file_header")
    assert file_seg.recompute_tokens == 0
    assert file_seg.hit_blocks == file_seg.tokens // 128
    assert hdr_seg.recompute_tokens == 0
    # only the history tail had to be recomputed
    assert rep.recompute_tokens <= 2 * 128


def test_edit_recompute_attribution(lines320: str) -> None:
    cache = KVCache(agent_id="t", namespace="ns")
    mgr = CacheManager(cache)
    hist = History()
    hist.append(Message("user", "goal"))
    written = lines320.replace("line 0165\n", "LINE 0165\n")

    mgr.prefill(build_prompt(STABLE, [FileSegment("a.txt", lines320, 1)], hist.messages))
    cache.on_file_write("a.txt", lines320, written, new_version=2)
    hist.append(Message("tool", "OK wrote a.txt v2"))

    rep = mgr.prefill(
        build_prompt(
            STABLE,
            [FileSegment("a.txt", written, 2), FileSegment("b.txt", "small\n", 1)],
            hist.messages,
        ),
        edited_paths=frozenset({"a.txt"}),
    )
    assert rep.edit_recompute_tokens == 640
    a_seg = next(s for s in rep.segments if s.key == "a.txt")
    assert a_seg.recompute_tokens == 640
    assert a_seg.first_miss_block == 5


def test_segment_reports_cover_prompt(lines320: str) -> None:
    cache = KVCache(agent_id="t", namespace="ns")
    mgr = CacheManager(cache)
    hist = History()
    hist.append(Message("user", "goal"))
    prompt = build_prompt(STABLE, [FileSegment("a.txt", lines320, 1)], hist.messages)
    rep = mgr.prefill(prompt)
    assert len(rep.segments) == len(prompt.segments)
    assert [s.kind for s in rep.segments] == ["stable", "history", "file_header", "file"]
    assert rep.prefill_tokens == prompt.total_tokens
