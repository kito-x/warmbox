from __future__ import annotations

from warmbox.agent.loop import AgentConfig, AgentLoop
from warmbox.agent.prompts import SYSTEM_PROMPT
from warmbox.agent.tools import TOOL_SCHEMAS
from warmbox.cache.block_cache import KVCache
from warmbox.cache.manager import CacheManager
from warmbox.config import BLOCK_SIZE
from warmbox.inference.client import MockBackend
from warmbox.inference.prompt_builder import StablePrefix
from warmbox.orchestrator import AgentSpec, Orchestrator
from warmbox.sandbox.runtime import Sandbox

GOAL = "replace line 0165"


def _stable() -> StablePrefix:
    return StablePrefix(SYSTEM_PROMPT, TOOL_SCHEMAS, "a.txt\n")


def _script() -> list[dict]:
    return [
        {"tool": "read_file", "args": {"path": "a.txt"}},
        {
            "tool": "edit_file",
            "args": {"path": "a.txt", "old_str": "line 0165\n", "new_str": "LINE 0165\n"},
        },
        {"tool": "finish", "args": {"summary": "done"}},
    ]


def test_three_turn_edit_in_turn_two(tmp_repo, tmp_path) -> None:
    sandbox = Sandbox(tmp_repo, "agent-0", workspaces_root=tmp_path / "ws")
    cache = KVCache(agent_id="agent-0", namespace="ns")
    loop = AgentLoop(
        "agent-0",
        sandbox=sandbox,
        client=MockBackend(_script()),
        cache_manager=CacheManager(cache),
        stable_prefix=_stable(),
        config=AgentConfig(working_set_max_files=4, working_set_max_tokens=8000),
    )
    result = loop.run(GOAL)
    turns = result.turns

    # ---- turn 1: nothing pinned yet, cold cache
    t1 = turns[0]
    assert [s.kind for s in t1.segments] == ["stable", "history"]
    assert t1.reused_tokens == 0

    # ---- turn 2: the read pinned a.txt; file and header are cold
    t2 = turns[1]
    kinds = [s.kind for s in t2.segments]
    assert kinds == ["stable", "history", "file_header", "file"]
    assert t2.edited_paths == ()
    file_seg = next(s for s in t2.segments if s.kind == "file")
    assert file_seg.recompute_tokens == 1280
    assert file_seg.hit_blocks == 0
    hdr_seg = next(s for s in t2.segments if s.kind == "file_header")
    assert hdr_seg.recompute_tokens > 0

    # ---- turn 3: edit invalidated only the divergent tail
    t3 = turns[2]
    stable_seg = next(s for s in t3.segments if s.kind == "stable")
    assert stable_seg.recompute_tokens == 0
    file_seg3 = next(s for s in t3.segments if s.kind == "file")
    assert file_seg3.first_miss_block == 5
    assert file_seg3.recompute_tokens == 640
    assert file_seg3.reused_tokens == 640
    hdr_seg3 = next(s for s in t3.segments if s.kind == "file_header")
    assert hdr_seg3.recompute_tokens == 0
    assert t3.edited_paths == ("a.txt",)
    assert t3.edit_recompute_tokens == 640
    hist_seg3 = next(s for s in t3.segments if s.kind == "history")
    assert hist_seg3.recompute_tokens <= 2 * BLOCK_SIZE

    # ---- end state
    assert sandbox.version_of("a.txt").version == 2
    assert cache.version_of("a.txt").version == 2
    assert result.finished is True
    assert "+LINE 0165" in result.patch
    assert len(result.turns) == 3

    tool_messages = [
        m for m in loop.history.messages if m.role == "tool"
    ]
    assert any("v2" in (m.content or "") and "first changed block 5" in (m.content or "")
               for m in tool_messages)
    sandbox.close()


def test_parallel_agents_share_l2(tmp_repo, tmp_path) -> None:
    # Distinct goals so only the stable prefix can be shared through L2 on turn 1.
    specs = [
        AgentSpec(f"agent-{i}", f"{GOAL} (agent {i})", MockBackend(_script()))
        for i in range(3)
    ]
    orch = Orchestrator(
        tmp_repo,
        model_id="mock-1",
        use_cache=True,
        shared_cache=True,
        workspaces_root=tmp_path / "ws",
    )
    results = orch.run(specs)

    stable_tokens = results[0].turns[0].segments[0].tokens
    assert stable_tokens > 0
    for follower in results[1:]:
        first = follower.turns[0]
        assert first.l2_reused_tokens > 0
        assert first.l2_reused_tokens == stable_tokens
    assert results[0].turns[0].l2_reused_tokens == 0
    for r in results:
        assert r.finished is True


def test_no_cache_baseline_recomputes_everything(tmp_repo, tmp_path) -> None:
    orch = Orchestrator(
        tmp_repo,
        model_id="mock-1",
        use_cache=False,
        shared_cache=False,
        workspaces_root=tmp_path / "ws",
    )
    results = orch.run([AgentSpec("agent-0", GOAL, MockBackend(_script()))])
    for turn in results[0].turns:
        assert turn.reused_tokens == 0
        assert turn.recompute_tokens == turn.prefill_tokens
