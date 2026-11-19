from __future__ import annotations

import pytest

from warmbox.cache.block_cache import KVCache

# 4 tokens per line, 32 lines per 128-token block -> exactly 10 full blocks.
LINES320: str = "".join(f"line {i:04d}\n" for i in range(320))


@pytest.fixture
def lines320() -> str:
    return LINES320


@pytest.fixture
def cache() -> KVCache:
    return KVCache(agent_id="t", namespace="ns", debug=True)


@pytest.fixture
def tmp_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text(LINES320, encoding="utf-8")
    (repo / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    return repo
