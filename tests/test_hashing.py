from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from warmbox.cache.hashing import (
    HASH_ALGO,
    chain_hash,
    chunk_hash,
    namespace_for,
    repo_hash,
    root_chain,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_chunk_hash_stable() -> None:
    first = chunk_hash("hello")
    assert len(first) == 32
    assert all(c in "0123456789abcdef" for c in first)
    for _ in range(1000):
        assert chunk_hash("hello") == first
    # A fresh interpreter with a different PYTHONHASHSEED must agree.
    out = subprocess.run(
        [sys.executable, "-c", "from warmbox.cache.hashing import chunk_hash;print(chunk_hash('hello'))"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
        check=True,
    )
    assert out.stdout.strip() == first


@pytest.mark.skipif(HASH_ALGO != "blake2b-128", reason="vectors are blake2b-128 specific")
def test_chunk_hash_known_vectors() -> None:
    assert chunk_hash("hello") == "46fb7408d4f285228f4af516ea25851b"
    assert chunk_hash("") == "cae66941d9efbd404e4d88758ea67670"


def test_chain_hash_depends_on_parent() -> None:
    h = chunk_hash("block")
    assert chain_hash("a", h) != chain_hash("b", h)
    assert chain_hash("a", h) == chain_hash("a", h)


def test_namespace_isolation() -> None:
    assert root_chain("x") != root_chain("y")
    assert root_chain("x") == root_chain("x")


def test_repo_hash_order_insensitive() -> None:
    a = ("a.py", chunk_hash("a"))
    b = ("b.py", chunk_hash("b"))
    assert repo_hash([a, b]) == repo_hash([b, a])
    assert namespace_for("m", repo_hash([a])) != namespace_for("m2", repo_hash([a]))
