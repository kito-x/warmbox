from __future__ import annotations

from ..config import HASH_DIGEST_BYTES
import hashlib
from typing import Iterable

try:  # optional accelerated backend
    import blake3 as _blake3

    HASH_ALGO: str = "blake3-128"
except ImportError:  # pragma: no cover - depends on the environment
    _blake3 = None
    HASH_ALGO = "blake2b-128"


def _h(data: bytes) -> str:
    if _blake3 is not None:
        return _blake3.blake3(data).hexdigest(length=HASH_DIGEST_BYTES)
    return hashlib.blake2b(data, digest_size=HASH_DIGEST_BYTES).hexdigest()


def chunk_hash(text: str) -> str:
    """128-bit content hash of a chunk, as 32 lowercase hex characters."""
    return _h(text.encode("utf-8"))


def chain_hash(parent_chain: str, content_hash: str) -> str:
    """Hash of a block together with everything before it (its causal context)."""
    return _h(f"{parent_chain}:{content_hash}".encode("utf-8"))


def root_chain(namespace: str) -> str:
    return _h(b"root:" + namespace.encode("utf-8"))


def repo_hash(files: Iterable[tuple[str, str]]) -> str:
    """Hash of a repo snapshot: (relpath, content_hash) pairs, sorted by relpath."""
    pairs = sorted((p, h) for p, h in files)
    return _h("\n".join(f"{p}\t{h}" for p, h in pairs).encode("utf-8"))


def namespace_for(model_id: str, repo_hash_hex: str) -> str:
    return f"{model_id}|{repo_hash_hex}"
