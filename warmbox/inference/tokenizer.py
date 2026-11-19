from __future__ import annotations

from typing import Sequence

from ..config import BLOCK_SIZE
import re

TOKEN_RE: re.Pattern[str] = re.compile(r"\s+|\w{1,4}|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Lossless regex approximation of a BPE tokenizer: "".join(tokenize(s)) == s."""
    return TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def chunk_tokens(tokens: Sequence[str], block_size: int = BLOCK_SIZE) -> list[tuple[str, int]]:
    """Cut a token sequence into (chunk_text, token_count) blocks of block_size."""
    return [
        ("".join(tokens[i : i + block_size]), len(tokens[i : i + block_size]))
        for i in range(0, len(tokens), block_size)
    ]
