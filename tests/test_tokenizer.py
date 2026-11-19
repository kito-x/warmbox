from __future__ import annotations

import random

from warmbox.config import BLOCK_SIZE
from warmbox.inference.tokenizer import chunk_tokens, count_tokens, tokenize


def test_lossless_roundtrip() -> None:
    rng = random.Random(7)
    alphabet = [chr(i) for i in range(32, 0x2FFF)] + ["\n", "\t", " ", "\u2028", "é", "汉"]
    for _ in range(200):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        assert "".join(tokenize(s)) == s


def test_reference_facts() -> None:
    assert tokenize("line 0001\n") == ["line", " ", "0001", "\n"]

    text = "".join(f"line {i:04d}\n" for i in range(320))
    assert count_tokens(text) == 1280
    blocks = chunk_tokens(tokenize(text))
    assert len(blocks) == 10
    assert all(n == BLOCK_SIZE for _, n in blocks)
    # line i sits in block i // 32
    for line in (0, 31, 32, 164, 165, 319):
        chunk_text = blocks[line // 32][0]
        assert f"line {line:04d}\n" in chunk_text

    assert tokenize("def foo_bar(x):\n    return x+1\n") == [
        "def", " ", "foo_", "bar", "(", "x", ")", ":", "\n    ",
        "retu", "rn", " ", "x", "+", "1", "\n",
    ]


def test_empty_content_has_no_blocks() -> None:
    assert chunk_tokens(tokenize("")) == []
