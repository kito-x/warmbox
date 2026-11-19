from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

from ..config import REPO_MAP_MAX_FILES, REPO_MAP_MAX_TOKENS
from ..inference.tokenizer import count_tokens

SKIP_DIRS = {".git", "__pycache__", ".warmbox", ".infercode", "node_modules", ".tmp", ".pytest_cache"}


def _names_for(source: str) -> str:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.append(f"class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(f"def {node.name}")
    return ", ".join(names)


def build_repo_map(
    root: Path,
    *,
    max_files: int = REPO_MAP_MAX_FILES,
    max_tokens: int = REPO_MAP_MAX_TOKENS,
) -> str:
    """Deterministic one-line-per-file map: path, plus top-level names for parseable .py files."""
    root = Path(root)
    files: list[PurePosixPath] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            parts = path.relative_to(root).parts
            if any(part in SKIP_DIRS for part in parts):
                continue
            try:
                with path.open("rb") as fh:
                    head = fh.read(8192)
            except OSError:
                continue
            if b"\x00" in head:
                continue
            files.append(PurePosixPath(*parts))

    lines: list[str] = []
    used_tokens = 0
    count = 0
    for rel in files:
        if count >= max_files:
            break
        line = rel.as_posix()
        if line.endswith(".py"):
            try:
                names = _names_for((root / rel).read_text("utf-8"))
            except (UnicodeDecodeError, OSError):
                names = ""
            if names:
                line = f"{line}: {names}"
        line += "\n"
        cost = count_tokens(line)
        if lines and used_tokens + cost > max_tokens:
            break
        lines.append(line)
        used_tokens += cost
        count += 1

    remaining = len(files) - count
    if remaining > 0:
        lines.append(f"... ({remaining} more files)\n")
    return "".join(lines)
