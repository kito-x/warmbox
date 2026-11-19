from __future__ import annotations

import json
from pathlib import Path

N_TASKS = 20
FNS = 8


def _reads(m: int) -> list[dict]:
    return [
        {"tool": "read_file", "args": {"path": "minicalc/__init__.py"}},
        {"tool": "read_file", "args": {"path": "minicalc/registry.py"}},
        {"tool": "read_file", "args": {"path": f"minicalc/op_{m:02d}.py"}},
    ]


def _agents(m: int, k: int, n: int) -> list[dict]:
    op = f"minicalc/op_{m:02d}.py"
    test = f"tests/test_op_{m:02d}.py"
    buggy = f"    y{k} = y{k} + {k} + 1\n"
    fixed = f"    y{k} = y{k} + {k}\n"
    docstring = f'"""minicalc.op_{m:02d}: arithmetic kernels (generated)."""\n'

    fixer = _reads(m) + [
        {"tool": "run", "args": {"argv": ["python", "-m", "pytest", "-q", test]}},
        {
            "tool": "edit_file",
            "args": {"path": op, "old_str": buggy, "new_str": fixed},
        },
        {"tool": "run", "args": {"argv": ["python", "-m", "pytest", "-q", test]}},
        {"tool": "finish", "args": {"summary": f"fixed op_{m:02d}"}},
    ]
    reviewer = _reads(m) + [
        {"tool": "read_file", "args": {"path": test}},
        {"tool": "search", "args": {"pattern": f"def f{k}", "path": "minicalc"}},
        {"tool": "finish", "args": {"summary": "reviewed"}},
    ]
    doc_writer = _reads(m) + [
        {
            "tool": "edit_file",
            "args": {
                "path": op,
                "old_str": docstring,
                "new_str": docstring + "# Reviewed by doc-writer.\n",
            },
        },
        {"tool": "finish", "args": {"summary": "annotated"}},
    ]
    appender = [
        {"tool": "read_file", "args": {"path": op}},
        {
            "tool": "edit_file",
            "args": {
                "path": op,
                "old_str": "    return y7\n",
                "new_str": "    return y7\n\n\ndef g_extra(x: int) -> int:\n    return x\n",
            },
        },
        {"tool": "run", "args": {"argv": ["python", "-m", "pytest", "-q", test]}},
        {"tool": "finish", "args": {"summary": "appended helper"}},
    ]
    explorer = _reads(m) + [
        {"tool": "read_file", "args": {"path": f"minicalc/op_{n:02d}.py"}},
        {"tool": "list_dir", "args": {"path": "minicalc"}},
        {"tool": "finish", "args": {"summary": "explored"}},
    ]

    return [
        {
            "role": "fixer",
            "goal": f"Tests in {test} fail. Find and fix the bug in {op}.",
            "script": fixer,
        },
        {
            "role": "reviewer",
            "goal": f"Review {op} and its tests. Do not modify files.",
            "script": reviewer,
        },
        {
            "role": "doc-writer",
            "goal": f"Add a review marker comment under the module docstring of {op}.",
            "script": doc_writer,
        },
        {
            "role": "appender",
            "goal": f"Append a helper g_extra(x) returning x to {op}.",
            "script": appender,
        },
        {
            "role": "explorer",
            "goal": f"Summarise how {op} relates to minicalc/op_{n:02d}.py.",
            "script": explorer,
        },
    ]


def generate(dest_dir: Path) -> list[Path]:
    """Write task_00.json .. task_19.json and return their paths."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for t in range(N_TASKS):
        m = t
        k = t % FNS
        n = (t + 1) % N_TASKS
        op = f"minicalc/op_{m:02d}.py"
        doc = {
            "id": f"custom-{t:02d}",
            "suite": "custom",
            "fixture": "minicalc",
            "setup": [
                {
                    "op": "replace",
                    "path": op,
                    "old": f"    y{k} = y{k} + {k}\n",
                    "new": f"    y{k} = y{k} + {k} + 1\n",
                }
            ],
            "agents": _agents(m, k, n),
            "check": {
                "agent": 0,
                "argv": ["python", "-m", "pytest", "-q", f"tests/test_op_{m:02d}.py"],
            },
        }
        path = dest_dir / f"task_{t:02d}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written
