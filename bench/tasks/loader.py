from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from warmbox.errors import BenchError

BENCH_DIR = Path(__file__).resolve().parents[1]
TASKS_DIR = BENCH_DIR / "tasks"
CUSTOM_DIR = TASKS_DIR / "custom"
SWE_FILE = TASKS_DIR / "swe_lite_subset.json"


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    suite: str
    agents: tuple[dict[str, Any], ...]
    raw: dict[str, Any]
    setup: tuple[dict[str, Any], ...] = ()
    check: dict[str, Any] | None = None
    fixture: str | None = None
    repo_path: str | None = None
    problem_statement: str | None = None

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(str(a.get("role", f"agent-{i}")) for i, a in enumerate(self.agents))


def _from_custom(doc: dict[str, Any]) -> Task:
    return Task(
        id=str(doc["id"]),
        suite="custom",
        agents=tuple(doc.get("agents") or ()),
        raw=doc,
        setup=tuple(doc.get("setup") or ()),
        check=doc.get("check"),
        fixture=doc.get("fixture"),
    )


def _from_swe(doc: dict[str, Any]) -> Task:
    agent = {
        "role": "solver",
        "goal": str(doc.get("problem_statement") or ""),
    }
    return Task(
        id=str(doc["id"]),
        suite="swe",
        agents=(agent,),
        raw=doc,
        repo_path=doc.get("repo_path"),
        problem_statement=doc.get("problem_statement"),
    )


def ensure_custom_tasks(force: bool = False) -> list[Path]:
    if force or not sorted(CUSTOM_DIR.glob("task_*.json")):
        from .custom_gen import generate

        return generate(CUSTOM_DIR)
    return sorted(CUSTOM_DIR.glob("task_*.json"))


def load_tasks(suite: str = "custom", n: int | None = None) -> list[Task]:
    if suite == "custom":
        paths = ensure_custom_tasks()
        tasks = [_from_custom(json.loads(p.read_text("utf-8"))) for p in paths]
    elif suite == "swe":
        if not SWE_FILE.exists():
            raise BenchError(
                f"{SWE_FILE} is missing; run `python bench/tasks/fetch_swe_lite.py` first"
            )
        docs = json.loads(SWE_FILE.read_text("utf-8"))
        tasks = [_from_swe(doc) for doc in docs]
    else:
        raise BenchError(f"unknown suite: {suite!r}")
    tasks.sort(key=lambda t: t.id)
    return tasks[:n] if n else tasks
