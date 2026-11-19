"""Download a deterministic SWE-bench Lite subset and materialise per-instance worktrees.

Network access happens here, on the host, never inside a sandbox.

    python bench/tasks/fetch_swe_lite.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
CACHE = BENCH_DIR / ".cache"
REPOS = CACHE / "repos"
OUT = Path(__file__).resolve().parent / "swe_lite_subset.json"

DATASET = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=princeton-nlp%2FSWE-bench_Lite&config=default&split=test"
    "&offset={offset}&length=100"
)
HEADERS = {"User-Agent": "warmbox-bench/0.1"}
N_TASKS = 20


def _fetch_rows() -> list[dict]:
    rows: list[dict] = []
    for offset in (0, 100, 200):
        req = urllib.request.Request(DATASET.format(offset=offset), headers=HEADERS)
        with urllib.request.urlopen(req, timeout=120) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
        rows.extend(item["row"] for item in doc.get("rows", []))
    return rows


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    rows = _fetch_rows()
    rows.sort(key=lambda r: r["instance_id"])
    if len(rows) < N_TASKS:
        print(f"only {len(rows)} rows returned", file=sys.stderr)
        return 1
    step = max(1, len(rows) // N_TASKS)
    picked = [rows[i] for i in range(0, len(rows), step)][:N_TASKS]

    REPOS.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    out: list[dict] = []
    for row in picked:
        instance_id = row["instance_id"]
        repo = row["repo"]
        base_commit = row["base_commit"]
        owner_name = repo.replace("/", "__")
        src = REPOS / "_src" / owner_name
        if repo not in sources:
            src.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                _run(["git", "clone", "--filter=blob:none", f"https://github.com/{repo}.git", str(src)])
            sources[repo] = src
        dest = REPOS / instance_id
        if not dest.exists():
            _run(["git", "-C", str(sources[repo]), "worktree", "add", str(dest), base_commit])
        out.append(
            {
                "id": instance_id,
                "suite": "swe",
                "repo": repo,
                "base_commit": base_commit,
                "problem_statement": row.get("problem_statement", ""),
                "hints_text": row.get("hints_text", ""),
                "FAIL_TO_PASS": row.get("FAIL_TO_PASS", "[]"),
                "PASS_TO_PASS": row.get("PASS_TO_PASS", "[]"),
                "repo_path": str(dest),
            }
        )

    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} tasks to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
