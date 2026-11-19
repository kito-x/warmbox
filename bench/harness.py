from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from bench.metrics import summarize
from bench.schema import validate_results
from bench.tasks.loader import Task, load_tasks
from warmbox.agent.loop import AgentConfig
from warmbox.cache.hashing import HASH_ALGO
from warmbox.config import BLOCK_SIZE, Settings
from warmbox.errors import BenchError
from warmbox.inference.client import MockBackend, make_client
from warmbox.inference.cost import CostModel
from warmbox.orchestrator import AgentSpec, Orchestrator
from warmbox.sandbox.policy import SandboxPolicy
from warmbox.sandbox.runtime import run_in_workspace

BENCH_DIR = Path(__file__).resolve().parent          # bench/
CACHE_DIR = BENCH_DIR / ".cache"
OUT_DIR = BENCH_DIR / "out"

# SWE baseline C fans one task out across five agents at a temperature ramp (§7.2).
TEMPERATURE_RAMP = (0.0, 0.2, 0.4, 0.6, 0.8)


# ---------------------------------------------------------------------- helpers


def _materialize(task: Task, work_root: Path) -> Path:
    dest = work_root / task.id
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if task.suite == "custom":
        from bench.fixtures.make_minicalc import write_minicalc

        write_minicalc(dest)
    elif task.suite == "swe":
        if not task.repo_path or not Path(task.repo_path).exists():
            raise BenchError(f"{task.id}: repo_path missing: {task.repo_path}")
        shutil.copytree(task.repo_path, dest)
    else:
        raise BenchError(f"{task.id}: unknown suite {task.suite!r}")

    for op in task.setup:
        if op.get("op") != "replace":
            raise BenchError(f"{task.id}: unsupported setup op {op.get('op')!r}")
        target = dest / op["path"]
        text = target.read_text("utf-8")
        old, new = op["old"], op["new"]
        if text.count(old) != 1:
            raise BenchError(
                f"{task.id}: setup replacement target occurs {text.count(old)} times in {op['path']}"
            )
        target.write_text(text.replace(old, new, 1), "utf-8")
    return dest


def _turn_record(t) -> dict[str, Any]:
    return {
        "turn": t.turn,
        "prefill_tokens": t.prefill_tokens,
        "reused_tokens": t.reused_tokens,
        "l2_reused_tokens": t.l2_reused_tokens,
        "recompute_tokens": t.recompute_tokens,
        "hit_blocks": t.hit_blocks,
        "miss_blocks": t.miss_blocks,
        "output_tokens": t.output_tokens,
        "ttft_ms": round(t.ttft_ms, 1),
        "ttft_simulated": t.ttft_simulated,
        "server_cached_tokens": t.server_cached_tokens,
        "usd": round(t.usd, 6),
        "edited_paths": list(t.edited_paths),
        "edit_recompute_tokens": t.edit_recompute_tokens,
        "tool_calls": list(t.tool_calls),
    }


def _specs(
    task: Task,
    baseline: str,
    n_agents: int,
    backend: str,
    settings: Settings,
    seed: int = 0,
) -> list[AgentSpec]:
    agents = list(task.agents)
    if baseline != "C":
        agents = agents[:1]
    else:
        agents = agents[: max(1, n_agents)]
    specs: list[AgentSpec] = []
    for i, agent in enumerate(agents):
        if backend == "mock":
            client = MockBackend(list(agent.get("script") or []), model_id=settings.model)
        else:
            client = make_client("openai", settings=settings)
        temperature = None
        if baseline == "C" and backend == "openai":
            temperature = TEMPERATURE_RAMP[(i + seed) % len(TEMPERATURE_RAMP)]
        specs.append(
            AgentSpec(
                f"agent-{i}",
                str(agent.get("goal") or task.problem_statement or ""),
                client,
                temperature=temperature,
            )
        )
    return specs


def environment_doc(settings: Settings, backend: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hash_algo": HASH_ALGO,
        "block_size": BLOCK_SIZE,
        "backend": backend,
        "model": settings.model,
        # KV tensors are simulated in every backend of this MVP (see the design document §11.1).
        "simulated_kv": True,
        "prices_per_mtok": {
            "input": settings.price_input,
            "cached": settings.price_cached,
            "output": settings.price_output,
        },
    }


# ---------------------------------------------------------------------- one run


def run_task(
    baseline: str,
    task: Task,
    run_index: int,
    *,
    backend: str,
    n_agents: int,
    sandbox: str,
    mode: str,
    settings: Settings,
    seed: int = 0,
) -> dict[str, Any]:
    if task.suite == "swe" and backend != "openai":
        raise BenchError("--suite swe requires --backend openai")

    repo = _materialize(task, CACHE_DIR / "work")
    ws_root = CACHE_DIR / "ws" / f"{baseline}-r{run_index}-{task.id}"
    if ws_root.exists():
        shutil.rmtree(ws_root)

    specs = _specs(task, baseline, n_agents, backend, settings, seed)
    orch = Orchestrator(
        repo,
        model_id=settings.model,
        use_cache=baseline in ("B", "C"),
        shared_cache=baseline == "C",
        sandbox_backend=sandbox,
        mode=mode,
        workspaces_root=ws_root,
        cost=CostModel.from_settings(settings),
        agent_config=AgentConfig(),
        keep_workspaces=True,
    )

    started = time.perf_counter()
    results = orch.run(specs)
    wall_ms = (time.perf_counter() - started) * 1000

    passed: dict[int, bool | None] = {}
    if task.check:
        idx = int(task.check.get("agent", 0))
        workspace = ws_root / f"agent-{idx}"
        if workspace.exists():
            try:
                res = run_in_workspace(
                    workspace, list(task.check["argv"]), policy=SandboxPolicy()
                )
                passed[idx] = res.exit_code == 0
            except Exception as exc:  # a broken check must not abort the sweep
                passed[idx] = False
                print(f"  check error on {task.id}: {exc}", file=sys.stderr)
        else:
            passed[idx] = False

    agents_out: list[dict[str, Any]] = []
    for i, result in enumerate(results):
        role = task.roles[i] if i < len(task.roles) else f"agent-{i}"
        agents_out.append(
            {
                "agent_id": result.agent_id,
                "role": role,
                "finished": result.finished,
                "passed": passed.get(i),
                "turns": [_turn_record(t) for t in result.turns],
            }
        )

    if task.suite == "swe":
        # Official-evaluator input (§7.6): one line per task, patch only.
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        predictions = OUT_DIR / f"predictions_{baseline}_r{run_index}.jsonl"
        with predictions.open("a", encoding="utf-8") as fh:
            for result in results:
                fh.write(
                    json.dumps(
                        {
                            "instance_id": task.id,
                            "model_name_or_path": f"warmbox-{baseline}",
                            "model_patch": result.patch,
                        }
                    )
                    + "\n"
                )

    shutil.rmtree(ws_root, ignore_errors=True)
    shutil.rmtree(repo, ignore_errors=True)

    all_turns = [t for r in results for t in r.turns]
    task_metrics = {
        "prefill_tokens": sum(t.prefill_tokens for t in all_turns),
        "reused_tokens": sum(t.reused_tokens for t in all_turns),
        "recompute_tokens": sum(t.recompute_tokens for t in all_turns),
        "usd": round(sum(t.usd for t in all_turns), 6),
        "passed": passed.get(0),
    }

    return {
        "run_id": f"{baseline}-r{run_index}-{task.id}",
        "baseline": baseline,
        "run_index": run_index,
        "task_id": task.id,
        "suite": task.suite,
        "n_agents": len(results),
        "wall_ms": round(wall_ms, 1),
        "agents": agents_out,
        "task_metrics": task_metrics,
    }


def run_baseline(
    baseline: str,
    tasks: Sequence[Task],
    runs: int,
    *,
    backend: str,
    n_agents: int,
    sandbox: str,
    mode: str,
    settings: Settings,
    seed: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if any(t.suite == "swe" for t in tasks):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for r in range(runs):
            stale = OUT_DIR / f"predictions_{baseline}_r{r}.jsonl"
            if stale.exists():
                stale.unlink()
    for task in tasks:
        for run_index in range(runs):
            records.append(
                run_task(
                    baseline,
                    task,
                    run_index,
                    backend=backend,
                    n_agents=n_agents,
                    sandbox=sandbox,
                    mode=mode,
                    settings=settings,
                    seed=seed,
                )
            )
            print(
                f"  {baseline} {task.id} run {run_index}: "
                f"{records[-1]['task_metrics']['reused_tokens']}/"
                f"{records[-1]['task_metrics']['prefill_tokens']} tokens reused, "
                f"passed={records[-1]['task_metrics']['passed']}",
                file=sys.stderr,
            )
    return records


# ---------------------------------------------------------------------- results doc


def merge_results(
    path: Path, new_runs: list[dict[str, Any]], baseline: str, env: dict[str, Any]
) -> dict[str, Any]:
    doc: dict[str, Any]
    if path.exists():
        try:
            doc = json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError:
            doc = {}
    else:
        doc = {}
    doc.setdefault("schema_version", "1.0")
    doc["environment"] = env
    runs = [r for r in doc.get("runs", []) if r.get("baseline") != baseline]
    runs.extend(new_runs)
    runs.sort(key=lambda r: (r["baseline"], r["task_id"], r["run_index"]))
    doc["runs"] = runs
    doc["summary"] = summarize(runs)
    doc["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", "utf-8")
    os.replace(tmp, path)
    return doc


# ---------------------------------------------------------------------- reporting


def _fmt(value: Any, spec: str = ".2f") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return format(value, spec)


def print_report(doc: dict[str, Any]) -> None:
    header = (
        f"{'baseline':<8} {'tasks':>5} {'runs':>5} {'turns':>6} "
        f"{'hit%(rep)':>10} {'saved%(rep)':>12} {'shared%':>8} "
        f"{'ttft_p50':>9} {'ttft_p95':>9} {'$/task':>10} {'pass@1':>7} {'edit_rc_p50':>12}"
    )
    print(header)
    for baseline in ("A", "B", "C"):
        s = doc.get("summary", {}).get(baseline)
        if not s:
            continue
        print(
            f"{baseline:<8} {s['n_tasks']:>5} {s['n_runs']:>5} {s['n_turns']:>6} "
            f"{_fmt(s['prefix_hit_rate'], '.4f') + '/' + _fmt(s['prefix_hit_rate_repeat'], '.4f'):>10} "
            f"{_fmt(s['prefill_saved_pct']) + '/' + _fmt(s['prefill_saved_pct_repeat']):>12} "
            f"{_fmt(s['shared_reuse_pct']):>8} "
            f"{_fmt(s['ttft_p50_ms'], '.1f'):>9} {_fmt(s['ttft_p95_ms'], '.1f'):>9} "
            f"{_fmt(s['usd_per_task'], '.6f'):>10} {_fmt(s['pass_at_1'], '.2f'):>7} "
            f"{_fmt(s['edit_recompute_ratio_p50'], '.4f'):>12}"
        )
    if doc.get("environment", {}).get("backend") == "mock":
        print("NOTE: mock backend -> KV, TTFT and pass@1 are SIMULATED (see the design document §11.1)")


def check_gates(doc: dict[str, Any]) -> list[tuple[str, bool, str]]:
    s = doc.get("summary", {})
    a = s.get("A", {})
    b = s.get("B", {})
    c = s.get("C", {})
    gates: list[tuple[str, bool, str]] = []

    gates.append(
        (
            "G-A0",
            a.get("prefill_saved_pct") == 0 and a.get("prefix_hit_rate") == 0,
            f"A saved={a.get('prefill_saved_pct')} hit={a.get('prefix_hit_rate')}",
        )
    )

    b_rep = b.get("prefill_saved_pct_repeat")
    gates.append(("G-B65", b_rep is not None and b_rep >= 65.0, f"B repeat saved={b_rep}"))

    c_rep = c.get("prefill_saved_pct_repeat")
    gates.append(
        (
            "G-C>=B",
            b_rep is not None and c_rep is not None and c_rep >= b_rep,
            f"C={c_rep} B={b_rep}",
        )
    )

    gates.append(
        (
            "G-SHARE",
            (c.get("shared_reuse_pct") or 0) > 0 and (c.get("followers_turn1_reuse") or 0) >= 0.5,
            f"C shared={c.get('shared_reuse_pct')} followers={c.get('followers_turn1_reuse')}",
        )
    )

    edit = b.get("edit_recompute_ratio_p50")
    gates.append(("G-EDIT", edit is not None and edit < 0.10, f"B edit_rc_p50={edit}"))

    gates.append(
        (
            "G-COST",
            (b.get("usd_per_task") or 0) < (a.get("usd_per_task") or 0),
            f"B={b.get('usd_per_task')} A={a.get('usd_per_task')}",
        )
    )

    pass_ok = all(s.get(x, {}).get("pass_at_1") == 1.0 for x in ("A", "B", "C"))
    gates.append(
        (
            "G-PASS",
            pass_ok,
            " ".join(f"{x}={s.get(x, {}).get('pass_at_1')}" for x in ("A", "B", "C")),
        )
    )

    det_ok = all(s.get(x, {}).get("determinism_ok") is True for x in ("A", "B", "C"))
    gates.append(
        (
            "G-DET",
            det_ok,
            " ".join(f"{x}={s.get(x, {}).get('determinism_ok')}" for x in ("A", "B", "C")),
        )
    )

    errors = validate_results(doc)
    gates.append(("G-SCHEMA", not errors, "valid" if not errors else "; ".join(errors[:3])))
    return gates


def ingest_swe_eval(path: Path, out_path: Path, baseline: str, run_index: int) -> dict[str, Any]:
    report = json.loads(Path(path).read_text("utf-8"))
    resolved = set(report.get("resolved_ids") or [])
    doc = json.loads(out_path.read_text("utf-8"))
    touched = 0
    for run in doc.get("runs", []):
        if run.get("baseline") != baseline or int(run.get("run_index", -1)) != run_index:
            continue
        ok = run.get("task_id") in resolved
        run["task_metrics"]["passed"] = ok
        for agent in run.get("agents", []):
            agent["passed"] = ok
        touched += 1
    doc["summary"] = summarize(doc["runs"])
    doc["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_path.write_text(json.dumps(doc, indent=2) + "\n", "utf-8")
    print(f"ingested {len(resolved)} resolved ids into {touched} runs of baseline {baseline}")
    return doc


# ---------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench/harness.py")
    parser.add_argument("--baseline", choices=["A", "B", "C"])
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--suite", choices=["custom", "swe"], default="custom")
    parser.add_argument("--backend", choices=["mock", "openai"], default="mock")
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--sandbox", choices=["local", "docker"], default="local")
    parser.add_argument("--mode", choices=["lockstep", "threads"], default=None)
    parser.add_argument("--out", default=str(BENCH_DIR / "results.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--ingest-swe-eval")
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    settings = Settings.from_env()

    if args.ingest_swe_eval:
        if not args.baseline:
            parser.error("--ingest-swe-eval needs --baseline")
        ingest_swe_eval(Path(args.ingest_swe_eval), out_path, args.baseline, args.run_index)
        return 0

    if args.report or args.check:
        if not out_path.exists():
            print(f"error: {out_path} does not exist", file=sys.stderr)
            return 2
        doc = json.loads(out_path.read_text("utf-8"))
        if args.report:
            print_report(doc)
        if args.check:
            gates = check_gates(doc)
            for gate_id, ok, detail in gates:
                print(f"{'PASS' if ok else 'FAIL'} {gate_id}: {detail}")
            return 0 if all(ok for _, ok, _ in gates) else 1
        return 0

    if not args.baseline:
        parser.error("--baseline is required to run a sweep")

    mode = args.mode or ("threads" if args.backend == "openai" else "lockstep")
    if args.suite == "swe" and args.backend != "openai":
        raise BenchError("--suite swe requires --backend openai")

    tasks = load_tasks(args.suite, args.tasks)
    print(f"running baseline {args.baseline}: {len(tasks)} tasks x {args.runs} runs "
          f"(backend={args.backend}, sandbox={args.sandbox}, mode={mode})", file=sys.stderr)
    runs = run_baseline(
        args.baseline,
        tasks,
        args.runs,
        backend=args.backend,
        n_agents=args.agents,
        sandbox=args.sandbox,
        mode=mode,
        settings=settings,
        seed=args.seed,
    )
    doc = merge_results(out_path, runs, args.baseline, environment_doc(settings, args.backend))
    print_report(doc)
    errors = validate_results(doc)
    if errors:
        print("schema errors:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
