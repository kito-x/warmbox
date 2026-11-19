from __future__ import annotations

import re
from typing import Any

RUN_ID_RE = re.compile(r"^[ABC]-r[0-9]+-.+$")
BASELINES = ("A", "B", "C")
TURN_FIELDS = (
    "turn",
    "prefill_tokens",
    "reused_tokens",
    "l2_reused_tokens",
    "recompute_tokens",
    "hit_blocks",
    "miss_blocks",
    "output_tokens",
    "ttft_ms",
    "ttft_simulated",
    "usd",
    "edited_paths",
    "edit_recompute_tokens",
    "tool_calls",
)
SUMMARY_FIELDS = (
    "n_tasks",
    "n_runs",
    "n_turns",
    "prefix_hit_rate",
    "prefix_hit_rate_repeat",
    "prefill_saved_pct",
    "prefill_saved_pct_repeat",
    "shared_reuse_pct",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "usd_per_task",
    "pass_at_1",
    "edit_recompute_ratio_p50",
    "followers_turn1_reuse",
)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _req(obj: dict, keys: tuple[str, ...], where: str, err) -> None:
    for key in keys:
        if key not in obj:
            err(f"{where}: missing key {key!r}")


def validate_results(doc: dict[str, Any]) -> list[str]:
    """Stdlib-only validator for results.json. Returns error strings; empty means valid."""
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    if not isinstance(doc, dict):
        return ["root: not an object"]

    _req(doc, ("schema_version", "generated_at", "environment", "runs", "summary"), "root", err)
    allowed = {"schema_version", "generated_at", "environment", "runs", "summary"}
    for key in doc:
        if key not in allowed:
            err(f"root: unexpected key {key!r}")
    if doc.get("schema_version") != "1.0":
        err("root.schema_version: must be '1.0'")
    if not isinstance(doc.get("generated_at"), str):
        err("root.generated_at: must be a string")

    env = doc.get("environment")
    if not isinstance(env, dict):
        err("environment: not an object")
    else:
        _req(
            env,
            ("python", "platform", "hash_algo", "block_size", "backend", "model", "simulated_kv"),
            "environment",
            err,
        )
        if env.get("hash_algo") not in ("blake3-128", "blake2b-128"):
            err("environment.hash_algo: bad enum")
        if env.get("block_size") != 128:
            err("environment.block_size: must be 128")
        if env.get("backend") not in ("mock", "openai"):
            err("environment.backend: bad enum")
        if not isinstance(env.get("simulated_kv"), bool):
            err("environment.simulated_kv: must be a boolean")
        prices = env.get("prices_per_mtok")
        if prices is not None and not isinstance(prices, dict):
            err("environment.prices_per_mtok: must be an object")

    runs = doc.get("runs")
    if not isinstance(runs, list):
        err("runs: not an array")
    else:
        for i, run in enumerate(runs):
            _validate_run(run, i, err)

    summary = doc.get("summary")
    if not isinstance(summary, dict):
        err("summary: not an object")
    else:
        for name, block in summary.items():
            if name not in BASELINES:
                err(f"summary: unexpected baseline {name!r}")
                continue
            _validate_summary(block, f"summary.{name}", err)

    return errors


def _validate_run(run: Any, i: int, err) -> None:
    where = f"runs[{i}]"
    if not isinstance(run, dict):
        err(f"{where}: not an object")
        return
    _req(
        run,
        ("run_id", "baseline", "run_index", "task_id", "suite", "n_agents", "wall_ms",
         "agents", "task_metrics"),
        where,
        err,
    )
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        err(f"{where}.run_id: must match {RUN_ID_RE.pattern}")
    if run.get("baseline") not in BASELINES:
        err(f"{where}.baseline: bad enum")
    if not isinstance(run.get("run_index"), int) or isinstance(run.get("run_index"), bool):
        err(f"{where}.run_index: must be an integer")
    if not isinstance(run.get("task_id"), str):
        err(f"{where}.task_id: must be a string")
    if run.get("suite") not in ("custom", "swe"):
        err(f"{where}.suite: bad enum")
    n_agents = run.get("n_agents")
    if not isinstance(n_agents, int) or isinstance(n_agents, bool) or n_agents < 1:
        err(f"{where}.n_agents: must be an integer >= 1")
    if not _is_num(run.get("wall_ms")) or run.get("wall_ms", -1) < 0:
        err(f"{where}.wall_ms: must be a number >= 0")

    agents = run.get("agents")
    if not isinstance(agents, list) or not agents:
        err(f"{where}.agents: must be a non-empty array")
    else:
        for j, agent in enumerate(agents):
            _validate_agent(agent, f"{where}.agents[{j}]", err)

    tm = run.get("task_metrics")
    if not isinstance(tm, dict):
        err(f"{where}.task_metrics: not an object")
    else:
        _req(tm, ("prefill_tokens", "reused_tokens", "recompute_tokens", "usd", "passed"),
             f"{where}.task_metrics", err)
        if tm.get("passed") is not None and not isinstance(tm.get("passed"), bool):
            err(f"{where}.task_metrics.passed: must be boolean or null")


def _validate_agent(agent: Any, where: str, err) -> None:
    if not isinstance(agent, dict):
        err(f"{where}: not an object")
        return
    _req(agent, ("agent_id", "role", "finished", "passed", "turns"), where, err)
    if not isinstance(agent.get("agent_id"), str):
        err(f"{where}.agent_id: must be a string")
    if not isinstance(agent.get("role"), str):
        err(f"{where}.role: must be a string")
    if not isinstance(agent.get("finished"), bool):
        err(f"{where}.finished: must be a boolean")
    if agent.get("passed") is not None and not isinstance(agent.get("passed"), bool):
        err(f"{where}.passed: must be boolean or null")
    turns = agent.get("turns")
    if not isinstance(turns, list):
        err(f"{where}.turns: not an array")
        return
    for k, turn in enumerate(turns):
        _validate_turn(turn, f"{where}.turns[{k}]", err)


def _validate_turn(turn: Any, where: str, err) -> None:
    if not isinstance(turn, dict):
        err(f"{where}: not an object")
        return
    _req(turn, TURN_FIELDS, where, err)
    for key in (
        "turn",
        "prefill_tokens",
        "reused_tokens",
        "l2_reused_tokens",
        "recompute_tokens",
        "hit_blocks",
        "miss_blocks",
        "output_tokens",
        "edit_recompute_tokens",
    ):
        value = turn.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            err(f"{where}.{key}: must be an integer")
    if isinstance(turn.get("turn"), int) and turn["turn"] < 1:
        err(f"{where}.turn: must be >= 1")
    if not _is_num(turn.get("ttft_ms")) or turn.get("ttft_ms", -1) < 0:
        err(f"{where}.ttft_ms: must be a number >= 0")
    if not isinstance(turn.get("ttft_simulated"), bool):
        err(f"{where}.ttft_simulated: must be a boolean")
    if not _is_num(turn.get("usd")) or turn.get("usd", -1) < 0:
        err(f"{where}.usd: must be a number >= 0")
    cached = turn.get("server_cached_tokens")
    if cached is not None and (not isinstance(cached, int) or isinstance(cached, bool)):
        err(f"{where}.server_cached_tokens: must be an integer or null")
    for key in ("edited_paths", "tool_calls"):
        value = turn.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            err(f"{where}.{key}: must be an array of strings")


def _validate_summary(block: Any, where: str, err) -> None:
    if not isinstance(block, dict):
        err(f"{where}: not an object")
        return
    _req(block, SUMMARY_FIELDS, where, err)
    for key in (
        "prefix_hit_rate",
        "prefix_hit_rate_repeat",
        "prefill_saved_pct",
        "prefill_saved_pct_repeat",
        "shared_reuse_pct",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "usd_per_task",
    ):
        if not _is_num(block.get(key)):
            err(f"{where}.{key}: must be a number")
    for key in ("pass_at_1", "edit_recompute_ratio_p50", "followers_turn1_reuse"):
        if block.get(key) is not None and not _is_num(block.get(key)):
            err(f"{where}.{key}: must be a number or null")
    if block.get("determinism_ok") is not None and not isinstance(block.get("determinism_ok"), bool):
        err(f"{where}.determinism_ok: must be a boolean or null")
