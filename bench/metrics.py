from __future__ import annotations

from statistics import median
from typing import Any, Iterable, Sequence

from warmbox.inference.cost import percentile


def _iter_turns(runs: Iterable[dict[str, Any]]):
    for run in runs:
        for agent in run["agents"]:
            for turn in agent["turns"]:
                yield run, agent, turn


def _pct(num: float, den: float) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(runs: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per baseline using the exact formulas of the design document (§7.4)."""
    out: dict[str, dict[str, Any]] = {}
    for baseline in ("A", "B", "C"):
        subset = [r for r in runs if r.get("baseline") == baseline]
        if subset:
            out[baseline] = _summarize_baseline(subset, baseline)
    return out


def _summarize_baseline(runs: Sequence[dict[str, Any]], baseline: str) -> dict[str, Any]:
    turns = [(r, a, t) for r, a, t in _iter_turns(runs)]
    repeat = [t for _, _, t in turns if int(t["turn"]) >= 2]

    hit = sum(int(t["hit_blocks"]) for _, _, t in turns)
    miss = sum(int(t["miss_blocks"]) for _, _, t in turns)
    hit_rep = sum(int(t["hit_blocks"]) for t in repeat)
    miss_rep = sum(int(t["miss_blocks"]) for t in repeat)

    prefill = sum(int(t["prefill_tokens"]) for _, _, t in turns)
    reused = sum(int(t["reused_tokens"]) for _, _, t in turns)
    prefill_rep = sum(int(t["prefill_tokens"]) for t in repeat)
    reused_rep = sum(int(t["reused_tokens"]) for t in repeat)
    l2 = sum(int(t["l2_reused_tokens"]) for _, _, t in turns)

    ttfts = [float(t["ttft_ms"]) for _, _, t in turns]

    usd_per_task = _mean(
        [sum(float(t["usd"]) for t in _turns_of_run(r)) for r in runs]
    )

    passed = [
        r["task_metrics"]["passed"]
        for r in runs
        if r.get("task_metrics", {}).get("passed") is not None
    ]
    pass_at_1 = (
        round(sum(1.0 for p in passed if p) / len(passed), 4) if passed else None
    )

    edit_ratios = [
        float(t["edit_recompute_tokens"]) / float(t["prefill_tokens"])
        for _, _, t in turns
        if t.get("edited_paths") and int(t["prefill_tokens"]) > 0
    ]
    edit_p50 = round(median(edit_ratios), 4) if edit_ratios else None

    followers: list[float] = []
    if baseline == "C":
        for run in runs:
            agents = run["agents"]
            for agent in agents[1:]:
                for t in agent["turns"]:
                    if int(t["turn"]) == 1 and int(t["prefill_tokens"]) > 0:
                        followers.append(
                            float(t["reused_tokens"]) / float(t["prefill_tokens"])
                        )

    return {
        "n_tasks": len({r["task_id"] for r in runs}),
        "n_runs": len(runs),
        "n_turns": len(turns),
        "prefix_hit_rate": round(hit / (hit + miss), 4) if (hit + miss) else 0.0,
        "prefix_hit_rate_repeat": (
            round(hit_rep / (hit_rep + miss_rep), 4) if (hit_rep + miss_rep) else 0.0
        ),
        "prefill_saved_pct": _pct(reused, prefill),
        "prefill_saved_pct_repeat": _pct(reused_rep, prefill_rep),
        "shared_reuse_pct": _pct(l2, prefill),
        "ttft_p50_ms": round(percentile(ttfts, 50), 1),
        "ttft_p95_ms": round(percentile(ttfts, 95), 1),
        "usd_per_task": round(usd_per_task, 6),
        "pass_at_1": pass_at_1,
        "edit_recompute_ratio_p50": edit_p50,
        "followers_turn1_reuse": round(_mean(followers), 4) if followers else None,
        "determinism_ok": _determinism(runs),
    }


def _turns_of_run(run: dict[str, Any]):
    for agent in run["agents"]:
        yield from agent["turns"]


TOKEN_FIELDS = ("prefill_tokens", "reused_tokens", "l2_reused_tokens", "recompute_tokens",
                "hit_blocks", "miss_blocks")


def _determinism(runs: Sequence[dict[str, Any]]) -> bool | None:
    """True when every (task, agent) produced identical token accounting across runs."""
    seen: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
    for run in runs:
        for agent in run["agents"]:
            key = (run["task_id"], agent["agent_id"])
            sig = tuple(
                tuple(t[f] for f in TOKEN_FIELDS) for t in agent["turns"]
            )
            seen.setdefault(key, []).append(sig)
    comparable = [sigs for sigs in seen.values() if len(sigs) > 1]
    if not comparable:
        return None
    return all(len(set(sigs)) == 1 for sigs in comparable)
