from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .agent.loop import AgentConfig
from .cache.block_cache import KVCache
from .cache.hashing import HASH_ALGO
from .config import BLOCK_SIZE, Settings
from .inference.client import MockBackend, make_client
from .inference.cost import CostModel
from .orchestrator import AgentSpec, Orchestrator
from .sandbox.policy import SandboxPolicy


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warmbox", description="Prefill-reuse coding agent in a sandboxed workspace."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="deterministic cache walkthrough: no sandbox, no LLM")
    demo.add_argument("--lines", type=int, default=320)
    demo.add_argument("--edit-line", type=int, default=165)
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("run", help="run an agent (or several) against a repo copy")
    run.add_argument("--repo", required=True)
    run.add_argument("--goal", required=True)
    run.add_argument("--backend", choices=["mock", "openai"], default="mock")
    run.add_argument("--script", help="JSON file with a mock script (list of steps)")
    run.add_argument("--agents", type=int, default=1)
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--no-shared", action="store_true")
    run.add_argument("--sandbox", choices=["local", "docker"], default="local")
    run.add_argument("--mode", choices=["lockstep", "threads"], default="lockstep")
    run.add_argument("--max-turns", type=int, default=30)
    run.add_argument("--max-command-timeout", type=float, default=None)
    run.add_argument("--allow-network", action="store_true")
    run.add_argument("--keep-workspaces", action="store_true")
    run.set_defaults(func=cmd_run)

    stats = sub.add_parser("cache-stats", help="inspect a persisted L1 cache file")
    stats.add_argument("--cache-file", required=True)
    stats.set_defaults(func=cmd_cache_stats)

    return parser


# ---------------------------------------------------------------------- demo


def cmd_demo(args: argparse.Namespace) -> int:
    cache = KVCache(agent_id="demo", namespace="demo", debug=True)
    text = "".join(f"line {i:04d}\n" for i in range(args.lines))
    idx = args.edit_line

    print(f"warmbox cache demo  hash={HASH_ALGO}  block={BLOCK_SIZE} tokens")

    r1 = cache.get_blocks("demo.txt", text)
    print(
        f"[1] {'cold read':<17}demo.txt v{r1.version}: {r1.total_tokens} tokens, "
        f"{len(r1.blocks)} blocks | hit {r1.hit_blocks}  computed {r1.computed_blocks} | "
        f"recompute {r1.recompute_tokens}"
    )

    r2 = cache.get_blocks("demo.txt", text)
    pct2 = 100.0 * r2.reused_tokens / r2.total_tokens if r2.total_tokens else 0.0
    print(
        f"[2] {'warm read':<17}demo.txt v{r2.version}: hit {r2.hit_blocks} "
        f"computed {r2.computed_blocks} | reuse {pct2:.1f}%"
    )

    text2 = text.replace(f"line {idx:04d}\n", f"LINE {idx:04d}\n")
    inv = cache.on_file_write("demo.txt", text, text2, new_version=2)
    print(
        f"[3] {f'write line {idx}':<17}demo.txt v{inv.new_version}: first divergent block "
        f"{inv.first_divergent_block} | kept {inv.blocks_kept} ({inv.tokens_kept} tok) "
        f"dropped {inv.blocks_dropped} ({inv.tokens_dropped} tok)"
    )

    r4 = cache.get_blocks("demo.txt", text2)
    pct4 = 100.0 * r4.recompute_tokens / r4.total_tokens if r4.total_tokens else 0.0
    print(
        f"[4] {'read after write':<17}demo.txt v{r4.version}: hit {r4.hit_blocks} "
        f"computed {r4.computed_blocks} | recompute {r4.recompute_tokens} ({pct4:.1f}% of file)"
    )

    text3 = text2 + f"line {args.lines:04d}\n"
    inv2 = cache.on_file_write("demo.txt", text2, text3, new_version=3)
    print(
        f"[5] {'append 1 line':<17}demo.txt v{inv2.new_version}: first divergent block "
        f"{inv2.first_divergent_block} | kept {inv2.blocks_kept} ({inv2.tokens_kept} tok) "
        f"dropped {inv2.blocks_dropped} ({inv2.tokens_dropped} tok)"
    )

    r6 = cache.get_blocks("demo.txt", text3)
    print(
        f"[6] {'read after app.':<17}demo.txt v{r6.version}: {r6.total_tokens} tokens, "
        f"{len(r6.blocks)} blocks | hit {r6.hit_blocks} computed {r6.computed_blocks} | "
        f"recompute {r6.recompute_tokens}"
    )

    s = cache.stats()
    print(
        f"stats: hit_rate={s.hit_rate:.3f} prefill_tokens={s.prefill_tokens} "
        f"tokens_saved={s.tokens_saved} recompute_tokens={s.recompute_tokens} "
        f"reuse={100.0 * s.reuse_ratio:.1f}%"
    )
    return 0


# ---------------------------------------------------------------------- run


def _load_script(path: str) -> list[dict]:
    doc = json.loads(Path(path).read_text("utf-8"))
    if isinstance(doc, dict):
        doc = doc.get("script") or doc.get("steps") or []
    return list(doc)


def cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"error: --repo is not a directory: {repo}", file=sys.stderr)
        return 2
    if args.backend == "openai" and not settings.base_url:
        print("error: WARMBOX_BASE_URL is not set", file=sys.stderr)
        return 2

    script = _load_script(args.script) if args.script else []
    specs: list[AgentSpec] = []
    for i in range(max(1, args.agents)):
        if args.backend == "mock":
            client = MockBackend(list(script), model_id=settings.model)
        else:
            client = make_client("openai", settings=settings)
        specs.append(AgentSpec(f"agent-{i}", args.goal, client))

    policy = SandboxPolicy(network=True) if args.allow_network else None

    orchestrator = Orchestrator(
        repo,
        model_id=settings.model,
        use_cache=not args.no_cache,
        shared_cache=not args.no_shared,
        sandbox_backend=args.sandbox,
        mode=args.mode,
        workspaces_root=Path(settings.home) / "workspaces",
        cost=CostModel.from_settings(settings),
        agent_config=AgentConfig(max_turns=args.max_turns),
        policy=policy,
    )
    if args.keep_workspaces:
        import os

        os.environ["WARMBOX_KEEP_WORKSPACES"] = "1"

    print(f"namespace: {orchestrator.namespace}")
    results = orchestrator.run(specs)

    print(f"{'turn':>4} {'prefill':>8} {'reused':>7} {'recomp':>7} {'reuse%':>7} "
          f"{'ttft':>8} {'usd':>9}  tools")
    for result in results:
        print(f"-- {result.agent_id}: {'finished' if result.finished else 'stopped'}")
        for t in result.turns:
            reuse = 100.0 * t.reused_tokens / t.prefill_tokens if t.prefill_tokens else 0.0
            print(
                f"{t.turn:>4} {t.prefill_tokens:>8} {t.reused_tokens:>7} {t.recompute_tokens:>7} "
                f"{reuse:>6.1f}% {t.ttft_ms:>7.1f} {t.usd:>9.6f}  {','.join(t.tool_calls)}"
            )
        patch_lines = len(result.patch.splitlines())
        print(f"summary: {result.summary or '<none>'}")
        print(f"patch: {patch_lines} diff lines")
    return 0


# ---------------------------------------------------------------------- cache-stats


def cmd_cache_stats(args: argparse.Namespace) -> int:
    from .errors import CacheCorruptError

    try:
        cache = KVCache.load(Path(args.cache_file))
    except CacheCorruptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    s = cache.stats()
    print(f"file: {args.cache_file}")
    print(f"namespace: {cache.namespace}  agent: {cache.agent_id}  blocks: {len(cache)}")
    for key, value in s.to_dict().items():
        print(f"  {key}: {value}")
    versions = [cache.version_of(p) for p in sorted(cache._by_path)]
    print("files:")
    for fv in versions:
        if fv is not None:
            print(f"  {fv.path} v{fv.version}  {fv.num_tokens} tokens  {len(fv.chunk_hashes)} blocks")
    return 0


# ---------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
