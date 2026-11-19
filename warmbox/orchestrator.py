from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

from .agent.loop import AgentConfig, AgentLoop, AgentResult
from .agent.prompts import SYSTEM_PROMPT
from .agent.repo_map import build_repo_map
from .agent.tools import TOOL_SCHEMAS
from .cache.hashing import chunk_hash, namespace_for, repo_hash
from .cache.manager import CacheManager
from .cache.block_cache import KVCache
from .cache.radix_tree import RadixTree
from .inference.cost import CostModel
from .inference.prompt_builder import StablePrefix
from .sandbox.backends import make_backend
from .sandbox.policy import SandboxPolicy
from .sandbox.runtime import Sandbox

if TYPE_CHECKING:  # pragma: no cover
    from .inference.client import LLMClient


@dataclass(frozen=True, slots=True)
class AgentSpec:
    agent_id: str
    goal: str
    client: "LLMClient"
    # DESIGN-GAP: the design document ramps per-agent temperature for SWE baseline C
    # (§7.2) but AgentSpec's documented fields are only (agent_id, goal, client); this adds
    # the per-agent override rather than sharing one temperature across the fan-out.
    temperature: float | None = None


class Orchestrator:
    def __init__(
        self,
        repo: Path,
        *,
        model_id: str,
        use_cache: bool = True,
        shared_cache: bool = True,
        sandbox_backend: Literal["local", "docker"] = "local",
        mode: Literal["lockstep", "threads"] = "lockstep",
        workspaces_root: Path | None = None,
        cost: CostModel | None = None,
        agent_config: AgentConfig | None = None,
        radix: RadixTree | None = None,
        # DESIGN-GAP: the design document's Orchestrator signature has no policy parameter, but
        # the CLI's --allow-network needs one; it defaults to the standard sandbox policy.
        policy: "SandboxPolicy | None" = None,
        # DESIGN-GAP: the harness must run its pass@1 check inside agent 0's workspace after the
        # run, so workspace cleanup is made a parameter instead of reading the env var directly.
        keep_workspaces: bool | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.model_id = model_id
        self.use_cache = use_cache
        self.shared_cache = shared_cache
        self.sandbox_backend = sandbox_backend
        self.mode = mode
        self.workspaces_root = workspaces_root
        self.cost = cost or CostModel()
        self.agent_config = agent_config or AgentConfig()
        self.policy = policy or SandboxPolicy()
        self.keep_workspaces = keep_workspaces
        self._radix = radix
        self._namespace = ""
        self._stable: StablePrefix | None = None

    # ------------------------------------------------------------------ properties

    @property
    def namespace(self) -> str:
        if not self._namespace:
            self._snapshot()
        return self._namespace

    @property
    def radix(self) -> RadixTree | None:
        return self._radix if (self.use_cache and self.shared_cache) else None

    @property
    def stable_prefix(self) -> StablePrefix:
        if self._stable is None:
            self._stable = StablePrefix(
                SYSTEM_PROMPT, TOOL_SCHEMAS, build_repo_map(self.repo)
            )
        return self._stable

    # ------------------------------------------------------------------ run

    def _snapshot(self) -> None:
        files: list[tuple[str, str]] = []
        for rel in Sandbox._walk_text_files(self.repo):  # static helper, no workspace created
            try:
                text = (self.repo / rel).read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            files.append((rel, chunk_hash(text)))
        self._namespace = namespace_for(self.model_id, repo_hash(files))

    def run(self, specs: Sequence[AgentSpec]) -> list[AgentResult]:
        self._snapshot()
        stable = self.stable_prefix
        radix: RadixTree | None = None
        if self.use_cache and self.shared_cache:
            radix = self._radix or RadixTree()
            self._radix = radix

        loops: list[AgentLoop] = []
        for spec in specs:
            sandbox = Sandbox(
                self.repo,
                spec.agent_id,
                backend=make_backend(self.sandbox_backend),
                policy=self.policy,
                workspaces_root=self.workspaces_root,
            )
            cache = (
                KVCache(agent_id=spec.agent_id, namespace=self.namespace, shared=radix)
                if self.use_cache
                else None
            )
            loop = AgentLoop(
                spec.agent_id,
                sandbox=sandbox,
                client=spec.client,
                cache_manager=CacheManager(cache),
                stable_prefix=stable,
                cost=self.cost,
                config=(
                    self.agent_config
                    if spec.temperature is None
                    else replace(self.agent_config, temperature=spec.temperature)
                ),
            )
            loop.start(spec.goal)
            loops.append(loop)

        if self.mode == "threads" and len(loops) > 1:
            with ThreadPoolExecutor(max_workers=len(loops)) as pool:
                list(pool.map(_drain, loops))
        else:
            while any(not loop.done for loop in loops):
                for loop in loops:
                    if not loop.done:
                        loop.step()

        results = [loop.result() for loop in loops]
        if self.keep_workspaces is None:
            keep = os.environ.get("WARMBOX_KEEP_WORKSPACES") == "1"
        else:
            keep = self.keep_workspaces
        for loop in loops:
            loop.sandbox.close(keep=keep)
        return results


def _drain(loop: AgentLoop) -> None:
    while not loop.done:
        loop.step()
