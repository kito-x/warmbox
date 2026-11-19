from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..cache.manager import CacheManager, SegmentReport
from ..cache.models import InvalidationResult
from ..config import (
    MAX_TURNS,
    WORKING_SET_MAX_FILES,
    WORKING_SET_MAX_TOKENS,
)
from ..errors import VersionConflictError
from ..inference.client import ChatResponse, Usage
from ..inference.cost import CostModel
from ..inference.prompt_builder import (
    BuiltPrompt,
    FileSegment,
    StablePrefix,
    build_prompt,
)
from ..sandbox.fs_watcher import WriteEvent
from ..sandbox.runtime import Sandbox
from .history import History, Message
from .tools import ToolExecutor
from .working_set import WorkingSet

if TYPE_CHECKING:  # pragma: no cover
    from ..inference.client import LLMClient

log = logging.getLogger("warmbox.agent")


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_turns: int = MAX_TURNS
    temperature: float = 0.0
    max_output_tokens: int = 1024
    working_set_max_files: int = WORKING_SET_MAX_FILES
    working_set_max_tokens: int = WORKING_SET_MAX_TOKENS


@dataclass(frozen=True, slots=True)
class TurnMetrics:
    agent_id: str
    turn: int                       # 1-based
    prefill_tokens: int
    reused_tokens: int
    l2_reused_tokens: int
    recompute_tokens: int
    hit_blocks: int
    miss_blocks: int
    output_tokens: int
    ttft_ms: float                  # measured if real backend and available, else simulated
    ttft_simulated: bool
    latency_ms: float
    usd: float
    server_cached_tokens: int | None
    edited_paths: tuple[str, ...]   # files written since the previous prompt (sorted)
    edit_recompute_tokens: int
    tool_calls: tuple[str, ...]     # tool names in order
    segments: tuple[SegmentReport, ...]


@dataclass(frozen=True, slots=True)
class AgentResult:
    agent_id: str
    goal: str
    finished: bool                  # True if finish tool was called
    summary: str
    turns: tuple[TurnMetrics, ...]
    patch: str                      # sandbox.diff()


def _cost_inputs(report, usage: Usage, simulated: bool) -> dict[str, int]:
    if not simulated and usage.cached_prompt_tokens is not None:
        cached = usage.cached_prompt_tokens
        uncached = max(0, usage.prompt_tokens - cached)
    else:
        cached = report.reused_tokens
        uncached = report.recompute_tokens
    return {
        "uncached_input": uncached,
        "cached_input": cached,
        "output": usage.completion_tokens,
    }


class AgentLoop:
    def __init__(
        self,
        agent_id: str,
        *,
        sandbox: Sandbox,
        client: "LLMClient",
        cache_manager: CacheManager,
        stable_prefix: StablePrefix,
        cost: CostModel | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.sandbox = sandbox
        self.client = client
        self.cache_manager = cache_manager
        self.stable_prefix = stable_prefix
        self.cost = cost or CostModel()
        self.config = config or AgentConfig()
        self._history = History()
        self.working_set = WorkingSet(
            self.config.working_set_max_files, self.config.working_set_max_tokens
        )
        self.tools = ToolExecutor(sandbox, self.working_set)
        self._turn = 0
        self._turns: list[TurnMetrics] = []
        self._done = False
        self._finished_ok = False
        self._summary = ""
        self._goal = ""
        self._pending_edits: set[str] = set()
        if self.cache_manager.cache is not None:
            sandbox.subscribe(self._on_write)

    # ------------------------------------------------------------------ lifecycle

    def start(self, goal: str) -> None:
        self._goal = goal
        self._turn = 0
        self._history.append(Message("user", goal))

    def step(self) -> TurnMetrics | None:
        if self._done:
            return None
        self._turn += 1

        events = self.sandbox.sync()          # D5: out-of-band writes -> invalidation
        edited = frozenset(self._pending_edits | {e.path for e in events})
        self._pending_edits.clear()

        files: list[FileSegment] = []
        for path in self.working_set.paths():
            try:
                content = self.sandbox.read_file(path)
            except FileNotFoundError:
                self.working_set.unpin(path)
                continue
            tf = self.sandbox.version_of(path)
            files.append(FileSegment(path, content, tf.version if tf else 1))

        prompt: BuiltPrompt = build_prompt(self.stable_prefix, files, self._history.messages)
        report = self.cache_manager.prefill(prompt, edited_paths=edited)

        resp: ChatResponse = self.client.chat(
            list(prompt.messages),
            list(prompt.tools),
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        self._history.append(resp.message)

        finished = False
        self.tools.turn = self._turn
        for call in resp.message.tool_calls:
            res = self.tools.execute(call)     # writes fire _on_write synchronously
            self._history.append(
                Message("tool", res.content, tool_call_id=call.id, name=call.name)
            )
            finished |= res.finished
            if res.finished:
                self._summary = str(call.arguments.get("summary", ""))

        if not resp.message.tool_calls:
            finished = True
            self._summary = resp.message.content or ""

        if finished or self._turn >= self.config.max_turns:
            self._done = True
            self._finished_ok = finished

        ttft, simulated = (
            (resp.ttft_ms, False) if resp.ttft_ms is not None else (report.sim_ttft_ms, True)
        )
        usd = self.cost.turn_cost(**_cost_inputs(report, resp.usage, self.client.is_simulated))
        metrics = TurnMetrics(
            agent_id=self.agent_id,
            turn=self._turn,
            prefill_tokens=report.prefill_tokens,
            reused_tokens=report.reused_tokens,
            l2_reused_tokens=report.l2_reused_tokens,
            recompute_tokens=report.recompute_tokens,
            hit_blocks=report.hit_blocks,
            miss_blocks=report.miss_blocks,
            output_tokens=resp.usage.completion_tokens,
            ttft_ms=ttft,
            ttft_simulated=simulated,
            latency_ms=resp.latency_ms,
            usd=usd,
            server_cached_tokens=resp.usage.cached_prompt_tokens,
            edited_paths=tuple(sorted(edited)),
            edit_recompute_tokens=report.edit_recompute_tokens,
            tool_calls=tuple(c.name for c in resp.message.tool_calls),
            segments=report.segments,
        )
        self._turns.append(metrics)
        return metrics

    def run(self, goal: str) -> AgentResult:
        self.start(goal)
        while not self._done:
            self.step()
        return self.result()

    def result(self) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            goal=self._goal,
            finished=self._finished_ok,
            summary=self._summary,
            turns=tuple(self._turns),
            patch=self.sandbox.diff(),
        )

    @property
    def done(self) -> bool:
        return self._done

    @property
    def history(self) -> History:
        return self._history

    @property
    def turns(self) -> tuple[TurnMetrics, ...]:
        return tuple(self._turns)

    # ------------------------------------------------------------------ writes

    def _on_write(self, ev: WriteEvent) -> InvalidationResult:
        cache = self.cache_manager.cache
        if cache is None:  # pragma: no cover - subscription only happens with a cache
            return InvalidationResult(ev.path, None, ev.new_version, 0, 0, 0, 0, 0, 0)
        try:
            res = cache.on_file_write(
                ev.path, ev.old_content, ev.new_content, new_version=ev.new_version
            )
        except VersionConflictError:
            log.warning("version conflict on %s; dropping its cached blocks", ev.path)
            cache.drop_file(ev.path)
            res = InvalidationResult(ev.path, None, ev.new_version, 0, 0, 0, 0, 0, 0)
        self._pending_edits.add(ev.path)
        return res
