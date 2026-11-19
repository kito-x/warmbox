# warmbox

A terminal coding agent whose core trick is **not recomputing what it already computed**.

Every turn, an agent resends the system prompt, tool schemas, repo map, conversation and open
files. warmbox cuts each of those segments into 128-token blocks, content-addresses them, and
reuses the KV of every block that has not changed. A file edit invalidates only the blocks from
the first divergent chunk onward; parallel agents on one repo share blocks through a process-wide
radix tree. All commands run inside a workspace jail with an argv allowlist and no network.

## Quickstart

```bash
python3.12 -m venv .venv && .venv/bin/pip install pytest
PYTHONPATH=. .venv/bin/python -m pytest -q     # 92 tests, no network, no Docker
PYTHONPATH=. .venv/bin/python -m warmbox.cli demo
PYTHONPATH=. .venv/bin/python bin/warmbox demo
```

`demo` walks one file through cold read → warm read → mid-file write → re-read → append and
prints the exact reuse accounting (hit rate 0.610, 3200 tokens saved of 5124).

## Run an agent

```bash
export WARMBOX_BACKEND=mock                      # or: openai
export WARMBOX_BASE_URL=http://localhost:8000/v1 # any OpenAI-compatible endpoint
export WARMBOX_MODEL=mock-1
.venv/bin/python -m warmbox.cli run --repo ./some-repo --goal "fix the failing test"
```

Add `--agents 4 --mode lockstep` for parallel agents sharing the L2 cache, `--no-cache` for the
vanilla baseline, `--sandbox docker` for enforced network isolation.

## Benchmark

```bash
.venv/bin/python bench/harness.py --baseline A --tasks 20 --runs 3
.venv/bin/python bench/harness.py --baseline B --tasks 20 --runs 3
.venv/bin/python bench/harness.py --baseline C --tasks 20 --runs 3 --agents 5
.venv/bin/python bench/harness.py --report
.venv/bin/python bench/harness.py --check     # exit 0 when every gate passes
```

`A` recomputes everything, `B` is single-agent with L1, `C` adds the shared radix tree. Results
land in `bench/results.json` and are validated against `bench/results.schema.json`.

**What is real vs. simulated:** block hashing, invalidation, the version registry, the radix tree,
the path jail, the argv allowlist and env scrubbing are real. KV tensors are simulated handles, so
reuse/recompute token counts are exact for this accounting model; TTFT comes from a latency model
and pass@1 on the mock backend is a scripted self-test. Real prefill reuse needs an engine with
modular attention (Phase 4).

## Layout

`warmbox/cache` (hashing, L1, L2, invalidation, manager) · `warmbox/sandbox` (policy, fs watcher,
backends, runtime) · `warmbox/inference` (tokenizer, prompt builder, client, cost) ·
`warmbox/agent` (loop, tools, working set, history, repo map) · `bench/` · `tests/`.

The design document this implements lives in `docs/DESIGN.md`.
