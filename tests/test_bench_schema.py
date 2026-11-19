from __future__ import annotations

from bench.schema import validate_results


def _turn(n: int = 1) -> dict:
    return {
        "turn": n,
        "prefill_tokens": 100,
        "reused_tokens": 60,
        "l2_reused_tokens": 0,
        "recompute_tokens": 40,
        "hit_blocks": 3,
        "miss_blocks": 2,
        "output_tokens": 10,
        "ttft_ms": 12.5,
        "ttft_simulated": True,
        "server_cached_tokens": None,
        "usd": 0.000123,
        "edited_paths": [],
        "edit_recompute_tokens": 0,
        "tool_calls": ["read_file"],
    }


def _summary() -> dict:
    return {
        "n_tasks": 1,
        "n_runs": 1,
        "n_turns": 1,
        "prefix_hit_rate": 0.6,
        "prefix_hit_rate_repeat": 0.7,
        "prefill_saved_pct": 60.0,
        "prefill_saved_pct_repeat": 70.0,
        "shared_reuse_pct": 0.0,
        "ttft_p50_ms": 12.5,
        "ttft_p95_ms": 12.5,
        "usd_per_task": 0.000123,
        "pass_at_1": 1.0,
        "edit_recompute_ratio_p50": 0.05,
        "followers_turn1_reuse": None,
        "determinism_ok": True,
    }


def minimal_doc() -> dict:
    return {
        "schema_version": "1.0",
        "generated_at": "2026-11-19T00:00:00+00:00",
        "environment": {
            "python": "3.12.3",
            "platform": "macOS",
            "hash_algo": "blake2b-128",
            "block_size": 128,
            "backend": "mock",
            "model": "mock-1",
            "simulated_kv": True,
            "prices_per_mtok": {"input": 3.0, "cached": 0.3, "output": 15.0},
        },
        "runs": [
            {
                "run_id": "B-r0-custom-00",
                "baseline": "B",
                "run_index": 0,
                "task_id": "custom-00",
                "suite": "custom",
                "n_agents": 1,
                "wall_ms": 12.0,
                "agents": [
                    {
                        "agent_id": "agent-0",
                        "role": "fixer",
                        "finished": True,
                        "passed": True,
                        "turns": [_turn()],
                    }
                ],
                "task_metrics": {
                    "prefill_tokens": 100,
                    "reused_tokens": 60,
                    "recompute_tokens": 40,
                    "usd": 0.000123,
                    "passed": True,
                },
            }
        ],
        "summary": {"B": _summary()},
    }


def test_validator_accepts_minimal() -> None:
    assert validate_results(minimal_doc()) == []


def test_validator_rejects_missing_keys() -> None:
    doc = minimal_doc()
    del doc["runs"]
    assert any("runs" in e for e in validate_results(doc))

    doc = minimal_doc()
    del doc["environment"]["hash_algo"]
    assert any("hash_algo" in e for e in validate_results(doc))

    doc = minimal_doc()
    del doc["runs"][0]["agents"][0]["turns"][0]["usd"]
    assert any("usd" in e for e in validate_results(doc))


def test_validator_rejects_bad_enums() -> None:
    doc = minimal_doc()
    doc["environment"]["backend"] = "vllm"
    assert any("backend" in e for e in validate_results(doc))

    doc = minimal_doc()
    doc["runs"][0]["baseline"] = "D"
    assert any("baseline" in e for e in validate_results(doc))

    doc = minimal_doc()
    doc["runs"][0]["run_id"] = "nope"
    assert any("run_id" in e for e in validate_results(doc))

    doc = minimal_doc()
    doc["runs"][0]["task_metrics"]["passed"] = "yes"
    assert any("passed" in e for e in validate_results(doc))


def test_validator_rejects_bad_types() -> None:
    doc = minimal_doc()
    doc["runs"][0]["agents"][0]["turns"][0]["turn"] = "1"
    assert any("turn" in e for e in validate_results(doc))

    doc = minimal_doc()
    doc["runs"][0]["agents"][0]["turns"][0]["edited_paths"] = "a.txt"
    assert any("edited_paths" in e for e in validate_results(doc))

    doc = minimal_doc()
    doc["summary"]["B"]["pass_at_1"] = "high"
    assert any("pass_at_1" in e for e in validate_results(doc))


def test_schema_file_matches_validator_keys() -> None:
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "bench" / "results.schema.json"
    schema = json.loads(path.read_text("utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert schema["properties"]["environment"]["properties"]["block_size"]["const"] == 128
    assert set(schema["properties"]["summary"]["propertyNames"]["enum"]) == {"A", "B", "C"}
