from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

BLOCK_SIZE: int = 128
L1_MAX_BLOCKS: int = 10_000
L2_MAX_BLOCKS: int = 10_000
HASH_DIGEST_BYTES: int = 16                  # 128-bit -> 32 hex chars
STABLE_KEY: str = "<stable>"
HISTORY_KEY: str = "<history>"
HEADER_KEY_PREFIX: str = "<hdr>:"            # header pseudo-file key = "<hdr>:" + path
WORKING_SET_MAX_FILES: int = 12
WORKING_SET_MAX_TOKENS: int = 24_000
MAX_TOOL_OUTPUT_CHARS: int = 4_000
MAX_FILE_BYTES: int = 1_048_576              # 1 MiB
DEFAULT_CMD_TIMEOUT_S: float = 60.0
MAX_CMD_TIMEOUT_S: float = 120.0
MAX_CMD_OUTPUT_BYTES: int = 65_536
MAX_TURNS: int = 30
REPO_MAP_MAX_FILES: int = 500
REPO_MAP_MAX_TOKENS: int = 4_000
SEARCH_MAX_MATCHES: int = 50
# Latency model (SIMULATED TTFT). Assumes ~10k tok/s prefill and cheap cached-block loads.
LATENCY_OVERHEAD_MS: float = 50.0
PREFILL_MS_PER_TOKEN: float = 0.10
CACHED_MS_PER_TOKEN: float = 0.002
# Cost model, USD per 1M tokens. PLACEHOLDERS: override via env.
PRICE_INPUT_PER_MTOK: float = 3.00
PRICE_CACHED_PER_MTOK: float = 0.30
PRICE_OUTPUT_PER_MTOK: float = 15.00
HOME_DIR: str = ".warmbox"                   # relative to CWD

CACHE_FORMAT: str = "warmbox-kvcache"
CACHE_FORMAT_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class Settings:
    backend: str
    base_url: str
    api_key: str
    model: str
    price_input: float
    price_cached: float
    price_output: float
    home: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        return cls(
            backend=e.get("WARMBOX_BACKEND", "mock"),
            base_url=e.get("WARMBOX_BASE_URL", "http://localhost:8000/v1"),
            api_key=e.get("WARMBOX_API_KEY", ""),
            model=e.get("WARMBOX_MODEL", "mock-1"),
            price_input=float(e.get("WARMBOX_PRICE_INPUT", PRICE_INPUT_PER_MTOK)),
            price_cached=float(e.get("WARMBOX_PRICE_CACHED", PRICE_CACHED_PER_MTOK)),
            price_output=float(e.get("WARMBOX_PRICE_OUTPUT", PRICE_OUTPUT_PER_MTOK)),
            home=Path(e.get("WARMBOX_HOME", HOME_DIR)),
        )
