"""Centralized env var readers with whitespace stripping.

Guards against docker-compose .env trailing newlines across all modules.
"""

from __future__ import annotations

import os


def env(key: str, default: str = "") -> str:
    """Read an env var, stripping whitespace."""
    return os.environ.get(key, default).strip()


def opt_env(key: str) -> str | None:
    """Read an optional env var. Returns None if absent, stripped value if present."""
    val = os.environ.get(key)
    return val.strip() if val is not None else None


def bool_env(key: str, default: str = "false") -> bool:
    """Read a boolean env var (true/1/yes)."""
    return env(key, default).lower() in ("true", "1", "yes")


def parse_custom_headers(raw: str) -> dict[str, str]:
    """Parse newline-separated ``key: value`` header pairs into a dict.

    Lines that don't contain ``": "`` are silently skipped.
    Used by both the Anthropic LLM and OpenAI embedder for gateway headers.
    """
    headers: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if ": " in line:
            key, _, value = line.partition(": ")
            headers[key.strip()] = value.strip()
    return headers
