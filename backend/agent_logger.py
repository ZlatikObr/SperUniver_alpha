"""Append-only JSONL logging for agent steps."""
import json
import os
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
_LOG_PATH = _ROOT / "logs" / "agent.jsonl"
_TOKENS_PATH = _ROOT / "logs" / "tokens.jsonl"


def log_agent_step(step: str, status: str = "ok", **payload: Any) -> None:
    if os.getenv("PULS_DISABLE_AGENT_LOGS") == "1":
        return
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), "step": step, "status": status, **payload}
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        # Logging must never break the user flow.
        return


def log_token_usage(step: str, usage, model: str = "") -> None:
    """Log token usage from an OpenAI response.usage object."""
    if os.getenv("PULS_DISABLE_AGENT_LOGS") == "1":
        return
    if usage is None:
        return
    try:
        _TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "step": step,
            "model": model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
        with open(_TOKENS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError):
        return
