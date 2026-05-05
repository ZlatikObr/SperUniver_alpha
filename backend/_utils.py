"""Shared prompt and JSON helpers."""
import json
from functools import lru_cache
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent


@lru_cache(maxsize=16)
def load_prompt(name: str) -> dict:
    path = _ROOT / "prompts" / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    return None


def extract_json_array(text: str) -> list | None:
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    return None
