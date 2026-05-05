"""Shared runtime configuration for backend modules."""
import os

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
