"""
DeepSeek via OpenRouter.

Default model: deepseek/deepseek-chat (V3) — fast, low-latency.
Override with env var DEEPSEEK_MODEL=deepseek/deepseek-r1 or via
  python main.py --deepseek-model reasoner
to use R1 for targeted chain-of-thought analysis runs.

max_tokens is capped at 800 because game responses are short JSON objects;
this prevents V3/R1 from padding output unnecessarily.
"""

from __future__ import annotations
import os

from .base import OpenRouterProvider

_CHAT_MODEL = "deepseek/deepseek-chat"
_REASONER_MODEL = "deepseek/deepseek-r1"


class DeepSeekProvider(OpenRouterProvider):
    _max_tokens: int = 800  # cap for fast JSON responses

    def __init__(self) -> None:
        # Resolve model from env var (set by --deepseek-model flag in main.py)
        model_key = os.getenv("DEEPSEEK_MODEL", "chat").lower().strip()
        if model_key in ("reasoner", "r1", _REASONER_MODEL):
            self._model = _REASONER_MODEL
            self._capture_reasoning = True   # R1 exposes CoT
        else:
            self._model = _CHAT_MODEL
            self._capture_reasoning = False  # V3 does not use extended reasoning

        super().__init__()
