"""Alibaba Qwen via OpenRouter."""
from .base import OpenRouterProvider

class QwenProvider(OpenRouterProvider):
    _model = "qwen/qwen-plus"   # the model the paper names (the June code used qwen3-235b-a22b)
