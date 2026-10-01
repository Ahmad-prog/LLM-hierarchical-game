"""Newer versions of two families (GPT-4o and DeepSeek V3 are older than Claude Sonnet 4.5)."""
from .base import OpenRouterProvider


class GPT5Provider(OpenRouterProvider):
    _model = "openai/gpt-5"


class DeepSeek31Provider(OpenRouterProvider):
    _model = "deepseek/deepseek-chat-v3.1"
