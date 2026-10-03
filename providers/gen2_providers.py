"""Newer versions of two families (GPT-4o and DeepSeek V3 are older than Claude Sonnet 4.5), and GPT-4.1 / GPT-5 minimal."""
from .base import OpenRouterProvider


class GPT5Provider(OpenRouterProvider):
    _model = "openai/gpt-5"


class GPT41Provider(OpenRouterProvider):
    _model = "openai/gpt-4.1"


class GPT5MinProvider(OpenRouterProvider):
    _model = "openai/gpt-5"
    _reasoning = {"effort": "minimal"}


class DeepSeek31Provider(OpenRouterProvider):
    _model = "deepseek/deepseek-chat-v3.1"
