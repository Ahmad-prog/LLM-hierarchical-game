"""Anthropic Claude via OpenRouter (with reasoning capture when available)."""
from .base import OpenRouterProvider

class AnthropicProvider(OpenRouterProvider):
    _model = "anthropic/claude-sonnet-4.5"   # "claude-sonnet-4-5" is not a valid OpenRouter ID
    _capture_reasoning = True   # OpenRouter passes back reasoning for Claude when supported
