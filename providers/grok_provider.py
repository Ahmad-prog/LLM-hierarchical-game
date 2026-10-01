"""xAI Grok via OpenRouter."""
from .base import OpenRouterProvider

class GrokProvider(OpenRouterProvider):
    _model = "x-ai/grok-4.3"
