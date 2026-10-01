"""GPT-4o via OpenRouter."""
from .base import OpenRouterProvider

class OpenAIProvider(OpenRouterProvider):
    _model = "openai/gpt-4o"
