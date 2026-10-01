"""Google Gemini via OpenRouter."""
from .base import OpenRouterProvider

class GeminiProvider(OpenRouterProvider):
    _model = "google/gemini-2.5-flash"
    _capture_reasoning = True   # Gemini 2.5 Flash exposes thinking via include_reasoning
