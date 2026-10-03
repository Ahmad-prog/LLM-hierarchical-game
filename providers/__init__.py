from .base import BaseProvider, LLMResponse
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .deepseek_provider import DeepSeekProvider
from .gemini_provider import GeminiProvider
from .grok_provider import GrokProvider
from .qwen_provider import QwenProvider
from .mock_provider import MockProvider
from .local_provider import LocalProvider
from .gen2_providers import GPT5Provider, DeepSeek31Provider, GPT41Provider, GPT5MinProvider
from config.enums import ModelType


def get_provider(model: ModelType) -> BaseProvider:
    """Factory: return the appropriate provider instance for a given ModelType."""
    mapping = {
        ModelType.GPT4O: OpenAIProvider,
        ModelType.CLAUDE: AnthropicProvider,
        ModelType.DEEPSEEK: DeepSeekProvider,
        ModelType.GEMINI: GeminiProvider,
        ModelType.GROK: GrokProvider,
        ModelType.QWEN: QwenProvider,
        ModelType.MOCK: MockProvider,
        ModelType.LOCAL: LocalProvider,
        ModelType.GPT5: GPT5Provider,
        ModelType.DEEPSEEK31: DeepSeek31Provider,
        ModelType.GPT41: GPT41Provider,
        ModelType.GPT5MIN: GPT5MinProvider,
    }
    cls = mapping.get(model)
    if cls is None:
        raise ValueError(f"Unknown model type: {model}")
    return cls()


__all__ = [
    "BaseProvider",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepSeekProvider",
    "GeminiProvider",
    "GrokProvider",
    "QwenProvider",
    "MockProvider",
    "get_provider",
]
