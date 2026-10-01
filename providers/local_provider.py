"""Open model served by a local vLLM server (OpenAI-compatible API).

The model name and endpoint come from game/settings.py (HG_LOCAL_MODEL, HG_LOCAL_BASE_URL).
Replies are constrained to a JSON object, so a reply can never be lost to a parse error
(which the game would otherwise score as a contribution of 0).
"""
from __future__ import annotations

import time

from game import settings
from .base import BaseProvider, LLMResponse, _parse_json_from_text


class LocalProvider(BaseProvider):
    _max_tokens: int = 4096   # room for reasoning models (gpt-oss) before the JSON answer

    def __init__(self) -> None:
        from openai import OpenAI

        if not settings.LOCAL_MODEL:
            raise EnvironmentError("HG_LOCAL_MODEL is not set")
        self._model = settings.LOCAL_MODEL
        self._client = OpenAI(api_key="EMPTY", base_url=settings.LOCAL_BASE_URL, timeout=600)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> LLMResponse:
        kwargs = dict(
            model=self._model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )
        last = None
        for attempt in range(4):
            try:
                response = self._client.chat.completions.create(**kwargs)
                if response.choices:
                    break
            except Exception as exc:  # server busy / restarting: back off and retry
                last = exc
            time.sleep(5 * 2 ** attempt)
        else:
            raise RuntimeError(f"local model {self._model} failed after retries: {last!r}")

        msg = response.choices[0].message
        raw_text = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        structured = _parse_json_from_text(raw_text)
        return LLMResponse(
            public_message=structured.get("public_message", raw_text.strip()),
            private_reasoning=reasoning,
            structured_action=structured,
            raw_text=raw_text,
            model_name=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    @property
    def model_id(self) -> str:
        return self._model
