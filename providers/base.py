"""
Base provider abstraction for all LLM backends.
"""

from __future__ import annotations
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# Per-thread API usage: hg_jobs resets it before each game (one game runs in one thread) and stores it
# with the result, so every result file carries the exact OpenRouter cost of that game.
_USAGE = threading.local()


def usage_reset() -> None:
    _USAGE.u = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def usage_get() -> dict:
    return dict(getattr(_USAGE, "u", None) or {})


@dataclass
class LLMResponse:
    """
    Unified response object returned by every provider.

    Attributes
    ----------
    public_message : str
        The text the agent 'says' publicly (shown to other agents when comm allows).
    private_reasoning : str | None
        Internal chain-of-thought / thinking trace, if the model exposes it.
        Only logged for research purposes — never shared with other agents.
    structured_action : dict
        Machine-readable action extracted from the response.
        Schema depends on context (action phase, election phase, manager phase).
    raw_text : str
        The full raw text returned by the LLM (for debugging / auditing).
    model_name : str
        The exact model identifier used for this call.
    input_tokens : int
        Approximate prompt token count.
    output_tokens : int
        Approximate completion token count.
    """

    public_message: str
    private_reasoning: str | None = None
    structured_action: dict = field(default_factory=dict)
    raw_text: str = ""
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_contribution(self) -> float:
        """Extract numeric contribution from structured_action."""
        val = self.structured_action.get("contribution", 0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def get_message(self) -> str | None:
        """Extract optional communication message from structured_action."""
        return self.structured_action.get("message")

    def get_vote(self) -> str | None:
        """Extract vote target (agent_id) from structured_action."""
        return self.structured_action.get("vote")

    def get_speech(self) -> str | None:
        """Extract campaign speech from structured_action."""
        return self.structured_action.get("speech")

    def get_private_deal(self) -> dict | None:
        """Extract private deal message from structured_action."""
        return self.structured_action.get("private_deal") or self.structured_action.get("private_note")

    def get_punish(self) -> dict:
        """Extract punishment allocations {agent_id: tokens_spent}."""
        return self.structured_action.get("punish", {})

    def get_reward(self) -> dict:
        """Extract reward allocations {agent_id: tokens_spent}."""
        return self.structured_action.get("reward", {})


def _parse_json_from_text(text: str) -> dict:
    """
    Attempt to parse a JSON object from model output.
    Handles code-fenced JSON blocks and bare JSON.
    Returns an empty dict if parsing fails.
    """
    # Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        inner = "\n".join(lines[1:] if lines[-1].strip() == "```" else lines[1:])
        cleaned = inner.strip().rstrip("`").strip()

    # Find first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return {}

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {}


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Subclasses must implement the `generate` method which calls the underlying
    API and returns an LLMResponse populated with public_message and optionally
    private_reasoning and structured_action.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Call the LLM API and return a unified LLMResponse.

        Parameters
        ----------
        system_prompt : str
            The system / context prompt describing the agent's role.
        user_prompt : str
            The current round state and question for the agent to answer.
        temperature : float
            Sampling temperature (0.0 = deterministic, 1.0 = high randomness).
        """

    @property
    def model_id(self) -> str:
        """Human-readable model identifier for logging."""
        return self.__class__.__name__


class OpenAICompatibleProvider(BaseProvider):
    """
    Shared base for providers that use the OpenAI-compatible REST API.
    Subclasses only need to set `_base_url`, `_api_key_env`, and `_model`.
    """

    _base_url: str = "https://api.openai.com/v1"
    _api_key_env: str = "OPENAI_API_KEY"
    _model: str = "gpt-4o"

    def __init__(self) -> None:
        import os
        from openai import OpenAI

        api_key = os.getenv(self._api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing environment variable: {self._api_key_env}"
            )
        self._client = OpenAI(api_key=api_key, base_url=self._base_url)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        choice = response.choices[0]
        raw_text = choice.message.content or ""
        structured = _parse_json_from_text(raw_text)
        public_msg = structured.get("public_message", raw_text)

        return LLMResponse(
            public_message=public_msg,
            private_reasoning=None,
            structured_action=structured,
            raw_text=raw_text,
            model_name=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    @property
    def model_id(self) -> str:
        return self._model


class OpenRouterProvider(BaseProvider):
    """
    Unified base for ALL real LLM providers via OpenRouter.
    Uses a single OPENROUTER_API_KEY and the OpenAI-compatible
    endpoint at https://openrouter.ai/api/v1.

    Subclasses only need to set `_model` (OpenRouter model ID).
    Set `_capture_reasoning = True` for models that expose chain-of-thought
    in `choices[0].message.reasoning` (e.g. DeepSeek R1).
    """

    _base_url: str = "https://openrouter.ai/api/v1"
    _api_key_env: str = "OPENROUTER_API_KEY"
    _model: str = "openai/gpt-4o"
    _capture_reasoning: bool = False  # set True for reasoning models
    _max_tokens: int | None = None    # set to cap output tokens (e.g. 800 for DeepSeek V3)

    def __init__(self) -> None:
        import os
        from openai import OpenAI

        api_key = os.getenv(self._api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing environment variable: {self._api_key_env}"
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/hierarchical-game",
                "X-Title": "MultiAgentPGG",
            },
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

        # Request reasoning tokens for models that support it
        if self._capture_reasoning:
            kwargs["extra_body"] = {"include_reasoning": True}
        if "openrouter" in str(self._base_url):
            kwargs.setdefault("extra_body", {})["usage"] = {"include": True}   # cost per call
        if getattr(self, "_reasoning", None):
            kwargs.setdefault("extra_body", {})["reasoning"] = self._reasoning

        # Cap output tokens if specified (e.g. DeepSeek V3 for faster JSON responses)
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens

        # Robust retry loop — handles three transient failure modes from OpenRouter:
        #   1. HTTP 200 but choices=None (Qwen/Alibaba upstream filter)
        #   2. HTTP 429 RateLimitError (upstream rate limit — wait 30s+)
        #   3. JSONDecodeError inside openai client (malformed HTTP body from provider)
        import time as _time
        import json as _json_mod
        from openai import RateLimitError as _RateLimitError
        from openai import InternalServerError as _ServerError, APIConnectionError as _ConnError, APITimeoutError as _TimeoutError
        _max_retries = 4
        response = None
        _last_exc: Exception | None = None
        for _attempt in range(_max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                if response.choices:
                    _last_exc = None
                    break
                # null/empty choices — transient upstream issue
                if _attempt < _max_retries - 1:
                    _time.sleep(2 ** _attempt)  # 1s, 2s, 4s
            except _RateLimitError as _exc:
                _last_exc = _exc
                if _attempt < _max_retries - 1:
                    _time.sleep(30 * (2 ** _attempt))  # 30s, 60s, 120s
            except (_ServerError, _ConnError, _TimeoutError) as _exc:
                # upstream 5xx (e.g. 504 after a provider rate limit), dropped connection, timeout
                _last_exc = _exc
                if _attempt < _max_retries - 1:
                    _time.sleep(20 * (2 ** _attempt))  # 20s, 40s, 80s
            except _json_mod.JSONDecodeError as _exc:
                # openai client failed to parse the HTTP response body
                _last_exc = _exc
                if _attempt < _max_retries - 1:
                    _time.sleep(5 * (2 ** _attempt))  # 5s, 10s, 20s
            except Exception as _exc:
                raise  # non-retryable (auth errors, bad request, etc.)
        if _last_exc is not None:
            raise _last_exc
        if not response or not response.choices:
            raise RuntimeError(
                f"OpenRouter returned empty/null choices after {_max_retries} attempts "
                f"(model={self._model}). This is a transient upstream issue — retry later."
            )
        u = getattr(_USAGE, "u", None)
        if u is not None and response.usage:
            u["calls"] += 1
            u["prompt_tokens"] += response.usage.prompt_tokens or 0
            u["completion_tokens"] += response.usage.completion_tokens or 0
            cost = getattr(response.usage, "cost", None)
            if cost is None:
                cost = (getattr(response.usage, "model_extra", None) or {}).get("cost")
            u["cost_usd"] += float(cost or 0)
        choice = response.choices[0]
        raw_text = choice.message.content or ""

        # Capture chain-of-thought if exposed by the model/router
        private_reasoning: str | None = getattr(choice.message, "reasoning", None)

        structured = _parse_json_from_text(raw_text)
        public_msg = structured.get("public_message", raw_text.strip())

        return LLMResponse(
            public_message=public_msg,
            private_reasoning=private_reasoning,
            structured_action=structured,
            raw_text=raw_text,
            model_name=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    @property
    def model_id(self) -> str:
        return self._model
