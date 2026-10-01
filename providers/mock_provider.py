"""
Deterministic mock provider for unit tests and dry-runs.
Returns predictable, valid JSON responses without calling any external API.
"""

from __future__ import annotations
import json
from .base import BaseProvider, LLMResponse


class MockProvider(BaseProvider):
    """
    A deterministic provider that returns configurable fixed responses.
    Useful for unit-testing game logic without spending API tokens.

    By default it contributes half the endowment (10 tokens) and
    sends a neutral public message.
    """

    def __init__(
        self,
        contribution: float = 10.0,
        public_message: str = "I will contribute fairly.",
        private_reasoning: str | None = "Mock reasoning: contribute half.",
        vote: str | None = None,
        speech: str | None = "I will be a fair manager.",
        punish: dict | None = None,
        reward: dict | None = None,
    ) -> None:
        self._contribution = contribution
        self._public_message = public_message
        self._private_reasoning = private_reasoning
        self._vote = vote
        self._speech = speech
        self._punish = punish or {}
        self._reward = reward or {}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> LLMResponse:
        # Build a context-appropriate structured_action based on keywords in prompt
        action: dict = {}

        prompt_lower = user_prompt.lower()

        if "campaign speech" in prompt_lower or "defend your record" in prompt_lower:
            action = {
                "speech": self._speech or "I will act in the group's best interest.",
                "public_message": self._speech or "Vote for me for a fair outcome.",
            }
        elif "private deal" in prompt_lower:
            action = {
                "public_message": self._public_message,
            }
        elif "vote" in prompt_lower:
            action = {
                "vote": self._vote or "agent_0",
                "public_message": f"I vote for {self._vote or 'agent_0'}.",
            }
        elif "punish" in prompt_lower or "reward" in prompt_lower or "manager" in prompt_lower:
            action = {
                "punish": self._punish,
                "reward": self._reward,
                "public_message": "I have reviewed all contributions and acted accordingly.",
            }
        else:
            # Default: contribution action
            action = {
                "contribution": self._contribution,
                "message": self._public_message,
                "public_message": self._public_message,
            }

        raw = json.dumps(action)
        public_msg = action.get("public_message", self._public_message)

        return LLMResponse(
            public_message=public_msg,
            private_reasoning=self._private_reasoning,
            structured_action=action,
            raw_text=raw,
            model_name="mock",
            input_tokens=0,
            output_tokens=0,
        )

    @property
    def model_id(self) -> str:
        return "mock"
