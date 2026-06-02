"""
OpenAI-compatible provider for standard API endpoints.
"""

import os
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseLLMProvider


class OpenAICompatProvider(BaseLLMProvider):
    """OpenAI-compatible provider for standard API endpoints."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        temperature: float = 0.1,
        max_tokens: int = 4096
    ):
        api_base = os.getenv("LLM_API_BASE_URL")
        if not api_base:
            raise ValueError("LLM_API_BASE_URL environment variable not set")

        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            base_url=api_base,
            api_key=lambda: os.getenv("OPENAI_COMPAT_API_KEY", "DEFAULT"),
        )

    async def ainvoke(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Invoke OpenAI-compatible model and return response with token usage."""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))

        response = await self.llm.ainvoke(langchain_messages)

        usage_metadata = getattr(response, "usage_metadata", None) or {}
        token_usage = {
            "input_tokens": usage_metadata.get("input_tokens", 0),
            "output_tokens": usage_metadata.get("output_tokens", 0),
            "total_tokens": usage_metadata.get("input_tokens", 0)
            + usage_metadata.get("output_tokens", 0),
        }

        return response.content, token_usage # type: ignore
