"""
Anthropic Claude provider.
"""

from typing import Any, Dict, List, Optional, Tuple
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(
        self, 
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ):
        self.model = model
        self.llm = ChatAnthropic( # type: ignore
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_control={'type': 'ephemeral'}
        )
    
    async def ainvoke(self, messages):
        """Invoke Claude and return response with token usage."""
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                langchain_messages.append(msg)
            else:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))

        if self.tools:
            response = await self.llm.bind_tools(self.tools).ainvoke(langchain_messages)
            self.tools = [] # Clear tools after invocation to avoid unintended reuse
        else:
            response = await self.llm.ainvoke(langchain_messages)
        tool_calls = self.normalize_tool_calls(response)
        usage_metadata = self.normalize_usage_metadata(response)

        return response, tool_calls, usage_metadata

    @staticmethod
    def normalize_tool_calls(response: Any) -> List[Dict[str, Any]]:
        return response.tool_calls

    @staticmethod
    def normalize_usage_metadata(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage_metadata", None) or {}
        if not usage:
            response_metadata = getattr(response, "response_metadata", None) or {}
            usage = response_metadata.get("usage", {}) if isinstance(response_metadata, dict) else {}

        input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)

        # usage_metadata nests cache info under input_token_details
        token_details = usage.get("input_token_details", {}) or {}
        cache_read = int(token_details.get("cache_read", usage.get("cache_read_input_tokens", 0)) or 0)
        cache_creation = int(token_details.get("cache_creation", usage.get("cache_creation_input_tokens", 0)) or 0)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        }
