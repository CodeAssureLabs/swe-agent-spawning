"""
OpenAI GPT provider.
"""

from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(
        self, 
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4096
    ):
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
    
    async def ainvoke(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Invoke GPT and return response with token usage."""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
        
        response = await self.llm.ainvoke(langchain_messages)
        
        # Extract token usage from response
        usage_metadata = getattr(response, 'usage_metadata', None) or {}
        token_usage = {
            "input_tokens": usage_metadata.get('input_tokens', 0),
            "output_tokens": usage_metadata.get('output_tokens', 0),
            "total_tokens": usage_metadata.get('input_tokens', 0) + usage_metadata.get('output_tokens', 0)
        }
        
        return response.content, token_usage # type: ignore
    
