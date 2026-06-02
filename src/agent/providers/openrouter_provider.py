"""
OpenRouter provider.
Allows access to various models via OpenRouter API.
"""

import os
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider."""
    
    def __init__(
        self, 
        model: str = "openrouter/free",  # Default to a free model on OpenRouter for testing
        temperature: float = 0.1,
        max_tokens: int = 4096
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
    
    async def ainvoke(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Invoke OpenRouter model and return response with token usage."""
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
        
        return response.content, token_usage
    