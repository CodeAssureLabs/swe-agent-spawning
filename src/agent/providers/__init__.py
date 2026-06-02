"""
LLM Provider interfaces.
"""

from typing import Optional
from .base import BaseLLMProvider
from .anthropic_provider import AnthropicProvider
from .openrouter_provider import OpenRouterProvider
from .openai_provider import OpenAIProvider
from .openai_compat_provider import OpenAICompatProvider


def get_provider(provider_name: str = "anthropic", model: Optional[str] = None) -> BaseLLMProvider:
    """
    Get an LLM provider by name.
    
    Args:
        provider_name: Name of the provider (default: anthropic)
        model: Name of the model (default: None)
        
    Returns:
        LLM provider instance
    """
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "openrouter": OpenRouterProvider,
        "api": OpenAICompatProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        # Fallback to anthropic if unknown, or raise error?
        # Better to raise error to be explicit
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    
    return provider_class(**{"model": model} if model else {})


__all__ = [
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "OpenAIProvider",
    "OpenAICompatProvider",
    "get_provider",
]
