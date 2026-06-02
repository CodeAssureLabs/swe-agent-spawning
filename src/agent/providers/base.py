"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    model: str
    llm: BaseChatModel
    tools: List[Any] = []
    
    def bind_tools(self, tools: List[Dict]):
        """Bind tools to the underlying LLM, returning a new LLM instance."""
        self.tools = tools
        return

    @abstractmethod
    async def ainvoke(self, messages: List[Dict]|List[BaseMessage]) -> Tuple[Any | str, List[Dict], Dict]:
        """Invoke the LLM with messages.
        
        Returns:
            (response_text, tool_calls, token_usage_dict)
        """
        pass

    @staticmethod
    @abstractmethod
    def normalize_tool_calls(response: Any) -> List[Dict[str, Any]]:
        """Normalize provider tool-call shape into ``[{id, name, args}]``."""

    @staticmethod
    @abstractmethod
    def normalize_usage_metadata(response: Any) -> Dict[str, int]:
        """Normalize provider usage shape into ``{input_tokens, output_tokens, total_tokens}``."""
