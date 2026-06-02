"""
Runtime registry for LangChain BaseTool instances.

The agent keeps tools in a mutable registry so subagent handles can be added
after graph construction while each node still sees the current tool set.
"""

from __future__ import annotations

from typing import Any

from langchain.tools import BaseTool


class ToolRegistry:
    """Mutable name-to-tool registry used by the LangGraph nodes."""

    def __init__(self, session_dir: str | None = None):
        self._session_dir = session_dir
        self._base_tools: dict[str, BaseTool] = {}

    def register_base_tool(self, tool: Any) -> None:
        """Register a BaseTool instance, replacing any tool with the same name."""
        if isinstance(tool, BaseTool):
            self._base_tools[tool.name] = tool

    def unregister_base_tool(self, name: str) -> None:
        """Remove a registered tool by name."""
        self._base_tools.pop(name, None)

    def get_all_base_tools(self) -> list[BaseTool]:
        """Return the currently registered tools."""
        return list(self._base_tools.values())

    def get_base_tools_by_name(self) -> dict[str, BaseTool]:
        """Return the current name-to-tool lookup for tool execution."""
        return dict(self._base_tools)
