"""
Codebase Maintainer Agent.

Fully agentic LLM-powered codebase analysis using LangGraph.

Key features:
- Custom workflow with notes injected at every step
- Session persistence with exploration memory
- Lightweight code parsing
- Agent-spawning through scoped folder agents
"""

from .maintainer import CodebaseAgent
from .state import AgentState, SessionMemory
from .memory import SessionManager
from .tool_registry import ToolRegistry
from .tools import (
    get_all_tools, 
    get_core_tools, 
    get_analysis_tools,
    set_codebase_path,
    set_session_context,
)

__all__ = [
    "CodebaseAgent",
    "AgentState",
    "SessionMemory", 
    "SessionManager",
    "ToolRegistry",
    "get_all_tools",
    "get_core_tools",
    "get_analysis_tools",
    "set_codebase_path",
    "set_session_context",
]
