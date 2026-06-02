"""
Shared context for all tools.
"""

import contextvars
from pathlib import Path
from typing import Optional, Dict

_UNSET = object()

# Global context - set once at agent init
_codebase_path: Optional[Path] = None
_session_manager = None
_session_id: Optional[str] = None
_llm_provider = None
_tools: list = []  # flat list of BaseTool instances, set at agent init

# Task-local context for concurrent invocations.
# Default is _UNSET so getters can distinguish "never set" from "set to falsy".
_codebase_path_var: contextvars.ContextVar = contextvars.ContextVar(
    "codebase_path", default=_UNSET
)
_session_manager_var: contextvars.ContextVar = contextvars.ContextVar(
    "session_manager", default=_UNSET
)
_session_id_var: contextvars.ContextVar = contextvars.ContextVar(
    "session_id", default=_UNSET
)
_llm_provider_var: contextvars.ContextVar = contextvars.ContextVar(
    "llm_provider", default=_UNSET
)
_tools_var: contextvars.ContextVar = contextvars.ContextVar("tools", default=_UNSET)

# Accumulator for token usage from inline llm() calls made inside python_repl.
# Reset at the start of each ask_stream invocation.
_repl_llm_token_usage: Dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}
_repl_llm_token_usage_var: contextvars.ContextVar = contextvars.ContextVar(
    "repl_llm_token_usage", default=_UNSET,
)


def set_codebase_path(path: str) -> None:
    """Set the codebase root path."""
    global _codebase_path
    _codebase_path = Path(path).resolve()
    _codebase_path_var.set(_codebase_path)


def get_codebase_path() -> Optional[Path]:
    """Get the codebase root path."""
    val = _codebase_path_var.get()
    return val if val is not _UNSET else _codebase_path


def set_session_context(session_manager, session_id: str) -> None:
    """Set the session context for notes."""
    global _session_manager, _session_id
    _session_manager = session_manager
    _session_id = session_id
    _session_manager_var.set(session_manager)
    _session_id_var.set(session_id)


def get_session_context():
    """Get session manager and ID."""
    sm = _session_manager_var.get()
    session_manager = sm if sm is not _UNSET else _session_manager
    sid = _session_id_var.get()
    session_id = sid if sid is not _UNSET else _session_id
    return session_manager, session_id


def set_llm_provider(provider) -> None:
    """Set the LLM provider for REPL use."""
    global _llm_provider
    _llm_provider = provider
    _llm_provider_var.set(provider)


def get_llm_provider():
    """Get the LLM provider."""
    val = _llm_provider_var.get()
    return val if val is not _UNSET else _llm_provider


def set_tools(tools: list) -> None:
    """Store the full tools list for REPL injection."""
    global _tools
    _tools = tools
    _tools_var.set(tools)


def get_tools() -> list:
    """Get the tools list."""
    val = _tools_var.get()
    return val if val is not _UNSET else _tools


def reset_repl_llm_token_usage() -> None:
    """Reset the REPL LLM token accumulator. Call at the start of each ask_stream."""
    global _repl_llm_token_usage
    _repl_llm_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    _repl_llm_token_usage_var.set({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})


def accumulate_repl_llm_tokens(usage: Dict[str, int]) -> None:
    """Add token counts from one inline llm() call into the accumulator."""
    val = _repl_llm_token_usage_var.get()
    current = dict(val) if val is not _UNSET else dict(_repl_llm_token_usage)
    current["input_tokens"] += usage.get("input_tokens", 0)
    current["output_tokens"] += usage.get("output_tokens", 0)
    current["total_tokens"] += usage.get("total_tokens", 0)
    _repl_llm_token_usage_var.set(current)

    # Keep process-global aggregate for backward compatibility in non-concurrent flows.
    _repl_llm_token_usage["input_tokens"] += usage.get("input_tokens", 0)
    _repl_llm_token_usage["output_tokens"] += usage.get("output_tokens", 0)
    _repl_llm_token_usage["total_tokens"] += usage.get("total_tokens", 0)


def get_repl_llm_token_usage() -> Dict[str, int]:
    """Return a copy of the current REPL LLM token accumulator."""
    val = _repl_llm_token_usage_var.get()
    return dict(val) if val is not _UNSET else dict(_repl_llm_token_usage)
