"""
Tool modules organized by responsibility.
"""

from .file_tools import read_file, write_file, update_file, list_files
from .search_tools import search_codebase
from .parse_tools import parse_file
from .notes_tools import update_note, delete_note, clear_all_notes
from .repl_tool import python_repl, clear_repl_session
from .agent_tools import build_agents_for, consult_agents

# Context setters
from .context import set_codebase_path, set_session_context, set_llm_provider, get_llm_provider, reset_repl_llm_token_usage, get_repl_llm_token_usage, set_tools, get_tools


def get_core_tools() -> list:
    """Minimal tools for token-constrained situations."""
    return [read_file, write_file, list_files, search_codebase]


def get_analysis_tools() -> list:
    """Repository tools without agent spawning."""
    return [
        read_file, write_file, update_file, list_files, search_codebase,
        parse_file,
    ]


def get_repl_only_tools() -> list:
    """Only the python_repl — for evals that must not touch the codebase."""
    return [python_repl]


def get_all_tools() -> list:
    """Full capability - notes management always available."""
    return [
        read_file, write_file, update_file, list_files, search_codebase,
        parse_file,
        build_agents_for, consult_agents,
        python_repl,
    ]


__all__ = [
    "read_file", "write_file", "update_file", "list_files",
    "search_codebase", "parse_file",
    "update_note", "delete_note", "clear_all_notes",
    "build_agents_for", "consult_agents",
    "python_repl", "clear_repl_session",
    "set_codebase_path", "set_session_context",
    "set_llm_provider", "get_llm_provider",
    "reset_repl_llm_token_usage", "get_repl_llm_token_usage",
    "set_tools", "get_tools",
    "get_core_tools", "get_analysis_tools", "get_repl_only_tools", "get_all_tools",
]
