# src/agent/tools/agent_tools.py
"""
Agent management tools — exposed to the orchestrator LLM.
"""

from typing import List, Optional

from ._tool_decorator import conditional_tool


# These will be set at init time by CodebaseAgent
_agent_builder = None


def set_agent_builder(builder) -> None:
    global _agent_builder
    _agent_builder = builder


def get_agent_builder():
    return _agent_builder


@conditional_tool(True, parse_docstring=True)
async def build_agents_for(folder_paths: list) -> str:
    """
    Analyze one or more folders and potentially create folder-agent specialists to manage them.

    Pass multiple related sibling folders together so the AgentBuilder can group them
    into shared agents rather than evaluating each in isolation. For example, pass all
    small feature folders at once so they can be combined into one agent if appropriate.
    New agents become available through consult_agents.

    Args:
        folder_paths: List of relative folder paths to analyze (e.g. ['src/api', 'src/routes']).
                      Can also be a single-element list for one folder.
    """
    builder = get_agent_builder()
    if not builder:
        return "Error: AgentBuilder not initialized"

    from .context import get_session_context
    session_manager, session_id = get_session_context()
    codebase_context = ""
    if session_manager and session_id:
        codebase_context = session_manager.notes_for_prompt(session_id)

    return await builder.build_agents_for(folder_paths, codebase_context=codebase_context, session_id=session_id or "")


@conditional_tool(True, parse_docstring=True)
async def consult_agents(
    question: str,
    agent_ids: Optional[List[str]] = None,
    paths: Optional[List[str]] = None,
) -> str:
    """
    Consult one or more existing folder-agent specialists.

    Use the injected AVAILABLE FOLDER AGENTS context to choose agent_ids directly.
    If you know relevant paths but not the exact agent IDs, pass paths and the
    system will route to agents whose owned folders overlap those paths.

    Args:
        question: The specific question to ask the selected folder agents.
        agent_ids: Optional list of folder-agent IDs to consult.
        paths: Optional list of repository paths used to select matching agents.
    """
    builder = get_agent_builder()
    if not builder:
        return "Error: AgentBuilder not initialized"

    from .context import get_session_context
    _, session_id = get_session_context()

    return await builder.consult_agents(
        question=question,
        agent_ids=agent_ids or [],
        paths=paths or [],
        session_id=session_id or "",
    )
