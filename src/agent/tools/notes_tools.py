"""
Notes tools - for agent to MANAGE exploration notes.

Notes are INJECTED into every agent step automatically.
"""

from ._tool_decorator import conditional_tool
from .context import get_session_context


@conditional_tool(True, parse_docstring=True)
def update_note(path: str, observation: str) -> str:
    """
    Save or replace a note about a file/directory.
    If a note already exists for this path, it will be REPLACED.
    You MUST use this tool to record any new/updated findings about files/directories you read or discover.
    
    Args:
        path: File or directory path
        observation: What you learned (replaces any existing note for this path)
    """
    session_manager, session_id = get_session_context()
    
    if not session_manager or not session_id:
        return "Error: Session not initialized"
    print(f"Updating note for {path}: {observation}")
    session_manager.add_note(session_id, path, observation)
    return f"✓ Note saved for {path}"


@conditional_tool(True, parse_docstring=True)
def delete_note(path: str) -> str:
    """
    Delete a note for a path (if you want to forget it or re-explore fresh).
    
    Args:
        path: File or directory path to delete note for
    """
    session_manager, session_id = get_session_context()
    
    if not session_manager or not session_id:
        return "Error: Session not initialized"
    
    if session_manager.remove_note(session_id, path):
        return f"✓ Note deleted for {path}"
    else:
        return f"No note found for {path}"


@conditional_tool(True, parse_docstring=True)
def clear_all_notes() -> str:
    """
    Clear ALL notes to start fresh exploration.
    Use this if your notes are outdated or you want to re-explore everything.
    """
    session_manager, session_id = get_session_context()
    
    if not session_manager or not session_id:
        return "Error: Session not initialized"
    
    session_manager.clear_notes(session_id)
    return "✓ All notes cleared"
