"""
Session memory management for persistent exploration state.

This module handles loading, saving, and updating exploration sessions
across multiple user interactions.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import uuid

from .state import SessionMemory, ExplorationState


class SessionManager:
    """Manages persistent session memory stored in JSON files."""
    
    def __init__(self, db_dir: Optional[Path] = None):
        """
        Initialize session manager.
        
        Args:
            db_dir: Directory to store session files (defaults to src/agent/.memory)
        """
        if db_dir is None:
            db_dir = Path(__file__).parent / ".memory"
        self.db_dir = Path(db_dir)
        self.sessions_dir = self.db_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(
        self, 
        codebase_path: str
    ) -> SessionMemory:
        """
        Create a new exploration session.
        
        Args:
            codebase_path: Path to the codebase to explore
            
        Returns:
            New SessionMemory object
        """
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()
        
        session: SessionMemory = {
            "session_id": session_id,
            "created_at": now,
            "last_active": now,
            "codebase_path": str(Path(codebase_path).resolve()),
            
            "visited_entities": {
                "files": [],
                "functions": [],
                "classes": [],
                "modules": []
            },
            "entity_cache": {},
            "conversation_history": [],
            
            # Token tracking
            "total_workflow_tokens": 0,
            "workflow_invocation_history": []
        }
        
        self._save_session(session)
        return session
    
    def load_session(self, session_id: str) -> Optional[SessionMemory]:
        """
        Load an existing session from disk.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            SessionMemory object or None if not found
        """
        session_file = self.sessions_dir / session_id / "exploration_state.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file, 'r') as f:
            session = json.load(f)
        
        # Update last active time
        session["last_active"] = datetime.now().isoformat()
        self._save_session(session)
        
        return session
    
    def load_or_create_session(
        self, 
        codebase_path: str, 
        session_id: Optional[str] = None
    ) -> SessionMemory:
        """
        Load an existing session or create a new one.
        
        Args:
            codebase_path: Path to the codebase
            session_id: Optional session ID to load
            
        Returns:
            SessionMemory object
        """
        if session_id:
            session = self.load_session(session_id)
            if session:
                return session
        
        # Create new session
        return self.create_session(codebase_path)
    
    def persist_session(
        self, 
        state: ExplorationState
    ) -> None:
        """
        Save the current exploration state to the session.
        
        Args:
            state: Current ExplorationState
        """
        session = state.get("session_memory")
        if not session:
            return
        
        # Update conversation history
        if state.get("user_question") and state.get("final_answer"):
            session["conversation_history"].append({
                "role": "user",
                "content": state["user_question"],
                "timestamp": datetime.now().isoformat(),
                "citations": None
            })
            session["conversation_history"].append({
                "role": "assistant",
                "content": state["final_answer"],
                "timestamp": datetime.now().isoformat(),
                "citations": state.get("citations", [])
            })
        
        # Update visited entities from findings
        for finding in state.get("findings", []):
            entity_type = finding.get("type", "")
            entity_name = finding.get("name", "")
            
            # Map entity types to visited_entities keys
            type_map = {
                "function": "functions",
                "class": "classes",
                "module": "modules",
                "file": "files"
            }
            
            key = type_map.get(entity_type)
            if key and entity_name:
                if entity_name not in session["visited_entities"].get(key, []):
                    if key not in session["visited_entities"]:
                        session["visited_entities"][key] = []
                    session["visited_entities"][key].append(entity_name)
            
            # Cache entity details
            cache_key = f"{entity_type}:{entity_name}"
            if cache_key not in session["entity_cache"]:
                session["entity_cache"][cache_key] = {
                    "file": finding.get("file"),
                    "line_range": finding.get("line_range"),
                    "signature": finding.get("signature"),
                    "docstring": finding.get("docstring")
                }
        
        # Update timestamp
        session["last_active"] = datetime.now().isoformat()
        
        # Save to disk
        self._save_session(session)
    
    def _save_session(self, session: SessionMemory) -> None:
        """Save session to disk."""
        session_dir = self.sessions_dir / session["session_id"]
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = session_dir / "exploration_state.json"
        with open(session_file, 'w') as f:
            json.dump(session, f, indent=2)
    
    def list_sessions(self) -> list[Dict]:
        """
        List all available sessions.
        
        Returns:
            List of session metadata dicts
        """
        sessions = []
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                session_file = session_dir / "exploration_state.json"
                if session_file.exists():
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                        sessions.append({
                            "session_id": session_data["session_id"],
                            "codebase_path": session_data["codebase_path"],
                            "created_at": session_data["created_at"],
                            "last_active": session_data["last_active"],
                            "conversation_count": len(session_data.get("conversation_history", [])) // 2
                        })
        
        # Sort by last active
        sessions.sort(key=lambda x: x["last_active"], reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its data.
        
        Args:
            session_id: ID of session to delete
            
        Returns:
            True if deleted, False if not found
        """
        session_dir = self.sessions_dir / session_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
            return True
        return False
    
    # ================================================================
    # EXPLORATION NOTES: Simple, LLM-Driven Path Notes
    # The LLM writes free-form notes about paths it has explored.
    # These are stored per-session and injected into system prompts.
    # ================================================================
    
    def _notes_file(self, session_id: str) -> Path:
        """Get path to the notes file for a session."""
        return self.sessions_dir / session_id / "exploration_notes.json"
    
    def load_notes(self, session_id: str) -> Dict[str, str]:
        """
        Load exploration notes for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict mapping path -> note
        """
        notes_file = self._notes_file(session_id)
        if notes_file.exists():
            try:
                with open(notes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("notes", {})
            except Exception:
                return {}
        return {}
    
    def save_notes(self, session_id: str, notes: Dict[str, str]) -> None:
        """
        Save exploration notes for a session.
        
        Args:
            session_id: Session ID
            notes: Dict mapping path -> note
        """
        notes_file = self._notes_file(session_id)
        notes_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "last_updated": datetime.now().isoformat(),
            "session_id": session_id,
            "notes": notes
        }
        
        with open(notes_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def add_note(self, session_id: str, path: str, observation: str) -> None:
        """
        Add or update a note about a path.
        
        The LLM calls this to record observations.
        The text is entirely LLM-determined.
        
        Args:
            session_id: Session ID
            path: File or directory path
            observation: What the LLM observed (free-form)
        """
        notes = self.load_notes(session_id)
        notes[path] = observation
        self.save_notes(session_id, notes)
    
    def get_note(self, session_id: str, path: str) -> Optional[str]:
        """Get note for a path, if any."""
        notes = self.load_notes(session_id)
        return notes.get(path)
    
    def remove_note(self, session_id: str, path: str) -> bool:
        """Remove a note (to re-explore a path)."""
        notes = self.load_notes(session_id)
        if path in notes:
            del notes[path]
            self.save_notes(session_id, notes)
            return True
        return False
    
    def notes_for_prompt(self, session_id: str) -> str:
        """
        Get notes formatted for system prompt injection.
        
        Returns a token-efficient string the LLM can scan
        to know what's already been explored.
        
        Example output:
        ```
        EXPLORED:
        src/utils/ : utility funcs, not relevant
        src/auth/login.py : handles login, uses JWT
        ```
        """
        notes = self.load_notes(session_id)
        if not notes:
            return ""
        
        lines = ["EXPLORED:"]
        for path, note in notes.items():
            lines.append(f"{path} : {note}")
        
        return "\n".join(lines)
    
    def clear_notes(self, session_id: str) -> None:
        """Clear all notes for a session."""
        notes_file = self._notes_file(session_id)
        if notes_file.exists():
            notes_file.unlink()
    
    def notes_stats(self, session_id: str) -> Dict:
        """Get notes statistics."""
        notes = self.load_notes(session_id)
        total_chars = sum(len(n) for n in notes.values())
        return {
            "entries": len(notes),
            "total_chars": total_chars,
            "approx_tokens": total_chars // 4
        }


def initialize_state(
    user_question: str,
    session_memory: SessionMemory,
    token_budget: int = 50000
) -> ExplorationState:
    """
    Initialize a new ExplorationState from session memory and user question.
    
    Args:
        user_question: User's question
        session_memory: Loaded session memory
        token_budget: Maximum tokens for this turn
        
    Returns:
        Initialized ExplorationState
    """
    # Collect all explored entities from session
    explored_entities = []
    for entity_type, entities in session_memory.get("visited_entities", {}).items():
        for entity in entities:
            explored_entities.append(f"{entity_type}:{entity}")
    
    state: ExplorationState = {
        # Input
        "user_question": user_question,
        "conversation_history": session_memory.get("conversation_history", []),
        
        # Exploration tracking
        "explored_entities": explored_entities,
        "findings": [],
        "tool_calls": [],
        
        # Agent reasoning
        "current_thought": "",
        "should_respond": False,
        
        # Output
        "final_answer": "",
        "citations": [],
        
        # Session context
        "session_id": session_memory["session_id"],
        "codebase_path": session_memory["codebase_path"],
        "session_memory": session_memory,
        
        # Token management
        "tokens_used": 0,
        "token_budget": token_budget
    }
    
    return state
