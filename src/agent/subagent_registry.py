# src/agent/subagent_registry.py
"""
SubAgent Registry — persistent storage for subagent definitions.

Stores agent metadata in a JSON file under the session directory.
Each agent entry tracks: name, folders, overall_context, job_description,
creation time, and notes file location.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SubAgentRegistry:
    """
    Manages subagent definitions on disk.

    File: {session_dir}/subagent_registry.json
    """

    def __init__(self, session_dir: str, codebase_path: str):
        self._session_dir = Path(session_dir)
        self._file = self._session_dir / "subagent_registry.json"
        self._codebase_path = codebase_path
        self._data = self._load()

    def _load(self) -> Dict:
        if self._file.exists():
            with open(self._file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"codebase_path": self._codebase_path, "agents": {}}

    def _save(self) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def add_agent(
        self,
        name: str,
        folders: List[str],
        overall_context: str,
        job_description: str,
        session_id: Optional[str] = None,
    ) -> Dict:
        entry = {
            "name": name,
            "folders": folders,
            "overall_context": overall_context,
            "job_description": job_description,
            "created_at": datetime.now().isoformat(),
            "notes_file": f"subagents/{name}/notes.json",
            "session_id": session_id or "",
        }
        self._data["agents"][name] = entry
        self._save()
        return entry

    def update_session_id(self, name: str, session_id: str) -> None:
        if name in self._data["agents"]:
            self._data["agents"][name]["session_id"] = session_id
            self._save()

    def get_agent(self, name: str) -> Optional[Dict]:
        return self._data["agents"].get(name)

    def list_agents(self) -> List[Dict]:
        return list(self._data["agents"].values())

    def remove_agent(self, name: str) -> bool:
        if name in self._data["agents"]:
            del self._data["agents"][name]
            self._save()
            return True
        return False

    def update_job_description(self, name: str, job_description: str) -> None:
        if name in self._data["agents"]:
            self._data["agents"][name]["job_description"] = job_description
            self._save()

    def summary_for_prompt(self) -> str:
        agents = self.list_agents()
        if not agents:
            return ""
        lines = [
            "AVAILABLE FOLDER AGENTS:",
            "You have folder-specialist coworkers you can consult through consult_agents.",
            "",
        ]
        for agent in agents:
            name = agent["name"]
            folders = ", ".join(agent.get("folders", []))
            desc = agent.get("job_description", "(no description yet)")
            lines.append(f"- {name} | paths: {folders} | use for: {desc}")
        lines.extend([
            "",
            "GUIDELINES:",
            "- You can answer directly using your own tools when that's faster",
            "- Consult folder agents with consult_agents when the question falls clearly within their folders and their scoped context is likely to help",
            "- For cross-cutting questions, pass multiple agent_ids to consult_agents and synthesize their answers",
            "- You can also build new subagents for uncovered areas using build_agents_for(folder)",
        ])
        return "\n".join(lines)
