# src/agent/agent_builder.py
"""
AgentBuilder — tool-invoked LLM workflow for creating subagents.

This module contains:
1. Folder scanning utilities
2. Validation rules (pure Python, no LLM)
3. The full build_agents_for workflow (LLM + validation + registration)
"""

import os
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Optional


# ================================================================
# FOLDER SCANNING
# ================================================================

def scan_folder_tree(codebase_path: str, folder: str) -> Dict:
    """
    Scan a folder and return its structure with file counts.

    Returns:
        {
            "root": folder,
            "total_files": int,
            "children": {
                "subfolder_name": {"file_count": int, "has_subfolders": bool},
                ...
            }
        }
    """
    root = Path(codebase_path) / folder
    if not root.exists():
        return {"root": folder, "total_files": 0, "children": {}}

    total_files = 0
    children = {}

    for item in sorted(root.iterdir()):
        if item.name.startswith(".") or item.name == "__pycache__" or item.name == "node_modules":
            continue
        if item.is_dir():
            file_count = _count_files_recursive(item)
            has_subfolders = any(c.is_dir() for c in item.iterdir() if not c.name.startswith("."))
            children[item.name] = {
                "file_count": file_count,
                "has_subfolders": has_subfolders,
            }
            total_files += file_count
        elif item.is_file():
            total_files += 1

    return {"root": folder, "total_files": total_files, "children": children}


def _count_files_recursive(path: Path) -> int:
    """Count all files recursively, excluding hidden and __pycache__."""
    count = 0
    for item in path.rglob("*"):
        if any(part.startswith(".") or part == "__pycache__" for part in item.parts):
            continue
        if item.is_file():
            count += 1
    return count


# ================================================================
# VALIDATION RULES
# ================================================================

MIN_FILES_THRESHOLD = 5
SOFT_FILE_LIMIT = 80
SOFT_BREADTH_RATIO = 0.4


def _direct_entries(folder_path: Path) -> tuple[List[Path], List[Path]]:
    direct_files = [f for f in folder_path.iterdir() if f.is_file() and not f.name.startswith(".")]
    direct_subdirs = [
        d
        for d in folder_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
    ]
    return direct_files, direct_subdirs


def _format_warning(code: str, detail: str) -> str:
    return f"{code}: {detail}"


def _error(reason: str, warnings: List[str], total_files: int = 0) -> Dict:
    return {"valid": False, "reason": reason, "warnings": warnings, "total_files": total_files}


def _scan_root_for_proposal(folders: List[str], scan_roots: List[str]) -> Optional[str]:
    matching_roots = []
    for root in scan_roots:
        root_prefix = root.rstrip("/") + "/"
        if all(folder == root or folder.startswith(root_prefix) for folder in folders):
            matching_roots.append(root)
    if not matching_roots:
        return None
    return max(matching_roots, key=len)


def assess_scope_warnings(
    folders: List[str],
    codebase_path: str,
    total_files: int,
    existing_agents: Dict,
    scan_root: Optional[str] = None,
) -> List[str]:
    """
    Return non-blocking scope quality warnings.

    These are Parnas-style locality/cohesion hints, not validity rules. A broad
    folder can be a good department if it is cohesive; warnings should guide
    later splitting rather than reject creation.
    """
    warnings: List[str] = []
    base = Path(codebase_path)

    if total_files < MIN_FILES_THRESHOLD:
        warnings.append(
            _format_warning(
                "small_scope",
                f"owns {total_files} file(s); may be too small to justify a persistent folder agent",
            )
        )

    for folder in folders:
        folder_path = base / folder
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        direct_files, direct_subdirs = _direct_entries(folder_path)
        if len(direct_files) == 0 and len(direct_subdirs) == 1:
            warnings.append(
                _format_warning(
                    "shallow_scope",
                    f"{folder} is a pass-through wrapper with one child; consider assigning the child folder",
                )
            )

    if total_files > SOFT_FILE_LIMIT:
        warnings.append(
            _format_warning(
                "large_scope",
                f"owns {total_files} files; monitor whether this remains locally understandable",
            )
        )

    if scan_root:
        scan_tree = scan_folder_tree(codebase_path, scan_root)
        scan_total = scan_tree["total_files"]
        if scan_total > 0 and (total_files / scan_total) > SOFT_BREADTH_RATIO:
            pct = int((total_files / scan_total) * 100)
            warnings.append(
                _format_warning(
                    "large_relative_scope",
                    f"owns {pct}% of scanned files; this is acceptable if the scope is cohesive",
                )
            )

    proposed_set = set(folders)
    for existing in existing_agents.values():
        existing_set = set(existing.get("folders", []))
        if proposed_set & existing_set and proposed_set != existing_set and not proposed_set.issubset(existing_set):
            warnings.append(
                _format_warning(
                    "overlap_scope",
                    f"shares folders with existing agent '{existing['name']}'; verify shared ownership is intentional",
                )
            )

    top_level_roots = {Path(folder).parts[0] for folder in folders if Path(folder).parts}
    if len(top_level_roots) > 1:
        warnings.append(
            _format_warning(
                "multi_root_scope",
                f"spans top-level roots {sorted(top_level_roots)}; verify the grouping reflects one responsibility",
            )
        )

    return warnings


def validate_proposal(
    proposal: Dict,
    codebase_path: str,
    existing_agents: Dict,
    scan_root: Optional[str] = None,
) -> Dict:
    """
    Validate a single agent proposal against hard ownership rules.

    Size and shape concerns are returned as warnings. They should not block
    creation because folder-agent quality is about cohesive local knowledge,
    not a fixed percentage of a parent directory.

    Args:
        proposal: {"name": str, "folders": List[str]}
        codebase_path: Absolute path to the target codebase
        existing_agents: Current registry agents dict (name -> agent_def)
        scan_root: The folder that was scanned, used only for advisory warnings.

    Returns:
        {"valid": bool, "reason": str, "warnings": List[str], "total_files": int}
    """
    folders = proposal.get("folders", [])
    name = proposal.get("name", "")
    base = Path(codebase_path)
    warnings: List[str] = []

    if not name or not re.match(r"^[a-z][a-z0-9_]*$", name):
        return _error(f"Invalid agent name: {name!r}. Use snake_case starting with a letter.", warnings)

    if name in existing_agents:
        return _error(f"Duplicate agent name: '{name}' already exists", warnings)

    if not folders:
        return _error("No folders proposed", warnings)
    if not isinstance(folders, list) or not all(isinstance(folder, str) for folder in folders):
        return _error("Folders must be a list of relative directory paths", warnings)

    # Hard rule 1: All proposed folders must exist and be directories.
    for folder in folders:
        folder_path = base / folder
        if not folder_path.exists():
            return _error(f"Folder does not exist: {folder}", warnings)
        if not folder_path.is_dir():
            return _error(f"Path is not a directory: {folder}", warnings)

    total_files = 0
    for folder in folders:
        folder_path = base / folder
        if folder_path.exists():
            total_files += _count_files_recursive(folder_path)

    # Hard rule 2: Duplicate ownership is not useful.
    proposed_set = set(folders)
    for existing in existing_agents.values():
        existing_set = set(existing.get("folders", []))
        if proposed_set == existing_set:
            return _error(
                f"Duplicate: existing agent '{existing['name']}' already covers {folders}",
                warnings,
                total_files=total_files,
            )

    # Hard rule 3: A strict subset is already covered by an existing agent.
    for existing in existing_agents.values():
        existing_set = set(existing.get("folders", []))
        if proposed_set and proposed_set.issubset(existing_set):
            return _error(
                f"Subset: proposed folders are entirely within existing agent '{existing['name']}' scope {list(existing_set)}",
                warnings,
                total_files=total_files,
            )

    warnings = assess_scope_warnings(folders, codebase_path, total_files, existing_agents, scan_root=scan_root)
    return {"valid": True, "reason": "", "warnings": warnings, "total_files": total_files}


# ================================================================
# AGENT BUILDER — LLM WORKFLOW
# ================================================================

import json
from langchain_core.messages import SystemMessage, HumanMessage

from .providers.base import BaseLLMProvider
from .subagent_registry import SubAgentRegistry
from .token_logger import log_token_usage


PARTITION_SYSTEM_PROMPT = """You are an expert at analyzing codebase structure and deciding how to partition it into manageable agent-owned zones.

You will receive:
1. A folder tree with file counts
2. The existing agent registry (what's already covered)
3. Product/codebase context

Your job: propose a grouping of folders into agents, or decide no agents are warranted.

RULES FOR GOOD AGENTS:
- Each agent should own a self-contained section of the codebase
- "Self-contained" means the folder has files and subfolders that all contribute to one clear purpose
- ALWAYS group related folders together into one agent — e.g. ["src/api", "src/routes"] as one agent, not two
- Small sibling folders that share a theme (e.g. calendar/, crm/, storage/ all being integrations) should be grouped into one agent with all of them in its "folders" list
- Don't create agents for trivial folders (just a couple of config files)
- Don't create agents for pass-through folders (folders with just one subfolder and no files)
- Overlapping scope with existing agents is OK if both agents genuinely need the shared folder
- Prefer fewer, broader agents over many narrow ones

CRITICAL: The "folders" field must contain ONLY paths that appear verbatim in the VALID FOLDER PATHS list.
Do NOT invent, rename, or combine folder names. Copy them exactly as shown.
One agent CAN and SHOULD reference multiple folders from the list when they are related.

Respond with EXACTLY one JSON object (no markdown fencing):
{
    "agents": [
        {
            "name": "snake_case_name",
            "folders": ["exact/path/from/the/tree", "another/exact/path"],
            "overall_context": "Product context + architectural direction + this agent's role. Written like an onboarding brief.",
            "reasoning": "Why this grouping makes sense"
        }
    ]
}

Or if no agents are warranted:
{
    "agents": [],
    "reasoning": "Why no agents make sense here"
}
"""


class AgentBuilder:
    """
    Builds subagents for a target folder.

    Invoked as a tool by CodebaseAgent. Runs:
    1. Scan folder structure
    2. LLM proposes partition
    3. Rules validate proposals
    4. Register + initialize valid agents
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        codebase_path: str,
        subagent_registry: SubAgentRegistry,
        tool_registry,
        model: str,
        provider: str = "anthropic",
    ):
        self.llm = llm
        self.codebase_path = codebase_path
        self.subagent_registry = subagent_registry
        self.tool_registry = tool_registry
        self.model = model
        self.provider = provider
        self.subagent_handles: Dict[str, object] = {}

    def register_subagent_handle(self, agent_def: Dict, handle: object) -> None:
        """Keep a subagent handle available for consult_agents without exposing it as a separate LLM tool."""
        self.subagent_handles[agent_def["name"]] = handle

    async def build_agents_for(self, folder_paths: list, codebase_context: str = "", session_id: str = "") -> str:
        """
        Full workflow: scan -> propose -> validate -> register.

        Args:
            folder_paths: One or more relative folder paths within the codebase.
                          All folders are scanned together and presented to the LLM
                          as a single partition problem, so it can group related
                          siblings into one agent.
            codebase_context: Orchestrator's current exploration notes.

        Returns:
            Summary string for the orchestrator
        """
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]

        # Step 1: Scan all requested folders
        trees = {fp: scan_folder_tree(self.codebase_path, fp) for fp in folder_paths}
        trees = {fp: t for fp, t in trees.items() if t["total_files"] > 0}
        if not trees:
            return f"No agents warranted: none of the requested folders exist or contain files."

        # Step 2: LLM proposes partition
        existing_agents = {a["name"]: a for a in self.subagent_registry.list_agents()}

        context_block = f"\nCODEBASE CONTEXT (orchestrator's exploration notes):\n{codebase_context}\n" if codebase_context else ""

        # Build tree section and valid paths across all scanned roots
        tree_sections = []
        valid_paths = []
        for fp, tree in trees.items():
            tree_sections.append(
                f"Root: {tree['root']}\n"
                f"Total files: {tree['total_files']}\n"
                f"Subfolders:\n{json.dumps(tree['children'], indent=2)}"
            )
            # Direct children are valid paths; also the root itself if it has files directly
            for name in tree["children"]:
                valid_paths.append(f"{fp}/{name}")
            valid_paths.append(fp)

        valid_paths_str = "\n".join(f"  - {p}" for p in valid_paths)
        trees_str = "\n\n".join(tree_sections)

        prompt = f"""Analyze this folder structure and propose agents:{context_block}
FOLDER TREES:
{trees_str}

VALID FOLDER PATHS (you may ONLY use these exact strings in the "folders" field):
{valid_paths_str}

EXISTING AGENTS (already covered — do not duplicate):
{json.dumps({n: a.get('folders', []) for n, a in existing_agents.items()}, indent=2) if existing_agents else '(none)'}

Propose your partition."""

        self.llm.bind_tools([])  # no tools for this call
        messages = [
            SystemMessage(content=PARTITION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response, _, usage = await self.llm.ainvoke(messages)

        if session_id:
            log_token_usage(session_id, "agent_builder", "partition_call", usage)

        # Parse LLM response
        response_text = response.content if hasattr(response, "content") else str(response)
        if isinstance(response_text, list):
            response_text = "".join(
                part["text"] for part in response_text if isinstance(part, dict) and part.get("type") == "text"
            )

        # Strip markdown code fences the model sometimes emits despite instructions
        stripped = response_text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]  # drop opening fence line
            stripped = stripped.rsplit("```", 1)[0]  # drop closing fence
            stripped = stripped.strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return f"AgentBuilder error: could not parse LLM response as JSON.\nRaw response:\n{response_text[:500]}"

        proposed = parsed.get("agents", [])
        if not proposed:
            reasoning = parsed.get("reasoning", "no reason given")
            return f"No agents warranted: {reasoning}"

        # Step 3: Validate each proposal
        created = []
        rejected = []
        warnings_by_agent = []

        for proposal in proposed:
            scan_root = _scan_root_for_proposal(proposal.get("folders", []), list(trees.keys()))
            result = validate_proposal(
                proposal, self.codebase_path, existing_agents, scan_root=scan_root
            )
            if not result["valid"]:
                rejected.append(f"  - {proposal.get('name', '?')}: {result['reason']}")
                continue
            if result.get("warnings"):
                warnings_by_agent.append(
                    f"  - {proposal.get('name', '?')}: " + "; ".join(result["warnings"])
                )

            # Step 4: Register
            agent_def = self.subagent_registry.add_agent(
                name=proposal["name"],
                folders=proposal["folders"],
                overall_context=proposal.get("overall_context", ""),
                job_description="",  # filled by self-init below
            )

            # Initialize the subagent
            await self._initialize_subagent(agent_def, session_id=session_id)
            created.append(proposal["name"])

        # Build summary
        parts = []
        if created:
            parts.append(f"Created {len(created)} agent(s): {', '.join(created)}")
        if rejected:
            parts.append(f"Rejected {len(rejected)} proposal(s):\n" + "\n".join(rejected))
        if warnings_by_agent:
            parts.append(
                "Scope warning(s) for created agents (non-blocking):\n"
                + "\n".join(warnings_by_agent)
            )
        if not created and not rejected:
            parts.append("No agents created.")

        return "\n".join(parts)

    async def consult_agents(
        self,
        question: str,
        agent_ids: Optional[List[str]] = None,
        paths: Optional[List[str]] = None,
        session_id: str = "",
    ) -> str:
        """
        Consult selected folder agents through one orchestrator-visible tool.

        Individual SubAgentHandle instances remain internal so the orchestrator
        has one stable consultation interface instead of a growing tool list.
        """
        agent_ids = agent_ids or []
        paths = paths or []
        if not question.strip():
            return "Error: consult_agents requires a non-empty question."

        selected = self._select_agents(agent_ids=agent_ids, paths=paths)
        if not selected:
            available = ", ".join(a["name"] for a in self.subagent_registry.list_agents()) or "(none)"
            return (
                "No matching folder agents found. "
                f"Available agents: {available}. "
                "Use build_agents_for(folder_paths) if the relevant area is not covered."
            )

        async def _ask(agent_def: Dict) -> str:
            name = agent_def["name"]
            handle = self.subagent_handles.get(name)
            if handle is None:
                return f"## {name}\nError: subagent handle is not loaded."
            result = await handle.ainvoke({"question": question})
            return f"## {name}\n{result}"

        answers = await asyncio.gather(*[_ask(agent_def) for agent_def in selected])
        return "\n\n".join(answers)

    def _select_agents(self, agent_ids: List[str], paths: List[str]) -> List[Dict]:
        agents = {a["name"]: a for a in self.subagent_registry.list_agents()}
        selected: Dict[str, Dict] = {}

        for agent_id in agent_ids:
            if agent_id in agents:
                selected[agent_id] = agents[agent_id]

        normalized_paths = [str(Path(path)) for path in paths if isinstance(path, str) and path.strip()]
        for path in normalized_paths:
            for agent in agents.values():
                if self._agent_owns_path(agent, path):
                    selected[agent["name"]] = agent

        return list(selected.values())

    @staticmethod
    def _agent_owns_path(agent_def: Dict, path: str) -> bool:
        normalized_path = str(Path(path))
        for folder in agent_def.get("folders", []):
            normalized_folder = str(Path(folder))
            if (
                normalized_path == normalized_folder
                or normalized_path.startswith(normalized_folder + "/")
                or normalized_folder.startswith(normalized_path.rstrip("/") + "/")
            ):
                return True
        return False

    async def _initialize_subagent(self, agent_def: Dict, session_id: str = "") -> None:
        """
        Run first-init for a subagent: explore folders, generate job_description,
        and register the SubAgentHandle in the orchestrator's ToolRegistry.
        """
        from .subagent_handle import SubAgentHandle
        from .maintainer import CodebaseAgent

        # Create a scoped CodebaseAgent for this subagent
        scoped_agent = CodebaseAgent(
            model=self.model,
            codebase_path=self.codebase_path,
            tool_set="core",
            provider=self.provider,
            scoped_folders=agent_def["folders"],
            overall_context=agent_def["overall_context"],
            agent_name=agent_def["name"],
        )

        # Run first-init exploration to generate job_description
        init_question = (
            f"You are being initialized as a subagent. "
            f"Your scope is: {agent_def['folders']}. "
            f"List all files in your folders, read the key entry points, "
            f"and write a concise job description (1-2 paragraphs) summarizing "
            f"what you manage, the key patterns you see, and the important files. "
            f"Respond with ONLY the job description text."
        )

        job_description = ""
        async for event in scoped_agent.ask_stream(init_question):
            if event.get("type") == "done":
                job_description = event.get("answer", "")

                if session_id:
                    tokens_usage = event.get("tokens_usage", {})
                    log_token_usage(
                        session_id,
                        f"subagent_{agent_def['name']}",
                        "init_call",
                        tokens_usage.get("total", tokens_usage),
                    )

                break

        # Save job_description and subagent session_id back to registry
        self.subagent_registry.update_job_description(agent_def["name"], job_description)
        self.subagent_registry.update_session_id(agent_def["name"], scoped_agent.session["session_id"])

        # Create handle and keep it internal for consult_agents.
        # Update agent_def with the generated job_description so handle gets it.
        agent_def["job_description"] = job_description
        handle = SubAgentHandle(agent_def, scoped_agent=scoped_agent, orchestrator_session_id=session_id)
        self.register_subagent_handle(agent_def, handle)
