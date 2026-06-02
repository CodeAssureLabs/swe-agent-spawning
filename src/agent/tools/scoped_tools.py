# src/agent/tools/scoped_tools.py
"""
Factory functions that produce path-restricted versions of existing tools.

Each factory takes a codebase_path and a list of allowed folders,
and returns a LangChain BaseTool that rejects paths outside scope.
"""

import subprocess
from pathlib import Path
from typing import List

from ._tool_decorator import conditional_tool


def _is_within_scope(file_path: str, allowed_folders: List[str]) -> bool:
    """Check if a relative file_path falls within any of the allowed folders."""
    normalized = str(Path(file_path))
    for folder in allowed_folders:
        normalized_folder = str(Path(folder))
        if normalized == normalized_folder or normalized.startswith(normalized_folder + "/"):
            return True
    return False


def _scope_error(file_path: str, allowed_folders: List[str]) -> str:
    return f"Path {file_path} is outside your scope: {allowed_folders}"


def make_scoped_read_file(codebase_path: str, allowed_folders: List[str]):
    """Create a read_file tool restricted to allowed_folders."""
    codebase = Path(codebase_path).resolve()
    from .file_tools import _READ_FILE_MAX_LINES, _READ_FILE_MAX_CHARS, _PREVIEW_LINES, _is_code_file

    @conditional_tool(True, parse_docstring=True)
    def read_file(file_path: str, offset: int = 0, limit: int = 0) -> str:
        """
        Read a source file and return its content with line numbers.

        For large files, pass offset and limit to read a specific range.

        Args:
            file_path: Path to the file, relative to codebase root
            offset: 1-indexed line number to start from (0 = beginning)
            limit: Number of lines to return (0 = all remaining)
        """
        if not _is_within_scope(file_path, allowed_folders):
            return _scope_error(file_path, allowed_folders)

        full_path = codebase / file_path
        if not full_path.exists():
            return f"Error: File not found: {file_path}"
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.split("\n")
            line_count = len(lines)

            if offset > 0 or limit > 0:
                start = max(0, offset - 1)
                end = start + limit if limit > 0 else line_count
                sliced = lines[start:end]
                numbered = [f"{start + i + 1:4d} | {line}" for i, line in enumerate(sliced)]
                header = f"[lines {start + 1}-{min(start + len(sliced), line_count)} of {line_count}]"
                return f"{header}\n" + "\n".join(numbered)

            is_large = line_count > _READ_FILE_MAX_LINES or len(content) > _READ_FILE_MAX_CHARS
            if not is_large:
                numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
                return "\n".join(numbered)

            if _is_code_file(file_path):
                preview_lines = []
                preview_chars = 0
                for i, line in enumerate(lines[:_PREVIEW_LINES]):
                    entry = f"{i+1:4d} | {line}"
                    if preview_chars + len(entry) > _READ_FILE_MAX_CHARS // 4:
                        preview_lines.append(f"  ... (preview truncated at {i} lines)")
                        break
                    preview_lines.append(entry)
                    preview_chars += len(entry)
                return (
                    f"File is large ({line_count} lines). To explore efficiently:\n"
                    f"- read_file(\"{file_path}\", offset=N, limit=M) to read M lines starting from line N\n"
                    f"- search_codebase(\"pattern\") to find specific code, then read surrounding lines\n"
                    f"\nFirst lines preview:\n"
                    + "\n".join(preview_lines)
                )
            else:
                from .repl_tool import inject_context_file
                from .context import get_session_context as _gsc
                _, sid = _gsc()
                inject_context_file(sid or "default", file_path, content)
                return (
                    f"File is large ({line_count} lines). "
                    f"Full content loaded into REPL as context[\"{file_path}\"].\n"
                    f"Use python_repl to explore, filter, or process it.\n\n"
                    f"Example: lines = context[\"{file_path}\"].splitlines(); print(lines[:20])"
                )
        except Exception as e:
            return f"Error reading file: {e}"

    return read_file


def make_scoped_list_files(codebase_path: str, allowed_folders: List[str]):
    """Create a list_files tool restricted to allowed_folders."""
    codebase = Path(codebase_path).resolve()

    @conditional_tool(True, parse_docstring=True)
    def list_files(directory: str = ".", recursive: bool = False, depth: int = 1) -> str:
        """
        List files and folders in a scoped directory.

        Non-recursive output lists child names relative to the requested directory.
        Directories end with "/". Recursive output is a compact raw tree rooted at
        "./" so parent paths are not repeated on every line.

        Args:
            directory: Directory path relative to codebase root
            recursive: Whether to include nested directories
            depth: Maximum recursive depth to include when recursive=True
        """
        if not _is_within_scope(directory, allowed_folders):
            return _scope_error(directory, allowed_folders)

        dir_path = codebase / directory
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {directory}"

        try:
            dir_path.resolve().relative_to(codebase)
        except ValueError:
            return f"Error: Path escapes codebase directory: {directory}"

        max_depth = max(1, min(depth, 8))

        def visible_children(path: Path) -> list[Path]:
            children = []
            for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                name = item.name
                if name.startswith(".") or name in {"__pycache__", ".DS_Store"}:
                    continue
                rel_path = str(item.relative_to(codebase))
                if _is_within_scope(rel_path, allowed_folders):
                    children.append(item)
            return children

        if not recursive:
            items = []
            for item in visible_children(dir_path):
                items.append(f"{item.name}{'/' if item.is_dir() else ''}")
            return "\n".join(items) if items else f"(empty directory: {directory})"

        lines = ["./"]
        visited: set[Path] = set()

        def visit(path: Path, depth_remaining: int, indent: str) -> None:
            if depth_remaining <= 0:
                return
            try:
                real_path = path.resolve()
            except OSError:
                return
            if real_path in visited:
                return
            visited.add(real_path)

            for item in visible_children(path):
                is_dir = item.is_dir()
                lines.append(f"{indent}{item.name}{'/' if is_dir else ''}")
                if is_dir:
                    visit(item, depth_remaining - 1, indent + "  ")

        visit(dir_path, max_depth, "  ")
        return "\n".join(lines) if len(lines) > 1 else f"(empty directory: {directory})"

    return list_files


def make_scoped_search(codebase_path: str, allowed_folders: List[str]):
    """Create a search tool restricted to allowed_folders."""
    codebase = Path(codebase_path).resolve()

    @conditional_tool(True, parse_docstring=True)
    def search_codebase(pattern: str, file_pattern: str = "**/*", is_regex: bool = False) -> str:
        """
        Search for a pattern within the scoped folders using ripgrep.

        Args:
            pattern: Text or regex pattern to search for
            file_pattern: Glob pattern for files (e.g., '**/*.py')
            is_regex: Whether to treat pattern as regex
        """
        cmd = ["rg", "--line-number", "--color=never", "-m", "50"]
        if file_pattern != "**/*":
            cmd.extend(["--glob", file_pattern])
        if is_regex:
            cmd.append("-e")
        else:
            cmd.append("-F")
        cmd.append(pattern)

        # Only search within allowed folders
        for folder in allowed_folders:
            cmd.append(str(codebase / folder))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                output = []
                for line in lines[:50]:
                    try:
                        path_part = line.split(":", 1)[0]
                        rest = line.split(":", 1)[1] if ":" in line else ""
                        rel_path = Path(path_part).relative_to(codebase)
                        output.append(f"{rel_path}:{rest}")
                    except Exception:
                        output.append(line)
                return "\n".join(output)
            else:
                return "No matches found"
        except FileNotFoundError:
            return "Error: ripgrep not installed"
        except subprocess.TimeoutExpired:
            return "Error: Search timed out"

    return search_codebase


def make_scoped_update_note(session_manager, session_id: str):
    """Create an update_note tool bound to a specific session."""

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
        session_manager.add_note(session_id, path, observation)
        return f"Note saved for {path}"

    return update_note


def make_scoped_delete_note(session_manager, session_id: str):
    """Create a delete_note tool bound to a specific session."""

    @conditional_tool(True, parse_docstring=True)
    def delete_note(path: str) -> str:
        """
        Delete a note for a path (if you want to forget it or re-explore fresh).

        Args:
            path: File or directory path to delete note for
        """
        if session_manager.remove_note(session_id, path):
            return f"Note deleted for {path}"
        else:
            return f"No note found for {path}"

    return delete_note
