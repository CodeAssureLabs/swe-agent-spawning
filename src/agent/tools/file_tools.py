"""
File reading tools.
"""

import os
from pathlib import Path
from typing import Optional
from ._tool_decorator import conditional_tool
from .context import get_codebase_path, get_session_context

_READ_FILE_MAX_LINES = int(os.getenv("READ_FILE_MAX_LINES", "300"))
_READ_FILE_MAX_CHARS = int(os.getenv("READ_FILE_MAX_CHARS", "80000"))  # ~20k tokens
_PREVIEW_LINES = 50

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".cs", ".scala", ".sh",
    ".bash", ".zsh", ".sql", ".r", ".lua", ".pl", ".ex", ".exs", ".clj",
    ".hs", ".ml", ".vim", ".el",
}


def _is_code_file(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in _CODE_EXTENSIONS


@conditional_tool(True, parse_docstring=True)
def read_file(file_path: str, offset: int = 0, limit: int = 0) -> str:
    """
    Read a source file and return its content with line numbers.

    For large files, pass offset and limit to read a specific range instead
    of the full file.

    Args:
        file_path: Path to the file, relative to codebase root
        offset: 1-indexed line number to start from (0 = beginning of file)
        limit: Number of lines to return (0 = all remaining lines from offset)
    """
    codebase = get_codebase_path()
    if not codebase:
        return "Error: Codebase path not set"

    full_path = codebase / file_path

    if not full_path.exists():
        return f"Error: File not found: {file_path}"

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        line_count = len(lines)

        # If offset/limit specified, return that range directly
        if offset > 0 or limit > 0:
            start = max(0, offset - 1)  # convert 1-indexed to 0-indexed
            end = start + limit if limit > 0 else line_count
            sliced = lines[start:end]
            numbered = [f"{start + i + 1:4d} | {line}" for i, line in enumerate(sliced)]
            header = f"[lines {start + 1}-{min(start + len(sliced), line_count)} of {line_count}]"
            return f"{header}\n" + '\n'.join(numbered)

        # Full read — check thresholds (line count OR character count)
        is_large = line_count > _READ_FILE_MAX_LINES or len(content) > _READ_FILE_MAX_CHARS
        if not is_large:
            numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
            return '\n'.join(numbered)

        # File exceeds threshold — handle by type
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
                + '\n'.join(preview_lines)
            )
        else:
            # Data/text file — inject into REPL context
            from .repl_tool import inject_context_file
            _, session_id = get_session_context()
            inject_context_file(session_id or "default", file_path, content)
            return (
                f"File is large ({line_count} lines). "
                f"Full content loaded into REPL as context[\"{file_path}\"].\n"
                f"Use python_repl to explore, filter, or process it.\n\n"
                f"Example: lines = context[\"{file_path}\"].splitlines(); print(lines[:20])"
            )
    except Exception as e:
        return f"Error reading file: {e}"


@conditional_tool(True, parse_docstring=True)
def list_files(directory: str = ".", recursive: bool = False, depth: int = 1) -> str:
    """
    List files and folders in a directory, respecting .gitignore.

    Non-recursive output lists child names relative to the requested directory.
    Directories end with "/". Recursive output is a compact raw tree rooted at
    "./" so parent paths are not repeated on every line.
    
    Args:
        directory: Directory path relative to codebase root
        recursive: Whether to include nested directories
        depth: Maximum recursive depth to include when recursive=True
    """
    codebase = get_codebase_path()
    if not codebase:
        return "Error: Codebase path not set"
    
    dir_path = codebase / directory
    
    if not dir_path.exists():
        return f"Error: Directory not found: {directory}"
    if not dir_path.is_dir():
        return f"Error: Not a directory: {directory}"
    
    try:
        dir_path.resolve().relative_to(codebase.resolve())
    except ValueError:
        return f"Error: Path escapes codebase directory: {directory}"

    gitignore_patterns = _load_gitignore(codebase)
    max_depth = max(1, min(depth, 8))

    if not recursive:
        items = []
        for item in _visible_children(dir_path, codebase, gitignore_patterns):
            suffix = "/" if item.is_dir() else ""
            items.append(f"{item.name}{suffix}")
        if not items:
            return f"(empty directory: {directory})"
        return "\n".join(items)

    sections = _compact_tree_sections(dir_path, codebase, gitignore_patterns, max_depth=max_depth)
    if not sections:
        return f"(empty directory: {directory})"
    return "\n".join(sections)


@conditional_tool(True, parse_docstring=True)
def write_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """
    Write content to a file on disk. Creates parent directories if needed.
    Absolute paths and paths with '..' are rejected for security.
    
    Args:
        file_path: Path to the file, relative to codebase root
        content: Content to write to the file
        overwrite: Whether to overwrite if file already exists (default: False)
    """
    codebase = get_codebase_path()
    if not codebase:
        return "Error: Codebase path not set"
    
    # Security: prevent path traversal
    if ".." in file_path or file_path.startswith("/"):
        return f"Error: Invalid file path (no absolute paths or '..' allowed): {file_path}"
    
    full_path = codebase / file_path
    
    # Ensure the path stays within codebase
    try:
        full_path.resolve().relative_to(codebase.resolve())
    except ValueError:
        return f"Error: Path escapes codebase directory: {file_path}"
    
    # Check if file exists and prevent overwriting without explicit flag
    if full_path.exists() and not overwrite:
        return f"Error: File already exists: {file_path}. Use overwrite=True to replace it."
    
    # Create parent directories if needed
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"Error creating directories: {e}"
    
    # Write the file
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        size_kb = len(content) / 1024
        action = "Overwritten" if full_path.exists() and overwrite else "Created"
        return f"✓ {action} file: {file_path} ({size_kb:.1f} KB, {len(content.splitlines())} lines)"
    except Exception as e:
        return f"Error writing file: {e}"


@conditional_tool(True, parse_docstring=True)
def update_file(file_path: str, content: str, start_line: Optional[int] = None, end_line: Optional[int] = None, mode: str = "replace") -> str:
    """
    Update part of an existing file or append content.
    
    Args:
        file_path: Path to the file, relative to codebase root
        content: Content to insert/replace/append
        start_line: Starting line number (1-indexed) for replace mode
        end_line: Ending line number (1-indexed, inclusive) for replace mode
        mode: Operation mode - 'replace' (default), 'append', or 'prepend'
    """
    codebase = get_codebase_path()
    if not codebase:
        return "Error: Codebase path not set"
    
    full_path = codebase / file_path
    
    if not full_path.exists():
        return f"Error: File not found: {file_path}. Use write_file to create new files."
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_line_count = len(lines)
        
        if mode == "append":
            # Append to end of file
            if not content.endswith('\n'):
                content += '\n'
            lines.append(content)
            operation = "Appended content"
            
        elif mode == "prepend":
            # Insert at beginning
            if not content.endswith('\n'):
                content += '\n'
            lines.insert(0, content)
            operation = "Prepended content"
            
        elif mode == "replace":
            # Replace line range
            if start_line is None or end_line is None:
                return "Error: start_line and end_line required for replace mode"
            
            if start_line < 1 or end_line < start_line or end_line > original_line_count:
                return f"Error: Invalid line range {start_line}-{end_line} (file has {original_line_count} lines)"
            
            # Convert to 0-indexed
            start_idx = start_line - 1
            end_idx = end_line  # end_line is inclusive, so this works for slicing
            
            # Replace the range
            new_lines = content.split('\n')
            if not content.endswith('\n') and len(new_lines) > 0:
                new_lines = [line + '\n' for line in new_lines[:-1]] + [new_lines[-1]]
            else:
                new_lines = [line + '\n' for line in new_lines]
            
            lines[start_idx:end_idx] = new_lines
            operation = f"Replaced lines {start_line}-{end_line}"
            
        else:
            return f"Error: Invalid mode '{mode}'. Use 'replace', 'append', or 'prepend'."
        
        # Write back
        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        new_line_count = len(lines)
        return f"✓ {operation} in {file_path} ({original_line_count} → {new_line_count} lines)"
        
    except Exception as e:
        return f"Error updating file: {e}"


def _load_gitignore(codebase_path):
    """Load .gitignore patterns from codebase root."""
    gitignore_file = codebase_path / ".gitignore"
    if not gitignore_file.exists():
        return None
    
    try:
        import pathspec
        with open(gitignore_file, 'r') as f:
            patterns = f.read().splitlines()
        return pathspec.PathSpec.from_lines('gitwildmatch', patterns)
    except ImportError:
        # Fallback: return raw patterns for simple matching
        with open(gitignore_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception:
        return None


def _is_gitignored(path: str, patterns) -> bool:
    """Check if a path matches gitignore patterns."""
    if patterns is None:
        return False
    
    # If pathspec is available
    if hasattr(patterns, 'match_file'):
        return patterns.match_file(path)
    
    # Simple fallback matching
    from fnmatch import fnmatch
    for pattern in patterns:
        if fnmatch(path, pattern) or fnmatch(path.split('/')[-1], pattern):
            return True
    return False


_ALWAYS_IGNORE = {
    ".git", ".svn", ".hg", "__pycache__", ".pytest_cache",
    "node_modules", ".venv", "venv", ".idea", ".vscode",
    ".DS_Store", "Thumbs.db", ".memory",
}


def _visible_children(dir_path: Path, codebase: Path, gitignore_patterns) -> list[Path]:
    """Return visible child paths sorted with directories first."""
    children = []
    for item in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        name = item.name
        if name in _ALWAYS_IGNORE or name.startswith("."):
            continue
        rel_path = str(item.relative_to(codebase))
        if _is_gitignored(rel_path, gitignore_patterns):
            continue
        children.append(item)
    return children


def _compact_tree_sections(
    root: Path,
    codebase: Path,
    gitignore_patterns,
    max_depth: int,
) -> list[str]:
    """Render a compact raw tree rooted at ./ with relative child names."""
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

        for child in _visible_children(path, codebase, gitignore_patterns):
            is_dir = child.is_dir()
            lines.append(f"{indent}{child.name}{'/' if is_dir else ''}")
            if is_dir:
                visit(child, depth_remaining - 1, indent + "  ")

    visit(root, max_depth, "  ")
    return lines
