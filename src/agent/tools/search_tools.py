"""
Codebase search tools.
"""

import subprocess
from pathlib import Path
from ._tool_decorator import conditional_tool
from .context import get_codebase_path


@conditional_tool(True, parse_docstring=True)
def search_codebase(pattern: str, file_pattern: str = "**/*", is_regex: bool = False) -> str:
    """
    Search for a pattern in the codebase using ripgrep.
    
    Args:
        pattern: Text or regex pattern to search for
        file_pattern: Glob pattern for files (e.g., "**/*.py")
        is_regex: Whether to treat pattern as regex
    """
    codebase = get_codebase_path()
    if not codebase:
        return "Error: Codebase path not set"
    
    try:
        cmd = ["rg", "--line-number", "--color=never", "-m", "50"]
        
        if file_pattern != "**/*":
            cmd.extend(["--glob", file_pattern])
        
        if is_regex:
            cmd.append("-e")
        else:
            cmd.append("-F")
        
        cmd.append(pattern)
        cmd.append(str(codebase))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            output = []
            for line in lines[:50]:
                try:
                    path_part = line.split(':', 1)[0]
                    rest = line.split(':', 1)[1] if ':' in line else ""
                    rel_path = Path(path_part).relative_to(codebase)
                    output.append(f"{rel_path}:{rest}")
                except:
                    output.append(line)
            return "\n".join(output)
        else:
            return "No matches found"
    except FileNotFoundError:
        return "Error: ripgrep not installed"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"


def search_codebase_with_graph_database(pattern: str) -> str:
    """
    Search for a pattern in the codebase using the graph database.
    """
    return "Not implemented yet"
