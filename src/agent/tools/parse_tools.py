"""
Code parsing tools.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

from ._tool_decorator import conditional_tool
from .context import get_codebase_path


FILE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
}

PARSER_MODULES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java": "tree_sitter_java",
    "go": "tree_sitter_go",
}

_parsers: dict[str, Any] = {}


def _get_parser(language: str) -> Any | None:
    if language in _parsers:
        return _parsers[language]
    module_name = PARSER_MODULES.get(language)
    if not module_name:
        return None
    try:
        import tree_sitter

        lang_module = importlib.import_module(module_name)
        lang_obj = tree_sitter.Language(lang_module.language())
        parser = tree_sitter.Parser(lang_obj)
        _parsers[language] = parser
        return parser
    except ImportError:
        return None


def _detect_language(file_path: str) -> str | None:
    return FILE_EXTENSIONS.get(Path(file_path).suffix.lower())


@conditional_tool(True, parse_docstring=True)
def parse_file(file_path: str) -> str:
    """
    Parse a file and extract its structure.

    Args:
        file_path: Path to the file relative to codebase root
    """
    codebase = get_codebase_path()
    if not codebase:
        return "Error: Codebase path not set"

    full_path = codebase / file_path
    if not full_path.exists():
        return f"Error: File not found: {file_path}"

    try:
        content = full_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

    language = _detect_language(file_path)
    if not language:
        return f"Error: Unknown language for {file_path}"

    parser = _get_parser(language)
    if parser:
        return _parse_with_treesitter(parser, content, file_path)
    return _parse_with_regex(content, file_path, language)


def _parse_with_regex(content: str, file_path: str, language: str) -> str:
    lines = content.split("\n")
    functions: list[str] = []
    classes: list[str] = []
    imports: list[str] = []

    patterns = {
        "python": {
            "func": r"^def\s+(\w+)",
            "class": r"^class\s+(\w+)",
            "import": r"^(?:import|from)\s+",
        },
        "javascript": {
            "func": r"^(?:function|const|let|var)\s+(\w+)",
            "class": r"^class\s+(\w+)",
            "import": r"^import\s+",
        },
    }
    selected = patterns.get(language, patterns["python"])

    for index, line in enumerate(lines, 1):
        stripped = line.strip()
        if match := re.match(selected["func"], stripped):
            functions.append(f"  {match.group(1)} (line {index})")
        if match := re.match(selected["class"], stripped):
            classes.append(f"  {match.group(1)} (line {index})")
        if re.match(selected["import"], stripped):
            imports.append(f"  {stripped[:60]}")

    result = [f"File: {file_path}", f"Language: {language}", ""]
    if functions:
        result.extend(["Functions:", *functions[:20]])
    if classes:
        result.extend(["\nClasses:", *classes[:20]])
    if imports:
        result.extend(["\nImports:", *imports[:10]])
    return "\n".join(result)


def _parse_with_treesitter(parser: Any, content: str, file_path: str) -> str:
    tree = parser.parse(bytes(content, "utf8"))

    def get_text(node: Any) -> str:
        return content[node.start_byte : node.end_byte] if node else ""

    functions: list[str] = []
    classes: list[str] = []
    imports: list[str] = []

    def traverse(node: Any) -> None:
        if node.type == "function_definition":
            name = node.child_by_field_name("name")
            if name:
                functions.append(f"  {get_text(name)} (line {node.start_point[0] + 1})")
        elif node.type == "class_definition":
            name = node.child_by_field_name("name")
            if name:
                classes.append(f"  {get_text(name)} (line {node.start_point[0] + 1})")
        elif node.type in {"import_statement", "import_from_statement"}:
            imports.append(f"  {get_text(node)[:60]}")
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)

    result = [f"File: {file_path}", ""]
    if functions:
        result.extend(["Functions:", *functions[:20]])
    if classes:
        result.extend(["\nClasses:", *classes[:20]])
    if imports:
        result.extend(["\nImports:", *imports[:10]])
    return "\n".join(result)
