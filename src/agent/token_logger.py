"""
Filesystem-based token usage logging for LLM calls.

Tracks token consumption across all LLM invocations (orchestrator, agent builder,
subagents) by appending JSONL entries to namespace-specific files within the
session's token_logs directory.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any


__all__ = ["log_token_usage", "read_and_purge", "purge_stale_logs"]


def _token_log_dir(session_id: str) -> Path:
    """
    Resolve the token logs directory for a session.

    Args:
        session_id: The session ID

    Returns:
        Path to the token_logs directory within the session
    """
    return Path(__file__).parent / ".memory" / "sessions" / session_id / "token_logs"


def _sanitise_namespace(namespace: str) -> None:
    """
    Validate that namespace contains only safe characters for use as a filename.

    Args:
        namespace: The namespace to validate

    Raises:
        ValueError: If namespace contains characters outside [A-Za-z0-9_-]
    """
    if not re.fullmatch(r'[A-Za-z0-9_-]+', namespace):
        raise ValueError(f"Invalid namespace '{namespace}': must contain only [A-Za-z0-9_-] characters")


def log_token_usage(
    session_id: str,
    namespace: str,
    source: str,
    usage: Dict[str, Any]
) -> None:
    """
    Log token usage for an LLM call to a namespace-specific JSONL file.

    Appends a single JSON line to token_logs/{namespace}.jsonl within the session.
    Creates the token_logs directory on first use.

    Args:
        session_id: The session ID
        namespace: The namespace/component making the call (e.g., "orchestrator", "agent_builder", "subagent")
        source: The source identifier for the LLM call (e.g., "route_query", "create_subagent")
        usage: Dict with token counts; keys default to 0 if missing:
               - input_tokens
               - output_tokens
               - total_tokens
               - tool_output_contributed_input_tokens
    """
    _sanitise_namespace(namespace)

    log_dir = _token_log_dir(session_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{namespace}.jsonl"

    entry = {
        "namespace": namespace,
        "source": source,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "tool_output_contributed_input_tokens": usage.get("tool_output_contributed_input_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
    }

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def read_and_purge(session_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read all token logs for a session and delete the log files.

    Aggregates all JSONL entries from token_logs/ into a dict keyed by namespace.
    Deletes the log files after reading. Skips corrupt JSON lines.

    Args:
        session_id: The session ID

    Returns:
        Dict mapping namespace -> list of log entries. Empty dict if token_logs dir
        doesn't exist or contains no files.
    """
    log_dir = _token_log_dir(session_id)

    if not log_dir.exists():
        return {}

    result: Dict[str, List[Dict[str, Any]]] = {}

    for log_file in log_dir.glob("*.jsonl"):
        namespace = log_file.stem
        entries = []

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip corrupt JSON lines so a bad entry doesn't abort collection
                        continue

        # Skip empty files from result
        if entries:
            result[namespace] = entries

        log_file.unlink()

    return result


def purge_stale_logs(session_id: str) -> None:
    """
    Purge any existing log files for a session at the start of a run.

    Deletes all JSONL files in token_logs/ for the given session_id.
    Called at the start of ask_stream to prevent leftover files from
    previous crashed runs from being picked up and inflating counts.

    Args:
        session_id: The session ID
    """
    log_dir = _token_log_dir(session_id)
    if not log_dir.exists():
        return
    for log_file in log_dir.glob("*.jsonl"):
        log_file.unlink()
