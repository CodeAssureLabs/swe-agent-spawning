#!/usr/bin/env python3
"""
Long-session codebase file-localization eval harness.

Runs benchmark instances against a checked-out repository and asks the agent to
predict the repo-relative files that need modification. Predictions are compared
with each instance's gold_files using precision/recall/F1.

Usage:
    uv run evals/run_eval.py \
        evals/benchmark_ansible.json \
        --repo repos/ansible__ansible \
        --out evals/ansible_report.json
"""

from dotenv import load_dotenv; load_dotenv()

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from langchain_anthropic.chat_models import AnthropicContextOverflowError as _AnthropicOverflow
except ImportError:
    _AnthropicOverflow = None  # type: ignore

from rich import box
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.maintainer import CodebaseAgent
from agent.memory import SessionManager
from agent.token_logger import read_and_purge
from agent.tools.context import set_tools
from agent.tools.repl_tool import clear_repl_session

console = Console()


SUBAGENT_TOOL_NAMES = ("build_agents_for", "consult_agents")


def disable_subagent_tools(agent: CodebaseAgent) -> None:
    for tool_name in SUBAGENT_TOOL_NAMES:
        agent.tool_registry.unregister_base_tool(tool_name)
    set_tools(agent.tool_registry.get_all_base_tools())


def _is_context_overflow(exc: BaseException) -> bool:
    if _AnthropicOverflow is not None and isinstance(exc, _AnthropicOverflow):
        return True
    msg = str(exc).lower()
    return "prompt is too long" in msg or ("context" in msg and "overflow" in msg)


def _normalize_path(path: str, repo_path: Path | None = None) -> str:
    cleaned = path.strip().strip("'\"` ,;")
    cleaned = re.sub(r"^\w+://", "", cleaned)
    cleaned = re.sub(r"^file:", "", cleaned)
    cleaned = re.sub(r"[:#]L?\d+(?:-L?\d+)?$", "", cleaned)
    cleaned = cleaned.replace("\\", "/")

    if repo_path is not None:
        repo_abs = str(repo_path.resolve()).replace("\\", "/").rstrip("/")
        if cleaned.startswith(repo_abs + "/"):
            cleaned = cleaned[len(repo_abs) + 1:]

        repo_name = repo_path.name.rstrip("/")
        for prefix in (f"repos/{repo_name}/", f"./repos/{repo_name}/", f"{repo_name}/"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]

    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def _dedupe_paths(paths: list[str], repo_path: Path | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        path = _normalize_path(str(raw), repo_path)
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(match.group(1).strip())

    starts = [idx for idx, char in enumerate(text) if char in "[{"]
    for start in starts:
        stack: list[str] = []
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "[{":
                stack.append("]" if char == "[" else "}")
            elif char in "]}":
                if not stack or char != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    candidates.append(text[start:idx + 1])
                    break
    return candidates


def extract_predicted_files(answer: str, repo_path: Path | None = None) -> tuple[list[str], str]:
    """Return predicted file paths and the extraction mode used."""
    for candidate in _json_candidates(answer):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            for key in ("files", "predicted_files", "paths", "modified_files"):
                value = parsed.get(key)
                if isinstance(value, list):
                    paths = [str(item) for item in value if isinstance(item, str)]
                    return _dedupe_paths(paths, repo_path), "json-object"
        elif isinstance(parsed, list):
            paths = [str(item) for item in parsed if isinstance(item, str)]
            return _dedupe_paths(paths, repo_path), "json-array"

    path_re = re.compile(
        r"(?<![\w.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+"
        r"(?:[:#]L?\d+(?:-L?\d+)?)?"
    )
    return _dedupe_paths(path_re.findall(answer), repo_path), "regex"


def score_files(predicted: list[str], gold: list[str]) -> dict[str, Any]:
    pred_set = set(predicted)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": pred_set == gold_set,
        "all_gold_found": gold_set.issubset(pred_set),
        "missing_files": sorted(gold_set - pred_set),
        "extra_files": sorted(pred_set - gold_set),
    }


def _is_doc_path(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return path.startswith("docs/") or suffix in {".md", ".rst", ".txt"}


def _score_subset(predicted: list[str], gold: list[str], *, docs: bool) -> dict[str, Any]:
    return score_files(
        [path for path in predicted if _is_doc_path(path) == docs],
        [path for path in gold if _is_doc_path(path) == docs],
    )


def _metric_summary(results: list[dict]) -> dict[str, Any]:
    total = len(results)
    total_tp = sum(r["file_score"]["tp"] for r in results)
    total_fp = sum(r["file_score"]["fp"] for r in results)
    total_fn = sum(r["file_score"]["fn"] for r in results)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    def avg(key: str) -> float:
        return round(mean(r["file_score"][key] for r in results), 4) if results else 0.0

    return {
        "total": total,
        "exact_matches": sum(1 for r in results if r["file_score"]["exact_match"]),
        "exact_match_rate": round(sum(1 for r in results if r["file_score"]["exact_match"]) / total, 4) if total else 0.0,
        "all_gold_found": sum(1 for r in results if r["file_score"]["all_gold_found"]),
        "all_gold_found_rate": round(sum(1 for r in results if r["file_score"]["all_gold_found"]) / total, 4) if total else 0.0,
        "macro_precision": avg("precision"),
        "macro_recall": avg("recall"),
        "macro_f1": avg("f1"),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "avg_predicted_files": round(mean(len(r["predicted_files"]) for r in results), 2) if results else 0.0,
        "avg_gold_files": round(mean(len(r["gold_files"]) for r in results), 2) if results else 0.0,
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "total_repl_llm_tokens": sum(r.get("repl_llm_tokens", 0) for r in results),
        "total_repl_calls": sum(r["repl_calls"] for r in results),
        "subagent_consult_cases": sum(1 for r in results if r.get("subagent_consult_calls", 0) > 0),
        "total_subagent_consult_calls": sum(r.get("subagent_consult_calls", 0) for r in results),
        "total_subagent_response_chars": sum(r.get("subagent_response_chars", 0) for r in results),
        "max_context_window_used": max((r.get("context_window_used", 0) for r in results), default=0),
        "total_elapsed_s": round(sum(r["elapsed_s"] for r in results), 2),
        "total_repl_time_s": round(sum(r.get("repl_time_s", 0.0) for r in results), 2),
        "total_llm_time_s": round(sum(r.get("llm_time_s", 0.0) for r in results), 2),
    }


def _group_summaries(results: list[dict]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict]] = {}
    for result in results:
        groups.setdefault(f"difficulty:{result.get('difficulty', 'unknown')}", []).append(result)
        groups.setdefault("gold_docs:any" if result.get("gold_doc_files") else "gold_docs:none", []).append(result)
    return {name: _metric_summary(group_results) for name, group_results in sorted(groups.items())}


PROMPT_TASK = (
    "Identify the repo-relative files that would need to be modified to implement the requested fix. "
    "Include code, test, and documentation files when they would need edits. Do not edit files."
    "Use the repository tools to inspect the codebase. Prefer exact existing file paths."
)
PROMPT_SUBAGENTS = (
    "If available folder agents clearly match the relevant subsystem, consult them when their scoped context "
    "is likely to be useful."
)
PROMPT_OUTPUT = (
    "Your final answer must be valid JSON only, with this schema:\n"
    '{"files": ["path/to/file.py"], "rationale": "short reason"}'
)


def _build_prompt(instance: dict, *, subagents_enabled: bool) -> str:
    guidance = ""
    if subagents_enabled:
        guidance = f"{PROMPT_SUBAGENTS}\n\n"
    return (
        f"{PROMPT_TASK}\n\n"
        f"{guidance}"
        f"{PROMPT_OUTPUT}\n\n"
        "Problem statement:\n"
        f"{instance.get('problem_statement', '')}"
    )


async def run_instance(
    *,
    agent: CodebaseAgent,
    index: int,
    instance: dict,
    repo_path: Path,
    case_timeout: float,
    skip_notes: bool,
    subagents_enabled: bool,
) -> dict:
    gold_files = _dedupe_paths(instance.get("gold_files", []), repo_path)
    prompt = _build_prompt(instance, subagents_enabled=subagents_enabled)

    t_start = time.perf_counter()
    answer = ""
    tokens: dict = {}
    repl_calls = 0
    repl_time_s = 0.0
    context_window_used = 0
    subagent_consult_calls = 0
    subagent_consults: list[dict[str, Any]] = []
    pending_subagent_call: dict[str, Any] | None = None
    conversation_history: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    trace_content_max_chars = int(os.getenv("EVAL_TRACE_CONTENT_MAX_CHARS", "120000"))
    repl_call_start: float | None = None

    def trace_content(content: str) -> str:
        if trace_content_max_chars <= 0 or len(content) <= trace_content_max_chars:
            return content
        head = trace_content_max_chars // 2
        tail = trace_content_max_chars - head
        omitted = len(content) - trace_content_max_chars
        return content[:head] + f"\n\n[eval trace truncated: omitted {omitted} characters]\n\n" + content[-tail:]

    def append_history(role: str, content: str) -> None:
        conversation_history.append({"role": role, "content": trace_content(content)})

    try:
        async for event in agent.ask_stream(prompt, skip_notes=skip_notes):
            event_type = event.get("type")
            if event_type == "tool_call":
                if event.get("name") == "consult_agents":
                    subagent_consult_calls += 1
                    args = event.get("args", {})
                    pending_subagent_call = {
                        "args": args if isinstance(args, dict) else {},
                        "result_preview": "",
                        "result_chars": 0,
                    }
                append_history("assistant", f"tool_call: {event.get('name', '')}")
            elif event_type == "tool_done":
                result = str(event.get("result", ""))
                if event.get("name") == "consult_agents":
                    if pending_subagent_call is None:
                        pending_subagent_call = {"args": {}, "result_preview": "", "result_chars": 0}
                    pending_subagent_call["result_preview"] = result[:1000]
                    pending_subagent_call["result_chars"] = len(result)
                    subagent_consults.append(pending_subagent_call)
                    pending_subagent_call = None
                preview = result[:500] + "..." if len(result) > 500 else result
                append_history("tool", f"{event.get('name', '')}: {preview}")
            elif event_type == "repl_code":
                repl_calls += 1
                repl_call_start = time.perf_counter()
                append_history("tool", f"python_repl input:\n{event.get('content', '')}")
            elif event_type == "repl_output":
                if repl_call_start is not None:
                    repl_time_s += time.perf_counter() - repl_call_start
                    repl_call_start = None
                append_history("tool", f"python_repl output:\n{event.get('content', '')}")
            elif event_type == "tokens_usage":
                cumulative = event.get("tokens", {})
                if isinstance(cumulative, dict):
                    tokens.update(cumulative)
                context_window_used = max(context_window_used, event.get("context_window_used", 0))
            elif event_type == "done":
                answer = event.get("answer", "")
                tokens_usage = event.get("tokens_usage", {})
                if isinstance(tokens_usage, dict):
                    tokens = tokens_usage
                    context_window_used = max(context_window_used, tokens_usage.get("context_window_used", 0))
                append_history("assistant", answer)
    except asyncio.TimeoutError:
        answer = f"|ERROR| timed out after {case_timeout}s"
        append_history("assistant", answer)
    except Exception as exc:
        prefix = "|CONTEXT_OVERFLOW|" if _is_context_overflow(exc) else "|ERROR|"
        answer = f"{prefix} {exc}"
        append_history("assistant", answer)
    finally:
        current_session_id = agent.session.get("session_id")
        if isinstance(current_session_id, str):
            clear_repl_session(current_session_id)

    elapsed = time.perf_counter() - t_start
    if repl_call_start is not None:
        repl_time_s += time.perf_counter() - repl_call_start

    total_bucket = tokens.get("total", tokens)
    predicted_files, extraction_mode = extract_predicted_files(answer, repo_path)
    file_score = score_files(predicted_files, gold_files)
    doc_score = _score_subset(predicted_files, gold_files, docs=True)
    code_score = _score_subset(predicted_files, gold_files, docs=False)
    llm_time_s = max(0.0, elapsed - repl_time_s)

    return {
        "index": index,
        "instance_id": instance.get("instance_id", ""),
        "date": instance.get("date", ""),
        "difficulty": instance.get("difficulty", "unknown"),
        "num_files": instance.get("num_files", len(gold_files)),
        "problem_statement": instance.get("problem_statement", ""),
        "gold_files": gold_files,
        "gold_doc_files": [path for path in gold_files if _is_doc_path(path)],
        "predicted_files": predicted_files,
        "prediction_extraction_mode": extraction_mode,
        "base_commit": instance.get("base_commit", ""),
        "actual_commit": _git_head(repo_path),
        "answer": answer,
        "file_score": file_score,
        "doc_file_score": doc_score,
        "code_file_score": code_score,
        "elapsed_s": round(elapsed, 2),
        "repl_time_s": round(repl_time_s, 2),
        "llm_time_s": round(llm_time_s, 2),
        "total_tokens": total_bucket.get("total_tokens", 0),
        "input_tokens": total_bucket.get("input_tokens", 0),
        "output_tokens": total_bucket.get("output_tokens", 0),
        "repl_llm_tokens": tokens.get("repl_llm_total_tokens", 0),
        "repl_calls": repl_calls,
        "subagent_consult_calls": subagent_consult_calls,
        "subagent_consults": subagent_consults,
        "subagent_response_chars": sum(item.get("result_chars", 0) for item in subagent_consults),
        "context_window_used": context_window_used,
        "session_id": agent.session.get("session_id"),
        "conversation_history": conversation_history,
    }


async def run_subagent_setup(agent: CodebaseAgent, case_timeout: float) -> dict[str, Any]:
    prompt = (
        "Explore this repository enough to understand its major maintenance areas. "
        "Create appropriate folder-specialist subagents for cohesive parts of the codebase. "
        "Use your repository tools and build folder agents where they would help future maintenance questions. "
        "Stop when the useful folder agents are available, then summarize the agents you created."
    )

    t_start = time.perf_counter()
    answer = ""
    tokens: dict = {}
    repl_calls = 0
    repl_time_s = 0.0
    context_window_used = 0
    repl_call_start: float | None = None

    try:
        async for event in agent.ask_stream(prompt, skip_notes=False):
            event_type = event.get("type")
            if event_type == "repl_code":
                repl_calls += 1
                repl_call_start = time.perf_counter()
            elif event_type == "repl_output":
                if repl_call_start is not None:
                    repl_time_s += time.perf_counter() - repl_call_start
                    repl_call_start = None
            elif event_type == "tokens_usage":
                cumulative = event.get("tokens", {})
                if isinstance(cumulative, dict):
                    tokens.update(cumulative)
                context_window_used = max(context_window_used, event.get("context_window_used", 0))
            elif event_type == "done":
                answer = event.get("answer", "")
                tokens_usage = event.get("tokens_usage", {})
                if isinstance(tokens_usage, dict):
                    tokens = tokens_usage
                    context_window_used = max(context_window_used, tokens_usage.get("context_window_used", 0))
    except asyncio.TimeoutError:
        answer = f"|ERROR| timed out after {case_timeout}s"
    except Exception as exc:
        prefix = "|CONTEXT_OVERFLOW|" if _is_context_overflow(exc) else "|ERROR|"
        answer = f"{prefix} {exc}"
    finally:
        current_session_id = agent.session.get("session_id")
        if isinstance(current_session_id, str):
            clear_repl_session(current_session_id)

    elapsed = time.perf_counter() - t_start
    if repl_call_start is not None:
        repl_time_s += time.perf_counter() - repl_call_start

    total_bucket = tokens.get("total", tokens)
    return {
        "answer": answer,
        "elapsed_s": round(elapsed, 2),
        "repl_time_s": round(repl_time_s, 2),
        "llm_time_s": round(max(0.0, elapsed - repl_time_s), 2),
        "total_tokens": total_bucket.get("total_tokens", 0),
        "input_tokens": total_bucket.get("input_tokens", 0),
        "output_tokens": total_bucket.get("output_tokens", 0),
        "repl_llm_tokens": tokens.get("repl_llm_total_tokens", 0),
        "repl_calls": repl_calls,
        "context_window_used": context_window_used,
        "tokens_usage": tokens,
    }


def _error_result(index: int, instance: dict, answer: str, session_id: str | None, elapsed_s: float = 0.0) -> dict:
    gold_files = _dedupe_paths(instance.get("gold_files", []))
    predicted_files, extraction_mode = extract_predicted_files(answer)
    return {
        "index": index,
        "instance_id": instance.get("instance_id", ""),
        "date": instance.get("date", ""),
        "difficulty": instance.get("difficulty", "unknown"),
        "num_files": instance.get("num_files", len(gold_files)),
        "problem_statement": instance.get("problem_statement", ""),
        "gold_files": gold_files,
        "gold_doc_files": [path for path in gold_files if _is_doc_path(path)],
        "predicted_files": predicted_files,
        "prediction_extraction_mode": extraction_mode,
        "base_commit": instance.get("base_commit", ""),
        "actual_commit": None,
        "answer": answer,
        "file_score": score_files(predicted_files, gold_files),
        "doc_file_score": _score_subset(predicted_files, gold_files, docs=True),
        "code_file_score": _score_subset(predicted_files, gold_files, docs=False),
        "elapsed_s": round(elapsed_s, 2),
        "repl_time_s": 0.0,
        "llm_time_s": round(elapsed_s, 2),
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "repl_llm_tokens": 0,
        "repl_calls": 0,
        "subagent_consult_calls": 0,
        "subagent_consults": [],
        "subagent_response_chars": 0,
        "context_window_used": 0,
        "session_id": session_id,
        "conversation_history": [],
    }


@lru_cache(maxsize=8)
def _repo_stats(repo_path: Path) -> dict[str, Any]:
    files = [path for path in repo_path.rglob("*") if path.is_file() and ".git" not in path.parts]
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "estimated_tokens": total_bytes // 4,
    }


def _git_head(repo_path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def _git_commit_exists(repo_path: Path, commit: str) -> bool:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _git_status_porcelain(repo_path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Could not inspect git status for {repo_path}: {exc}")
    return proc.stdout.strip()


def checkout_commit(repo_path: Path, commit: str) -> None:
    if not commit:
        raise SystemExit("Missing commit to checkout.")
    try:
        subprocess.run(
            ["git", "checkout", "--quiet", commit],
            cwd=repo_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Could not checkout {commit} in {repo_path}.\n"
            f"{exc.stderr.strip()}"
        )


def validate_repo(repo_path: Path, expected_commit: str | None, *, skip_commit_check: bool) -> None:
    if not repo_path.exists():
        raise SystemExit(
            f"Repo path does not exist: {repo_path}\n"
            "Expected setup:\n"
            "  git clone https://github.com/ansible/ansible repos/ansible__ansible\n"
            "  cd repos/ansible__ansible\n"
            "  git checkout 01e7915b0a9778a934a0f0e9e9d110dbef7e31ec"
        )
    if not (repo_path / ".git").exists():
        raise SystemExit(f"Repo path is not a git checkout: {repo_path}")
    if expected_commit and not skip_commit_check:
        if not _git_commit_exists(repo_path, expected_commit):
            raise SystemExit(f"Expected commit is not available in {repo_path}: {expected_commit}")


def validate_instance_commits(repo_path: Path, selected_instances: list[tuple[int, dict]], fallback_commit: str | None) -> None:
    missing: list[str] = []
    for _, instance in selected_instances:
        commit = instance.get("base_commit") or fallback_commit
        if commit and not _git_commit_exists(repo_path, commit):
            missing.append(commit)
    if missing:
        unique_missing = sorted(set(missing))
        raise SystemExit(
            "Some instance base commits are not available in the target repo:\n"
            + "\n".join(f"  - {commit}" for commit in unique_missing)
        )


def _managed_session_dirs(agent: CodebaseAgent) -> list[Path]:
    paths = [agent.session_manager.sessions_dir / agent.session["session_id"]]
    base_memory_dir = SessionManager().db_dir
    for agent_def in agent.subagent_registry.list_agents():
        subagent_session_id = agent_def.get("session_id")
        if subagent_session_id:
            paths.append(base_memory_dir / "subagents" / agent_def["name"] / "sessions" / subagent_session_id)
    return paths


def snapshot_agent_state(agent: CodebaseAgent) -> dict[str, Any]:
    snapshot_root = Path(tempfile.mkdtemp(prefix="long_session_eval_snapshot_"))
    entries: list[dict[str, str]] = []
    for idx, source in enumerate(_managed_session_dirs(agent)):
        if not source.exists():
            continue
        target = snapshot_root / str(idx)
        shutil.copytree(source, target)
        entries.append({"source": str(source), "snapshot": str(target)})
    return {"root": str(snapshot_root), "entries": entries}


def restore_agent_state(snapshot: dict[str, Any]) -> None:
    for entry in snapshot.get("entries", []):
        source = Path(entry["source"])
        snap = Path(entry["snapshot"])
        if source.exists():
            shutil.rmtree(source)
        if snap.exists():
            shutil.copytree(snap, source)


def delete_snapshot(snapshot: dict[str, Any]) -> None:
    root = snapshot.get("root")
    if root:
        shutil.rmtree(root, ignore_errors=True)


def delete_managed_sessions(agent: CodebaseAgent) -> None:
    for path in _managed_session_dirs(agent):
        shutil.rmtree(path, ignore_errors=True)


def _sum_token_entries(entries: list[dict[str, Any]]) -> dict[str, int]:
    bucket = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "num_calls": len(entries),
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    for entry in entries:
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            bucket[key] += int(entry.get(key, 0) or 0)
    return bucket


def _timeout_setup_metrics(agent: CodebaseAgent, timeout_s: float, elapsed_s: float) -> dict[str, Any]:
    session_id = agent.session.get("session_id")
    log_data = read_and_purge(session_id) if isinstance(session_id, str) else {}
    orchestrator_bucket = _sum_token_entries(log_data.get("orchestrator", []))
    agent_builder_bucket = _sum_token_entries(log_data.get("agent_builder", []))
    subagent_entries = [
        entry
        for namespace, entries in log_data.items()
        if namespace.startswith("subagent_")
        for entry in entries
    ]
    subagents_bucket = _sum_token_entries(subagent_entries)
    total_bucket = {
        key: orchestrator_bucket[key] + agent_builder_bucket[key] + subagents_bucket[key]
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "num_calls",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    }
    tokens = {
        "orchestrator": orchestrator_bucket,
        "agent_builder": agent_builder_bucket,
        "subagents": subagents_bucket,
        "total": total_bucket,
    }
    return {
        "answer": f"|ERROR| subagent initialization timed out after {timeout_s}s",
        "timed_out": True,
        "elapsed_s": round(elapsed_s, 2),
        "repl_time_s": 0.0,
        "llm_time_s": round(elapsed_s, 2),
        "total_tokens": total_bucket["total_tokens"],
        "input_tokens": total_bucket["input_tokens"],
        "output_tokens": total_bucket["output_tokens"],
        "repl_llm_tokens": 0,
        "repl_calls": 0,
        "context_window_used": 0,
        "tokens_usage": tokens,
    }


def _build_report(results: list[dict], benchmark: dict, args: argparse.Namespace, repo_path: Path) -> dict:
    summary = _metric_summary(results)
    setup_metrics = (getattr(args, "subagent_initialization", None) or {}).get("metrics", {})
    summary["initialization_tokens"] = setup_metrics.get("total_tokens", 0)
    summary["initialization_input_tokens"] = setup_metrics.get("input_tokens", 0)
    summary["initialization_output_tokens"] = setup_metrics.get("output_tokens", 0)
    summary["total_tokens_with_initialization"] = summary["total_tokens"] + summary["initialization_tokens"]
    return {
        "run_at": datetime.now().isoformat(),
        "benchmark": {
            key: value for key, value in benchmark.items()
            if key != "instances"
        },
        "config": {
            "repo_path": str(repo_path),
            "provider": os.getenv("LLM_PROVIDER", "anthropic"),
            "model": os.getenv("LLM_MODEL", ""),
            "tool_set": args.tool_set,
            "skip_notes": args.skip_notes,
            "timeout": args.timeout,
            "initialization_timeout": args.initialization_timeout,
            "reuse_initialized_session": args.reuse_initialized_session,
            "no_subagents": args.no_subagents,
            "limit": args.limit,
            "difficulty": args.difficulty,
            "session_id": args.session_id,
        },
        "subagent_initialization": getattr(args, "subagent_initialization", None),
        "repo": _repo_stats(repo_path),
        "summary": summary,
        "groups": _group_summaries(results),
        "results": results,
    }


def save_report(results: list[dict], benchmark: dict, args: argparse.Namespace, repo_path: Path, out_path: Path, *, announce: bool = True) -> None:
    report = _build_report(results, benchmark, args, repo_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2))
    tmp_path.replace(out_path)
    if announce:
        console.print(f"\n[dim]Report saved -> {out_path}[/dim]")


def print_results_table(results: list[dict]) -> None:
    table = Table(title="Long-Session Codebase File Localization", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Difficulty", width=10)
    table.add_column("Gold", justify="right", width=5)
    table.add_column("Pred", justify="right", width=5)
    table.add_column("P", justify="right", width=6)
    table.add_column("R", justify="right", width=6)
    table.add_column("F1", justify="right", width=6)
    table.add_column("Exact", justify="center", width=6)
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("Peak In", justify="right", width=8)
    table.add_column("REPL", justify="right", width=6)
    table.add_column("Agents", justify="right", width=7)
    table.add_column("s", justify="right", width=7)

    for result in results:
        score = result["file_score"]
        table.add_row(
            str(result["index"] + 1),
            str(result["difficulty"]),
            str(len(result["gold_files"])),
            str(len(result["predicted_files"])),
            f"{score['precision']:.2f}",
            f"{score['recall']:.2f}",
            f"{score['f1']:.2f}",
            "[green]yes[/green]" if score["exact_match"] else "[red]no[/red]",
            str(result["total_tokens"]),
            str(result.get("context_window_used", 0)),
            str(result["repl_calls"]),
            str(result.get("subagent_consult_calls", 0)),
            str(result["elapsed_s"]),
        )

    console.print(table)
    summary = _metric_summary(results)
    console.print(
        f"\n[bold]Macro P/R/F1:[/bold] {summary['macro_precision']:.3f} / "
        f"{summary['macro_recall']:.3f} / {summary['macro_f1']:.3f}"
    )
    console.print(
        f"[bold]Micro P/R/F1:[/bold] {summary['micro_precision']:.3f} / "
        f"{summary['micro_recall']:.3f} / {summary['micro_f1']:.3f}"
    )
    console.print(
        f"[bold]Exact matches:[/bold] {summary['exact_matches']}/{summary['total']}  |  "
        f"[bold]All gold found:[/bold] {summary['all_gold_found']}/{summary['total']}"
    )
    console.print(
        f"[bold]Tokens:[/bold] {summary['total_tokens']} total  "
        f"({summary['total_input_tokens']}/{summary['total_output_tokens']} in/out)  |  "
        f"[bold]Peak input:[/bold] {summary['max_context_window_used']}  |  "
        f"[bold]REPL calls:[/bold] {summary['total_repl_calls']}  |  "
        f"[bold]Subagent consults:[/bold] {summary['total_subagent_consult_calls']} "
        f"({summary['subagent_consult_cases']} case(s))"
    )


def print_initialization_summary(initialization: dict[str, Any] | None) -> None:
    if not initialization:
        return
    metrics = initialization.get("metrics", {})
    agents = initialization.get("agents", [])
    console.print(
        f"[bold]Subagent initialization:[/bold] {len(agents)} agent(s)  |  "
        f"tokens={metrics.get('total_tokens', 0)}  "
        f"in/out={metrics.get('input_tokens', 0)}/{metrics.get('output_tokens', 0)}  "
        f"peak_in={metrics.get('context_window_used', 0)}  "
        f"repl={metrics.get('repl_calls', 0)}  "
        f"{metrics.get('elapsed_s', 0.0)}s"
    )


def _select_instances(instances: list[dict], args: argparse.Namespace) -> list[tuple[int, dict]]:
    selected = list(enumerate(instances))
    if args.difficulty:
        selected = [(i, item) for i, item in selected if item.get("difficulty") == args.difficulty]
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [(i, item) for i, item in selected if item.get("instance_id") in wanted]
    if args.limit is not None:
        selected = selected[:args.limit]
    return selected


async def main(args: argparse.Namespace) -> None:
    benchmark_path = args.benchmark.resolve()
    benchmark = json.loads(benchmark_path.read_text())
    instances = benchmark.get("instances", [])
    selected_instances = _select_instances(instances, args)
    if not selected_instances:
        console.print("[red]No benchmark instances selected.[/red]")
        raise SystemExit(1)

    repo_path = args.repo.resolve()
    validate_repo(repo_path, benchmark.get("fixed_commit"), skip_commit_check=args.skip_commit_check)
    validate_instance_commits(repo_path, selected_instances, benchmark.get("fixed_commit"))
    if not args.allow_dirty_repo:
        dirty = _git_status_porcelain(repo_path)
        if dirty:
            raise SystemExit(
                f"Target repo has uncommitted changes and this eval checks out commits:\n{dirty}\n"
                "Commit/stash those changes, or pass --allow-dirty-repo if you accept checkout side effects."
            )
    out_path = (args.out.resolve() if args.out else benchmark_path.with_name(benchmark_path.stem + "_report.json"))

    provider = os.getenv("LLM_PROVIDER", "anthropic")
    model = os.getenv("LLM_MODEL")
    if not model:
        console.print("[red]LLM_MODEL env var not set.[/red]")
        raise SystemExit(1)
    if args.no_subagents and args.reuse_initialized_session:
        raise SystemExit("--no-subagents cannot be combined with --reuse-initialized-session")
    if args.no_subagents and args.session_id:
        raise SystemExit("--no-subagents must use a fresh session; omit --session-id")

    console.print(
        f"\n[bold blue]Long-Session Codebase Eval[/bold blue] - {len(selected_instances)} instances | "
        f"{provider}/{model}"
    )
    console.print(
        f"[dim]Repo: {repo_path} | Timeout: {args.timeout}s[/dim]\n"
    )

    previous_cwd = Path.cwd()
    original_head = _git_head(repo_path)
    initialization_snapshot: dict[str, Any] | None = None
    setup_agent: CodebaseAgent | None = None
    created_session_id: str | None = None

    try:
        os.chdir(repo_path)
        setup_commit = benchmark.get("fixed_commit")
        if setup_commit:
            checkout_commit(repo_path, setup_commit)

        setup_agent = CodebaseAgent(
            codebase_path=".",
            provider=provider,
            model=model,
            session_id=args.session_id,
            tool_set=args.tool_set,
        )
        if args.no_subagents:
            disable_subagent_tools(setup_agent)

        created_session_id = setup_agent.session.get("session_id")
        if args.no_subagents:
            setup_metrics = {
                "answer": "Subagent initialization disabled for this run.",
                "elapsed_s": 0.0,
                "repl_time_s": 0.0,
                "llm_time_s": 0.0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "repl_llm_tokens": 0,
                "repl_calls": 0,
                "context_window_used": 0,
                "tokens_usage": {},
            }
            console.print("[cyan]Subagent tools disabled; skipping folder-agent initialization.[/cyan]")
        elif args.reuse_initialized_session:
            if not args.session_id:
                raise SystemExit("--reuse-initialized-session requires --session-id")
            setup_metrics = {
                "answer": "Reused existing initialized session.",
                "elapsed_s": 0.0,
                "repl_time_s": 0.0,
                "llm_time_s": 0.0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "repl_llm_tokens": 0,
                "repl_calls": 0,
                "context_window_used": 0,
                "tokens_usage": {},
            }
            console.print("[cyan]Reusing initialized folder agents from session.[/cyan]")
        else:
            console.print("[cyan]Initializing folder agents through the agent...[/cyan]")
            setup_started = time.perf_counter()
            try:
                setup_metrics = await asyncio.wait_for(
                    run_subagent_setup(setup_agent, args.initialization_timeout),
                    timeout=args.initialization_timeout,
                )
            except asyncio.TimeoutError:
                setup_elapsed = time.perf_counter() - setup_started
                setup_metrics = _timeout_setup_metrics(
                    setup_agent,
                    args.initialization_timeout,
                    setup_elapsed,
                )
                console.print(
                    "[yellow]Subagent initialization timed out; continuing with "
                    f"{len(setup_agent.subagent_registry.list_agents())} partial folder agent(s).[/yellow]"
                )
        subagent_init = {
            "metrics": setup_metrics,
            "agents": [] if args.no_subagents else setup_agent.subagent_registry.list_agents(),
        }
        args.subagent_initialization = subagent_init
        initialization_snapshot = snapshot_agent_state(setup_agent)
        console.print(
            f"[dim]Initialized {len(subagent_init['agents'])} folder agent(s); "
            f"setup tokens={setup_metrics['total_tokens']}.[/dim]\n"
        )
        save_report([], benchmark, args, repo_path, out_path, announce=False)
    except Exception:
        if original_head:
            try:
                checkout_commit(repo_path, original_head)
            except SystemExit as exc:
                console.print(f"[yellow]Could not restore repo HEAD after setup failure: {exc}[/yellow]")
        os.chdir(previous_cwd)
        raise

    results_by_index: dict[int, dict] = {}
    completed = 0
    interrupted = False

    async def run_one(original_index: int, instance: dict) -> tuple[int, dict]:
        if initialization_snapshot is not None:
            restore_agent_state(initialization_snapshot)
        checkout_commit(repo_path, instance.get("base_commit") or benchmark.get("fixed_commit"))

        agent = CodebaseAgent(
            codebase_path=".",
            provider=provider,
            model=model,
            session_id=created_session_id,
            tool_set=args.tool_set,
        )
        if args.no_subagents:
            disable_subagent_tools(agent)
        preview = instance.get("problem_statement", "").splitlines()[0][:70]
        console.print(f"[cyan][{completed + 1}/{len(selected_instances)}][/cyan] {instance.get('difficulty', '?')} {preview}")
        try:
            result = await asyncio.wait_for(
                run_instance(
                    agent=agent,
                    index=original_index,
                    instance=instance,
                    repo_path=repo_path,
                    case_timeout=args.timeout,
                    skip_notes=args.skip_notes,
                    subagents_enabled=not args.no_subagents,
                ),
                timeout=args.timeout,
            )
        except asyncio.TimeoutError:
            result = _error_result(
                original_index,
                instance,
                f"|ERROR| timed out after {args.timeout}s",
                agent.session.get("session_id"),
                elapsed_s=args.timeout,
            )
        finally:
            if initialization_snapshot is not None:
                restore_agent_state(initialization_snapshot)

        return original_index, result

    tasks: list[asyncio.Task] = []
    loop = asyncio.get_running_loop()
    interrupt_event = asyncio.Event()
    interrupt_task: asyncio.Task | None = None
    signal_handler_installed = False

    try:
        loop.add_signal_handler(signal.SIGINT, interrupt_event.set)
        signal_handler_installed = True
        interrupt_task = asyncio.create_task(interrupt_event.wait())
    except (NotImplementedError, RuntimeError):
        pass

    try:
        for original_index, instance in selected_instances:
            if interrupted:
                break
            task = asyncio.create_task(run_one(original_index, instance))
            tasks.append(task)
            wait_for = {task}
            if interrupt_task is not None:
                wait_for.add(interrupt_task)
            done, _pending = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)
            if interrupt_task is not None and interrupt_task in done:
                interrupted = True
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                break
            _, result = task.result()
            results_by_index[original_index] = result
            completed += 1
            score = result["file_score"]
            console.print(
                f"  F1={score['f1']:.3f} P={score['precision']:.3f} R={score['recall']:.3f} "
                f"tokens={result['total_tokens']} peak_in={result['context_window_used']} "
                f"[dim]({completed}/{len(selected_instances)} done)[/dim]"
            )
            save_report(
                [results_by_index[i] for i in sorted(results_by_index)],
                benchmark,
                args,
                repo_path,
                out_path,
                announce=False,
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
    finally:
        if interrupted:
            console.print("\n[yellow]Interrupted - cancelling remaining tasks...[/yellow]")
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if interrupt_task is not None:
            interrupt_task.cancel()
            await asyncio.gather(interrupt_task, return_exceptions=True)
        if signal_handler_installed:
            loop.remove_signal_handler(signal.SIGINT)
        if original_head:
            try:
                checkout_commit(repo_path, original_head)
            except SystemExit as exc:
                console.print(f"[yellow]Could not restore repo HEAD: {exc}[/yellow]")
        os.chdir(previous_cwd)

    ordered_results = [results_by_index[i] for i in sorted(results_by_index)]
    if ordered_results:
        console.print()
        print_initialization_summary(getattr(args, "subagent_initialization", None))
        print_results_table(ordered_results)
        save_report(ordered_results, benchmark, args, repo_path, out_path)
    else:
        console.print("[dim]No results to report.[/dim]")

    if args.discard_session:
        if setup_agent is not None and isinstance(created_session_id, str) and created_session_id != args.session_id:
            delete_managed_sessions(setup_agent)
    if initialization_snapshot is not None:
        delete_snapshot(initialization_snapshot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run file-localization evals on long-session codebase benchmarks")
    parser.add_argument("benchmark", type=Path, help="Path to benchmark JSON")
    parser.add_argument("--repo", type=Path, default=Path("repos/ansible__ansible"), help="Path to checked-out repository")
    parser.add_argument("--out", type=Path, default=None, help="Output report path")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per case in seconds")
    parser.add_argument("--initialization-timeout", type=float, default=1200.0, help="Timeout for initial repository exploration and folder-agent setup")
    parser.add_argument("--session-id", default=None, help="Existing agent session ID to resume")
    parser.add_argument("--reuse-initialized-session", action="store_true", help="Load subagents from --session-id and skip the initialization pass")
    parser.add_argument("--no-subagents", action="store_true", help="Disable build_agents_for and consult_agents, and skip subagent initialization")
    parser.add_argument("--discard-session", action="store_true", help="Delete the initialized agent and subagent session files after the run")
    parser.add_argument("--skip-commit-check", action="store_true", help="Skip checking that the benchmark fixed commit exists before setup")
    parser.add_argument("--allow-dirty-repo", action="store_true", help="Allow running even when the target repo has uncommitted changes")
    parser.add_argument("--tool-set", choices=["core", "analysis", "all", "repl"], default="all", help="Agent tool set")
    parser.add_argument("--skip-notes", action="store_true", help="Disable session note updates between turns")
    parser.add_argument("--difficulty", choices=["easy", "hard"], default=None, help="Only run one difficulty")
    parser.add_argument("--instance-id", action="append", default=None, help="Only run the given instance ID; can be repeated")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected cases")
    asyncio.run(main(parser.parse_args()))
