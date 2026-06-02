#!/usr/bin/env python3
"""
Run file-localization evals against external coding CLIs.

This is intentionally separate from run_eval.py because CLI agents have
different sandboxing, state, and token observability behavior than the in-process
CodebaseAgent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


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


def _metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    error_cases = sum(1 for r in results if str(r.get("answer", "")).startswith("|ERROR|"))
    timeout_cases = sum(1 for r in results if r.get("cli_timed_out") or "timed out after" in str(r.get("answer", "")))
    unsupported_model_cases = sum(1 for r in results if "invalid_request_error" in str(r.get("answer", "")))
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
        "error_cases": error_cases,
        "timeout_cases": timeout_cases,
        "unsupported_model_cases": unsupported_model_cases,
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
        "total_repl_llm_tokens": 0,
        "total_repl_calls": 0,
        "subagent_consult_cases": 0,
        "total_subagent_consult_calls": 0,
        "total_subagent_response_chars": 0,
        "max_context_window_used": max((r.get("context_window_used", 0) for r in results), default=0),
        "total_elapsed_s": round(sum(r["elapsed_s"] for r in results), 2),
        "total_repl_time_s": 0.0,
        "total_llm_time_s": round(sum(r["elapsed_s"] for r in results), 2),
        "initialization_tokens": 0,
        "initialization_input_tokens": 0,
        "initialization_output_tokens": 0,
        "total_tokens_with_initialization": sum(r["total_tokens"] for r in results),
        "cli_failures": sum(1 for r in results if r.get("cli_returncode") not in (0, None) or r.get("cli_timed_out")),
        "permission_failures": sum(1 for r in results if r.get("permission_failure")),
    }


def _group_summaries(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(f"difficulty:{result.get('difficulty', 'unknown')}", []).append(result)
        groups.setdefault("gold_docs:any" if result.get("gold_doc_files") else "gold_docs:none", []).append(result)
    return {name: _metric_summary(group_results) for name, group_results in sorted(groups.items())}


def _repo_stats(repo_path: Path) -> dict[str, Any]:
    files = [path for path in repo_path.rglob("*") if path.is_file() and ".git" not in path.parts]
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "estimated_tokens": total_bytes // 4,
    }


def _git(repo_path: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _git_head(repo_path: Path) -> str | None:
    try:
        return _git(repo_path, ["rev-parse", "HEAD"]).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_commit_exists(repo_path: Path, commit: str) -> bool:
    try:
        _git(repo_path, ["cat-file", "-e", f"{commit}^{{commit}}"])
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _git_status_porcelain(repo_path: Path) -> str:
    try:
        return _git(repo_path, ["status", "--porcelain"]).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Could not inspect git status for {repo_path}: {exc}") from exc


def clean_untracked(repo_path: Path) -> None:
    try:
        _git(repo_path, ["clean", "-fd"])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Could not clean untracked files in {repo_path}: {exc}") from exc


def checkout_commit(repo_path: Path, commit: str) -> None:
    if not commit:
        raise SystemExit("Missing commit to checkout.")
    try:
        _git(repo_path, ["checkout", "--quiet", commit])
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Could not checkout {commit} in {repo_path}.\n"
            f"{exc.stderr.strip()}"
        ) from exc


def validate_repo(repo_path: Path, expected_commit: str | None, *, skip_commit_check: bool) -> None:
    if not repo_path.exists():
        raise SystemExit(f"Repo path does not exist: {repo_path}")
    if not (repo_path / ".git").exists():
        raise SystemExit(f"Repo path is not a git checkout: {repo_path}")
    if expected_commit and not skip_commit_check and not _git_commit_exists(repo_path, expected_commit):
        raise SystemExit(f"Expected commit is not available in {repo_path}: {expected_commit}")


def validate_instance_commits(repo_path: Path, selected_instances: list[tuple[int, dict[str, Any]]], fallback_commit: str | None) -> None:
    missing: list[str] = []
    for _, instance in selected_instances:
        commit = instance.get("base_commit") or fallback_commit
        if commit and not _git_commit_exists(repo_path, commit):
            missing.append(commit)
    if missing:
        raise SystemExit(
            "Some instance base commits are not available in the target repo:\n"
            + "\n".join(f"  - {commit}" for commit in sorted(set(missing)))
        )


PROMPT_TASK = (
    "Identify the repo-relative files that would need to be modified to implement the requested fix. "
    "Include code, test, and documentation files when they would need edits. Do not edit files."
    "Use the repository tools to inspect the codebase. Prefer exact existing file paths."
)
PROMPT_OUTPUT = (
    "Your final answer must be valid JSON only, with this schema:\n"
    '{"files": ["path/to/file.py"], "rationale": "short reason"}'
)


def _build_prompt(instance: dict[str, Any]) -> str:
    return (
        f"{PROMPT_TASK}\n\n"
        f"{PROMPT_OUTPUT}\n\n"
        "Problem statement:\n"
        f"{instance.get('problem_statement', '')}"
    )


def _json_lines(text: str) -> list[Any]:
    events: list[Any] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return events


def _walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_json(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def _extract_text_from_json(value: Any) -> str:
    preferred_keys = ("response", "text", "content", "message", "answer", "output", "final")
    strings: list[str] = []
    for item in _walk_json(value):
        for key in preferred_keys:
            nested = item.get(key)
            if isinstance(nested, str):
                strings.append(nested)
            elif isinstance(nested, dict):
                strings.append(_extract_text_from_json(nested))
            elif isinstance(nested, list):
                for entry in nested:
                    if isinstance(entry, str):
                        strings.append(entry)
                    elif isinstance(entry, dict):
                        strings.append(_extract_text_from_json(entry))
    strings = [item for item in strings if item]
    if not strings:
        return ""
    return max(strings, key=len)


def _usage_from_json(values: list[Any]) -> dict[str, Any]:
    input_keys = ("input_tokens", "prompt_tokens", "promptTokenCount")
    output_keys = ("output_tokens", "completion_tokens", "candidatesTokenCount")
    total_keys = ("total_tokens", "totalTokenCount")
    cached_keys = ("cached_input_tokens", "cachedContentTokenCount")

    usage_candidates: list[dict[str, int]] = []
    for value in values:
        for item in _walk_json(value):
            usage: dict[str, int] = {}
            for key in input_keys:
                if isinstance(item.get(key), int):
                    usage["input_tokens"] = max(usage.get("input_tokens", 0), int(item[key]))
            for key in output_keys:
                if isinstance(item.get(key), int):
                    usage["output_tokens"] = max(usage.get("output_tokens", 0), int(item[key]))
            for key in total_keys:
                if isinstance(item.get(key), int):
                    usage["total_tokens"] = max(usage.get("total_tokens", 0), int(item[key]))
            for key in cached_keys:
                if isinstance(item.get(key), int):
                    usage["cached_input_tokens"] = max(usage.get("cached_input_tokens", 0), int(item[key]))
            if usage:
                usage_candidates.append(usage)

    input_tokens = max((u.get("input_tokens", 0) for u in usage_candidates), default=0)
    output_tokens = max((u.get("output_tokens", 0) for u in usage_candidates), default=0)
    total_tokens = max((u.get("total_tokens", 0) for u in usage_candidates), default=0)
    cached_input_tokens = max((u.get("cached_input_tokens", 0) for u in usage_candidates), default=0)
    if not total_tokens and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens

    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "candidates": usage_candidates,
    }


def _permission_failure(stdout: str, stderr: str, returncode: int | None) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    if returncode == 0:
        return False
    markers = (
        "permission denied",
        "operation not permitted",
        "approval",
        "not allowed",
        "sandbox",
        "read-only file system",
    )
    return any(marker in text for marker in markers)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _resolve_existing_path(path: Path, *, fallback_dir: Path) -> Path:
    if path.exists():
        return path.resolve()
    fallback = fallback_dir / path.name
    if fallback.exists():
        return fallback.resolve()
    return path.resolve()


def _run_codex(args: argparse.Namespace, repo_path: Path, prompt: str) -> dict[str, Any]:
    codex_bin = shutil.which(args.codex_bin)
    if not codex_bin:
        raise SystemExit(f"Could not find Codex CLI binary: {args.codex_bin}")

    schema_path = _resolve_existing_path(args.schema, fallback_dir=Path(__file__).resolve().parent)
    if not schema_path.exists():
        raise SystemExit(f"Output schema does not exist: {schema_path}")

    with tempfile.TemporaryDirectory(prefix="cli_eval_codex_") as temp_dir:
        last_message_path = Path(temp_dir) / "last_message.json"
        command = [
            codex_bin,
            "-a",
            "never",
        ]
        if args.codex_model:
            command.extend(["-m", args.codex_model])
        command.extend([
            "exec",
            "--cd",
            str(repo_path),
            "--sandbox",
            args.codex_sandbox,
            "--ephemeral",
            "--json",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(last_message_path),
            "-",
        ])
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                input=prompt,
                cwd=repo_path,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - started
            return {
                "answer": f"|ERROR| timed out after {args.timeout}s",
                "stdout": _to_text(exc.stdout),
                "stderr": _to_text(exc.stderr),
                "returncode": None,
                "elapsed_s": elapsed,
                "timed_out": True,
                "events": [],
                "usage": {},
                "command": command,
            }

        elapsed = time.perf_counter() - started
        events = _json_lines(proc.stdout)
        answer = ""
        if last_message_path.exists():
            answer = last_message_path.read_text()
        if not answer:
            answer = _extract_text_from_json(events)
        if not answer:
            answer = proc.stdout
        usage = _usage_from_json(events)
        return {
            "answer": answer,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "timed_out": timed_out,
            "events": events,
            "usage": usage,
            "command": command,
        }


def _run_gemini(args: argparse.Namespace, repo_path: Path, prompt: str) -> dict[str, Any]:
    gemini_bin = shutil.which(args.gemini_bin)
    if not gemini_bin:
        raise SystemExit(f"Could not find Gemini CLI binary: {args.gemini_bin}")

    command = [gemini_bin, "--output-format", "json", "--prompt", ""]
    if args.gemini_model:
        command.extend(["--model", args.gemini_model])
    if args.gemini_sandbox:
        command.append("--sandbox")
    if args.gemini_approval_mode:
        command.extend(["--approval-mode", args.gemini_approval_mode])

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            input=prompt,
            cwd=repo_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        return {
            "answer": f"|ERROR| timed out after {args.timeout}s",
            "stdout": _to_text(exc.stdout),
            "stderr": _to_text(exc.stderr),
            "returncode": None,
            "elapsed_s": elapsed,
            "timed_out": True,
            "events": [],
            "usage": {},
            "command": command,
        }

    elapsed = time.perf_counter() - started
    events = _json_lines(proc.stdout)
    parsed_stdout: Any | None = None
    try:
        parsed_stdout = json.loads(proc.stdout)
    except json.JSONDecodeError:
        parsed_stdout = None

    values = events[:]
    if parsed_stdout is not None:
        values.append(parsed_stdout)

    answer = _extract_text_from_json(values) or proc.stdout
    usage = _usage_from_json(values)
    return {
        "answer": answer,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "timed_out": timed_out,
        "events": events,
        "usage": usage,
        "command": command,
    }


def run_cli(args: argparse.Namespace, repo_path: Path, prompt: str) -> dict[str, Any]:
    if args.cli == "codex":
        return _run_codex(args, repo_path, prompt)
    if args.cli == "gemini":
        return _run_gemini(args, repo_path, prompt)
    raise SystemExit(f"Unsupported CLI: {args.cli}")


def run_instance(args: argparse.Namespace, index: int, instance: dict[str, Any], repo_path: Path) -> dict[str, Any]:
    gold_files = _dedupe_paths(instance.get("gold_files", []), repo_path)
    prompt = _build_prompt(instance)
    cli_result = run_cli(args, repo_path, prompt)

    answer = cli_result["answer"]
    predicted_files, extraction_mode = extract_predicted_files(answer, repo_path)
    file_score = score_files(predicted_files, gold_files)
    doc_score = _score_subset(predicted_files, gold_files, docs=True)
    code_score = _score_subset(predicted_files, gold_files, docs=False)
    usage = cli_result.get("usage", {})
    returncode = cli_result.get("returncode")
    stdout = cli_result.get("stdout", "")
    stderr = cli_result.get("stderr", "")

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
        "elapsed_s": round(cli_result.get("elapsed_s", 0.0), 2),
        "repl_time_s": 0.0,
        "llm_time_s": round(cli_result.get("elapsed_s", 0.0), 2),
        "total_tokens": usage.get("total_tokens", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cached_input_tokens": usage.get("cached_input_tokens", 0),
        "repl_llm_tokens": 0,
        "repl_calls": 0,
        "subagent_consult_calls": 0,
        "subagent_consults": [],
        "subagent_response_chars": 0,
        "context_window_used": usage.get("input_tokens", 0),
        "session_id": None,
        "conversation_history": [
            {"role": "user", "content": prompt[: args.trace_content_max_chars]},
            {"role": "assistant", "content": answer[: args.trace_content_max_chars]},
        ],
        "cli": args.cli,
        "cli_command": " ".join(str(part) for part in cli_result.get("command", [])),
        "cli_returncode": returncode,
        "cli_timed_out": cli_result.get("timed_out", False),
        "cli_event_count": len(cli_result.get("events", [])),
        "cli_token_usage_candidates": usage.get("candidates", []),
        "permission_failure": _permission_failure(stdout, stderr, returncode),
        "stderr_preview": stderr[:2000],
        "stdout_preview": stdout[:2000],
    }


def _select_instances(instances: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    selected = list(enumerate(instances))
    if args.difficulty:
        selected = [(i, item) for i, item in selected if item.get("difficulty") == args.difficulty]
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [(i, item) for i, item in selected if item.get("instance_id") in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def _build_report(results: list[dict[str, Any]], benchmark: dict[str, Any], args: argparse.Namespace, repo_path: Path) -> dict[str, Any]:
    summary = _metric_summary(results)
    return {
        "run_at": datetime.now().isoformat(),
        "benchmark": {
            key: value for key, value in benchmark.items()
            if key != "instances"
        },
        "config": {
            "runner": "cli",
            "repo_path": str(repo_path),
            "cli": args.cli,
            "provider": args.cli,
            "model": args.gemini_model if args.cli == "gemini" else "",
            "timeout": args.timeout,
            "limit": args.limit,
            "difficulty": args.difficulty,
            "instance_id": args.instance_id,
            "codex_sandbox": args.codex_sandbox,
            "codex_model": args.codex_model,
            "gemini_sandbox": args.gemini_sandbox,
            "gemini_approval_mode": args.gemini_approval_mode,
        },
        "subagent_initialization": {
            "metrics": {
                "answer": "External CLI runner has no subagent initialization step.",
                "elapsed_s": 0.0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "agents": [],
        },
        "repo": _repo_stats(repo_path),
        "summary": summary,
        "groups": _group_summaries(results),
        "results": results,
    }


def save_report(results: list[dict[str, Any]], benchmark: dict[str, Any], args: argparse.Namespace, repo_path: Path, out_path: Path, *, announce: bool = True) -> None:
    report = _build_report(results, benchmark, args, repo_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(_json_safe(report), indent=2))
    tmp_path.replace(out_path)
    if announce:
        console.print(f"\n[dim]Report saved -> {out_path}[/dim]")


def load_completed_results(out_path: Path, selected_indices: set[int]) -> dict[int, dict[str, Any]]:
    if not out_path.exists():
        return {}
    try:
        report = json.loads(out_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not resume from invalid report JSON: {out_path}\n{exc}") from exc

    completed: dict[int, dict[str, Any]] = {}
    for result in report.get("results", []):
        index = result.get("index")
        if isinstance(index, int) and index in selected_indices:
            completed[index] = result
    return completed


def print_results_table(results: list[dict[str, Any]]) -> None:
    table = Table(title="CLI File Localization", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Difficulty", width=10)
    table.add_column("Gold", justify="right", width=5)
    table.add_column("Pred", justify="right", width=5)
    table.add_column("P", justify="right", width=6)
    table.add_column("R", justify="right", width=6)
    table.add_column("F1", justify="right", width=6)
    table.add_column("Exact", justify="center", width=6)
    table.add_column("All", justify="center", width=5)
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("rc", justify="right", width=4)
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
            "[green]yes[/green]" if score["all_gold_found"] else "[red]no[/red]",
            str(result["total_tokens"]),
            str(result.get("cli_returncode")),
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
        f"[bold]All gold found:[/bold] {summary['all_gold_found']}/{summary['total']}  |  "
        f"[bold]CLI failures:[/bold] {summary['cli_failures']}  |  "
        f"[bold]Permission failures:[/bold] {summary['permission_failures']}"
    )
    console.print(
        f"[bold]Tokens:[/bold] {summary['total_tokens']} total  "
        f"({summary['total_input_tokens']}/{summary['total_output_tokens']} in/out)"
    )


def main(args: argparse.Namespace) -> None:
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
    if args.clean_untracked:
        clean_untracked(repo_path)
    if not args.allow_dirty_repo:
        dirty = _git_status_porcelain(repo_path)
        if dirty:
            raise SystemExit(
                f"Target repo has uncommitted changes and this eval checks out commits:\n{dirty}\n"
                "Commit/stash those changes, or pass --allow-dirty-repo if you accept checkout side effects."
            )

    out_path = (args.out.resolve() if args.out else benchmark_path.with_name(f"{benchmark_path.stem}_{args.cli}_report.json"))
    selected_indices = {index for index, _ in selected_instances}
    results_by_index: dict[int, dict[str, Any]] = (
        load_completed_results(out_path, selected_indices) if args.resume else {}
    )
    if args.resume and results_by_index:
        console.print(f"[cyan]Resuming from {out_path}; loaded {len(results_by_index)} completed result(s).[/cyan]")

    remaining_instances = [
        (index, instance) for index, instance in selected_instances
        if index not in results_by_index
    ]
    original_head = _git_head(repo_path)
    console.print(
        f"\n[bold blue]CLI File Localization Eval[/bold blue] - {args.cli} | "
        f"{len(selected_instances)} instance(s)"
    )
    console.print(f"[dim]Repo: {repo_path} | Timeout: {args.timeout}s[/dim]\n")

    try:
        for original_index, instance in remaining_instances:
            case_number = len(results_by_index) + 1
            checkout_commit(repo_path, instance.get("base_commit") or benchmark.get("fixed_commit"))
            preview = instance.get("problem_statement", "").splitlines()[0][:70]
            console.print(f"[cyan][{case_number}/{len(selected_instances)}][/cyan] {instance.get('difficulty', '?')} {preview}")
            result = run_instance(args, original_index, instance, repo_path)
            results_by_index[original_index] = result
            score = result["file_score"]
            console.print(
                f"  F1={score['f1']:.3f} P={score['precision']:.3f} R={score['recall']:.3f} "
                f"tokens={result['total_tokens']} rc={result['cli_returncode']} "
                f"[dim]({len(results_by_index)}/{len(selected_instances)} done)[/dim]"
            )
            if not args.allow_dirty_repo:
                dirty_after = _git_status_porcelain(repo_path)
                if dirty_after:
                    if args.clean_untracked:
                        clean_untracked(repo_path)
                        dirty_after = _git_status_porcelain(repo_path)
                    if not dirty_after:
                        save_report(
                            [results_by_index[i] for i in sorted(results_by_index)],
                            benchmark,
                            args,
                            repo_path,
                            out_path,
                            announce=False,
                        )
                        continue
                    save_report(
                        [results_by_index[i] for i in sorted(results_by_index)],
                        benchmark,
                        args,
                        repo_path,
                        out_path,
                        announce=True,
                    )
                    raise SystemExit(
                        "CLI left the target repo dirty. Report saved with completed cases.\n"
                        f"{dirty_after}\n"
                        "Use a disposable checkout or pass --allow-dirty-repo if you accept this."
                    )
            save_report(
                [results_by_index[i] for i in sorted(results_by_index)],
                benchmark,
                args,
                repo_path,
                out_path,
                announce=False,
            )
    finally:
        if original_head:
            try:
                checkout_commit(repo_path, original_head)
            except SystemExit as exc:
                console.print(f"[yellow]Could not restore repo HEAD: {exc}[/yellow]")

    ordered_results = [results_by_index[i] for i in sorted(results_by_index)]
    if ordered_results:
        console.print()
        print_results_table(ordered_results)
        save_report(ordered_results, benchmark, args, repo_path, out_path)
    else:
        console.print("[dim]No results to report.[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run file-localization evals against external coding CLIs")
    parser.add_argument("benchmark", type=Path, help="Path to benchmark JSON")
    parser.add_argument("--repo", type=Path, default=Path("evals/repos/ansible"), help="Path to checked-out repository")
    parser.add_argument("--out", type=Path, default=None, help="Output report path")
    parser.add_argument("--cli", choices=["codex", "gemini"], default="codex", help="CLI backend to run")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per case in seconds")
    parser.add_argument("--skip-commit-check", action="store_true", help="Skip checking that the benchmark fixed commit exists before setup")
    parser.add_argument("--allow-dirty-repo", action="store_true", help="Allow running even when the target repo has uncommitted changes")
    parser.add_argument("--clean-untracked", action="store_true", help="Run git clean -fd in the target repo before and between cases. Use only with disposable eval checkouts.")
    parser.add_argument("--resume", action="store_true", help="Load completed results from --out and skip those instances")
    parser.add_argument("--difficulty", choices=["easy", "hard"], default=None, help="Only run one difficulty")
    parser.add_argument("--instance-id", action="append", default=None, help="Only run the given instance ID; can be repeated")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected cases")
    parser.add_argument("--trace-content-max-chars", type=int, default=120000, help="Max prompt/answer chars retained in conversation_history")
    parser.add_argument("--schema", type=Path, default=Path("evals/file_prediction_schema.schema"), help="JSON schema used by CLIs that support structured output")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI binary")
    parser.add_argument("--codex-model", default="", help="Optional Codex model override, for example gpt-5.4-mini")
    parser.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="read-only", help="Codex sandbox mode")
    parser.add_argument("--gemini-bin", default="gemini", help="Gemini CLI binary")
    parser.add_argument("--gemini-model", default="", help="Optional Gemini model override")
    parser.add_argument("--gemini-sandbox", action="store_true", help="Enable Gemini CLI sandbox mode")
    parser.add_argument("--gemini-approval-mode", choices=["default", "auto_edit", "yolo"], default="default", help="Gemini CLI approval mode")
    main(parser.parse_args())
