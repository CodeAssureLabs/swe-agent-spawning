#!/usr/bin/env python3
"""Regenerate Ansible benchmark task text with separate Codex workers.

This script intentionally does not preload PR bodies, issue comments, or diffs
into one large prompt. Each instance is handed to a fresh non-interactive Codex
process with the same generation instruction and a small per-instance payload.
The worker is expected to inspect GitHub directly with `gh` and/or web search,
manage its own context, and return structured JSON.

Default usage from this repository:

    uv run python evals/swebench_style_extended/generate_ansible_problem_statements.py

By default this reads the two selected Ansible benchmark JSONs from `evals/`
and writes regenerated copies under `evals/swebench_style_extended/artifacts/`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = REPO_ROOT
DEFAULT_CANDIDATE_CACHE = PACKAGE_DIR / "audit/ansible_test_sets/ansible_labelled_candidates_cache.json"
DEFAULT_INPUTS = [
    DEFAULT_SOURCE_DIR / "benchmark_ansible_2025_stress.json",
    DEFAULT_SOURCE_DIR / "benchmark_ansible_2026_ytd.json",
]
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_AUDIT_DIR = PACKAGE_DIR / "audit/problem_statement_generation"
OUTPUT_SCHEMA_PATH = PACKAGE_DIR / "docs/generated_problem_statement.schema.json"


GENERATOR_INSTRUCTION = """\
You are generating benchmark task metadata for an agentic coding evaluation.
Use only the concrete style and field rules below; do not rely on any external
benchmark style guide or named benchmark convention.

You must gather source context yourself. Use GitHub directly through available
tools such as:
- `gh pr view <PR_NUMBER> --repo ansible/ansible --json number,title,body,labels,closingIssuesReferences,comments,commits,files,url`
- `gh issue view <ISSUE_NUMBER> --repo ansible/ansible --json number,title,body,comments,labels,url,state`
- `gh pr diff <PR_NUMBER> --repo ansible/ansible --patch`
- web/search tools, if available, for public GitHub pages when `gh` is blocked.

Evidence priority:
1. Linked issue title/body/comments and PR discussion that describe the user
   problem, reproduction, observed behavior, expected behavior, or maintainer
   clarification.
2. PR body, PR title, commit messages, changelog fragments, and tests.
3. Reference diff only as evidence of externally visible behavior, not as an
   implementation recipe.

problem_statement field:
- Write Markdown that looks like a well-scoped GitHub issue, not a PR summary.
- Start with a single H1 title that names the user-visible problem or requested
  capability. Do not use the PR title verbatim if it only describes the patch.
- Use these sections when evidence supports them:
  - "## Problem" for the scenario and why current behavior is wrong or missing.
  - "## Actual Behavior" for what happens before the fix.
  - "## Expected Behavior" for what the user should observe after the fix.
  - "## Steps to Reproduce" only when concrete commands, playbook snippets,
    config snippets, module parameters, or workflows are available.
  - "## Additional Context" only for compatibility constraints, regression
    notes, or examples needed to understand the task.
- If evidence is sparse, omit unsupported sections instead of inventing details.
- Keep the statement self-contained enough that a human engineer could start
  investigating without reading the original PR.
- Write from the perspective of the unresolved task. Do not say "this PR fixes",
  "the patch changes", "the solution should modify", or similar.
- Do not include a "Requirements" or "Interface" section inside the
  problem_statement. Those belong only in the separate JSON fields below.
- Do not include internal source file paths, gold_files, line numbers, patch
  hunk descriptions, private helper names, or implementation strategy.
- Exact public names are allowed when they are part of the user-facing surface:
  CLI commands/options, config keys, module names/options, callback options,
  plugin names, documented return fields, public classes/functions, warning or
  error text when exact text is required by behavior.

requirements field:
- Store acceptance criteria here as a separate Markdown bullet list.
- Each bullet should describe externally observable behavior that must hold
  after the fix.
- Do not prescribe internal implementation, changed files, private functions, or
  code structure.
- Include edge cases, backwards compatibility, warnings/errors, formatting, or
  regression constraints only when supported by the evidence.
- If a criterion is inferred from tests or diff rather than stated in an issue
  or PR discussion, phrase it behaviorally and conservatively.

interface field:
- Use null if no explicit public interface contract is needed.
- Otherwise write a short Markdown bullet list of public surfaces that tests or
  users must be able to call or configure.
- Include signatures or exact option/config names only for public APIs, CLI
  options, module parameters, callback options, config keys, or documented data
  fields. Do not list private helper functions or source file paths.

General constraints:
- Do not invent facts that are not supported by the public PR, issue, comments,
  commits, tests, changelog, or diff context you inspect.
- If linked issues include placeholder references such as #1234, ignore them as
  evidence unless their content clearly matches the task.
- If the PR is implementation-oriented, infer the task from linked issues,
  comments, tests, changelog, and externally visible behavior.
- If the context is genuinely sparse, set confidence to "low" and keep both the
  problem_statement and requirements conservative.

Return exactly one JSON object with these keys:
- problem_statement: string
- requirements: string
- interface: string or null
- confidence: "high", "medium", or "low"
- source_notes: array of short strings explaining what evidence you relied on
"""


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "problem_statement": {"type": "string", "minLength": 200},
        "requirements": {"type": "string", "minLength": 80},
        "interface": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "source_notes": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
        },
    },
    "required": ["problem_statement", "requirements", "interface", "confidence", "source_notes"],
}


def run(args: list[str], *, input_text: str | None = None) -> str:
    proc = subprocess.run(
        args,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def commit_sha_from_instance_id(instance_id: str) -> str:
    match = re.match(r"instance_ansible__ansible-([0-9a-f]{40})-v[0-9a-f]{40}$", instance_id)
    if not match:
        raise ValueError(f"Cannot parse commit SHA from instance_id: {instance_id}")
    return match.group(1)


def load_candidate_lookup(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text())
    return {row["sha"]: row for row in rows}


def enrich_instance(instance: dict[str, Any], candidate_lookup: dict[str, dict[str, Any]], index: int) -> dict[str, Any]:
    if "relevant_pr" in instance:
        return instance
    sha = commit_sha_from_instance_id(instance["instance_id"])
    if sha not in candidate_lookup:
        raise KeyError(f"Commit SHA not found in candidate cache: {sha}")
    candidate = candidate_lookup[sha]
    enriched = dict(instance)
    enriched["human_instance_number"] = index
    enriched["relevant_pr"] = {
        "number": candidate["pr"],
        "title": candidate.get("title") or candidate.get("subject"),
        "url": candidate.get("url"),
    }
    enriched["relevant_issue_links"] = []
    enriched["source_file_count_for_difficulty_only"] = candidate.get("source_count")
    enriched["new_file_count"] = candidate.get("added_count")
    enriched["modified_file_count"] = candidate.get("modified_count")
    return enriched


def instance_payload(instance: dict[str, Any], benchmark_path: Path) -> dict[str, Any]:
    pr = instance["relevant_pr"]
    return {
        "repo": "ansible/ansible",
        "benchmark_file": str(benchmark_path),
        "instance_id": instance["instance_id"],
        "human_instance_number": instance.get("human_instance_number"),
        "date": instance.get("date"),
        "difficulty": instance.get("difficulty"),
        "base_commit": instance.get("base_commit"),
        "relevant_pr": {
            "number": pr["number"],
            "title": pr.get("title"),
            "url": pr.get("url"),
        },
        "known_relevant_issue_links": instance.get("relevant_issue_links", []),
        "changed_files_summary": {
            "num_files": instance.get("num_files"),
            "source_file_count_for_difficulty_only": instance.get("source_file_count_for_difficulty_only", instance.get("num_files")),
            "new_file_count": instance.get("new_file_count"),
            "modified_file_count": instance.get("modified_file_count"),
        },
    }


def prompt_for_instance(instance: dict[str, Any], benchmark_path: Path) -> str:
    payload = instance_payload(instance, benchmark_path)
    return (
        GENERATOR_INSTRUCTION
        + "\n\nInstance payload follows. Use it to identify the PR and benchmark instance, then gather public GitHub context yourself before writing the output JSON.\n\n"
        + json.dumps(payload, indent=2)
    )


def codex_command(schema_path: Path, output_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        "codex",
        "--search",
    ]
    if args.danger_full_access:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.extend(["--ask-for-approval", "never", "--sandbox", args.codex_sandbox])
    if args.model:
        command.extend(["--model", args.model])
    command.extend(
        [
        "exec",
        "--ephemeral",
        "--cd",
        str(REPO_ROOT),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
        ]
    )
    return command


def normalize_generation(generated: dict[str, Any]) -> dict[str, Any]:
    interface = generated.get("interface")
    if isinstance(interface, str):
        interface = interface.strip() or None
    return {
        "problem_statement": generated["problem_statement"].strip(),
        "requirements": generated["requirements"].strip(),
        "interface": interface,
        "problem_statement_generation": {
            "method": "codex_exec_per_instance_github_direct",
            "style": "issue_style_task_contract",
            "confidence": generated["confidence"],
            "source_notes": generated["source_notes"],
        },
    }


def process_benchmark(path: Path, args: argparse.Namespace) -> None:
    data = json.loads(path.read_text())
    candidate_lookup = load_candidate_lookup(args.candidate_cache)
    output_path = args.output_dir / path.name
    raw_dir = args.audit_dir / "codex_raw" / path.stem
    prompt_dir = args.audit_dir / "prompts" / path.stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    for index, instance in enumerate(data["instances"], 1):
        instance = enrich_instance(instance, candidate_lookup, index)
        pr_number = int(instance["relevant_pr"]["number"])
        raw_path = raw_dir / f"pr_{pr_number}.json"
        prompt_path = prompt_dir / f"pr_{pr_number}.txt"
        prompt = prompt_for_instance(instance, path)
        prompt_path.write_text(prompt)

        if args.dry_run:
            continue
        if raw_path.exists() and not args.force:
            generated = json.loads(raw_path.read_text())
        else:
            command = codex_command(OUTPUT_SCHEMA_PATH, raw_path, args)
            run(command, input_text=prompt)
            generated = json.loads(raw_path.read_text())

        instance.update(normalize_generation(generated))

    if not args.dry_run:
        note = (
            "problem_statement, requirements, and interface were regenerated by per-instance "
            "non-interactive Codex workers that gathered GitHub context directly."
        )
        if note not in data.setdefault("notes", []):
            data["notes"].append(note)
        output_path.write_text(json.dumps(data, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, dest="inputs", help="Benchmark JSON to regenerate. Can be passed multiple times.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--candidate-cache", type=Path, default=DEFAULT_CANDIDATE_CACHE)
    parser.add_argument("--model", help="Optional Codex model override.")
    parser.add_argument("--codex-sandbox", default="workspace-write", choices=["read-only", "workspace-write", "danger-full-access"])
    parser.add_argument(
        "--danger-full-access",
        action="store_true",
        help="Pass --dangerously-bypass-approvals-and-sandbox to Codex workers. Useful only if local sandboxing blocks gh network reads.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun Codex even when raw generation output already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts and method docs, but do not call Codex or write regenerated benchmark JSONs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing fixed output schema: {OUTPUT_SCHEMA_PATH}")

    inputs = args.inputs or DEFAULT_INPUTS
    missing = [path for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input benchmark(s): " + ", ".join(str(path) for path in missing))

    for path in inputs:
        process_benchmark(path, args)


if __name__ == "__main__":
    main()
