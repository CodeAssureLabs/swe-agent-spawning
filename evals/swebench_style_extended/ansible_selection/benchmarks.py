from __future__ import annotations

import json
from pathlib import Path
import re

from .config import (
    SELECTED_PRS,
    WINDOWS,
    Candidate,
)
from .github import gh_graphql_pull, relevant_issues
from .schema import BenchmarkFile, BenchmarkInstance
from .utils import clean_section, git, is_source, is_swebench_like_gold, normalize_body


def problem_statement(pr: dict, issues: list[dict]) -> str:
    labels = {node["name"] for node in pr["labels"]["nodes"]}
    if "feature" in labels and "bug" not in labels:
        issue_type = "Feature Pull Request"
    elif "bug" in labels:
        issue_type = "Bugfix Pull Request"
    else:
        issue_type = "Pull Request"

    body = normalize_body(pr.get("body"))
    summary = clean_section(body, "SUMMARY") or clean_section(body, "Summary")
    if not summary:
        summary = body.split("\n\n", 1)[0] if body else pr["title"]
    if re.fullmatch(r"https?://\S+", summary.strip()) and issues:
        summary = f"See linked issue #{issues[0]['number']}: {issues[0]['title']}"
    if len(summary) > 1800:
        summary = summary[:1800].rsplit("\n", 1)[0].strip() + "\n\n[truncated]"

    parts = [f"# {pr['title']}", "", "## Summary", "", summary.strip(), "", "## Issue Type", "", issue_type]
    if issues:
        parts += ["", "## Linked Issues", ""]
        parts += [f"- #{issue['number']}: {issue['title']}" for issue in issues]
    return "\n".join(parts).strip()


def write_benchmark_jsons(rows: list[Candidate], repo: Path, out: Path) -> None:
    row_by_pr = {row.pr: row for row in rows}
    issue_cache: dict[int, dict | None] = {}
    window_bounds = {window: (since, until) for window, since, until in WINDOWS}

    for window, pr_numbers in SELECTED_PRS.items():
        instances: list[BenchmarkInstance] = []
        window_start, _ = window_bounds[window]
        fixed_commit = git(repo, ["rev-list", "--all", f"--before={window_start}T00:00:00Z", "--max-count=1"])
        for pr_number in pr_numbers:
            row = row_by_pr[pr_number]
            pr = gh_graphql_pull(pr_number)
            files = pr["files"]["nodes"]
            touched_files = [file["path"] for file in files]
            swebench_like_gold_files = [path for path in touched_files if is_swebench_like_gold(path)]
            source_files = [path for path in touched_files if is_source(path)]
            other = [file for file in files if file["changeType"] not in {"ADDED", "MODIFIED"}]
            if other:
                raise RuntimeError(f"PR #{pr_number} has unsupported file statuses: {other}")

            base_commit = git(repo, ["rev-parse", f"{row.sha}^"])
            issues = relevant_issues(pr, issue_cache)
            instance_id = f"instance_ansible__ansible-{row.sha}-v{base_commit}"
            problem = problem_statement(pr, issues)
            difficulty = "easy" if len(source_files) == 1 else "hard"
            instances.append(
                BenchmarkInstance(
                    instance_id=instance_id,
                    date=row.date,
                    base_commit=base_commit,
                    difficulty=difficulty,
                    gold_files=swebench_like_gold_files,
                    problem_statement=problem,
                )
            )

        stem = f"benchmark_ansible_{window.lower().replace('-', '_')}"
        target = out / f"{stem}.json"
        data = BenchmarkFile(fixed_commit=fixed_commit, window=window, instances=instances)
        target.write_text(json.dumps(data.to_json(), indent=2) + "\n")
