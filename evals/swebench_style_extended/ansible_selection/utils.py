from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Iterable

from .config import (
    BENCHMARK_SET_PRS,
    Candidate,
    EASY_TOTAL_FILE_CAP,
    HARD_TOTAL_FILE_CAP,
    LOW_SIGNAL_TITLE_FRAGMENTS,
    SOURCE_PREFIXES,
    SOURCE_SUFFIXES,
)


def run(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=str(cwd) if cwd else None, text=True)


def git(repo: Path, args: list[str]) -> str:
    return run(["git", "-C", str(repo), *args]).strip()


def is_source(path: str) -> bool:
    if path.startswith("bin/") and not path.endswith(".rst"):
        return True
    return path.startswith(SOURCE_PREFIXES) and path.endswith(SOURCE_SUFFIXES)


def is_swebench_like_gold(path: str) -> bool:
    if is_source(path):
        return True
    return path.startswith("docs/") or path.endswith(".rst")


def is_low_signal_title(title: str) -> bool:
    lowered = title.lower()
    return any(fragment in lowered for fragment in LOW_SIGNAL_TITLE_FRAGMENTS)


def source_difficulty(row: Candidate) -> str:
    if row.source_count == 1:
        return "easy"
    if row.source_count >= 2:
        return "hard"
    return "non_source"


def size_cap(row: Candidate) -> int:
    if row.source_count == 1:
        return EASY_TOTAL_FILE_CAP
    return HARD_TOTAL_FILE_CAP


def rejection_reason(row: Candidate) -> str:
    selected = row.pr in {pr for prs in BENCHMARK_SET_PRS.values() for pr in prs}
    if selected:
        return "selected"
    if row.source_count == 0:
        return "no_source_file"
    if row.other_status_count:
        return "rename_or_remove_status"
    if is_low_signal_title(row.title):
        return "low_signal_mechanical_title"
    if row.source_count == 1 and row.file_count > EASY_TOTAL_FILE_CAP:
        return "oversize_easy"
    if row.source_count >= 2 and row.file_count > HARD_TOTAL_FILE_CAP:
        return "oversize_hard"
    return "eligible_not_selected"


def eligible(row: Candidate, kind: str) -> bool:
    if row.other_status_count:
        return False
    if is_low_signal_title(row.title):
        return False
    if kind == "easy":
        return row.source_count == 1 and row.file_count <= EASY_TOTAL_FILE_CAP
    if kind == "hard":
        return row.source_count >= 2 and row.file_count <= HARD_TOTAL_FILE_CAP
    raise ValueError(kind)


def load_candidates(path: Path) -> list[Candidate]:
    rows = json.loads(path.read_text())
    return [
        Candidate(
            window=r["window"],
            sha=r["sha"],
            date=r["date"],
            pr=int(r["pr"]),
            subject=r["subject"],
            file_count=int(r["file_count"]),
            source_count=int(r["source_count"]),
            source_files=tuple(r["source_files"]),
            added_count=int(r["added_count"]),
            modified_count=int(r["modified_count"]),
            other_status_count=int(r["other_status_count"]),
            labels=tuple(r["labels"]),
            title=r["title"],
            url=r["url"],
        )
        for r in rows
    ]


def normalize_body(body: str | None) -> str:
    body = body or ""
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"\r\n?", "\n", body).strip()
    return re.sub(r"\n{3,}", "\n\n", body)


def clean_section(body: str, heading: str) -> str:
    pattern = re.compile(
        r"(?:^|\n)#{2,6}\s*" + re.escape(heading) + r"\s*\n(?P<section>.*?)(?=\n#{2,6}\s|\Z)",
        re.I | re.S,
    )
    match = pattern.search(body)
    return match.group("section").strip() if match else ""


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def rows_in_date_range(rows: list[Candidate], start: date, end: date) -> list[Candidate]:
    return [row for row in rows if start <= date.fromisoformat(row.date) <= end]


def diff_paths(repo: Path, sha: str) -> list[str]:
    output = git(repo, ["diff-tree", "--no-commit-id", "--name-only", "-r", sha])
    return [line for line in output.splitlines() if line]


def category_for(path: str) -> str:
    if is_source(path):
        return "source"
    if path.startswith("test/"):
        return "tests"
    if path.startswith("docs/"):
        return "docs"
    if path.startswith("changelogs/"):
        return "changelog"
    if path.startswith(".github/") or path.startswith("packaging/"):
        return "ci_packaging"
    if path.startswith("test/sanity/") or path.startswith("test/lib/ansible_test/"):
        return "test_infra"
    return "other"


def dirs_for(paths: Iterable[str]) -> set[str]:
    dirs: set[str] = set()
    for path in paths:
        parts = Path(path).parts[:-1]
        for index in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:index]))
    return dirs


def tree_paths(repo: Path, commit: str) -> set[str]:
    output = git(repo, ["ls-tree", "-r", "--name-only", commit])
    return {line for line in output.splitlines() if line}


def snapshot_commit(repo: Path, date_value: str) -> str:
    return git(repo, ["rev-list", "-1", f"--before={date_value}T23:59:59Z", "origin/devel"])


def selected_rows(rows: list[Candidate], window: str) -> list[Candidate]:
    row_by_pr = {row.pr: row for row in rows}
    return [row_by_pr[pr] for pr in BENCHMARK_SET_PRS[window] if pr in row_by_pr]


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def month_key(value: str) -> str:
    return value[:7]
