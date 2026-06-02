from __future__ import annotations

import json
import re
from pathlib import Path

from .config import WINDOWS
from .utils import git, is_source, run


def refresh_candidates(repo: Path, out: Path) -> None:
    pr_re = re.compile(r"\(#(\d+)\)")
    rows: list[dict] = []

    for window, since, until in WINDOWS:
        log = git(
            repo,
            [
                "log",
                "origin/devel",
                f"--since={since}T00:00:00Z",
                f"--until={until}T23:59:59Z",
                "--pretty=format:%H%x09%cI%x09%s",
            ],
        )
        for line in log.splitlines():
            if not line:
                continue
            sha, commit_date, subject = line.split("\t", 2)
            match = pr_re.search(subject)
            if not match:
                continue
            name_status = git(repo, ["diff-tree", "--no-commit-id", "--name-status", "-r", sha])
            files: list[str] = []
            source_files: list[str] = []
            added = modified = other = 0
            for entry in name_status.splitlines():
                parts = entry.split("\t")
                status, path = parts[0], parts[-1]
                files.append(path)
                if status.startswith("A"):
                    added += 1
                elif status.startswith("M"):
                    modified += 1
                else:
                    other += 1
                if is_source(path):
                    source_files.append(path)
            rows.append(
                {
                    "window": window,
                    "sha": sha,
                    "date": commit_date[:10],
                    "pr": int(match.group(1)),
                    "subject": subject,
                    "file_count": len(files),
                    "source_count": len(source_files),
                    "source_files": source_files,
                    "added_count": added,
                    "modified_count": modified,
                    "other_status_count": other,
                }
            )

    labelled: dict[int, dict] = {}
    for _, since, until in WINDOWS:
        window_query = f"{since}..{until}"
        for label in ("bug", "feature"):
            query = f"repo:ansible/ansible is:pr is:merged merged:{window_query} label:{label}"
            output = run(
                [
                    "gh",
                    "api",
                    "-X",
                    "GET",
                    "search/issues",
                    "-f",
                    f"q={query}",
                    "-f",
                    "per_page=100",
                    "--paginate",
                    "--jq",
                    ".items[] | {number,title,html_url,labels:[.labels[].name],closed_at}",
                ]
            )
            for line in output.splitlines():
                item = json.loads(line)
                existing = labelled.setdefault(item["number"], item)
                existing["labels"] = sorted(set(existing["labels"]) | set(item["labels"]))

    merged: list[dict] = []
    for row in rows:
        meta = labelled.get(row["pr"])
        if not meta:
            continue
        merged.append(
            {
                **row,
                "labels": meta["labels"],
                "title": meta["title"],
                "url": meta["html_url"],
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2) + "\n")
