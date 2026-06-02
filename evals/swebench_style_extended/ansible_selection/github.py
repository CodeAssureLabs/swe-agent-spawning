from __future__ import annotations

import json
import re
import subprocess

from .utils import run


def gh_json(args: list[str]) -> dict:
    return json.loads(run(["gh", "api", *args]))


def gh_graphql_pull(pr_number: int) -> dict:
    query = (
        'query($n:Int!){ repository(owner:"ansible", name:"ansible") { '
        "pullRequest(number:$n) { number title url mergedAt body "
        "labels(first:30){nodes{name}} "
        "closingIssuesReferences(first:10){nodes{number title url}} "
        "files(first:100){nodes{path changeType}} } } }"
    )
    data = json.loads(run(["gh", "api", "graphql", "-f", f"query={query}", "-F", f"n={pr_number}"]))
    return data["data"]["repository"]["pullRequest"]


def fetch_issue(issue_number: int, cache: dict[int, dict | None]) -> dict | None:
    if issue_number in cache:
        return cache[issue_number]
    try:
        data = gh_json([f"repos/ansible/ansible/issues/{issue_number}"])
    except subprocess.CalledProcessError:
        cache[issue_number] = None
        return None
    if "pull_request" in data:
        cache[issue_number] = None
    else:
        cache[issue_number] = {"number": data["number"], "title": data["title"], "url": data["html_url"]}
    return cache[issue_number]


def relevant_issues(pr: dict, cache: dict[int, dict | None]) -> list[dict]:
    found = {
        node["number"]: {"number": node["number"], "title": node["title"], "url": node["url"]}
        for node in pr["closingIssuesReferences"]["nodes"]
    }
    body = pr.get("body") or ""
    issue_numbers = {int(n) for n in re.findall(r"github\.com/ansible/ansible/issues/(\d+)", body)}
    issue_numbers.update(
        int(n)
        for n in re.findall(r"(?i)(?:fixes|fixed|closes|closed|resolves|resolved|see|issue)\s+#(\d+)", body)
    )
    for issue_number in issue_numbers:
        issue = fetch_issue(issue_number, cache)
        if issue:
            found[issue_number] = issue
    return [found[n] for n in sorted(found)]
