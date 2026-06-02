# Issue-Style Problem Statement Generation

This benchmark extension regenerates task metadata using one non-interactive Codex process per instance.

The parent script passes only a small instance payload: repository, PR number, known linked issues, base commit, and file-count metadata. It does not preload PR bodies, issue comments, or diffs.

Each Codex worker is instructed to gather public GitHub context itself using `gh`, web/search tools, or both. This keeps the parent prompt small and lets the worker decide which issue comments, PR comments, diffs, tests, or changelog evidence are relevant.

Generated benchmark fields are kept separate:

- `problem_statement`: the issue-style task text currently passed to coding agents.
- `requirements`: behavior-level acceptance criteria for audit or future prompt variants.
- `interface`: optional public API/CLI/config contract for audit or future prompt variants.

The exact output schema is stored in `generated_problem_statement.schema.json`.

The constant Codex generation instruction is embedded in `generate_ansible_problem_statements.py` as `GENERATOR_INSTRUCTION`. It is intentionally kept in code because it is the instruction passed to each worker.
