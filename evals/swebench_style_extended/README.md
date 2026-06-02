# swebench_style_extended

This folder contains local benchmark-extension tooling for issue-style task
metadata. It is separate from the original extractor code because the prompt
generation policy is ours, not a direct dataset export.

## Ansible Test-Set Selection

`python -m evals.swebench_style_extended.ansible_selection.cli` regenerates the
selected Ansible benchmark windows.

Benchmark JSONs are written under
`evals/swebench_style_extended/artifacts/ansible_test_sets/`.

The selection pipeline is split into modules under
`evals/swebench_style_extended/ansible_selection/`:

- `config.py`: windows, source-file definition, size thresholds, and final curated PR lists.
- `candidates.py`: candidate-cache construction from git history and GitHub labels.
- `github.py`: PR/issue metadata helpers used while writing benchmark JSONs.
- `benchmarks.py`: SWE-bench-like benchmark JSON writing and lightweight PR-body problem statements.
- `cli.py`: command-line orchestration.

The selection rationale is a static document at
`evals/swebench_style_extended/docs/selection_methodology.md`. It explains the
candidate pool, date-stratified first pass, curated replacement rules, and
explicit exclusions, with each rule tied back to measured characteristics of
the official SWE-bench Pro-derived Ansible reference set where possible. The
script does not generate that document.

Load an existing candidate cache:

```bash
uv run python -m evals.swebench_style_extended.ansible_selection.cli
```

Refresh the candidate cache and regenerate benchmark JSONs from a local Ansible
checkout:

```bash
uv run python -m evals.swebench_style_extended.ansible_selection.cli \
  --repo evals/repos/ansible \
  --refresh-candidates \
  --write-benchmarks
```

If the local Ansible checkout under `evals/repos/ansible` is stale, pass a
checkout with the relevant `origin/devel` history:

```bash
uv run python -m evals.swebench_style_extended.ansible_selection.cli \
  --repo /path/to/ansible
```

## Ansible Problem Statement Regeneration

`generate_ansible_problem_statements.py` reads selected Ansible benchmark JSONs
and writes regenerated copies under `evals/swebench_style_extended/artifacts/`.

The script launches one non-interactive `codex exec` worker per instance. The
parent prompt stays small: it passes the repository, PR number, known issue
links, base commit, and file-count metadata. The worker must gather GitHub
context itself using `gh`, web/search tools, or both.

Generated fields stay separate:

- `problem_statement`: issue-style task text passed to coding agents.
- `requirements`: behavior-level acceptance criteria for audit or future prompt variants.
- `interface`: optional public API/CLI/config contract for audit or future prompt variants.

Run. On this machine, Codex workers need `--danger-full-access` for `gh` to reach GitHub:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python evals/swebench_style_extended/generate_ansible_problem_statements.py --danger-full-access
```

For a no-network/no-Codex preview that only writes the exact audit prompts:

```bash
uv run python evals/swebench_style_extended/generate_ansible_problem_statements.py --dry-run
```

The sandboxed variant is available, but it may fail if `gh` cannot reach GitHub
from the Codex worker sandbox:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python evals/swebench_style_extended/generate_ansible_problem_statements.py
```

The generation method is documented statically in
`evals/swebench_style_extended/docs/problem_statement_generation_method.md`; the
script does not generate that document.

Problem-statement generation audit prompts are written under
`evals/swebench_style_extended/audit/problem_statement_generation/`.
