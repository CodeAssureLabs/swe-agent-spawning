# Long-Session Codebase Eval

This eval measures file localization on repeated issues from the Ansible
codebase. Each instance asks an agent to predict repo-relative files that would
need edits for a requested fix. The runner compares predictions with
`gold_files` and reports precision, recall, F1, all-gold-found rate, token use,
REPL use, and subagent-consult use.

## Benchmark Files

- `benchmark_ansible.json`: original SWE-bench-like Ansible subset.
- `benchmark_ansible_2025_stress.json`: SWE-bench-like stress window.
- `benchmark_ansible_2026_ytd.json`: SWE-bench-like modern control window.

All included benchmark files use the same SWE-bench-like gold convention.

## Target Repo Setup

```bash
mkdir -p evals/repos
git clone https://github.com/ansible/ansible evals/repos/ansible
```

The runners check out each instance's `base_commit`, so use a disposable or
clean Ansible checkout.

## Agent-Spawning Run

```bash
uv run python evals/run_eval.py \
  evals/benchmark_ansible.json \
  --repo evals/repos/ansible \
  --out evals/ansible_report_agents.json
```

The runner first initializes folder agents, snapshots the initialized registry,
then restores that same starting state before each instance.

## No-Subagent Ablation

```bash
uv run python evals/run_eval.py \
  evals/benchmark_ansible.json \
  --repo evals/repos/ansible \
  --no-subagents \
  --out evals/ansible_report_no_subagents.json
```

## External CLI Baseline

```bash
uv run python evals/run_cli_eval.py \
  evals/benchmark_ansible.json \
  --repo evals/repos/ansible \
  --cli codex \
  --limit 1 \
  --out evals/ansible_report_codex_smoke.json
```

The Codex runner uses `codex -a never exec`, read-only sandboxing, JSON event
output, and `file_prediction_schema.schema`.

## Plain LLM Baseline

```bash
uv run python evals/run_plain_llm.py \
  --benchmark evals/benchmark_ansible.json \
  --trial 1
```

This baseline uses the issue text only and does not inspect the repository.

## Single-Agent RLM Baseline

```bash
uv run python evals/run_rlm.py \
  --benchmark evals/benchmark_ansible.json \
  --repo evals/repos/ansible \
  --trial 1 \
  --timeout 3600
```

This baseline uses the RLM package with a single repository REPL context. It
defaults to the Ansible checkout at `evals/repos/ansible`; pass `--repo` to use
another checkout location.

To allow the root RLM to spawn one child RLM from the REPL via `rlm_query(...)`,
set `--max-depth 2`:

```bash
uv run python evals/run_rlm.py \
  --benchmark evals/benchmark_ansible.json \
  --repo evals/repos/ansible \
  --trial 1 \
  --timeout 7200 \
  --max-depth 2
```

RLM reports include inclusive per-instance token counts in `rlm_usage_summary`,
with `rlm_root_usage_summary` and `nested_rlm_usage_summary` kept separately so
child-RLM usage can be audited.

## Dataset Generation

The dataset-selection code lives under `evals/swebench_style_extended/`.

To regenerate benchmark JSONs, clone Ansible, refresh or provide a candidate
cache, then run:

```bash
uv run python -m evals.swebench_style_extended.ansible_selection.cli \
  --repo evals/repos/ansible \
  --write-benchmarks
```

Generated files are written under:

```text
evals/swebench_style_extended/artifacts/ansible_test_sets/
```

Raw baseline reports can be placed in `evals/results/raw/` if you want to
archive or post-process them. Generated reports are ignored by default.
