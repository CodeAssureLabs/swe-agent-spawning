# SWE Agent Spawning

This repository is a cleaned reconstruction of the agent-spawning code used for
software-maintenance file-localization experiments.

The core implementation is a LangGraph codebase agent that can:

- inspect a repository through bounded file, search, parse, and Python REPL tools;
- maintain persistent notes across a session;
- spawn folder-scoped specialist agents with restricted repository access;
- consult those specialists through a persistent subagent registry; and
- record token and tool-use metrics during evaluation.

The release intentionally excludes old notebooks, generated paper artifacts,
session archives, local memory state, baseline-specific providers, and
exploratory eval folders that were not part of the focused agent-spawning
results.

## Layout

```text
src/
  main.py                         Interactive CLI
  agent/
    maintainer.py                 Main LangGraph orchestrator
    agent_builder.py              Folder-agent proposal and creation workflow
    subagent_registry.py          Persistent registry for spawned agents
    subagent_handle.py            Tool wrapper for consulting scoped agents
    tools/                        Repository, note, REPL, and subagent tools
    providers/                    Anthropic, OpenAI, OpenRouter, API-compatible providers

evals/
  run_eval.py                     In-process agent-spawning eval runner
  run_cli_eval.py                 External CLI baseline runner
  run_plain_llm.py                Plain LLM baseline without repository access
  run_rlm.py                      Single-agent RLM baseline
  benchmark_ansible*.json         Ansible file-localization benchmark variants
  swebench_style_extended/        Dataset selection and generation code
  results/                        Raw-report drop zone
```

## Setup

```bash
uv sync
cp .env.example .env
```

Set `LLM_MODEL` and the API key for the provider you plan to use.

## Interactive Use

```bash
uv run python src/main.py /path/to/repo
```

Inside the session, the orchestrator can call `build_agents_for(...)` to create
folder specialists and `consult_agents(...)` to ask them scoped questions.
Runtime memory is written under `src/agent/.memory/`, which is gitignored.

## Evaluation

The retained evaluation targets file localization on Ansible issues. See
`evals/README.md` for full setup and commands.

The shortest smoke test after cloning Ansible is:

```bash
uv run python evals/run_eval.py \
  evals/benchmark_ansible.json \
  --repo evals/repos/ansible \
  --limit 1 \
  --out evals/smoke_report.json
```

For a no-subagent ablation, add `--no-subagents`. For external CLI baselines,
use `run_cli_eval.py`.
