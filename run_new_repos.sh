#!/usr/bin/env bash
# Runs the LLM configurations on openlibrary and qutebrowser with the protocol in
# revision/RUNBOOK_new_repos.md. Resumable: a run whose report already exists is skipped.
#
#   ./run_new_repos.sh smoke        # 1 issue per repo for each Haiku method (cheap check)
#   ./run_new_repos.sh p1           # priority 1: plain/RLM/coordinator-only/domain agents, Haiku
#   ./run_new_repos.sh p2           # priority 2: plain/RLM Sonnet, Codex
#   ./run_new_repos.sh p1 openlibrary   # one repository only (run repos in parallel)
#
# Requires ANTHROPIC_API_KEY in .env (and a logged-in `codex` CLI for p2).
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
# Preflight: the agents' search_codebase tool shells out to ripgrep.
if ! uv run python -c "import sys; sys.path.insert(0,'src'); from agent.tools.search_tools import *; import subprocess; subprocess.run(['rg','--version'],check=True,capture_output=True)" 2>/dev/null; then
  echo "ripgrep (rg) not callable from the harness; aborting"; exit 1
fi
OUT=../new_runs
LOG=../new_runs/logs
mkdir -p "$LOG"
MODE=${1:-p1}
REPOS=(${2:-openlibrary qutebrowser})   # optional 2nd argument: one repository
RUNS=3
LIMIT=()
if [[ $MODE == smoke ]]; then RUNS=1; LIMIT=(--limit 1); OUT=../new_runs/smoke; fi

run() {  # run <outfile> <cmd...>
  local out=$1; shift
  if [[ -s $out ]]; then echo "skip  $out"; return; fi
  # No internet 23:00-07:30: do not start a run after 21:30 or before 07:35.
  while :; do
    hm=$((10#$(date +%H%M)))
    if (( hm >= 2130 || hm < 735 )); then sleep 300; else break; fi
  done
  mkdir -p "$(dirname "$out")"
  echo "start $out  $(date '+%F %T')"
  local tmp="$out.partial"
  "$@" --out "$tmp" > "$LOG/$(basename "$(dirname "$out")")_$(basename "$out" .json).log" 2>&1
  # Some runners record API failures as empty results; reject reports with errors or no tokens.
  if python3 - "$tmp" <<'PY'
import json, sys
try:
    rs = json.load(open(sys.argv[1]))["results"]
except Exception:
    sys.exit(1)
# Per-issue timeouts are legitimate outcomes (scored as empty predictions, as in the Ansible runs).
timeout = lambda r: "timed out" in str(r.get("answer", ""))
bad = [r for r in rs if not timeout(r) and (r.get("error") or r.get("status") == "error" or not r.get("total_tokens"))]
sys.exit(1 if not rs or bad else 0)
PY
  then mv "$tmp" "$out"; echo "done  $out  $(date '+%F %T')"
  else mv "$tmp" "$out.failed" 2>/dev/null; echo "FAIL  $out (errors or zero tokens; see log)"; fi
}

for repo in "${REPOS[@]}"; do
  B=evals/benchmark_$repo.json
  # Checkout folders follow the owner__repo convention: the harness strips the folder
  # name from predicted paths, which breaks repositories whose package shares its name.
  case $repo in openlibrary) R=evals/repos/internetarchive__openlibrary;; qutebrowser) R=evals/repos/qutebrowser__qutebrowser;; esac
  for i in $(seq 1 $RUNS); do
    if [[ $MODE == p1 || $MODE == smoke ]]; then
      run "$OUT/$repo/plain_haiku_$i.json" uv run python evals/run_plain_llm.py "$B" --model claude-haiku-4-5 ${LIMIT[@]+"${LIMIT[@]}"}
      run "$OUT/$repo/rlm_haiku_$i.json"   uv run python evals/run_rlm.py "$B" --repo "$R" --model claude-haiku-4-5 --timeout 3600 ${LIMIT[@]+"${LIMIT[@]}"}
      LLM_MODEL=claude-haiku-4-5 run "$OUT/$repo/ds_nospawn_$i.json" uv run python evals/run_eval.py "$B" --repo "$R" --timeout 600 --no-subagents --discard-session ${LIMIT[@]+"${LIMIT[@]}"}
      LLM_MODEL=claude-haiku-4-5 run "$OUT/$repo/ds_adaptive_$i.json" uv run python evals/run_eval.py "$B" --repo "$R" --timeout 600 ${LIMIT[@]+"${LIMIT[@]}"}
    fi
    if [[ $MODE == p2 ]]; then
      run "$OUT/$repo/plain_sonnet_$i.json" uv run python evals/run_plain_llm.py "$B" --model claude-sonnet-4-6
      run "$OUT/$repo/rlm_sonnet_$i.json"   uv run python evals/run_rlm.py "$B" --repo "$R" --model claude-sonnet-4-6 --timeout 3600
      run "$OUT/$repo/codex_$i.json"        uv run python evals/run_cli_eval.py "$B" --repo "$R" --cli codex --codex-model gpt-5.5 --codex-reasoning-effort high --timeout 600 --clean-untracked
    fi
  done
done
