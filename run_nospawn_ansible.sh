#!/usr/bin/env bash
# Coordinator-only (no registry) on the Ansible windows: runs 2 and 3, 600 s timeout, per-issue checkout.
#   ./run_nospawn_ansible.sh <window 2020|2025|2026>
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
w=$1
case $w in 2020) B=evals/benchmark_ansible.json;; 2025) B=evals/benchmark_ansible_2025_stress.json;; 2026) B=evals/benchmark_ansible_2026_ytd.json;; esac
R=evals/repos/nospawn_$w/ansible__ansible
OUT=../new_runs/ansible_$w; LOG=../new_runs/logs; mkdir -p "$OUT" "$LOG"
for k in 2 3; do
  out=$OUT/ds_nospawn_$k.json
  if [[ -s $out ]]; then echo "skip $out"; continue; fi
  # No internet 22:30-07:30: do not start a run after 21:00 or before 07:35.
  while :; do hm=$((10#$(date +%H%M))); if (( hm >= 2100 || hm < 735 )); then sleep 300; else break; fi; done
  echo "start $out $(date '+%F %T')"
  LLM_MODEL=claude-haiku-4-5 uv run python evals/run_eval.py "$B" --repo "$R" --timeout 600 --no-subagents --discard-session \
    --out "$out.partial" > "$LOG/ansible_${w}_ds_nospawn_$k.log" 2>&1
  if python3 - "$out.partial" <<'PY'
import json, sys
try: rs = json.load(open(sys.argv[1]))["results"]
except Exception: sys.exit(1)
timeout = lambda r: "timed out" in str(r.get("answer", ""))
bad = [r for r in rs if not timeout(r) and (r.get("error") or r.get("status") == "error" or not r.get("total_tokens"))]
sys.exit(1 if len(rs) != 19 or bad else 0)
PY
  then mv "$out.partial" "$out"; echo "done $out $(date '+%F %T')"
  else mv "$out.partial" "$out.failed" 2>/dev/null; echo "FAIL $out"; fi
done
