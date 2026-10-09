#!/usr/bin/env bash
# Seeding ablation: domain agents (adaptive, Haiku, 600 s) with Agentless file-level candidates
# shown to the coordinator. Seeded run k uses Agentless run k. 3 runs per benchmark.
#   ./run_seed_ablation.sh <name> <benchmark.json> <repo checkout>
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
name=$1; B=$2; R=$3
P=../seed_ablation
LOG=$P/logs; mkdir -p "$LOG" "$P/$name"
n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['instances']))" "$B")
for k in 1 2 3; do
  out=$P/$name/ds_seeded_$k.json
  if [[ -s $out ]]; then echo "skip $out"; continue; fi
  # No internet 22:30-07:30: do not start a run after 21:00 or before 07:35.
  while :; do hm=$((10#$(date +%H%M))); if (( hm >= 2100 || hm < 735 )); then sleep 300; else break; fi; done
  echo "start $out $(date '+%F %T')"
  LLM_MODEL=claude-haiku-4-5 uv run python evals/run_eval.py "$B" --repo "$R" --timeout 600 \
    --seed-files "$P/seeds_agentless_run$k.json" --out "$out.partial" > "$LOG/${name}_$k.log" 2>&1
  if python3 - "$out.partial" "$n" <<'PY'
import json, sys
try: rs = json.load(open(sys.argv[1]))["results"]
except Exception: sys.exit(1)
timeout = lambda r: "timed out" in str(r.get("answer", ""))
bad = [r for r in rs if not timeout(r) and (r.get("error") or r.get("status") == "error" or not r.get("total_tokens"))]
sys.exit(1 if len(rs) != int(sys.argv[2]) or bad else 0)
PY
  then mv "$out.partial" "$out"; echo "done $out $(date '+%F %T')"
  else mv "$out.partial" "$out.failed" 2>/dev/null; echo "FAIL $out"; fi
done
