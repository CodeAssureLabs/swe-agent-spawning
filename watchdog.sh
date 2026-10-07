#!/usr/bin/env bash
# Kills an RLM runner whose newest trace file has not changed for 75 minutes
# (hung API call; the runner's own per-issue timeout does not fire in that case).
# Killed runs leave no report; rerun them with run_rlm.py --rerun-failed.
cd "$(dirname "$0")/../new_runs"
while pgrep -f "run_new_repos|run_rlm.py" >/dev/null; do
  for pid in $(pgrep -f "MacOS/Python evals/run_rlm.py"); do
    out=$(ps -o command= -p "$pid" | sed -E 's/.*--out ([^ ]+).*/\1/')
    logdir="$(dirname "$out")/$(basename "$out" .partial)_logs"
    [ -d "../harness/$logdir" ] && logdir="../harness/$logdir"
    newest=$(ls -t "$logdir" 2>/dev/null | head -1)
    [ -z "$newest" ] && continue
    age=$(( $(date +%s) - $(stat -f %m "$logdir/$newest") ))
    if (( age > 4500 )); then echo "$(date '+%F %T') killing $pid ($out), idle ${age}s"; kill "$pid"; fi
  done
  sleep 120
done
