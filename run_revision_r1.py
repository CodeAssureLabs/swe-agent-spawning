"""
Runs the IEEE Access revision (R1) conditions with the corrected harness.

Every job uses an isolated single-commit checkout (--history-repo) and the access guard.
Conditions (Claude Haiku 4.5):
  nospawn   coordinator only, all repository tools          (run_eval.py --no-subagents)
  adaptive  coordinator + domain-agent registry, adaptive     (run_eval.py)
  repl      coordinator only, python_repl as its only tool    (run_eval.py --no-subagents --tool-set repl)
  rlm       single-agent RLM, one REPL action per turn        (run_rlm.py, RLM_ONE_ACTION_PER_TURN=1)

  uv run python run_revision_r1.py smoke            # one issue per condition on Ansible 2025
  uv run python run_revision_r1.py run -j 10        # all jobs, 10 at a time (resumable)
  uv run python run_revision_r1.py run -j 10 --conditions nospawn repl --runs 1

Reports go to ../new_runs_r1/<window>/<condition>_<run>.json; a report with any error
(other than a per-issue timeout) or zero tokens is kept as .failed and the job is retried
on the next invocation. No job starts between 21:00 and 07:35 (network outage 22:30-07:30).

--guard-mode sets AGENT_GUARD_MODE for the agent processes (see src/agent/sandbox.py). The issue runs
reported in the IEEE Access resubmission used shared-tmp, except the configurations that consult domain agents
(adaptive, nudged, seeded; run with strict into ../new_runs_r2 via --out-root) and the issue runs repeated with
rerun_tmpguard.py; to reproduce the others, pass --guard-mode shared-tmp.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "new_runs_r1"
ISO = HERE / "evals" / "repos_isolated"

WINDOWS = {
    "ansible_2020": ("evals/benchmark_ansible.json", "ansible__ansible", "evals/repos/ansible_2020/ansible__ansible"),
    "ansible_2025": ("evals/benchmark_ansible_2025_stress.json", "ansible__ansible", "evals/repos/ansible_2020/ansible__ansible"),
    "ansible_2026": ("evals/benchmark_ansible_2026_ytd.json", "ansible__ansible", "evals/repos/ansible_2020/ansible__ansible"),
    "openlibrary": ("evals/benchmark_openlibrary.json", "internetarchive__openlibrary", "evals/repos/internetarchive__openlibrary"),
    "qutebrowser": ("evals/benchmark_qutebrowser.json", "qutebrowser__qutebrowser", "evals/repos/qutebrowser__qutebrowser"),
}
CONDITIONS = ["nospawn", "adaptive", "repl", "rlm"]
REFERENCES = ["rlm_sonnet", "codex"]  # larger-model references, rerun because the old runs could reach later history
GUIDANCE = ["nudged", "seeded", "maponly"]  # RQ3 variants and the registry-map ablation
IDLE_KILL_S = 75 * 60
print_lock = threading.Lock()
GUARD_MODE = "strict"


def log(msg):
    with print_lock:
        print(f"{datetime.now():%F %T}  {msg}", flush=True)


def command(cond, bench, repo, history, out, limit, run=1):
    lim = ["--limit", str(limit)] if limit else []
    if cond == "codex":
        return ["uv", "run", "python", "evals/run_cli_eval.py", bench, "--repo", repo, "--history-repo", history,
                "--cli", "codex", "--codex-model", "gpt-6.1-terra", "--codex-reasoning-effort", "high",  # CLI default model on 2026-11-18
                "--codex-sandbox", "read-only", "--timeout", "600", "--out", out, *lim]
    if cond == "rlm_sonnet":
        return ["uv", "run", "python", "evals/run_rlm.py", bench, "--repo", repo, "--history-repo", history,
                "--model", "claude-sonnet-4-6", "--timeout", "3600", "--out", out, *lim]
    if cond == "rlm":
        return ["uv", "run", "python", "evals/run_rlm.py", bench, "--repo", repo, "--history-repo", history,
                "--model", "claude-haiku-4-5", "--timeout", "3600", "--out", out, *lim]
    base = ["uv", "run", "python", "evals/run_eval.py", bench, "--repo", repo, "--history-repo", history,
            "--timeout", "600", "--out", out, *lim]
    if cond == "maponly":
        return base + ["--registry-map-only"]
    if cond == "nudged":
        return base + ["--consultation", "nudged"]
    if cond == "seeded":
        return base + ["--seed-files", str(HERE.parent / "seed_ablation" / f"seeds_agentless_run{run}.json")]
    if cond == "nospawn":
        return base + ["--no-subagents", "--discard-session"]
    if cond == "repl":
        return base + ["--no-subagents", "--tool-set", "repl", "--discard-session"]
    return base  # adaptive: keep the session so the registry is recorded


REGISTRY_CONDITIONS = {"adaptive", "maponly", "nudged", "seeded"}


def resume_args(cond, out, tmp):
    """Continue a stopped or failed job: keep its completed issues and, for the registry
    conditions, its registry (same session), so a run never mixes two registries."""
    prior = next((p for p in (out.with_suffix(".failed.json"), tmp) if p.exists()), None)
    if prior is None:
        return []
    try:
        d = json.load(open(prior))
    except Exception:
        return []
    if cond == "codex":
        done = [r for r in d.get("results", []) if r.get("cli_returncode") == 0 and r.get("total_tokens")]
    else:
        done = [r for r in d.get("results", []) if r.get("status") in ("ok", "completed", "timeout") and not r.get("error")]
    if not done:
        return []
    if cond in ("rlm", "rlm_sonnet"):
        if prior != tmp:
            shutil.copy(prior, tmp)
        return ["--rerun-failed"]
    if cond == "codex":
        if prior != tmp:
            shutil.copy(prior, tmp)
        return ["--resume"]
    keep = out.with_suffix(".resume.json")
    shutil.copy(prior, keep)
    args = ["--resume-from", str(keep)]
    if cond in REGISTRY_CONDITIONS:
        sid = next((r.get("session_id") for r in d["results"] if r.get("session_id")), None)
        if not sid:
            return []
        args += ["--session-id", sid, "--reuse-initialized-session"]
    return args


def report_ok(path, expected):
    try:
        rs = json.load(open(path))["results"]
    except Exception:
        return False, "unreadable report"
    if len(rs) != expected:
        return False, f"{len(rs)} of {expected} issues"
    timeout = lambda r: "timed out" in str(r.get("answer", "")) or r.get("status") == "timeout"
    bad = [r["instance_id"][-12:] for r in rs
           if not timeout(r) and (r.get("error") or r.get("status") == "error" or not r.get("total_tokens"))]
    return (not bad), (f"errors in {bad}" if bad else "ok")


def wait_for_window():
    while True:
        hm = int(datetime.now().strftime("%H%M"))
        if 735 <= hm < 2100:
            return
        time.sleep(300)


def run_job(job):
    cond, window, k, limit = job
    bench, folder, history = WINDOWS[window]
    out = OUT / window / f"{cond}_{k}.json"
    if out.exists() and out.stat().st_size:
        log(f"skip  {out.relative_to(OUT)}")
        return
    wait_for_window()
    expected = limit or len(json.load(open(HERE / bench))["instances"])
    repo = ISO / f"{cond}_{window}_{k}" / folder
    repo.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.json")
    logf = OUT / "logs" / f"{window}_{cond}_{k}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "LLM_MODEL": "claude-haiku-4-5", "AGENT_GUARD_MODE": GUARD_MODE}
    if cond in ("rlm", "rlm_sonnet"):
        env["RLM_ONE_ACTION_PER_TURN"] = "1"
        env["RLM_PROMPT_CACHE"] = "1"
    cmd = command(cond, bench, str(repo), history, str(tmp), limit, k)
    cmd += resume_args(cond, out, tmp)
    log(f"start {out.relative_to(OUT)}")
    with open(logf, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(30)
            # Hung API call: the log and partial report stop changing.
            last = max((p.stat().st_mtime for p in (logf, tmp) if p.exists()), default=time.time())
            logs_dir = tmp.with_name(tmp.stem + "_logs")
            if logs_dir.exists():
                last = max([last] + [p.stat().st_mtime for p in logs_dir.iterdir()])
            if time.time() - last > IDLE_KILL_S:
                log(f"kill  {out.relative_to(OUT)} (idle {IDLE_KILL_S}s)")
                proc.kill()
    ok, why = report_ok(tmp, expected)
    if proc.returncode == 0 and ok:
        tmp.rename(out)
        for suffix in ("_logs", "_events"):
            side = tmp.with_name(tmp.stem + suffix)
            if side.exists():
                side.rename(out.with_name(out.stem + suffix))
        log(f"done  {out.relative_to(OUT)}")
    else:
        if tmp.exists():
            tmp.rename(out.with_suffix(".failed.json"))
        log(f"FAIL  {out.relative_to(OUT)} (exit {proc.returncode}; {why}; see {logf.name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run"])
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS + REFERENCES + GUIDANCE)
    ap.add_argument("--windows", nargs="+", default=list(WINDOWS), choices=list(WINDOWS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--guard-mode", choices=["strict", "shared-tmp"], default="strict")
    ap.add_argument("--out-root", type=Path, default=None, help="report folder (default ../new_runs_r1)")
    a = ap.parse_args()
    global GUARD_MODE, OUT
    GUARD_MODE = a.guard_mode
    if a.out_root:
        OUT = a.out_root.resolve()
    if a.mode == "smoke":
        OUT = OUT / "smoke"
        jobs = [(c, "ansible_2025", 1, 1) for c in a.conditions]
    else:
        # Interleave so that every condition and window progresses together.
        jobs = [(c, w, k, None) for k in range(1, a.runs + 1) for w in a.windows for c in a.conditions]
    log(f"{len(jobs)} jobs, {a.jobs} at a time -> {OUT}")
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(run_job, jobs))
    log("all jobs finished")


if __name__ == "__main__":
    sys.exit(main())
