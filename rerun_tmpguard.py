"""Repeat selected issue runs under the guard with a private temporary directory: same condition,
same registry session, only the given instances (used for the issue runs that the access audit
found to have read the shared temporary directory).

  uv run python rerun_tmpguard.py JOBS.json      -> ../new_runs_r1/rerun_tmpguard/<window>_<cond>_<run>.json
"""
import json, os, subprocess, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from run_revision_r1 import HERE, OUT, ISO, WINDOWS, REGISTRY_CONDITIONS, command

jobs = defaultdict(lambda: {"iids": []})
for j in json.load(open(sys.argv[1])):
    key = (j["window"], j["cond"], j["run"])
    jobs[key]["iids"].append(j["iid"]); jobs[key]["sid"] = j["sid"]
dest = OUT / "rerun_tmpguard"; dest.mkdir(exist_ok=True)

def run(item):
    (window, cond, k), j = item
    bench, folder, history = WINDOWS[window]
    out = dest / f"{window}_{cond}_{k}.json"
    if out.exists():
        return f"skip {out.name}"
    repo = ISO / f"{cond}_{window}_{k}_tmpguard" / folder
    repo.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(cond, bench, str(repo), history, str(out), None, int(k))
    if cond in REGISTRY_CONDITIONS:
        cmd += ["--session-id", j["sid"], "--reuse-initialized-session"]
    for iid in j["iids"]:
        cmd += ["--instance-id", iid]
    env = {**os.environ, "LLM_MODEL": "claude-haiku-4-5", "AGENT_GUARD_MODE": "strict"}
    with open(dest / f"{out.stem}.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return f"{out.name} exit {rc}"

with ThreadPoolExecutor(8) as ex:
    for r in ex.map(run, jobs.items()):
        print(r, flush=True)
