"""
RLM File Localization

Usage:
    uv run evals/run_rlm.py evals/benchmark_ansible.json
    uv run evals/run_rlm.py evals/benchmark_ansible_2025_stress.json
    uv run evals/run_rlm.py evals/benchmark_ansible.json --repo /path/to/ansible
    uv run evals/run_rlm.py evals/benchmark_ansible.json --timeout 7200
    uv run evals/run_rlm.py evals/benchmark_ansible.json --max-depth 2
"""

import os, json, time, argparse, re, ast, subprocess
from datetime import datetime
from pathlib import Path

from rlm import RLM
from rlm.utils.prompts import RLM_SYSTEM_PROMPT
try:
    from evals.rlm_runtime_patches import (
        CompactRLMLogger,
        UsageAccumulator,
        capture_rlm_usage,
        drop_repl_locals,
    )
except ModuleNotFoundError:
    from rlm_runtime_patches import (
        CompactRLMLogger,
        UsageAccumulator,
        capture_rlm_usage,
        drop_repl_locals,
    )
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

MODEL     = "claude-haiku-4-5"
DEFAULT_REPO_PATH = "repos/ansible"
DEFAULT_TIMEOUT = 3600
DEFAULT_MAX_DEPTH = 1

PROMPT_TASK = (
    "Identify the repo-relative files that would need to be modified to implement the requested fix. "
    "Include code, test, and documentation files when they would need edits. Do not edit files. "
    "Use the repository tools to inspect the codebase. Prefer exact existing file paths."
)
PROMPT_OUTPUT = (
    "Your final answer must be a Python list of file paths:\n"
    'FINAL(["path/to/file.py", "docs/docsite/rst/guide.rst"])\n'
    "Use exact repo-relative paths. No leading ./ or absolute paths."
)


def build_system_prompt(repo_path):
    prompt_repl = (
        "\nYou have access to the repository via the REPL.\n"
        "Use this exact XML form for every REPL action:\n"
        "<invoke name=\"repl\">\n"
        "<parameter name=\"code\">\n"
        "print(\"example\")\n"
        "</parameter>\n"
        "</invoke>\n"
        "This runner accepts the XML form above; follow it even if earlier generic instructions mention ```repl blocks.\n"
        "Do not nest <invoke> tags, do not use <invoke name=\"call\">, and do not include </repl>.\n"
        "Prefer one XML REPL action per assistant message when you need to inspect REPL output before deciding "
        "the next step. If you already have enough evidence, you may include FINAL(...) in the same assistant "
        "message after XML REPL actions; the runner executes REPL code blocks in order before honoring FINAL(...).\n"
        "IMPORTANT: Your very first REPL action must run this exact setup code inside the XML code parameter:\n"
        "import os, subprocess\n"
        f"repo_path = {repo_path!r}\n"
        "Then use repo_path as the base path for all file operations.\n"
        "To submit, set answer[\"content\"] to the Python list of files and answer[\"ready\"] = True "
        "inside the same XML form.\n"
    )
    return RLM_SYSTEM_PROMPT + "\n\n" + PROMPT_TASK + prompt_repl + "\n" + PROMPT_OUTPUT

DOC_PREFIXES = ("docs/", "changelogs/")


def is_doc(path):
    return any(path.startswith(p) for p in DOC_PREFIXES)


def score_set(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p  = tp / len(pred_set) if pred_set else 0.0
    r  = tp / len(gold_set) if gold_set else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "exact_match": pred_set == gold_set,
        "all_gold_found": gold_set.issubset(pred_set),
        "missing_files": sorted(gold_set - pred_set),
        "extra_files":   sorted(pred_set - gold_set),
    }


def empty_usage_summary():
    return {"model_usage_summaries": {}}


def merge_usage_summaries(*summaries):
    merged = empty_usage_summary()
    for summary in summaries:
        if not summary:
            continue
        for model, usage in summary.get("model_usage_summaries", {}).items():
            acc = merged["model_usage_summaries"].setdefault(
                model,
                {"total_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0},
            )
            acc["total_calls"] += usage.get("total_calls", 0) or 0
            acc["total_input_tokens"] += usage.get("total_input_tokens", 0) or 0
            acc["total_output_tokens"] += usage.get("total_output_tokens", 0) or 0
    return merged


def usage_totals(summary):
    model_summaries = summary.get("model_usage_summaries", {})
    input_tokens = sum(v.get("total_input_tokens", 0) or 0 for v in model_summaries.values())
    output_tokens = sum(v.get("total_output_tokens", 0) or 0 for v in model_summaries.values())
    calls = sum(v.get("total_calls", 0) or 0 for v in model_summaries.values())
    return input_tokens, output_tokens, calls


def nested_rlm_usage_from_trajectory(trajectory):
    """Usage from child RLM completions spawned via rlm_query.

    Plain llm_query calls are already counted by the parent LM handler and have
    no child metadata, so counting only metadata-bearing calls avoids double
    counting the common max_depth=1 fallback path.
    """
    nested = empty_usage_summary()
    for iteration in (trajectory or {}).get("iterations", []):
        for code_block in iteration.get("code_blocks", []):
            result = code_block.get("result") or {}
            for call in result.get("rlm_calls", []):
                metadata = call.get("metadata")
                if metadata is None:
                    continue
                nested = merge_usage_summaries(
                    nested,
                    call.get("usage_summary"),
                    nested_rlm_usage_from_trajectory(metadata),
                )
    return nested


def extract_files(response):
    matches = re.findall(r'\[([^\[\]]+)\]', response)
    for match in reversed(matches):
        try:
            items = ast.literal_eval(f'[{match}]')
            if all(isinstance(i, str) for i in items):
                cleaned = []
                for path in items:
                    path = path.strip()
                    if path.startswith('./'): path = path[2:]
                    if 'ansible__ansible/' in path:
                        path = path.split('ansible__ansible/')[-1]
                    cleaned.append(path)
                return cleaned
        except Exception:
            continue
    py_files = re.findall(r'[\w/.-]+\.(?:py|yml|yaml|rst|cs|js)\b', response)
    return list(dict.fromkeys(py_files)) if py_files else []


def build_summary(results, elapsed_total):
    n = len(results)
    if n == 0:
        return {}
    error_cases = sum(1 for r in results if r.get("status") == "error")
    tp_total   = sum(r['tp'] for r in results)
    pred_total = sum(r['tp'] + r['fp'] for r in results)
    gold_total = sum(r['tp'] + r['fn'] for r in results)
    mp = tp_total / pred_total if pred_total else 0.0
    mr = tp_total / gold_total if gold_total else 0.0
    micro_f1 = 2*mp*mr/(mp+mr) if (mp+mr) else 0.0
    macro_p  = sum(r['precision'] for r in results) / n
    macro_r  = sum(r['recall']    for r in results) / n
    macro_f1 = sum(r['f1']        for r in results) / n
    exact_matches = sum(1 for r in results if r['exact_match'])
    all_gold      = sum(1 for r in results if r['all_gold_found'])
    total_tok = sum(r['total_tokens']  for r in results)
    total_in  = sum(r['input_tokens']  for r in results)
    total_out = sum(r['output_tokens'] for r in results)
    total_llm_calls = sum(r.get('llm_calls', 0) for r in results)
    total_repl_calls = sum(r.get('repl_calls', 0) for r in results)
    total_turns = sum(r.get('turn_count', len(r.get('conversation_history', []))) for r in results)
    avg_pred  = sum(len(r['predicted_files']) for r in results) / n
    avg_gold  = sum(r['num_files'] for r in results) / n
    return {
        "total":               n,
        "completed":           n - error_cases,
        "error_cases":         error_cases,
        "exact_matches":       exact_matches,
        "exact_match_rate":    round(exact_matches / n, 4),
        "all_gold_found":      all_gold,
        "all_gold_found_rate": round(all_gold / n, 4),
        "macro_precision":     round(macro_p,  4),
        "macro_recall":        round(macro_r,  4),
        "macro_f1":            round(macro_f1, 4),
        "micro_precision":     round(mp,       4),
        "micro_recall":        round(mr,       4),
        "micro_f1":            round(micro_f1, 4),
        "avg_predicted_files": round(avg_pred, 2),
        "avg_gold_files":      round(avg_gold, 2),
        "total_tokens":        total_tok,
        "total_input_tokens":  total_in,
        "total_output_tokens": total_out,
        "total_llm_calls":     total_llm_calls,
        "total_repl_calls":    total_repl_calls,
        "total_turns":         total_turns,
        "total_elapsed_s":     round(elapsed_total, 2),
    }


def build_groups(results):
    groups = {}
    splits = {
        "difficulty:easy": lambda r: r['difficulty'] == 'easy',
        "difficulty:hard": lambda r: r['difficulty'] == 'hard',
        "gold_docs:any":   lambda r: len(r['gold_doc_files']) > 0,
        "gold_docs:none":  lambda r: len(r['gold_doc_files']) == 0,
    }
    for name, fn in splits.items():
        subset = [r for r in results if fn(r)]
        if not subset:
            continue
        n = len(subset)
        tp_t = sum(r['tp'] for r in subset)
        pr_t = sum(r['tp'] + r['fp'] for r in subset)
        go_t = sum(r['tp'] + r['fn'] for r in subset)
        mp = tp_t/pr_t if pr_t else 0.0
        mr = tp_t/go_t if go_t else 0.0
        mf = 2*mp*mr/(mp+mr) if (mp+mr) else 0.0
        ag = sum(1 for r in subset if r['all_gold_found'])
        groups[name] = {
            "total": n,
            "exact_matches": sum(1 for r in subset if r['exact_match']),
            "all_gold_found": ag,
            "all_gold_found_rate": round(ag/n, 4),
            "macro_f1": round(sum(r['f1'] for r in subset)/n, 4),
            "micro_precision": round(mp, 4),
            "micro_recall":    round(mr, 4),
            "micro_f1":        round(mf, 4),
        }
    return groups


def make_result_entry(
    *,
    idx,
    inst,
    model,
    base_commit,
    response,
    predicted,
    elapsed,
    total_in,
    total_out,
    llm_calls,
    repl_calls,
    usage_by_model,
    root_usage_summary,
    nested_usage_summary,
    trajectory,
    trace_path,
    status,
    error=None,
):
    pred_set = set(predicted)
    gold_set = set(inst['gold_files'])
    gold_docs = set(f for f in gold_set if is_doc(f))
    file_score = score_set(pred_set, gold_set)

    iterations = (trajectory or {}).get("iterations", []) or []
    trajectory_code_blocks = [
        code_block
        for iteration in iterations
        for code_block in (iteration.get("code_blocks", []) or [])
    ]
    final_answer_repl_calls = sum(
        1
        for code_block in trajectory_code_blocks
        if (code_block.get("code") or "").lstrip().startswith('answer["content"]')
    )
    trajectory_repl_calls = len(trajectory_code_blocks)
    reported_repl_calls = repl_calls or 0
    repl_calls = max(reported_repl_calls, trajectory_repl_calls)
    conversation_history = []
    if iterations:
        final_iteration = iterations[-1]
        last_prompt = final_iteration.get("prompt")
        if isinstance(last_prompt, list):
            conversation_history = [
                {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
                for message in last_prompt
                if isinstance(message, dict)
            ]
        elif isinstance(last_prompt, str):
            conversation_history = [{"role": "user", "content": last_prompt}]

        if final_iteration.get("response"):
            conversation_history.append({"role": "assistant", "content": str(final_iteration["response"])})

        repl_outputs = []
        code_blocks = final_iteration.get("code_blocks", []) or []
        multi = len(code_blocks) > 1
        for index, code_block in enumerate(code_blocks, 1):
            result = code_block.get("result") or {}
            result_parts = []
            if result.get("stdout"):
                result_parts.append(f"\n{result['stdout']}")
            if result.get("stderr"):
                result_parts.append(f"\n{result['stderr']}")

            formatted = "\n\n".join(result_parts) if result_parts else "No output"
            max_character_length = 20000
            if len(formatted) > max_character_length:
                formatted = formatted[:max_character_length] + f"... + [{len(formatted) - max_character_length} chars...]"
            header = f"REPL output (block {index}):" if multi else "REPL output:"
            repl_outputs.append(f"{header}\n{formatted}")

        if repl_outputs:
            conversation_history.append({"role": "user", "content": "\n\n".join(repl_outputs)})

    return {
        "index":             idx,
        "instance_id":       inst['instance_id'],
        "date":              inst.get('date', ''),
        "difficulty":        inst['difficulty'],
        "num_files":         len(inst['gold_files']),
        "problem_statement": inst['problem_statement'],
        "gold_files":        inst['gold_files'],
        "gold_doc_files":    sorted(gold_docs),
        "predicted_files":   predicted,
        "base_commit":       base_commit,
        "actual_commit":     base_commit,
        "answer":            response,
        "file_score":        file_score,
        "tp":                file_score['tp'],
        "fp":                file_score['fp'],
        "fn":                file_score['fn'],
        "precision":         file_score['precision'],
        "recall":            file_score['recall'],
        "f1":                file_score['f1'],
        "exact_match":       file_score['exact_match'],
        "all_gold_found":    file_score['all_gold_found'],
        "missing_files":     file_score['missing_files'],
        "extra_files":       file_score['extra_files'],
        "elapsed_s":         round(elapsed, 2),
        "total_tokens":      total_in + total_out,
        "input_tokens":      total_in,
        "output_tokens":     total_out,
        "llm_calls":         llm_calls,
        "rlm_usage_summary":  usage_by_model,
        "rlm_root_usage_summary": root_usage_summary,
        "nested_rlm_usage_summary": nested_usage_summary,
        "repl_calls":        repl_calls,
        "reported_repl_calls": reported_repl_calls,
        "trajectory_repl_calls": trajectory_repl_calls,
        "final_answer_repl_calls": final_answer_repl_calls,
        "exploration_repl_calls": max(0, trajectory_repl_calls - final_answer_repl_calls),
        "turn_count":        len((trajectory or {}).get("iterations", []) or []),
        "conversation_history_message_count": len(conversation_history),
        "conversation_history": conversation_history,
        "rlm_trajectory":     trajectory,
        "trace_path":         trace_path,
        "trace_format":       "rlm_logger_jsonl_embedded",
        "any_correct":       file_score['tp'] > 0,
        "model":             model,
        "pred_files":        predicted,
        "status":            status,
        "error":             error,
    }


def run(
    model=MODEL,
    benchmark_path="evals/benchmark_ansible.json",
    repo_path=DEFAULT_REPO_PATH,
    timeout=DEFAULT_TIMEOUT,
    max_depth=DEFAULT_MAX_DEPTH,
    limit=None,
    rerun_failed_from=None,
    log_dir=None,
    out_path=None,
):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY in .env")
    if not os.path.exists(repo_path):
        raise ValueError(f"Repo not found at {repo_path}")

    benchmark_path = Path(benchmark_path).resolve()
    repo_path = str(Path(repo_path).resolve())
    with benchmark_path.open() as f:
        bench = json.load(f)

    instances    = bench['instances']
    selected_instances = list(enumerate(instances))
    rerun_source = None
    rerun_failed_ids = set()
    if rerun_failed_from:
        rerun_source = os.path.abspath(rerun_failed_from)
        if not os.path.exists(rerun_source):
            raise ValueError(f"Previous result path not found: {rerun_source}")
        if rerun_source.endswith(".jsonl"):
            previous_results = []
            with open(rerun_source) as f:
                for line in f:
                    if line.strip():
                        previous_results.append(json.loads(line))
        else:
            with open(rerun_source) as f:
                previous_data = json.load(f)
            if isinstance(previous_data, list):
                previous_results = previous_data
            elif isinstance(previous_data, dict):
                previous_results = previous_data.get("results", [])
            else:
                raise ValueError(f"Unsupported previous result format: {rerun_source}")

        previous_by_instance_id = {}
        previous_by_index = {}
        for result in previous_results:
            instance_id = result.get("instance_id")
            index = result.get("index")
            if instance_id:
                previous_by_instance_id[instance_id] = result
            if isinstance(index, int):
                previous_by_index[index] = result

        for idx, inst in selected_instances:
            previous = previous_by_instance_id.get(inst.get("instance_id"), previous_by_index.get(idx))
            failed = (
                previous is None
                or previous.get("status") == "error"
                or previous.get("error")
                or str(previous.get("answer", "")).startswith("|ERROR|")
            )
            if failed:
                rerun_failed_ids.add(inst.get("instance_id"))
        selected_instances = [
            (idx, inst) for idx, inst in selected_instances
            if inst.get("instance_id") in rerun_failed_ids
        ]
        if not selected_instances:
            print(f"No failed or missing instances found in {rerun_source}.")
            return
    if limit is not None:
        selected_instances = selected_instances[:limit]

    # Top-level fixed_commit is optional; new benchmarks may use per-instance base_commit.
    bench_commit = bench.get('fixed_commit', 'HEAD')
    out_path_was_defaulted = out_path is None
    if out_path:
        out_path = Path(out_path).resolve()
    else:
        report_suffix = "_rlm_failed_rerun_report.json" if rerun_failed_from else "_rlm_report.json"
        out_path = benchmark_path.with_name(benchmark_path.stem + report_suffix)
    if rerun_failed_from and out_path_was_defaulted:
        suffix = 2
        base_out_path = out_path
        while out_path.exists():
            out_path = base_out_path.with_name(f"{base_out_path.stem}_{suffix}{base_out_path.suffix}")
            suffix += 1
    if log_dir is None:
        log_dir = out_path.with_name(out_path.stem + "_logs")
    log_dir = str(log_dir)
    suffix = 2
    base_log_dir = log_dir
    while os.path.exists(log_dir):
        log_dir = f"{base_log_dir}_{suffix}"
        suffix += 1
    os.makedirs(log_dir, exist_ok=True)
    index_path   = os.path.join(log_dir, "index.jsonl")
    open(index_path, "w").close()

    print(f"\n{'='*65}")
    print(f"RLM File Localization (final) | model={model}")
    print(f"Benchmark: {benchmark_path}  ({len(selected_instances)} selected / {len(instances)} total instances)")
    print(f"Repo: {repo_path}")
    print(f"Timeout: {timeout}s  |  Max depth: {max_depth}  |  Logs -> {log_dir}")
    print(f"Report -> {out_path}")
    if rerun_failed_from:
        print(f"Failed-only rerun from: {rerun_source}  |  failed/missing found: {len(rerun_failed_ids)}")
    print(f"{'='*65}")

    all_results = []
    t_start = time.perf_counter()

    for display_idx, (idx, inst) in enumerate(selected_instances, 1):
        # Use instance-level base_commit if present, else fall back to benchmark-level
        base_commit = inst.get('base_commit', bench_commit)

        print(f"\n[{display_idx}/{len(selected_instances)}] [{inst['difficulty'].upper()}] "
              f"{inst['instance_id'][:50]}...")
        print(f"  Commit: {base_commit[:12]}  Q: {inst['problem_statement'][:80]}...")
        print(f"  Gold: {inst['gold_files']}")

        # Checkout the repo to the correct commit for this instance
        checkout = subprocess.run(
            ['git', '-C', repo_path, 'checkout', base_commit],
            capture_output=True, text=True
        )
        if checkout.returncode != 0:
            print(f"  WARN: git checkout failed: {checkout.stderr.strip()}")

        logger = CompactRLMLogger(log_dir=log_dir,
                                  file_name=f"sample_{idx:03d}_{inst['difficulty']}")

        prompt_context = (
            f"import os, subprocess\n"
            f"repo_path = {repr(repo_path)}\n"
            f"print('Ready. repo_path =', repo_path)\n\n"
            f"# Commit: {base_commit}\n"
            f"# Use repo_path as the base path for all file operations."
        )

        rlm = RLM(
            backend="anthropic",
            backend_kwargs={"api_key": api_key, "model_name": model},
            logger=logger,
            custom_system_prompt=build_system_prompt(repo_path),
            max_depth=max_depth,
            max_timeout=timeout,
            verbose=False,
        )

        t0 = time.perf_counter()
        usage_accumulator = UsageAccumulator()
        try:
            with capture_rlm_usage(usage_accumulator):
                result = rlm.completion(prompt=prompt_context,
                                        root_prompt=inst['problem_statement'])
            response  = result.response
            root_usage_summary = result.usage_summary.to_dict()
            trajectory = result.metadata or logger.get_trajectory() or {"run_metadata": None, "iterations": []}
            drop_repl_locals(trajectory)
            nested_usage_summary = nested_rlm_usage_from_trajectory(trajectory)
            usage_by_model = merge_usage_summaries(root_usage_summary, nested_usage_summary)
            total_in, total_out, llm_calls = usage_totals(usage_by_model)
            total_tok = total_in + total_out
            repl_calls = getattr(result.usage_summary, "total_repl_calls", 0) or 0
        except Exception as e:
            print(f"  ERROR: {e}")
            elapsed = time.perf_counter() - t0
            trajectory = logger.get_trajectory() or {"run_metadata": None, "iterations": []}
            drop_repl_locals(trajectory)
            usage_by_model = usage_accumulator.to_usage_summary()
            total_in, total_out, llm_calls = usage_totals(usage_by_model)
            response = f"|ERROR| {e}"
            entry = make_result_entry(
                idx=idx,
                inst=inst,
                model=model,
                base_commit=base_commit,
                response=response,
                predicted=[],
                elapsed=elapsed,
                total_in=total_in,
                total_out=total_out,
                llm_calls=llm_calls,
                repl_calls=0,
                usage_by_model=usage_by_model,
                root_usage_summary=usage_by_model,
                nested_usage_summary=empty_usage_summary(),
                trajectory=trajectory,
                trace_path=logger.log_file_path,
                status="error",
                error=str(e),
            )
            all_results.append(entry)
            with open(index_path, "a") as f:
                f.write(json.dumps(entry) + '\n')
            continue
        elapsed = time.perf_counter() - t0

        predicted       = extract_files(response)
        entry = make_result_entry(
            idx=idx,
            inst=inst,
            model=model,
            base_commit=base_commit,
            response=response,
            predicted=predicted,
            elapsed=elapsed,
            total_in=total_in,
            total_out=total_out,
            llm_calls=llm_calls,
            repl_calls=repl_calls,
            usage_by_model=usage_by_model,
            root_usage_summary=root_usage_summary,
            nested_usage_summary=nested_usage_summary,
            trajectory=trajectory,
            trace_path=logger.log_file_path,
            status="completed",
            error=None,
        )
        file_score = entry["file_score"]

        print(f"  Predicted: {predicted}")
        print(f"  P={file_score['precision']:.2f} R={file_score['recall']:.2f}"
              f"  F1={file_score['f1']:.2f} | all_gold={file_score['all_gold_found']}")
        print(f"  Missing: {file_score['missing_files']}")
        print(f"  Tokens: {total_tok:,}  Time: {elapsed:.1f}s")

        all_results.append(entry)

        with open(index_path, "a") as f:
            f.write(json.dumps(entry) + '\n')

    elapsed_total = time.perf_counter() - t_start

    report = {
        "run_at":    datetime.now().isoformat(),
        "benchmark": {
            "repo":            "ansible/ansible",
            "fixed_commit":    bench_commit,
            "window":          bench.get('window', ''),
            "total_instances": len(selected_instances),
            "source_total_instances": len(instances),
            "hard_instances":  sum(1 for _, i in selected_instances if i['difficulty'] == 'hard'),
        },
        "config": {
            "model":          model,
            "repo_path":      repo_path,
            "timeout_s":      timeout,
            "max_depth":      max_depth,
            "max_iterations": 30,
            "condition":      "rlm_single_agent_failed_rerun" if rerun_failed_from else "rlm_single_agent",
            "limit":          limit,
            "rerun_failed_from": rerun_source,
        },
        "summary": build_summary(all_results, elapsed_total),
        "groups":  build_groups(all_results),
        "results": all_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2))
    tmp_path.replace(out_path)

    s = report['summary']
    print(f"\n{'='*65}")
    print(f"RESULTS ({len(all_results)} instances)")
    print(f"  Micro F1: {s['micro_f1']:.3f}  "
          f"P={s['micro_precision']:.3f} R={s['micro_recall']:.3f}")
    print(f"  Macro F1: {s['macro_f1']:.3f}")
    print(f"  All-gold: {s['all_gold_found']}/{s['total']}")
    print(f"  index.jsonl -> {index_path}")
    print(f"  report JSON -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", type=Path, help="Path to benchmark JSON")
    p.add_argument("--model",     default=MODEL)
    p.add_argument("--repo", default=DEFAULT_REPO_PATH,
                   help="Path to the checked-out Ansible repository")
    p.add_argument("--out", type=Path, default=None, help="Output report path")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="Per-instance RLM timeout in seconds")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help="RLM recursion depth. Use 2 to allow one child RLM from the REPL.")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit selected instances; useful for smoke tests")
    p.add_argument("--rerun-failed-from", default=None,
                   help="Path to a previous RLM report JSON or index.jsonl; rerun only missing/error instances in a fresh log directory")
    p.add_argument("--log-dir", default=None,
                   help="Override output log directory")
    args = p.parse_args()
    run(
        model=args.model,
        benchmark_path=args.benchmark,
        repo_path=args.repo,
        timeout=args.timeout,
        max_depth=args.max_depth,
        limit=args.limit,
        rerun_failed_from=args.rerun_failed_from,
        log_dir=args.log_dir,
        out_path=args.out,
    )
