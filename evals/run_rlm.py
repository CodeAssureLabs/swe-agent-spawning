"""
RLM File Localization

Usage:
    uv run python evals/run_rlm.py --trial 1
    uv run python evals/run_rlm.py --trial 1 --benchmark evals/benchmark_ansible_2025_stress.json
    uv run python evals/run_rlm.py --trial 1 --repo /path/to/ansible
    uv run python evals/run_rlm.py --trial 1 --timeout 7200
    uv run python evals/run_rlm.py --trial 1 --max-depth 2
"""

import os, json, time, argparse, re, ast, subprocess
from datetime import datetime

try:
    __import__("_rlm_import_fix")
except ImportError:
    pass

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

MODEL     = "claude-sonnet-4-6"
DEFAULT_REPO_PATH = "evals/repos/ansible"
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
        "IMPORTANT: Your very first action must be to run this exact setup code:\n"
        "    import os, subprocess\n"
        f"    repo_path = {repo_path!r}\n"
        "Then use repo_path as the base path for all file operations.\n"
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


def run(
    trial=1,
    model=MODEL,
    benchmark_path="evals/benchmark_ansible.json",
    repo_path=DEFAULT_REPO_PATH,
    timeout=DEFAULT_TIMEOUT,
    max_depth=DEFAULT_MAX_DEPTH,
):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY in .env")
    if not os.path.exists(repo_path):
        raise ValueError(f"Repo not found at {repo_path}")

    with open(benchmark_path) as f:
        bench = json.load(f)

    instances    = bench['instances']
    # Top-level fixed_commit is optional; new benchmarks may use per-instance base_commit.
    bench_commit = bench.get('fixed_commit', 'HEAD')
    log_dir      = f"evals/logs/rlm_ansible_trial{trial}"
    os.makedirs(log_dir, exist_ok=True)
    index_path   = os.path.join(log_dir, "index.jsonl")
    report_path  = os.path.join(log_dir, f"report_trial{trial}.json")

    print(f"\n{'='*65}")
    print(f"RLM File Localization (final) | model={model}")
    print(f"Benchmark: {benchmark_path}  ({len(instances)} instances)")
    print(f"Repo: {repo_path}")
    print(f"Trial: {trial}  |  Timeout: {timeout}s  |  Max depth: {max_depth}  |  Logs -> {log_dir}")
    print(f"{'='*65}")

    all_results = []
    t_start = time.perf_counter()

    for idx, inst in enumerate(instances):
        # Use instance-level base_commit if present, else fall back to benchmark-level
        base_commit = inst.get('base_commit', bench_commit)

        print(f"\n[{idx+1}/{len(instances)}] [{inst['difficulty'].upper()}] "
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

        logger = RLMLogger(log_dir=log_dir,
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
            max_iterations=15,
            max_depth=max_depth,
            max_timeout=timeout,
            verbose=False,
        )

        t0 = time.perf_counter()
        try:
            result    = rlm.completion(prompt=prompt_context,
                                       root_prompt=inst['problem_statement'])
            response  = result.response
            root_usage_summary = result.usage_summary.to_dict()
            trajectory = result.metadata or logger.get_trajectory() or {"run_metadata": None, "iterations": []}
            nested_usage_summary = nested_rlm_usage_from_trajectory(trajectory)
            usage_by_model = merge_usage_summaries(root_usage_summary, nested_usage_summary)
            total_in, total_out, llm_calls = usage_totals(usage_by_model)
            total_tok = total_in + total_out
            repl_calls = getattr(result.usage_summary, "total_repl_calls", 0) or 0
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0
        conversation_history = trajectory.get("iterations", [])

        predicted       = extract_files(response)
        pred_set        = set(predicted)
        gold_set        = set(inst['gold_files'])
        gold_docs       = set(f for f in gold_set if is_doc(f))
        file_score      = score_set(pred_set,  gold_set)

        print(f"  Predicted: {predicted}")
        print(f"  P={file_score['precision']:.2f} R={file_score['recall']:.2f} "
              f"F1={file_score['f1']:.2f} | all_gold={file_score['all_gold_found']}")
        print(f"  Missing: {file_score['missing_files']}")
        print(f"  Tokens: {total_tok:,}  Time: {elapsed:.1f}s")

        entry = {
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
            "total_tokens":      total_tok,
            "input_tokens":      total_in,
            "output_tokens":     total_out,
            "llm_calls":         llm_calls,
            "rlm_usage_summary":  usage_by_model,
            "rlm_root_usage_summary": root_usage_summary,
            "nested_rlm_usage_summary": nested_usage_summary,
            "repl_calls":        repl_calls,
            "turn_count":        len(conversation_history),
            "conversation_history": conversation_history,
            "rlm_trajectory":     trajectory,
            "trace_path":         logger.log_file_path,
            "trace_format":       "rlm_logger_jsonl_embedded",
            "any_correct":       file_score['tp'] > 0,
            "model":             model,
            "trial":             trial,
            "pred_files":        predicted,
        }
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
            "total_instances": len(instances),
            "hard_instances":  sum(1 for i in instances if i['difficulty'] == 'hard'),
        },
        "config": {
            "model":          model,
            "trial":          trial,
            "repo_path":      repo_path,
            "timeout_s":      timeout,
            "max_depth":      max_depth,
            "max_iterations": 15,
            "condition":      "rlm_single_agent",
        },
        "summary": build_summary(all_results, elapsed_total),
        "groups":  build_groups(all_results),
        "results": all_results,
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    s = report['summary']
    print(f"\n{'='*65}")
    print(f"RESULTS ({len(all_results)} instances, trial {trial})")
    print(f"  Micro F1: {s['micro_f1']:.3f}  "
          f"P={s['micro_precision']:.3f} R={s['micro_recall']:.3f}")
    print(f"  Macro F1: {s['macro_f1']:.3f}")
    print(f"  All-gold: {s['all_gold_found']}/{s['total']}")
    print(f"  index.jsonl -> {index_path}")
    print(f"  report JSON -> {report_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trial",     type=int, default=1,
                   help="Trial number (any integer)")
    p.add_argument("--model",     default=MODEL)
    p.add_argument("--benchmark", default="evals/benchmark_ansible.json",
                   help="Path to benchmark JSON")
    p.add_argument("--repo", default=DEFAULT_REPO_PATH,
                   help="Path to the checked-out Ansible repository")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="Per-instance RLM timeout in seconds")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help="RLM recursion depth. Use 2 to allow one child RLM from the REPL.")
    args = p.parse_args()
    run(
        trial=args.trial,
        model=args.model,
        benchmark_path=args.benchmark,
        repo_path=args.repo,
        timeout=args.timeout,
        max_depth=args.max_depth,
    )
