"""
Plain LLM baseline for ansible file localization.

No REPL, no file system access: the model answers from the issue text alone.
This is the lowest baseline. Expected to struggle since it can't explore the repo.

Usage:
    uv run python evals/run_plain_llm.py --trial 1
"""

import os
import json
import time
import argparse
from datetime import datetime

import anthropic
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

MODEL    = "claude-haiku-4-5"
BENCHMARK = "evals/benchmark_ansible.json"
DOC_PREFIXES = ("docs/", "changelogs/")

SYSTEM = (
    "You are a software engineering assistant. "
    "Given a GitHub issue for the ansible/ansible repository, "
    "identify which files in the repository would need to be modified to fix the issue. "
    "Include source and documentation files that would be modified. "
    "Do not include test files unless the issue specifically requires changing test infrastructure.\n\n"
    "You do NOT have access to the repository. Use your knowledge of ansible's structure "
    "to make your best guess.\n\n"
    "Respond with ONLY a JSON object in this format:\n"
    '{"files": ["lib/ansible/cli/galaxy.py", "docs/docsite/rst/guide.rst"]}\n\n'
    "Use repo-relative paths. No leading ./ or absolute paths."
)


def extract_files_from_response(response: str) -> list[str]:
    import re, json as _json

    # Try JSON parse first
    try:
        data = _json.loads(response.strip())
        if isinstance(data, dict) and "files" in data:
            return [f.strip() for f in data["files"] if isinstance(f, str)]
    except Exception:
        pass

    # Try to find JSON object in response
    match = re.search(r'\{[^{}]*"files"\s*:\s*\[[^\]]*\][^{}]*\}', response, re.DOTALL)
    if match:
        try:
            data = _json.loads(match.group())
            return [f.strip() for f in data.get("files", [])]
        except Exception:
            pass

    # Fallback: extract file paths
    paths = re.findall(r'[\w/.-]+\.(?:py|yml|yaml|rst|cs|js)\b', response)
    return list(dict.fromkeys(paths))


def is_doc(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in DOC_PREFIXES)


def score_set(pred_set: set[str], gold_set: set[str]) -> dict:
    true_pos = len(pred_set & gold_set)
    false_pos = len(pred_set - gold_set)
    false_neg = len(gold_set - pred_set)
    precision = true_pos / len(pred_set) if pred_set else 0.0
    recall = true_pos / len(gold_set) if gold_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": true_pos,
        "fp": false_pos,
        "fn": false_neg,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": pred_set == gold_set,
        "all_gold_found": gold_set.issubset(pred_set),
        "missing_files": sorted(gold_set - pred_set),
        "extra_files": sorted(pred_set - gold_set),
    }


def build_summary(results: list[dict], elapsed_total: float) -> dict:
    n = len(results)
    if n == 0:
        return {}
    tp_total = sum(r["tp"] for r in results)
    pred_total = sum(r["tp"] + r["fp"] for r in results)
    gold_total = sum(r["tp"] + r["fn"] for r in results)
    micro_p = tp_total / pred_total if pred_total else 0.0
    micro_r = tp_total / gold_total if gold_total else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    exact_matches = sum(1 for r in results if r["exact_match"])
    all_gold = sum(1 for r in results if r["all_gold_found"])
    return {
        "total": n,
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / n, 4),
        "all_gold_found": all_gold,
        "all_gold_found_rate": round(all_gold / n, 4),
        "macro_precision": round(sum(r["precision"] for r in results) / n, 4),
        "macro_recall": round(sum(r["recall"] for r in results) / n, 4),
        "macro_f1": round(sum(r["f1"] for r in results) / n, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "avg_predicted_files": round(sum(len(r["predicted_files"]) for r in results) / n, 2),
        "avg_gold_files": round(sum(r["num_files"] for r in results) / n, 2),
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "total_repl_calls": 0,
        "total_turns": sum(r.get("turn_count", len(r.get("conversation_history", []))) for r in results),
        "total_elapsed_s": round(elapsed_total, 2),
    }


def build_groups(results: list[dict]) -> dict:
    groups = {}
    splits = {
        "difficulty:easy": lambda r: r["difficulty"] == "easy",
        "difficulty:hard": lambda r: r["difficulty"] == "hard",
        "gold_docs:any": lambda r: len(r["gold_doc_files"]) > 0,
        "gold_docs:none": lambda r: len(r["gold_doc_files"]) == 0,
    }
    for name, fn in splits.items():
        subset = [r for r in results if fn(r)]
        if not subset:
            continue
        n = len(subset)
        tp_total = sum(r["tp"] for r in subset)
        pred_total = sum(r["tp"] + r["fp"] for r in subset)
        gold_total = sum(r["tp"] + r["fn"] for r in subset)
        micro_p = tp_total / pred_total if pred_total else 0.0
        micro_r = tp_total / gold_total if gold_total else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
        all_gold = sum(1 for r in subset if r["all_gold_found"])
        groups[name] = {
            "total": n,
            "exact_matches": sum(1 for r in subset if r["exact_match"]),
            "all_gold_found": all_gold,
            "all_gold_found_rate": round(all_gold / n, 4),
            "macro_f1": round(sum(r["f1"] for r in subset) / n, 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
        }
    return groups


def run(trial: int = 1, model: str = MODEL, benchmark_path: str = BENCHMARK):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY in .env")

    client = anthropic.Anthropic(api_key=api_key)

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    instances = benchmark["instances"]
    log_dir   = f"evals/logs/plain_llm_ansible_trial{trial}"
    os.makedirs(log_dir, exist_ok=True)
    index_path = os.path.join(log_dir, "index.jsonl")
    report_path = os.path.join(log_dir, f"report_trial{trial}.json")

    print(f"\n{'='*65}")
    print(f"Plain LLM Baseline | ansible/ansible | model={model}")
    print(f"NO REPL: model answers from issue text only (no repo access)")
    print(f"Instances: {len(instances)} | Trial: {trial}")
    print(f"Logs -> {log_dir}")
    print(f"{'='*65}")

    results = []
    t_start = time.perf_counter()
    for i, inst in enumerate(instances):
        print(f"\n[{i+1}/{len(instances)}] [{inst['difficulty'].upper()}] "
              f"{inst['instance_id'][:50]}...")
        print(f"  Q: {inst['problem_statement'][:100]}...")
        print(f"  Gold: {inst['gold_files']}")

        t0 = time.perf_counter()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                system=SYSTEM,
                messages=[{"role": "user", "content": inst["problem_statement"]}],
            )
            output        = resp.content[0].text.strip()
            input_tokens  = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
            total_tokens  = input_tokens + output_tokens
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0

        predicted = extract_files_from_response(output)
        pred_set = set(predicted)
        gold_set = set(inst["gold_files"])
        gold_docs = {path for path in gold_set if is_doc(path)}
        file_score = score_set(pred_set, gold_set)
        conversation_history = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": inst["problem_statement"]},
            {"role": "assistant", "content": output},
        ]

        print(f"  Predicted: {predicted}")
        print(f"  P={file_score['precision']:.2f} R={file_score['recall']:.2f} "
              f"F1={file_score['f1']:.2f} | exact={file_score['exact_match']}")
        print(f"  Tokens: {total_tokens:,} ({input_tokens} in / {output_tokens} out) "
              f"Time: {elapsed:.1f}s")

        entry = {
            "index": i,
            "instance_id": inst["instance_id"],
            "date": inst.get("date", ""),
            "difficulty": inst["difficulty"],
            "num_files": len(inst["gold_files"]),
            "problem_statement": inst["problem_statement"],
            "gold_files": inst["gold_files"],
            "gold_doc_files": sorted(gold_docs),
            "predicted_files": predicted,
            "base_commit": inst.get("base_commit", benchmark.get("fixed_commit")),
            "actual_commit": inst.get("base_commit", benchmark.get("fixed_commit")),
            "answer": output,
            "tp": file_score["tp"],
            "fp": file_score["fp"],
            "fn": file_score["fn"],
            "precision": file_score["precision"],
            "recall": file_score["recall"],
            "f1": file_score["f1"],
            "exact_match": file_score["exact_match"],
            "all_gold_found": file_score["all_gold_found"],
            "missing_files": file_score["missing_files"],
            "extra_files": file_score["extra_files"],
            "elapsed_s": round(elapsed, 2),
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "repl_calls": 0,
            "turn_count": len(conversation_history),
            "conversation_history": conversation_history,
            "trace_format": "single_anthropic_messages_call_embedded",
            "any_correct": file_score["tp"] > 0,
            "model": model,
            "trial": trial,
            "pred_files": predicted,
        }
        results.append(entry)

        with open(index_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    # Summary
    if results:
        elapsed_total = time.perf_counter() - t_start
        report = {
            "run_at": datetime.now().isoformat(),
            "benchmark": {
                "repo": benchmark.get("repo", "ansible/ansible"),
                "fixed_commit": benchmark.get("fixed_commit"),
                "window": benchmark.get("window", ""),
                "total_instances": len(instances),
                "hard_instances": sum(1 for inst in instances if inst["difficulty"] == "hard"),
            },
            "config": {
                "model": model,
                "trial": trial,
                "condition": "plain_llm",
                "tool_set": "none",
            },
            "summary": build_summary(results, elapsed_total),
            "groups": build_groups(results),
            "results": results,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        s = report["summary"]

        print(f"\n{'='*65}")
        print(f"RESULTS ({s['total']} instances, trial {trial})")
        print(f"  Micro F1: {s['micro_f1']:.3f}  "
              f"(P={s['micro_precision']:.3f} R={s['micro_recall']:.3f})")
        print(f"  Macro F1: {s['macro_f1']:.3f}")
        print(f"  All-gold: {s['all_gold_found']}/{s['total']}")
        print(f"  index.jsonl -> {index_path}")
        print(f"  report JSON -> {report_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trial",     type=int, default=1)
    p.add_argument("--model",     default=MODEL)
    p.add_argument("--benchmark", default=BENCHMARK)
    args = p.parse_args()
    run(trial=args.trial, model=args.model, benchmark_path=args.benchmark)
