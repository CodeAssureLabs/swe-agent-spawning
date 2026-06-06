"""
Plain LLM baseline for ansible file localization.

No REPL, no file system access: the model answers from the issue text alone.
This is the lowest baseline. Expected to struggle since it can't explore the repo.

Usage:
    uv run python evals/run_plain_llm.py evals/benchmark_ansible.json
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

MODEL    = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4096
DOC_PREFIXES = ("docs/", "changelogs/")

PROMPT_TASK = (
    "Identify the repo-relative files that would need to be modified to implement the requested fix. "
    "Include code, test, and documentation files when they would need edits. Do not edit files. "
    # "Use the repository tools to inspect the codebase. "  # Plain LLM has no tool/repo access.
    "Prefer exact existing file paths."
)
PROMPT_OUTPUT = (
    "Your final answer must be valid JSON only, with this schema:\n"
    '{"files": ["path/to/file.py"], "rationale": "short reason"}'
)
SYSTEM_PROMPT = f"{PROMPT_TASK}\n\n{PROMPT_OUTPUT}"


def normalize_path(path: str) -> str:
    cleaned = path.strip().strip("'\"` ,;")
    cleaned = cleaned.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def dedupe_paths(paths: list[str]) -> list[str]:
    seen = set()
    out = []
    for raw in paths:
        path = normalize_path(raw)
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def extract_files_from_response(response: str) -> tuple[list[str], str]:
    import re, json as _json

    # Try JSON parse first
    try:
        data = _json.loads(response.strip())
        if isinstance(data, dict) and "files" in data:
            return dedupe_paths([f for f in data["files"] if isinstance(f, str)]), "json-object"
        if isinstance(data, list):
            return dedupe_paths([f for f in data if isinstance(f, str)]), "json-array"
    except Exception:
        pass

    # Try to find JSON object in response
    match = re.search(r'\{[^{}]*"files"\s*:\s*\[[^\]]*\][^{}]*\}', response, re.DOTALL)
    if match:
        try:
            data = _json.loads(match.group())
            return dedupe_paths([f for f in data.get("files", []) if isinstance(f, str)]), "json-object"
        except Exception:
            pass

    # Fallback: extract file paths
    paths = re.findall(r'[\w/.-]+\.(?:py|yml|yaml|rst|cs|js)\b', response)
    return dedupe_paths(paths), "regex"


def is_doc(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return any(path.startswith(prefix) for prefix in DOC_PREFIXES) or suffix in {".md", ".rst", ".txt"}


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


def score_subset(predicted: list[str], gold: list[str], *, docs: bool) -> dict:
    return score_set(
        {path for path in predicted if is_doc(path) == docs},
        {path for path in gold if is_doc(path) == docs},
    )


def build_summary(results: list[dict], elapsed_total: float) -> dict:
    n = len(results)
    if n == 0:
        return {}
    error_cases = sum(1 for r in results if r.get("status") == "error")
    tp_total = sum(r["file_score"]["tp"] for r in results)
    pred_total = sum(r["file_score"]["tp"] + r["file_score"]["fp"] for r in results)
    gold_total = sum(r["file_score"]["tp"] + r["file_score"]["fn"] for r in results)
    micro_p = tp_total / pred_total if pred_total else 0.0
    micro_r = tp_total / gold_total if gold_total else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    exact_matches = sum(1 for r in results if r["file_score"]["exact_match"])
    all_gold = sum(1 for r in results if r["file_score"]["all_gold_found"])
    return {
        "total": n,
        "completed": n - error_cases,
        "error_cases": error_cases,
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / n, 4),
        "all_gold_found": all_gold,
        "all_gold_found_rate": round(all_gold / n, 4),
        "macro_precision": round(sum(r["file_score"]["precision"] for r in results) / n, 4),
        "macro_recall": round(sum(r["file_score"]["recall"] for r in results) / n, 4),
        "macro_f1": round(sum(r["file_score"]["f1"] for r in results) / n, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "avg_predicted_files": round(sum(len(r["predicted_files"]) for r in results) / n, 2),
        "avg_gold_files": round(sum(r["num_files"] for r in results) / n, 2),
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
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
        tp_total = sum(r["file_score"]["tp"] for r in subset)
        pred_total = sum(r["file_score"]["tp"] + r["file_score"]["fp"] for r in subset)
        gold_total = sum(r["file_score"]["tp"] + r["file_score"]["fn"] for r in subset)
        micro_p = tp_total / pred_total if pred_total else 0.0
        micro_r = tp_total / gold_total if gold_total else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
        all_gold = sum(1 for r in subset if r["file_score"]["all_gold_found"])
        groups[name] = {
            "total": n,
            "exact_matches": sum(1 for r in subset if r["file_score"]["exact_match"]),
            "all_gold_found": all_gold,
            "all_gold_found_rate": round(all_gold / n, 4),
            "macro_f1": round(sum(r["file_score"]["f1"] for r in subset) / n, 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
        }
    return groups


def default_out_path(benchmark_path: Path) -> Path:
    return benchmark_path.with_name(benchmark_path.stem + "_plain_llm_report.json")


def default_log_dir(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_logs")


def uniquify_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.name}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2))
    tmp_path.replace(out_path)


def make_result_entry(
    *,
    index: int,
    inst: dict,
    benchmark: dict,
    output: str,
    predicted: list[str],
    extraction_mode: str,
    elapsed: float,
    input_tokens: int,
    output_tokens: int,
    status: str,
    error: str | None = None,
) -> dict:
    gold_set = set(inst["gold_files"])
    pred_set = set(predicted)
    gold_docs = {path for path in gold_set if is_doc(path)}
    file_score = score_set(pred_set, gold_set)
    doc_score = score_subset(predicted, inst["gold_files"], docs=True)
    code_score = score_subset(predicted, inst["gold_files"], docs=False)
    assistant_content = output if status == "completed" else f"|ERROR| {error}"
    conversation_history = [
        {"role": "user", "content": inst["problem_statement"]},
        {"role": "assistant", "content": assistant_content},
    ]
    entry = {
        "index": index,
        "instance_id": inst["instance_id"],
        "date": inst.get("date", ""),
        "difficulty": inst["difficulty"],
        "num_files": len(inst["gold_files"]),
        "problem_statement": inst["problem_statement"],
        "gold_files": inst["gold_files"],
        "gold_doc_files": sorted(gold_docs),
        "predicted_files": predicted,
        "prediction_extraction_mode": extraction_mode,
        "base_commit": inst.get("base_commit", benchmark.get("fixed_commit")),
        "answer": assistant_content,
        "file_score": file_score,
        "doc_file_score": doc_score,
        "code_file_score": code_score,
        "elapsed_s": round(elapsed, 2),
        "total_tokens": input_tokens + output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "conversation_history": conversation_history,
        "any_correct": file_score["tp"] > 0,
        "status": status,
    }
    if error is not None:
        entry["error"] = error
    return entry


def run(
    model: str = MODEL,
    benchmark_path: Path | str = Path("evals/benchmark_ansible.json"),
    limit: int | None = None,
    log_dir: Path | str | None = None,
    out_path: Path | str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY in .env")

    client = anthropic.Anthropic(api_key=api_key)

    benchmark_path = Path(benchmark_path).resolve()
    with benchmark_path.open() as f:
        benchmark = json.load(f)

    all_instances = benchmark["instances"]
    selected_instances = list(enumerate(all_instances))
    if limit is not None:
        selected_instances = selected_instances[:limit]
    out_path = Path(out_path).resolve() if out_path else default_out_path(benchmark_path)
    if log_dir is None:
        log_dir = default_log_dir(out_path)
    log_dir = uniquify_path(Path(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    index_path = os.path.join(log_dir, "index.jsonl")
    open(index_path, "w").close()

    print(f"\n{'='*65}")
    print(f"Plain LLM Baseline | ansible/ansible | model={model}")
    print(f"NO REPL: model answers from issue text only (no repo access)")
    print(f"Instances: {len(selected_instances)} selected / {len(all_instances)} total")
    print(f"Max output tokens: {max_tokens}")
    print(f"Logs -> {log_dir}")
    print(f"Report -> {out_path}")
    print(f"{'='*65}")

    results = []
    t_start = time.perf_counter()
    for display_idx, (i, inst) in enumerate(selected_instances, 1):
        print(f"\n[{display_idx}/{len(selected_instances)}] [{inst['difficulty'].upper()}] "
              f"{inst['instance_id'][:50]}...")
        print(f"  Q: {inst['problem_statement'][:100]}...")
        print(f"  Gold: {inst['gold_files']}")

        t0 = time.perf_counter()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": inst["problem_statement"]}],
            )
            output        = resp.content[0].text.strip()
            input_tokens  = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
            total_tokens  = input_tokens + output_tokens
        except Exception as e:
            print(f"  ERROR: {e}")
            elapsed = time.perf_counter() - t0
            entry = make_result_entry(
                index=i,
                inst=inst,
                benchmark=benchmark,
                output="",
                predicted=[],
                extraction_mode="error",
                elapsed=elapsed,
                input_tokens=0,
                output_tokens=0,
                status="error",
                error=str(e),
            )
            results.append(entry)
            with open(index_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")
            continue
        elapsed = time.perf_counter() - t0

        predicted, extraction_mode = extract_files_from_response(output)
        entry = make_result_entry(
            index=i,
            inst=inst,
            benchmark=benchmark,
            output=output,
            predicted=predicted,
            extraction_mode=extraction_mode,
            elapsed=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status="completed",
            error=None,
        )
        file_score = entry["file_score"]

        print(f"  Predicted: {predicted}")
        print(f"  P={file_score['precision']:.2f} R={file_score['recall']:.2f} "
              f"F1={file_score['f1']:.2f} | exact={file_score['exact_match']}")
        print(f"  Tokens: {total_tokens:,} ({input_tokens} in / {output_tokens} out) "
              f"Time: {elapsed:.1f}s")

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
                key: value for key, value in benchmark.items()
                if key != "instances"
            },
            "config": {
                "runner": "plain_llm",
                "provider": "anthropic",
                "model": model,
                "max_tokens": max_tokens,
                "condition": "plain_llm",
                "system_prompt": SYSTEM_PROMPT,
            },
            "summary": build_summary(results, elapsed_total),
            "groups": build_groups(results),
            "results": results,
        }
        write_report(report, out_path)

        s = report["summary"]

        print(f"\n{'='*65}")
        print(f"RESULTS ({s['total']} instances)")
        print(f"  Micro F1: {s['micro_f1']:.3f}  "
              f"(P={s['micro_precision']:.3f} R={s['micro_recall']:.3f})")
        print(f"  Macro F1: {s['macro_f1']:.3f}")
        print(f"  All-gold: {s['all_gold_found']}/{s['total']}")
        print(f"  index.jsonl -> {index_path}")
        print(f"  report JSON -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", type=Path, help="Path to benchmark JSON")
    p.add_argument("--model",     default=MODEL)
    p.add_argument("--out", type=Path, default=None, help="Output report path")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit selected instances; useful for smoke tests")
    p.add_argument("--log-dir", default=None,
                   help="Override output log directory")
    args = p.parse_args()
    run(
        model=args.model,
        benchmark_path=args.benchmark,
        limit=args.limit,
        log_dir=args.log_dir,
        out_path=args.out,
    )
