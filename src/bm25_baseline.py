"""BM25 baseline for multi-file change localization on the 19-instance Ansible benchmark.

Indexes files in the ansible repo at base commit 01e7915b0a97.
For each issue, scores files against the issue's problem_statement; sweeps K to build a P/R envelope.
"""
import json
import os
import re
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi

REPO = Path("/Users/lwynter/Downloads/RLM/ansible_repo")
GOLD_PATH = Path("/Users/lwynter/Downloads/RLM/benchmark_ansible_pr_gold.json")
OUT_DIR = Path("/Users/lwynter/Downloads/RLM/analysis")
OUT_DIR.mkdir(exist_ok=True)

EXTS = {".py", ".rst", ".txt", ".yml", ".yaml", ".cs"}
SKIP_DIRS = {".git", "node_modules", "__pycache__"}

STOPWORDS = set("""
a an the and or but if then else when while for of in on at to from by with as is are was were be been being
do does did done not no yes that this these those it its his her their there here so such which what who whom
how why where when than into about over under between among across through self def class import return yield
true false none null pass raise except try finally with as global nonlocal lambda print str int float bool list
dict set tuple type len range map filter open read write close call test will would should could may might can
issue file files use using used uses make makes made get gets got set sets way ways one two three new old
""".split())

CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
SPLIT = re.compile(r"[^A-Za-z0-9]+")

def tokenize(text: str):
    if not text:
        return []
    # camelCase -> spaces
    text = CAMEL.sub(" ", text)
    toks = SPLIT.split(text.lower())
    return [t for t in toks if t and t not in STOPWORDS and len(t) > 1]

def file_tokens(rel: str, content: str):
    path_toks = tokenize(rel)
    body_toks = tokenize(content)
    return path_toks + body_toks

def build_corpus():
    files = []
    contents = []
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in names:
            if Path(n).suffix.lower() not in EXTS:
                continue
            p = Path(root) / n
            rel = str(p.relative_to(REPO))
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                txt = ""
            files.append(rel)
            contents.append(txt)
    return files, contents

def score_set(pred, gold):
    pred = set(pred); gold = set(gold)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1, all_gold=gold.issubset(pred))

def main():
    print("Loading gold...")
    gold_db = json.load(open(GOLD_PATH))
    curated_gold = {x["instance_id"]: list(x["original_gold_files"]) for x in gold_db["instances"]}
    pr_gold = {x["instance_id"]: list(x["gold_files"]) for x in gold_db["instances"]}
    problems = {x["instance_id"]: x["problem_statement"] for x in gold_db["instances"]}

    print("Building corpus...")
    files, contents = build_corpus()
    print(f"  {len(files)} files")
    print("Tokenizing...")
    docs = [file_tokens(f, c) for f, c in zip(files, contents)]
    print(f"  avg tokens/file: {sum(len(d) for d in docs)/len(docs):.0f}")

    print("Building BM25...")
    bm = BM25Okapi(docs, k1=1.2, b=0.75)

    # Run queries
    print("Running queries...")
    per_inst_ranking = {}
    for iid, q in problems.items():
        q_toks = tokenize(q)
        scores = bm.get_scores(q_toks)
        ranked = sorted(range(len(files)), key=lambda i: -scores[i])
        # Keep top-50 to be safe (max K we'll sweep)
        per_inst_ranking[iid] = [files[i] for i in ranked[:50]]

    # Sweep K and compute micro-P, micro-R, micro-F1, all-gold across instances for each gold set
    Ks = list(range(1, 31))
    results = {"curated": {}, "pr": {}}
    for kind, gold_map in [("curated", curated_gold), ("pr", pr_gold)]:
        for K in Ks:
            sum_tp = sum_fp = sum_fn = 0
            all_gold_count = 0
            for iid, ranking in per_inst_ranking.items():
                pred = ranking[:K]
                s = score_set(pred, gold_map[iid])
                sum_tp += s["tp"]; sum_fp += s["fp"]; sum_fn += s["fn"]
                if s["all_gold"]:
                    all_gold_count += 1
            P = sum_tp/(sum_tp+sum_fp) if (sum_tp+sum_fp) else 0.0
            R = sum_tp/(sum_tp+sum_fn) if (sum_tp+sum_fn) else 0.0
            F1 = 2*P*R/(P+R) if (P+R) else 0.0
            results[kind][K] = {"P": P, "R": R, "F1": F1, "all_gold": all_gold_count}
            if K in (1, 3, 5, 10, 20, 30):
                print(f"  K={K:2d}  [{kind:7}]  P={P:.3f}  R={R:.3f}  F1={F1:.3f}  all_gold={all_gold_count}/19")

    out = {"per_K": results, "per_instance_top50": per_inst_ranking}
    json.dump(out, open(OUT_DIR/"bm25_results.json", "w"), indent=2)
    print("\nSaved", OUT_DIR/"bm25_results.json")

if __name__ == "__main__":
    main()
