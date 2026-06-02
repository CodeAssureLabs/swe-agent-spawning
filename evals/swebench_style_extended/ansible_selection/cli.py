from __future__ import annotations

import argparse

from .candidates import refresh_candidates
from .config import DEFAULT_AUDIT, DEFAULT_OUT, DEFAULT_REPO
from .benchmarks import write_benchmark_jsons
from .utils import load_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=type(DEFAULT_REPO), default=DEFAULT_REPO)
    parser.add_argument("--out", type=type(DEFAULT_OUT), default=DEFAULT_OUT)
    parser.add_argument("--audit-dir", type=type(DEFAULT_AUDIT), default=DEFAULT_AUDIT)
    parser.add_argument("--refresh-candidates", action="store_true")
    parser.add_argument("--write-benchmarks", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    cache = args.audit_dir / "ansible_labelled_candidates_cache.json"

    if args.refresh_candidates or not cache.exists():
        refresh_candidates(args.repo, cache)

    rows = load_candidates(cache)
    if args.write_benchmarks:
        write_benchmark_jsons(rows, args.repo, args.out)
        print(f"Wrote SWE-bench-like benchmark JSONs to {args.out}")
    else:
        print(f"Loaded {len(rows)} candidate rows. Pass --write-benchmarks to write JSON files.")


if __name__ == "__main__":
    main()
