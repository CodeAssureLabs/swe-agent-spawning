from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO = REPO_ROOT / "repos/ansible"
DEFAULT_OUT = PACKAGE_DIR / "artifacts/ansible_test_sets"
DEFAULT_AUDIT = PACKAGE_DIR / "audit/ansible_test_sets"
BASELINE_BENCHMARK = DEFAULT_OUT / "benchmark_ansible.json"

WINDOWS = [
    ("SWE-bench-Pro-2020", "2020-05-01", "2020-11-30"),
    ("2023-H1", "2023-01-01", "2023-06-30"),
    ("2023-H2", "2023-07-01", "2023-12-31"),
    ("2024-H1", "2024-01-01", "2024-06-30"),
    ("2024-H2", "2024-07-01", "2024-12-31"),
    ("2025-H1", "2025-01-01", "2025-06-30"),
    ("2025-Stress", "2025-04-01", "2025-09-30"),
    ("2025-H2", "2025-07-01", "2025-12-31"),
    ("2026-YTD", "2026-01-01", "2026-05-30"),
]

WINDOW_LABELS = {
    "SWE-bench-Pro-2020": "May 2020 - Nov 2020",
    "2023-H2": "July 2023 - Dec 2023",
    "2024-H2": "July 2024 - Dec 2024",
    "2025-H1": "Jan 2025 - Jun 2025",
    "2025-Stress": "Apr 2025 - Sep 2025",
    "2025-H2": "Jul 2025 - Dec 2025",
    "2026-YTD": "Jan 2026 - May 2026",
}

# Source file definition used for difficulty classification. The original
# SWE-bench Pro-derived Ansible gold includes non-Python runtime files under
# lib/ansible, including .yml and .cs; recent candidate windows also include
# PowerShell, YAML, and template runtime files under lib/ansible. Tests,
# changelogs, CI files, and generated metadata do not count as SWE-bench-like
# gold files unless they match the explicit style filter in utils.py.
SOURCE_PREFIXES = ("lib/ansible/",)
SOURCE_SUFFIXES = (".py", ".yml", ".yaml", ".ps1", ".psm1", ".cs", ".j2")

BENCHMARK_FILENAMES = {
    "SWE-bench-Pro-2020": "benchmark_ansible.json",
    "2025-Stress": "benchmark_ansible_2025_stress.json",
    "2026-YTD": "benchmark_ansible_2026_ytd.json",
    "2023-H2": "benchmark_ansible_2023_h2.json",
    "2024-H2": "benchmark_ansible_2024_h2.json",
    "2025-H2": "benchmark_ansible_2025_h2.json",
}

EASY_TOTAL_FILE_CAP = 10
HARD_TOTAL_FILE_CAP = 16
TARGET_EASY = 6
TARGET_HARD = 13

# These were treated as weak localization tasks even when label-valid.
LOW_SIGNAL_TITLE_FRAGMENTS = [
    "typo",
    "docstring",
    "type hint",
    "type annotation",
    "mypy",
    "pylint",
    "unused",
    "copyright",
    "inclusive word",
    "noqa",
    "lint",
    "formatting",
    "remove python 2 traces",
    "deprecated imports",
    "obsolete todo",
]

# Final curated sets. See docs/selection_methodology.md for the selection process.
SELECTED_PRS = {
    "2025-Stress": [
        84690,
        84912,
        84754,
        85121,
        85129,
        85222,
        82314,
        85266,
        85351,
        85385,
        85390,
        85476,
        85590,
        85628,
        85690,
        85724,
        85763,
        85801,
        85889,
    ],
    "2026-YTD": [
        86341,
        85857,
        85498,
        86473,
        86493,
        85758,
        86237,
        86561,
        86183,
        86619,
        86705,
        86727,
        86603,
        86793,
        86885,
        86687,
        86919,
        86931,
        86977,
    ],
    "2023-H2": [
        81218,
        81178,
        81423,
        81531,
        81469,
        81524,
        79781,
        81450,
        81678,
        81780,
        81700,
        81599,
        81772,
        79945,
        82002,
        81809,
        82165,
        82319,
        82339,
    ],
    "2024-H2": [
        83601,
        83726,
        83712,
        83711,
        83353,
        83803,
        83834,
        83953,
        83290,
        84007,
        83332,
        83014,
        84126,
        84211,
        84292,
        84398,
        84240,
        80566,
        84442,
    ],
    "2025-H2": [
        85440,
        85291,
        83659,
        85524,
        85653,
        85690,
        85497,
        85782,
        85889,
        85871,
        85805,
        86012,
        85909,
        85372,
        86144,
        84661,
        86141,
        86219,
        86294,
    ],
}

SWEBENCH_PRO_2020_PRS = [
    67843,
    66569,
    69376,
    58278,
    68177,
    69154,
    69822,
    69959,
    70624,
    70610,
    70762,
    74797,
    70221,
    71070,
    50909,
    72070,
    72288,
    71230,
    71904,
]

BENCHMARK_SET_PRS = {
    "SWE-bench-Pro-2020": SWEBENCH_PRO_2020_PRS,
    **SELECTED_PRS,
}

RECOMMENDED_ADDITIONAL_WINDOWS = ["2025-Stress", "2026-YTD"]
AUXILIARY_WINDOWS = ["2025-H2", "2024-H2", "2023-H2"]
COMPARISON_WINDOWS = ["SWE-bench-Pro-2020", *RECOMMENDED_ADDITIONAL_WINDOWS, *AUXILIARY_WINDOWS]

@dataclass(frozen=True)
class Candidate:
    window: str
    sha: str
    date: str
    pr: int
    subject: str
    file_count: int
    source_count: int
    source_files: tuple[str, ...]
    added_count: int
    modified_count: int
    other_status_count: int
    labels: tuple[str, ...]
    title: str
    url: str
