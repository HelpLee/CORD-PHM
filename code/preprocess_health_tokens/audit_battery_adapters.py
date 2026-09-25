"""Read-only, bounded QA for the cycle-level battery adapters.

This script never writes health-token NPZ files.  It samples native cycles from
each adapter and reports whether the common capacity-window contract accepts
them, while preserving dataset-specific parsing and discharge segmentation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np

# Allow direct execution from the repository root, matching main.py usage.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from preprocess_health_tokens.common.battery_health_token_utils import tokenize_cycle


DATASETS = (
    "nasa_battery",
    "oxford_battery",
    "xjtu_battery",
    "isu_ilcc_battery",
    "hust_battery",
    "rwth_battery",
    "sdu_battery",
    "mich_exp_battery",
    "calce_cs2_downstream_4cells",
)


def audit(dataset: str, sample_cycles: int, limit_files: int) -> dict:
    module = importlib.import_module(f"preprocess_health_tokens.datasets.battery.{dataset}")
    counts: Counter[int] = Counter()
    failures: Counter[str] = Counter()
    sampled = 0
    for cycle in islice(module.iter_cycles(limit_files=limit_files), sample_cycles):
        sampled += 1
        try:
            tokenized = tokenize_cycle(cycle)
            counts[int(np.sum(tokenized["token_mask"]))] += 1
        except (TypeError, ValueError, IndexError) as exc:
            reason = str(exc).split(": ", 1)[-1]
            failures[reason] += 1
    accepted = int(sum(counts.values()))
    return {
        "dataset": dataset,
        "sampled": sampled,
        "accepted": accepted,
        "rejected": sampled - accepted,
        "valid_token_histogram": dict(sorted(counts.items())),
        "failure_examples": dict(failures.most_common(5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--sample-cycles", type=int, default=30)
    parser.add_argument("--limit-files", type=int, default=1)
    args = parser.parse_args()
    for dataset in args.datasets:
        print(json.dumps(audit(dataset, args.sample_cycles, args.limit_files), indent=2))


if __name__ == "__main__":
    main()
