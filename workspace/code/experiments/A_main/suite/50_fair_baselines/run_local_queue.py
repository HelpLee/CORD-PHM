"""Resumable local queue for CPU trees and compact controlled models."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from run_all import DOMAINS, FRACTIONS, SEEDS, HERE

MODELS = ("native_xgboost", "native_random_forest",
          "controlled_mlp", "controlled_tcn")


def write_status(completed, failed, current):
    target = HERE / "local_queue_status.json"
    target.write_text(json.dumps({
        "total": len(MODELS) * len(DOMAINS) * len(FRACTIONS) * len(SEEDS),
        "completed": completed, "failed": failed, "current": current,
    }, indent=2), encoding="utf-8")


def main():
    logs = HERE / "local_logs"
    logs.mkdir(exist_ok=True)
    completed, failed = [], []
    for model in MODELS:
        for domain in DOMAINS:
            for fraction in FRACTIONS:
                for seed in SEEDS:
                    key = f"{model}_{domain}_p{int(fraction * 100)}_s{seed}"
                    metric = (HERE / "results" / model / domain /
                              f"fraction{int(fraction * 100)}" /
                              f"seed{seed}" / "metrics.json")
                    if metric.exists():
                        completed.append(key)
                        continue
                    write_status(completed, failed, key)
                    command = [sys.executable, "-u", str(HERE / "run.py"),
                               "--model", model, "--domain", domain,
                               "--fraction", str(fraction), "--seed", str(seed),
                               "--gpu", "0"]
                    with (logs / f"{key}.log").open("w", encoding="utf-8") as log:
                        result = subprocess.run(command, cwd=HERE, env=os.environ.copy(),
                                                stdout=log, stderr=subprocess.STDOUT,
                                                check=False)
                    if result.returncode:
                        failed.append(key)
                    else:
                        completed.append(key)
                    write_status(completed, failed, None)
    subprocess.run([sys.executable, str(HERE / "aggregate.py")],
                   cwd=HERE, check=True)
    if failed:
        raise SystemExit(f"{len(failed)} local jobs failed; see local_logs")


if __name__ == "__main__":
    main()
