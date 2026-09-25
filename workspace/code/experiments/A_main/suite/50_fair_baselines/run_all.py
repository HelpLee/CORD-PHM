"""Stage or run both baseline families on the canonical 3 x 3 x 5 grid."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOMAINS = ("bearing", "battery", "milling")
FRACTIONS = (0.1, 0.2, 1.0)
SEEDS = (42, 43, 44, 45, 46)
MODELS = (
    "native_xgboost", "native_random_forest", "native_cnn_lstm_attention",
    "native_tcn", "native_patchtst", "native_itransformer",
    "native_moment_frozen", "native_moment_full",
    "controlled_mlp", "controlled_tcn", "controlled_cnn_lstm_attention",
    "controlled_patchtst", "controlled_itransformer",
    "controlled_moment_frozen", "controlled_moment_full")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    parser.add_argument("--fractions", nargs="+", type=float, choices=FRACTIONS, default=FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    jobs = [{"model": model, "domain": domain, "fraction": fraction, "seed": seed}
            for model in args.models for domain in args.domains
            for fraction in args.fractions for seed in args.seeds]
    (HERE / "job_manifest.json").write_text(json.dumps(
        {"jobs": jobs, "count": len(jobs), "legacy_results_reused": False}, indent=2),
        encoding="utf-8")
    if args.stage_only:
        print(f"Staged {len(jobs)} jobs")
        return
    for job in jobs:
        target = (HERE / "results" / job["model"] / job["domain"] /
                  f"fraction{int(job['fraction'] * 100)}" / f"seed{job['seed']}" /
                  "metrics.json")
        if target.exists():
            continue
        command = [sys.executable, "-u", str(HERE / "run.py"),
                   "--model", job["model"], "--domain", job["domain"],
                   "--fraction", str(job["fraction"]), "--seed", str(job["seed"]),
                   "--gpu", args.gpu]
        subprocess.run(command, cwd=HERE, env=os.environ.copy(), check=True)
    subprocess.run([sys.executable, str(HERE / "aggregate.py")], cwd=HERE, check=True)


if __name__ == "__main__":
    main()
