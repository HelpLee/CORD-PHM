"""Validated retry chain for experiment-28 Bearing/Battery downstream jobs.

Two representative three4 cells run first.  The other sixteen cells are held
behind afterok dependencies, so an interface failure cannot waste the whole
allocation again.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
MODELS = ("three4", "three5", "single5")
DOMAINS = ("bearing", "battery")
FRACTIONS = (.1, .2, 1.0)
SMOKE = (("three4", "bearing", .1), ("three4", "battery", .1))


def submit(*arguments: str) -> str:
    return subprocess.check_output(
        ["sbatch", "--parsable", *arguments], cwd=HERE, text=True
    ).strip().split(";")[0]


def downstream(model: str, domain: str, fraction: float,
               dependency: Optional[str] = None) -> str:
    tag = f"p{round(fraction * 100)}"
    options = [f"--job-name=e28v3_{model}_{domain[:3]}_{tag}"]
    if dependency:
        options.extend((f"--dependency=afterok:{dependency}",
                        "--kill-on-invalid-dep=yes"))
    return submit(
        *options, "--time=02:00:00", "run_gpua100.sh", "run_downstream.py",
        "--model", model, "--domain", domain, "--fraction", str(fraction),
    )


def main() -> None:
    for model in MODELS:
        status = json.loads((HERE / "models" / model / "training" /
                             "status.json").read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            raise RuntimeError(f"Incomplete upstream checkpoint: {model}")

    jobs: dict[str, str] = {}
    smoke_ids = []
    for model, domain, fraction in SMOKE:
        key = f"{model}_{domain}_p{round(fraction * 100)}"
        jobs[key] = downstream(model, domain, fraction)
        smoke_ids.append(jobs[key])
        print("VALIDATION", key, jobs[key], flush=True)

    gate = ":".join(smoke_ids)
    for model in MODELS:
        for domain in DOMAINS:
            for fraction in FRACTIONS:
                if (model, domain, fraction) in SMOKE:
                    continue
                key = f"{model}_{domain}_p{round(fraction * 100)}"
                jobs[key] = downstream(model, domain, fraction, gate)
                print("DOWNSTREAM", key, jobs[key], flush=True)

    summary = submit(
        "--job-name=e28v3_summary",
        "--dependency=afterok:" + ":".join(jobs.values()),
        "--kill-on-invalid-dep=yes", "run_cpu.sh", "summarize.py",
    )
    record = {"validation_jobs": smoke_ids, "downstream": jobs,
              "summary": summary}
    (HERE / "resubmission_failed_downstream_v3.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
