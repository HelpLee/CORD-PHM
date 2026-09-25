"""Resubmit only the failed experiment-28 branches.

Keeps the currently running three4 upstream/downstream dependency chain,
restarts three5 after the shared battery scalers are ready, and reuses the
already-complete single5 upstream checkpoint.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
THREE4_DOWNSTREAM = tuple(str(job) for job in range(1904533, 1904542))


def submit(*arguments: str) -> str:
    return subprocess.check_output(
        ["sbatch", "--parsable", *arguments], cwd=HERE, text=True
    ).strip().split(";")[0]


def downstream(model: str, dependency: Optional[str]) -> dict[str, str]:
    jobs = {}
    for domain in ("bearing", "battery", "milling"):
        for fraction in (.1, .2, 1.0):
            tag = f"p{round(fraction * 100)}"
            key = f"{model}_{domain}_{tag}"
            options = [f"--job-name=e28r_{model}_{domain[:3]}_{tag}"]
            if dependency is not None:
                options.extend([f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"])
            jobs[key] = submit(
                *options, "--time=02:00:00", "run_gpua100.sh", "run_downstream.py",
                "--model", model, "--domain", domain, "--fraction", str(fraction),
            )
            print("DOWNSTREAM", key, jobs[key], flush=True)
    return jobs


def main() -> None:
    single_status = json.loads(
        (HERE / "models/single5/training/status.json").read_text(encoding="utf-8")
    )
    if single_status.get("state") != "complete":
        raise RuntimeError("single5 checkpoint is not complete")

    three5 = submit(
        "--job-name=e28r_up_three5", "--time=06:00:00", "run_gpua100.sh",
        "run_upstream.py", "--model", "three5",
    )
    print("UPSTREAM three5", three5, flush=True)
    three5_downstream = downstream("three5", three5)
    single5_downstream = downstream("single5", None)
    all_downstream = list(THREE4_DOWNSTREAM) + list(three5_downstream.values()) + list(single5_downstream.values())
    summary = submit(
        "--job-name=e28r_summary",
        "--dependency=afterok:" + ":".join(all_downstream),
        "--kill-on-invalid-dep=yes", "run_cpu.sh", "summarize.py",
    )
    record = {
        "kept_three4_downstream": list(THREE4_DOWNSTREAM),
        "three5_upstream": three5,
        "three5_downstream": three5_downstream,
        "single5_upstream": "reused complete checkpoint (epoch 347, best 317)",
        "single5_downstream": single5_downstream,
        "summary": summary,
    }
    (HERE / "resubmission_failed_chain.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
