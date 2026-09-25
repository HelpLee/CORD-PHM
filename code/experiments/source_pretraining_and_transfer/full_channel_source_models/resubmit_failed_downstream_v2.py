"""Resubmit only the failed Bearing/Battery downstream cells for experiment 28.

All three upstream checkpoints and every Milling downstream cell are already
complete.  This submission therefore preserves those artifacts and schedules
only the 18 failed cross-domain downstream cells, followed by one summary job.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODELS = ("three4", "three5", "single5")
DOMAINS = ("bearing", "battery")
FRACTIONS = (.1, .2, 1.0)


def submit(*arguments: str) -> str:
    return subprocess.check_output(
        ["sbatch", "--parsable", *arguments], cwd=HERE, text=True
    ).strip().split(";")[0]


def main() -> None:
    for model in MODELS:
        status_path = HERE / "models" / model / "training" / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            raise RuntimeError(f"Upstream checkpoint is incomplete: {status_path}")

    jobs: dict[str, str] = {}
    for model in MODELS:
        for domain in DOMAINS:
            for fraction in FRACTIONS:
                tag = f"p{round(fraction * 100)}"
                key = f"{model}_{domain}_{tag}"
                jobs[key] = submit(
                    f"--job-name=e28v2_{model}_{domain[:3]}_{tag}",
                    "--time=02:00:00",
                    "run_gpua100.sh", "run_downstream.py",
                    "--model", model, "--domain", domain,
                    "--fraction", str(fraction),
                )
                print("DOWNSTREAM", key, jobs[key], flush=True)

    summary = submit(
        "--job-name=e28v2_summary",
        "--dependency=afterok:" + ":".join(jobs.values()),
        "--kill-on-invalid-dep=yes",
        "run_cpu.sh", "summarize.py",
    )
    record = {"reused_upstreams": list(MODELS), "downstream": jobs,
              "summary": summary}
    (HERE / "resubmission_failed_downstream_v2.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
