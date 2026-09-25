"""Submit only incomplete experiment-28 downstream cells after validation."""
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


def complete(model: str, domain: str, fraction: float) -> bool:
    path = (HERE / "downstream" / model / domain /
            f"p{round(fraction * 100)}" / "status.json")
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("state") == "complete"


def main() -> None:
    jobs: dict[str, str] = {}
    reused = []
    for model in MODELS:
        upstream = json.loads((HERE / "models" / model / "training" /
                               "status.json").read_text(encoding="utf-8"))
        if upstream.get("state") != "complete":
            raise RuntimeError(f"Incomplete upstream checkpoint: {model}")
        for domain in DOMAINS:
            for fraction in FRACTIONS:
                tag = f"p{round(fraction * 100)}"
                key = f"{model}_{domain}_{tag}"
                if complete(model, domain, fraction):
                    reused.append(key)
                    print("REUSE", key, flush=True)
                    continue
                jobs[key] = submit(
                    f"--job-name=e28v4_{model}_{domain[:3]}_{tag}",
                    "--time=02:00:00", "run_gpua100.sh", "run_downstream.py",
                    "--model", model, "--domain", domain,
                    "--fraction", str(fraction),
                )
                print("DOWNSTREAM", key, jobs[key], flush=True)

    summary_options = ["--job-name=e28v4_summary"]
    if jobs:
        summary_options.extend(("--dependency=afterok:" + ":".join(jobs.values()),
                                "--kill-on-invalid-dep=yes"))
    summary = submit(*summary_options, "run_cpu.sh", "summarize.py")
    record = {"reused_complete": reused, "submitted": jobs, "summary": summary}
    (HERE / "resubmission_incomplete_downstream_v4.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
