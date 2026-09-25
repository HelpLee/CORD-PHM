"""Submit three upstreams, 27 dependent downstream jobs, and one summary."""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def submit(*arguments):
    return subprocess.check_output(
        ["sbatch", "--parsable", *arguments], cwd=str(HERE), text=True
    ).strip().split(";")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-prepared", action="store_true",
        help="Reuse the already verified runtime after a downstream/upstream-only retry.",
    )
    args = parser.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)
    prepare = None
    if args.reuse_prepared:
        audit = HERE / "runtime/milling/upstream_splits_scalers.json"
        if not audit.is_file():
            raise FileNotFoundError(f"Prepared runtime is incomplete: {audit}")
    else:
        prepare = submit("--job-name=e28_prepare", "--time=02:00:00",
                         "run_gpua100.sh", "prepare_runtime.py")
    upstream = {}
    for model in ("three4", "three5", "single5"):
        dependency = [] if prepare is None else [f"--dependency=afterok:{prepare}"]
        upstream[model] = submit(
            f"--job-name=e28_up_{model}",
            *dependency, "--kill-on-invalid-dep=yes",
            "--time=06:00:00", "run_gpua100.sh", "run_upstream.py", "--model", model,
        )
        print("UPSTREAM", model, upstream[model], flush=True)

    downstream = {}
    downstream_ids = []
    for model in ("three4", "three5", "single5"):
        for domain in ("bearing", "battery", "milling"):
            for fraction in (.1, .2, 1.0):
                tag = f"p{round(fraction * 100)}"
                key = f"{model}_{domain}_{tag}"
                job = submit(
                    f"--job-name=e28_{model}_{domain[:3]}_{tag}",
                    f"--dependency=afterok:{upstream[model]}", "--kill-on-invalid-dep=yes",
                    "--time=02:00:00", "run_gpua100.sh", "run_downstream.py",
                    "--model", model, "--domain", domain, "--fraction", str(fraction),
                )
                downstream[key] = job
                downstream_ids.append(job)
                print("DOWNSTREAM", key, job, flush=True)

    summary = submit(
        "--job-name=e28_summary",
        "--dependency=afterok:" + ":".join(downstream_ids), "--kill-on-invalid-dep=yes",
        "run_cpu.sh", "summarize.py",
    )
    record = {
        "partition": "gpua100", "prepare": prepare, "upstream": upstream,
        "downstream": downstream, "summary": summary,
        "models": {
            "three4": "three-domain; original four Milling sources; all channels",
            "three5": "three-domain; original four plus HMoTP; all channels",
            "single5": "Milling-only; original four plus HMoTP; all channels",
        },
        "downstream_protocol": {
            "domains": ["bearing", "battery", "milling"],
            "fractions": [.1, .2, 1.0], "seeds": [42, 43, 44, 45, 46],
            "arm": "full_finetune", "encoder_lr": 3e-4, "head_lr": 1e-3,
        },
    }
    (HERE / "submissions.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("PREPARE", prepare if prepare is not None else "reused", flush=True)
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
