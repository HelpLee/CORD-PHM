"""Idempotently rerun only incomplete E29 downstream cells away from cluster-gpu13."""
import json
import subprocess
from pathlib import Path

from submit import ARMS, DOMAINS

HERE = Path(__file__).resolve().parent


def complete(path):
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return data.get("complete") and {row["seed"] for row in data.get("rows", [])} == set(range(42, 47))


def main():
    (HERE / "logs").mkdir(exist_ok=True)
    ledger_path = HERE / "resubmissions_gpu13.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    missing = []
    for arm in (*ARMS, "joint_selection"):
        domains = DOMAINS if arm.startswith("joint_") else (arm.split("_")[1],)
        for domain in domains:
            for fraction in (.1, .2, 1.):
                result = HERE / "downstream" / arm / domain / f"p{round(100*fraction)}" / "results.json"
                if not complete(result):
                    missing.append((arm, domain, fraction))
    tails = [None] * 4
    jobs = []
    for index, (arm, domain, fraction) in enumerate(missing):
        key = f"retry_{arm}_{domain}_p{round(100*fraction)}"
        if key in ledger:
            jobs.append(ledger[key])
            tails[index % 4] = ledger[key]
            continue
        command = ["sbatch", "--parsable", "--partition=gpua100", "--exclude=cluster-gpu13",
                   "--job-name=e29r_" + key, "--time=01:00:00"]
        if tails[index % 4]:
            command += ["--dependency=afterany:" + tails[index % 4]]
        command += ["run_gpu.sh", "downstream.py", "--model", arm, "--domain", domain,
                    "--fraction", str(fraction)]
        job = subprocess.check_output(command, cwd=HERE, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(job)
        ledger[key] = job
        ledger_path.write_text(json.dumps(ledger, indent=2))
        jobs.append(job); tails[index % 4] = job
        print(key, job, flush=True)
    if jobs and "summary" not in ledger:
        command = ["sbatch", "--parsable", "--partition=gpua100", "--exclude=cluster-gpu13",
                   "--job-name=e29r_summary", "--time=00:10:00",
                   "--dependency=afterany:" + ":".join(jobs), "run_gpu.sh", "summarize.py"]
        job = subprocess.check_output(command, cwd=HERE, text=True).strip().split(";")[0]
        ledger["summary"] = job
        ledger_path.write_text(json.dumps(ledger, indent=2))
    print("INCOMPLETE_CELLS", len(missing), "RESUBMITTED", len(jobs), flush=True)


if __name__ == "__main__":
    main()
