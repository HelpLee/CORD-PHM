"""Idempotent submission ledger; every downstream depends on its own upstream."""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOMAINS = ("bearing", "battery", "milling")
ARMS = ("joint_raw", "joint_norm", "joint_pcgrad", "joint_cagrad") + tuple(
    f"single_{d}_{mode}" for d in DOMAINS for mode in ("raw", "norm"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)
    path = HERE / "submissions.json"
    ledger = json.loads(path.read_text()) if path.exists() else {}
    def submit(key, command, hours, dependencies=(), lane=None):
        if key in ledger:
            return ledger[key]
        options = ["sbatch", "--parsable", "--job-name=e29_" + key, f"--time={hours}"]
        if dependencies:
            dependency = "afterok:" + ":".join(dict.fromkeys(dependencies))
            if lane is not None:
                dependency += ",afterany:" + lane
            options += ["--dependency=" + dependency, "--kill-on-invalid-dep=yes"]
        job = subprocess.check_output(options + ["run_gpu.sh"] + command, cwd=HERE, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(job)
        ledger[key] = job
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(ledger, indent=2)); temp.replace(path)
        print(key, job, flush=True)
        return job
    smoke = submit("smoke", ["smoke.py"], "00:30:00")
    if args.smoke_only:
        return
    # Four upstream lanes bound concurrent data loading. No pre-existing job is touched.
    lanes = [smoke] * 4
    upstream = {}
    for i, arm in enumerate(ARMS):
        job = submit("up_" + arm, ["train.py", "--arm", arm, "--resume"], "04:00:00", [smoke], lanes[i % 4])
        upstream[arm] = job
        lanes[i % 4] = job
    # Four downstream lanes start after all upstream jobs to cap total concurrency.
    # afterany lane links do not cancel unrelated experiments if another cell fails.
    last = [None] * 4
    cells = []
    for arm in (*ARMS, "joint_selection"):
        targets = DOMAINS if arm.startswith("joint_") else (arm.split("_")[1],)
        for domain in targets:
            for fraction in (.1, .2, 1.):
                cells.append((arm, domain, fraction))
    for i, (arm, domain, fraction) in enumerate(cells):
        key = f"down_{arm}_{domain}_p{round(fraction*100)}"
        own = upstream["joint_raw" if arm == "joint_selection" else arm]
        if key in ledger:
            last[i % 4] = ledger[key]
            continue
        deps = "afterok:" + own + ",afterany:" + ":".join(lanes + ([last[i % 4]] if last[i % 4] else []))
        command = ["sbatch", "--parsable", "--job-name=e29_" + key, "--time=01:00:00",
                   "--dependency=" + deps, "--kill-on-invalid-dep=yes", "run_gpu.sh", "downstream.py",
                   "--model", arm, "--domain", domain, "--fraction", str(fraction)]
        job = subprocess.check_output(command, cwd=HERE, text=True).strip().split(";")[0]
        assert job.isdigit(), job
        ledger[key] = job
        temp = path.with_suffix(".tmp"); temp.write_text(json.dumps(ledger, indent=2)); temp.replace(path)
        last[i % 4] = job
        print(key, job, flush=True)
    if "summary" not in ledger:
        dependencies = [v for k, v in ledger.items() if k.startswith("down_")]
        job = subprocess.check_output(["sbatch", "--parsable", "--job-name=e29_summary",
            "--dependency=afterany:" + ":".join(dependencies), "run_cpu.sh"], cwd=HERE, text=True).strip().split(";")[0]
        ledger["summary"] = job
        temp = path.with_suffix(".tmp"); temp.write_text(json.dumps(ledger, indent=2)); temp.replace(path)
        print("summary", job, flush=True)


if __name__ == "__main__":
    main()
