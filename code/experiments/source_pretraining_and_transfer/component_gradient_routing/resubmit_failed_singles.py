"""Idempotently rebuild the six COMPONENT_ROUTING single-domain pipelines after the prepare fix."""
import json
import subprocess

from config import FRACTIONS, HERE, SINGLE_ARMS, settings


def main():
    (HERE / "logs").mkdir(exist_ok=True)
    path = HERE / "resubmissions_failed_singles.json"
    ledger = json.loads(path.read_text()) if path.exists() else {}

    def save():
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(ledger, indent=2))
        temp.replace(path)

    def submit(key, arguments, walltime, dependency=""):
        if key in ledger:
            return ledger[key]["job_id"]
        command = ["sbatch", "--parsable", "--partition=gpua100",
                   "--exclude=cluster-gpu13", "--job-name=e34r_" + key,
                   "--time=" + walltime]
        if arguments[0] == "train.py":
            command += ["--signal=USR1@300"]
        if dependency:
            command += ["--dependency=" + dependency, "--kill-on-invalid-dep=yes"]
        full = command + ["run_gpu.sh"] + list(arguments)
        job = subprocess.check_output(full, cwd=HERE, universal_newlines=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(job)
        ledger[key] = dict(job_id=job, command=full, dependency=dependency)
        save()
        print(key, job, flush=True)
        return job

    lanes = [None, None]
    downstream = []
    for index, arm in enumerate(SINGLE_ARMS):
        lane = index % 2
        dependency = "afterany:" + lanes[lane] if lanes[lane] else ""
        upstream = submit("up_" + arm, ["train.py", "--arm", arm, "--resume"],
                          "05:00:00", dependency)
        tail = None
        for domain in settings(arm)["domains"]:
            for fraction in FRACTIONS:
                dependency = "afterok:" + upstream
                if tail:
                    dependency += ",afterany:" + tail
                key = f"down_{arm}_{domain}_p{round(100*fraction)}"
                tail = submit(key, ["downstream.py", "--model", arm, "--domain", domain,
                                    "--fraction", str(fraction)], "02:00:00", dependency)
                downstream.append(tail)
        lanes[lane] = tail
    submit("summary", ["summarize.py"], "00:10:00",
           "afterany:" + ":".join(downstream))
    print("RESUBMITTED", len(ledger), "jobs", flush=True)


if __name__ == "__main__":
    main()
