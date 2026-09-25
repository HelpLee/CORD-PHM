"""Submit bounded, restartable COMPONENT_ROUTING pipelines to gpua100."""
import json
import subprocess
from config import ARMS, FRACTIONS, HERE, settings


def main():
    (HERE / "logs").mkdir(exist_ok=True)
    path = HERE / "submissions.json"; ledger = json.loads(path.read_text()) if path.exists() else {}
    def save():
        temp = path.with_suffix(".tmp"); temp.write_text(json.dumps(ledger, indent=2)); temp.replace(path)
    def submit(key, args, walltime, dependency=""):
        if key in ledger: return ledger[key]["job_id"]
        command = ["sbatch", "--parsable", "--partition=gpua100", "--exclude=cluster-gpu13",
                   "--job-name=cord_component_" + key, "--time=" + walltime]
        if args and args[0] == "train.py": command += ["--signal=USR1@300"]
        if dependency: command += ["--dependency=" + dependency, "--kill-on-invalid-dep=yes"]
        full = command + ["run_gpu.sh"] + list(args)
        job = subprocess.check_output(full, cwd=HERE, text=True).strip().split(";")[0]
        if not job.isdigit(): raise RuntimeError(job)
        ledger[key] = dict(job_id=job, partition="gpua100", exclude="cluster-gpu13",
                           walltime=walltime, dependency=dependency, command=full)
        save(); print(key, job, flush=True); return job
    smoke = submit("smoke", ["smoke.py"], "00:45:00")
    lanes = [None, None]; downstream = []
    order = ("joint_obsprivate", "joint_obsprivate_cagrad", "joint_component") + tuple(
        arm for arm in ARMS if arm.startswith("single_"))
    for index, arm in enumerate(order):
        lane = index % 2; dependency = "afterok:" + smoke
        if lanes[lane]: dependency += ",afterany:" + lanes[lane]
        upstream = submit("up_" + arm, ["train.py", "--arm", arm, "--resume"], "10:00:00", dependency)
        tail = None
        for domain in settings(arm)["domains"]:
            for fraction in FRACTIONS:
                dependency = "afterok:" + upstream
                if tail: dependency += ",afterany:" + tail
                key = f"down_{arm}_{domain}_p{round(100*fraction)}"
                tail = submit(key, ["downstream.py", "--model", arm, "--domain", domain,
                                    "--fraction", str(fraction)], "02:00:00", dependency)
                downstream.append(tail)
        lanes[lane] = tail
    submit("summary", ["summarize.py"], "00:10:00", "afterany:" + ":".join(downstream))
    print("COMPONENT_ROUTING_SUBMITTED", len(ledger), "jobs", flush=True)


if __name__ == "__main__": main()
