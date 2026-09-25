import json
import subprocess
from config import DOMAINS, FRACTIONS, HERE, MODELS, checkpoint, result_path

# Observed gpua100 E39 downstream walltimes are 1--24 minutes, including the
# slowest 100%-label partial fine-tune.  Forty-five minutes retains substantial
# queue-safe slack without asking Slurm for the previous 2--3 hour allocations.
DOWNSTREAM_WALLTIME = "00:45:00"

def main():
    # Refuse to submit downstream work until every immutable upstream is complete.
    for model in MODELS:
        for domain in DOMAINS:
            cp=checkpoint(model,domain)
            status=json.loads((cp.parent/"status.json").read_text())
            if status.get("state") != "complete" or not cp.is_file():
                raise RuntimeError(f"Incomplete upstream: {cp}")
    (HERE/"logs").mkdir(exist_ok=True)
    ledger_path=HERE/"submissions.json"
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    def submit(key,args,walltime,dependency=""):
        if key in ledger:return ledger[key]["job_id"]
        cmd=["sbatch","--parsable","--partition=gpua100","--exclude=cluster-gpu13",
             "--job-name=e39_"+key,"--time="+walltime]
        if dependency:cmd += ["--dependency="+dependency,"--kill-on-invalid-dep=yes"]
        cmd += ["run_gpu.sh",*args]
        jid=subprocess.check_output(cmd,cwd=HERE,text=True).strip().split(";")[0]
        if not jid.isdigit():raise RuntimeError(jid)
        ledger[key]=dict(job_id=jid,dependency=dependency,command=cmd)
        tmp=ledger_path.with_suffix(".tmp");tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(ledger_path)
        print(key,jid,dependency,flush=True);return jid
    jobs=[]
    for model in MODELS:
        for mode in ("frozen","partial"):
            for domain in DOMAINS:
                for fraction in FRACTIONS:
                    if result_path(model,mode,domain,fraction).is_file():continue
                    key=f'{model}_{mode}_{domain}_p{round(fraction*100)}'
                    jobs.append(submit(key,["downstream.py","--model",model,"--mode",mode,
                                      "--domain",domain,"--fraction",str(fraction)],
                                      DOWNSTREAM_WALLTIME))
    submit("summary",["summarize.py"],"00:10:00",
           "afterany:"+":".join(jobs) if jobs else "")
    print("SUBMITTED",len(ledger),"jobs")


if __name__ == "__main__":main()
