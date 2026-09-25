"""gpu_test smoke -> independent gpua100 upstreams -> own downstreams."""
import argparse
import json
import subprocess
from config import HERE, ARMS, FRACTIONS, settings


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dry-run',action='store_true'); args=p.parse_args()
    file=HERE/'submissions.json'; ledger=json.loads(file.read_text()) if file.exists() else {}
    if not args.dry_run: (HERE/'logs').mkdir(exist_ok=True)
    def submit(key, arguments, walltime, dependency='', partition='gpua100'):
        if key in ledger: return ledger[key]['job_id']
        command=['sbatch','--parsable','--partition='+partition,'--job-name=cord_source_'+key,'--time='+walltime]
        if partition=='gpua100': command+=['--exclude=cluster-gpu13']
        if arguments[0]=='train.py': command+=['--signal=USR1@300']
        if dependency: command+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
        command+=['run_gpu.sh']+arguments
        job=str(9900000+len(ledger)) if args.dry_run else subprocess.check_output(command,cwd=HERE,text=True).strip().split(';')[0]
        if not job.isdigit(): raise RuntimeError(job)
        ledger[key]=dict(job_id=job,partition=partition,dependency=dependency,walltime=walltime,command=command)
        if not args.dry_run:
            temp=file.with_suffix('.tmp'); temp.write_text(json.dumps(ledger,indent=2));temp.replace(file)
        print(key,job,partition,dependency,flush=True)
        return job
    smoke=submit('smoke',['smoke.py'],'01:00:00',partition='gpu_test')
    downstream=[]
    for arm in ARMS:
        up=submit('up_'+arm,['train.py','--arm',arm,'--resume'],
                  '10:00:00' if arm.startswith('joint_') else '04:00:00','afterok:'+smoke)
        for domain in settings(arm)['domains']:
            for f in FRACTIONS:
                downstream.append(submit(f'down_{arm}_{domain}_p{round(100*f)}',
                    ['downstream.py','--model',arm,'--domain',domain,'--fraction',str(f)],
                    '02:00:00','afterok:'+up))
    submit('summary',['summarize.py'],'00:10:00','afterany:'+':'.join(downstream))
    print('53 jobs: smoke + 9 upstream + 42 downstream cells (210 seeds) + summary',flush=True)


if __name__=='__main__': main()
