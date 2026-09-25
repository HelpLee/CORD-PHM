"""Submit tasks A, B, C in user order; independent jobs depend on own smoke/upstream."""
import argparse
import json
import subprocess
from config import HERE,ARMS,settings
from summarize import cells

def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    path=HERE/'submissions.json';ledger=json.loads(path.read_text()) if path.exists() else {}
    if not args.dry_run:(HERE/'logs').mkdir(exist_ok=True)
    def submit(key,command,time,dependency=''):
        if key in ledger:return ledger[key]['job_id']
        cmd=['sbatch','--parsable','--partition=gpua100','--exclude=cluster-gpu13','--job-name=e38_'+key,'--time='+time]
        if command[0]=='train.py':cmd+=['--signal=USR1@300']
        if dependency:cmd+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
        cmd+=['run_gpu.sh']+command
        jid=str(9800000+len(ledger)) if args.dry_run else subprocess.check_output(cmd,cwd=HERE,text=True).strip().split(';')[0]
        if not jid.isdigit():raise RuntimeError(jid)
        ledger[key]=dict(job_id=jid,dependency=dependency,partition='gpua100',command=cmd,walltime=time)
        if not args.dry_run:
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(path)
        print(key,jid,dependency,flush=True);return jid
    for phase in ('baseline','adaptive','bearing10'):
        smoke=submit(phase+'_smoke',['smoke.py','--phase',phase],'01:00:00')
        upstream={}
        if phase=='adaptive':
            for arm in ARMS:
                upstream[arm]=submit('up_'+arm,['train.py','--arm',arm,'--resume'],
                    '10:00:00' if arm.startswith('joint_') else '04:00:00','afterok:'+smoke)
        downs=[]
        for model,mode,domain,f in cells(phase):
            if phase=='baseline' and model!='scratch' and f!=1.:continue
            dependency=upstream.get(model,smoke)
            downs.append(submit(f'{phase}_{model}_{mode}_{domain}_p{round(100*f)}',
                ['downstream.py','--phase',phase,'--model',model,'--mode',mode,'--domain',domain,'--fraction',str(f)],
                '03:00:00' if f==1. else '02:00:00','afterok:'+dependency))
        submit(phase+'_summary',['summarize.py','--phase',phase],'00:10:00','afterany:'+':'.join(downs))
    print('SUBMITTED',len(ledger),'jobs',flush=True)

if __name__=='__main__':main()
