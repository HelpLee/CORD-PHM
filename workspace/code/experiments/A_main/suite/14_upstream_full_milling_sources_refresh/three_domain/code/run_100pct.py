"""Append 100% labelled-data downstream runs without touching 10/20% results."""
import json, os, subprocess, sys
from pathlib import Path
from pipeline import PACKAGE, CODE, OUT, environment
from data import ROOT

def main():
    rows=[]
    for seed in (42,43,44,45,46):
        env=environment(seed)
        env.update(BEARING_FEWSHOT_NAME=f'downstream100_seed{seed}',
                   BEARING_FEWSHOT_FRACTIONS='1.00',
                   BEARING_FEWSHOT_ARMS='scratch,finetune',
                   PYTHONUNBUFFERED='1')
        log=(PACKAGE/'logs'/f'downstream100_seed{seed}.log').open('w',encoding='utf8')
        try:
            subprocess.run([sys.executable,'-u',str(CODE/'run_bearing_delta6_no_b24_fewshot.py')],
                           cwd=ROOT,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        finally: log.close()
        summary=json.loads((OUT/f'downstream100_seed{seed}/development_summary.json').read_text())
        assert summary['complete'] and len(summary['rows'])==2
        rows.extend(summary['rows'])
    aggregate=[]
    for arm in ('scratch','finetune'):
        g=[r for r in rows if r['arm']==arm]
        aggregate.append({'fraction':1.0,'arm':arm,'n':len(g),
          **{m:{'mean':sum(r['metrics'][m] for r in g)/len(g)} for m in ('rmse','mae','r2','bias')}})
    (PACKAGE/'results_100pct.json').write_text(json.dumps({'rows':rows,'summary':aggregate},indent=2),encoding='utf8')

if __name__=='__main__': main()
