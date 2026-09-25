"""Battery validation study first, then fresh no-validation bearing runs."""
import os,json,subprocess,sys
from pipeline import ROOT,PACKAGE,CODE,OUT,environment
from data import write

def main():
    battery=ROOT/'outputs/local_transfer_research/battery_reproduction_packages/02_interval_val200_nonnested'
    write(PACKAGE/'battery_then_bearing_status.json',dict(state='battery',pid=os.getpid()))
    with (battery/'battery_first.log').open('a',encoding='utf8') as log:
        subprocess.run([sys.executable,'-u',str(battery/'code/queue_battery_val200.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    done=json.loads((battery/'outputs/results.json').read_text())
    assert done['complete'] and len(done['rows'])==30
    rows=[]
    for seed in (42,43,44,45,46):
        env=environment(seed)
        env.update(BEARING_FEWSHOT_NAME=f'restored500_seed{seed}',BEARING_FEWSHOT_FRACTIONS='0.1,0.2,1.0')
        write(PACKAGE/'battery_then_bearing_status.json',dict(state='bearing',seed=seed,pid=os.getpid()))
        with (PACKAGE/f'restored500_seed{seed}.log').open('w') as log:
            subprocess.run([sys.executable,'-u',str(CODE/'run_bearing_delta6_no_b24_fewshot.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((OUT/f'restored500_seed{seed}/development_summary.json').read_text())
        assert result['complete'] and len(result['rows'])==6
        assert all(r['optimizer_updates']==500 for r in result['rows'])
        rows.extend(result['rows'])
        write(PACKAGE/'results_restored500.json',dict(complete=False,rows=rows))
    write(PACKAGE/'results_restored500.json',dict(complete=True,rows=rows))
    write(PACKAGE/'battery_then_bearing_status.json',dict(state='completed',runs=60))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(PACKAGE/'battery_then_bearing_status.json',dict(state='failed',error=repr(e)))
        raise
