"""One-shot gate: Raw Patch Battery completion -> F1 pipeline."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'code/data_phm').is_dir());RAW=ROOT/'code/experiments/healthtoken_raw_patch_input_ablation';STATUS=HERE/'queue_status.json'
def write(value):
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');os.replace(tmp,STATUS)
def complete(path):
    try:return json.loads(path.read_text()).get('complete') is True
    except (OSError,json.JSONDecodeError):return False
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',default='0');parser.add_argument('--poll',type=int,default=30);args=parser.parse_args();logs=HERE/'logs';logs.mkdir(parents=True,exist_ok=True)
    while True:
        try:raw=json.loads((RAW/'queue_status.json').read_text())
        except (OSError,json.JSONDecodeError):raw={}
        milling=complete(RAW/'outputs/milling/results.json');battery=complete(RAW/'outputs/battery/results.json')
        if raw.get('state')=='failed':write(dict(state='blocked_raw_failed',raw_status=raw,pid=os.getpid()));return 2
        if raw.get('state')=='complete' and milling and battery:break
        write(dict(state='waiting_for_raw_battery',raw_state=raw.get('state'),raw_domain=raw.get('domain'),milling_complete=milling,battery_complete=battery,pid=os.getpid(),gpu=args.gpu));time.sleep(args.poll)
    write(dict(state='starting_f1',pid=os.getpid(),gpu=args.gpu));env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    with (logs/'pipeline.stdout.log').open('a',buffering=1,encoding='utf8') as out,(logs/'pipeline.stderr.log').open('a',buffering=1,encoding='utf8') as err:result=subprocess.run([sys.executable,str(HERE/'run_pipeline.py'),'--gpu',str(args.gpu)],cwd=HERE,env=env,stdout=out,stderr=err)
    write(dict(state='complete' if result.returncode==0 else 'failed',returncode=result.returncode,pid=os.getpid(),gpu=args.gpu));return result.returncode
if __name__=='__main__':raise SystemExit(main())
