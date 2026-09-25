"""Isolated package runner. Validate, pretrain, then automatically run five seeds."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from data import ROOT, OUT, write

PACKAGE=Path(__file__).resolve().parents[1]
CODE=PACKAGE/'code'
OLD=ROOT/'outputs/local_transfer_research/bearing_reproduction_packages/03_all_bearings_variable_channels'
FIXED=ROOT/'outputs/local_transfer_research/bearing_reproduction_packages/01_fixed500_nested'
STEMS=('cwru','femto','ferrara','ims','kaist','seu','unsw')

def environment(seed=42):
    env=os.environ.copy()
    env.update(BEARING_DYNAMICS_NAME='attention65_upstream_seed42',BEARING_INDEPENDENT_CHANNELS='0',
        BEARING_FEWSHOT_SEED=str(seed),BEARING_FEWSHOT_NAME=f'downstream_seed{seed}',
        BEARING_FEWSHOT_UNITS='Bearing2_1,Bearing2_2,Bearing2_3,Bearing2_4,Bearing2_5',
        BEARING_FEWSHOT_HELD='Bearing2_1',BEARING_FEWSHOT_EXCLUDED='none',
        BEARING_FEWSHOT_FRACTIONS='0.10,0.20',BEARING_FEWSHOT_ARMS='scratch,finetune',
        BEARING_FEWSHOT_EPOCHS='1000000',BEARING_FEWSHOT_MAX_UPDATES='500',
        BEARING_FEWSHOT_DETERMINISTIC='1',BEARING_FEWSHOT_NESTED_LABELS='1',
        BEARING_FEWSHOT_ENCODER_LR='1e-4',BEARING_FEWSHOT_HEAD_LR='1e-3',
        BEARING_FEWSHOT_DROPOUT='0.05',BEARING_FEWSHOT_SWA_EPOCHS='0',
        BEARING_FEWSHOT_CHANNELS='2',BEARING_FEWSHOT_VARIABLE_CHANNELS='1',
        BEARING_UPSTREAM_CHECKPOINT=str(OUT/'attention65_upstream_seed42/upstream/encoder.pt'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    return env

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    copies=[(OLD/'results/all_bearing_temporal_cache_v4',OUT/'all_bearing_temporal_cache_v4'),
            (ROOT/'outputs/local_transfer_research/global_feature_cache/row_scoped_v4',OUT/'global_feature_cache/row_scoped_v4'),
            (ROOT/'outputs/local_transfer_research/global_feature_cache/xjtu_condition2',OUT/'global_feature_cache/xjtu_condition2'),
            (ROOT/'outputs/local_transfer_research/global_local_cache/xjtu_condition2',OUT/'global_local_cache/xjtu_condition2')]
    for source,dest in copies:
        if source.exists() and not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            print('COPY_VERIFIED_DERIVED_CACHE',source,flush=True)
            shutil.copytree(source,dest)
    for stem in STEMS:
        manifest=OUT/f'all_bearing_temporal_cache_v4/{stem}_bearing/manifest.json'
        if not manifest.exists():
            continue  # Original builder reconstructs from canonical NPZ/raw signals.
        m=json.loads(manifest.read_text()); p=Path(m['signature']['path'])
        assert p.parent.resolve()==(ROOT/'code/data_phm/processed_health_tokens/bearing').resolve()
        assert p.stat().st_size==m['signature']['bytes'] and p.stat().st_mtime_ns==m['signature']['mtime_ns']
    protocol=dict(architecture='native channels -> attention Local/Global fusion -> fixed65 Transformer -> snapshot96',
        upstream_sources=list(STEMS),upstream_reference=str(OLD),downstream_reference=str(FIXED),
        upstream=dict(seed=42,epochs=200,patience=20,updates_per_epoch=20,batch=32,microbatch=8,
                      dropout=.1,lr=1e-4,weight_decay=1e-4,mask_ratio=.3,dynamics_weight=.2,
                      sampling='dataset round robin, uniform unit/row, uniform temporal window; no R2F weighting',
                      selection='same valid >=7 sequences; UNSW6Hz; no snapshot cap',
                      calibration='original first retained snapshot RMS and train-only normalization'),
        downstream=dict(seeds=[42,43,44,45,46],fractions=[.1,.2],arms=['scratch','finetune'],
                        test='Bearing2_1',train=['Bearing2_2','Bearing2_3','Bearing2_4','Bearing2_5'],
                        max_updates=500,nested_labels=True,dropout=.05,deterministic=True,
                        encoder_lr=1e-4,head_lr=1e-3,scratch_lr=1e-3,freeze=False,
                        validation=None,batch=32,microbatch=8,huber_beta=.05,history=6,
                        labels='same historical FPT map and linear post-FPT normalized RUL'),
        change_note='Upstream data protocol from03; downstream optimization from01. Not a single-change comparison to03.',
        python=sys.executable,
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob('*.py')})
    write(PACKAGE/'protocol.json',protocol)

def execute(script,seed=None):
    name=Path(script).stem+(f'_seed{seed}' if seed is not None else '')
    logs=PACKAGE/'logs';logs.mkdir(exist_ok=True)
    with (logs/(name+'.log')).open('a',encoding='utf8') as stream:
        subprocess.run([sys.executable,'-u',str(CODE/script)],cwd=ROOT,env=environment(seed or 42),
                       stdout=stream,stderr=subprocess.STDOUT,check=True)

def aggregate():
    rows=[]
    for seed in (42,43,44,45,46):
        done=json.loads((OUT/f'downstream_seed{seed}/development_summary.json').read_text())
        assert done['complete'] and len(done['rows'])==4
        for row in done['rows']:
            assert row['optimizer_updates']==500 and row['nested_labels'] and row['deterministic'] and row['dropout']==.05
        rows.extend(done['rows'])
    summary=[]
    for f in (.1,.2):
        for a in ('scratch','finetune'):
            group=[r for r in rows if r['fraction']==f and r['arm']==a]
            summary.append(dict(fraction=f,arm=a,n=len(group),**{k:dict(mean=float(np.mean([r['metrics'][k] for r in group])),
                std=float(np.std([r['metrics'][k] for r in group],ddof=1))) for k in ('rmse','mae','r2','bias')}))
    write(PACKAGE/'results.json',dict(rows=rows,summary=summary))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    prepare()
    if args.prepare_only:
        execute('verify.py')
        return
    lock=PACKAGE/'RUNNING.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        write(PACKAGE/'status.json',dict(state='validating',pid=os.getpid()))
        execute('verify.py')
        write(PACKAGE/'status.json',dict(state='upstream',pid=os.getpid()))
        execute('run_bearing_all_temporal_variable_channels_upstream.py')
        assert (OUT/'attention65_upstream_seed42/upstream/summary.json').exists()
        for seed in (42,43,44,45,46):
            write(PACKAGE/'status.json',dict(state='downstream',seed=seed,pid=os.getpid()))
            execute('run_bearing_delta6_no_b24_fewshot.py',seed)
        aggregate()
        write(PACKAGE/'status.json',dict(state='completed',runs=20))
    except BaseException as error:
        write(PACKAGE/'status.json',dict(state='failed',error=repr(error),pid=os.getpid()))
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
