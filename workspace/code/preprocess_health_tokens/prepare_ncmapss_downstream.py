"""Native DS02 flights -> one unscaled [14,64,26] HealthToken per flight.

Run from code: python -m preprocess_health_tokens.prepare_ncmapss_downstream
No engine selection, resampling, learned scaling or flight truncation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import time
import os
from concurrent.futures import ProcessPoolExecutor
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import h5py
import numpy as np
import pycatch22

CODE = Path(__file__).resolve().parents[1]
CHANNELS = ['Wf','Nf','Nc','T24','T30','T48','T50','P15','P2','P21','P24','Ps30','P40','P50']

def descriptors(y):
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite raw signal: refusing to interpolate or remove samples')
    result = pycatch22.catch22_all(y.tolist())
    t = np.arange(len(y), dtype=np.float64)
    tc = t-t.mean()
    slope = np.dot(tc,y-y.mean())/np.dot(tc,tc) if len(y)>1 else np.nan
    v = np.asarray(result['values']+[float(y.mean()),float(y.std()),float(slope),float(y[-1]-y[0])])
    valid = np.isfinite(v) & (np.abs(v)<=np.finfo(np.float32).max)
    return np.where(valid,v,0).astype(np.float32),valid

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def channel_descriptors(y):
    edges=np.linspace(0,len(y),65,dtype=np.int64)
    pairs=[descriptors(y[edges[j]:edges[j+1]]) for j in range(64)]
    g,gm=descriptors(y)
    return np.stack([p[0] for p in pairs]),np.stack([p[1] for p in pairs]),g,gm

def build(raw,output):
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    start=time.time()
    names=pycatch22.catch22_all(list(range(100)))['names']+['mean','std','linear_slope','end_start_delta']
    xs=[]; masks=[]; globals_=[]; gmasks=[]; contexts=[]; units=[]; cycles=[]; splits=[]; labels=[]; counts=[]; spans=[]
    with ProcessPoolExecutor(max_workers=4) as pool, h5py.File(raw,'r') as f:
        decode=lambda k:[x.decode() for x in f[k][:]]
        sensor_names=decode('X_s_var'); auxiliary=decode('A_var'); context_names=decode('W_var')
        cols=[sensor_names.index(c) for c in CHANNELS]
        for split in ['dev','test']:
            a=f['A_'+split][:]
            unit=a[:,auxiliary.index('unit')].astype(int); cycle=a[:,auxiliary.index('cycle')].astype(int)
            bounds=np.r_[0,np.flatnonzero((np.diff(unit)!=0)|(np.diff(cycle)!=0))+1,len(a)]
            seen=set()
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                key=(int(unit[lo]),int(cycle[lo]))
                if key in seen: raise ValueError(f'Noncontiguous flight {key}')
                seen.add(key)
                signal=f['X_s_'+split][lo:hi][:,cols]
                rul=f['Y_'+split][lo:hi].reshape(-1)
                if not np.allclose(rul,rul[0]): raise ValueError(f'RUL changes within flight {key}')
                if len(signal)<64*10: raise ValueError(f'Flight too short for Catch22: {key}')
                edges=np.linspace(0,len(signal),65,dtype=np.int64)
                x=np.zeros((14,64,26),np.float32); mask=np.zeros_like(x,dtype=bool)
                glob=np.zeros((14,26),np.float32); gmask=np.zeros_like(glob,dtype=bool)
                for c,result in enumerate(pool.map(channel_descriptors,[signal[:,c] for c in range(14)])):
                    x[c],mask[c],glob[c],gmask[c]=result
                xs.append(x);masks.append(mask);globals_.append(glob);gmasks.append(gmask)
                contexts.append(f['W_'+split][lo:hi].mean(axis=0))
                units.append(key[0]);cycles.append(key[1]);splits.append(split);labels.append(float(rul[0]));counts.append(hi-lo);spans.append([lo,hi])
                if len(xs)%10==0:
                    progress=dict(flights=len(xs),split=split,unit=key[0],flight=key[1],elapsed_seconds=time.time()-start)
                    output.with_suffix('.progress.json').write_text(json.dumps(progress,indent=2))
                    print(json.dumps(progress),flush=True)
    unit=np.asarray(units);cycle=np.asarray(cycles);rul=np.asarray(labels,dtype=np.float32)
    norm=np.zeros_like(rul)
    for u in np.unique(unit):
        ids=np.flatnonzero(unit==u)
        if not np.all(np.diff(cycle[ids])>0): raise ValueError('Flight order invalid')
        norm[ids]=rul[ids]/max(float(rul[ids].max()),1.)
    x=np.stack(xs);fm=np.stack(masks)
    metadata=dict(dataset='ncmapss_ds02_downstream',snapshot='one complete flight',x_health_scaled=False,
        scaling_deferred_to_experiment=True,window_definition='64 contiguous equal-sample windows, complete native flight',
        resampling=False,clipping=False,channel_names=CHANNELS,feature_names=names,
        slope_time_unit='native sample interval (slope per sample)',std_ddof=0,
        invalid_features='zero with feature_mask false; no interpolation',
        official_split='annotation only; dev and test combined in one NPZ',
        target='Y from raw H5; y_rul_norm=Y/max(Y) separately per engine, labels only',
        global_definition='same 26 descriptors computed on complete flight per sensor',
        context_definition='mean W per complete flight, kept separate from health channels',
        source_path=str(raw.resolve()),source_sha256=sha256(raw),script_sha256=sha256(Path(__file__)),
        versions={p:importlib.metadata.version(p) for p in ['numpy','h5py','pycatch22']},
        shape=list(x.shape),elapsed_seconds=time.time()-start)
    np.savez_compressed(output,x_health=x,feature_mask=fm,token_mask=np.ones(x.shape[:3],bool),
        c_mask=np.ones(x.shape[:2],bool),x_global=np.stack(globals_),global_feature_mask=np.stack(gmasks),
        global_context=np.asarray(contexts,np.float32),context_names=np.asarray(context_names),
        feature_names=np.asarray(names),channel_names=np.asarray(CHANNELS),
        sample_unit_id=np.asarray(['U'+str(u) for u in unit]),sample_engine_id=unit,
        sample_cycle_index=cycle,sample_flight_index=cycle,sample_official_split=np.asarray(splits),
        sample_source_row_range=np.asarray(spans),sample_num_points=np.asarray(counts),
        sample_dataset=np.full(len(x),'ncmapss_ds02'),y_rul=rul,y_rul_norm=norm,
        meta_json=np.asarray(json.dumps(metadata)))
    with np.load(output) as z:
        assert z['x_health'].shape==(len(x),14,64,26)
        assert np.isfinite(z['x_health']).all() and (z['x_health'][~z['feature_mask']]==0).all()
        assert len(set(zip(z['sample_engine_id'],z['sample_flight_index'])))==len(x)
    metadata['output_sha256']=sha256(output)
    metadata['engine_flights']={str(u):int(sum(unit==u)) for u in np.unique(unit)}
    metadata['completed']=True
    output.with_suffix('.manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=CODE/'data_phm/raw/Engine/N-CMAPSS_DS02-006.h5')
    p.add_argument('--output',type=Path,default=CODE/'data_phm/processed_health_tokens/engine/ncmapss_ds02_downstream_health_tokens.npz')
    a=p.parse_args();build(a.raw,a.output)
