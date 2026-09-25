"""The agreed three-domain source-only pretraining, maximum 500 epochs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, json, random, time, hashlib, sys
from pathlib import Path
import numpy as np
import torch
import data
sys.path.insert(1, str(data.ROOT / 'code'))
import global_local_data as gl, feature_observations as fo
import run_bearing_all_temporal_variable_channels_upstream as bearing
import run_battery_fullrows_65token as battery
from sensor_selection import select_sensors
from joint_model import JointModel, DOMAINS

BASE=Path(__file__).resolve().parents[1]
ROOT=data.ROOT
SUITE=Path(__file__).resolve().parents[2]
EXPERIMENT=BASE.parent
A_SUITE=EXPERIMENT.parent
SHARED=A_SUITE/'01_shared_dependencies'
MILLING_RUNTIME=EXPERIMENT/'runtime/milling'
OUT=BASE/'training'
SEED=42
MILLING=('luh_milling','matwi_milling','nonastreda_milling','qit_cemc_milling',
         'piecuch_milling','hmotp_milling','nasa_milling')

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)

def save(path,obj):
    tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp); os.replace(tmp,path)

class Pool:
    def __init__(self,domain,name,store,train,val,norm,tw,vw):
        self.domain,self.name,self.store,self.norm=domain,name,store,norm
        self.groups=store.eligible(train)
        self.tw=tw
        vg=store.eligible(val)
        self.vr=np.concatenate(list(vg.values())) if vg else np.empty(0,dtype=int)
        self.vw=vw
        self.audit=dict(domain=domain,name=name,source=store.info['signature'],train=train,val=val,
                       train_windows=len(tw),val_windows=len(vw),rows=len(store.x),
                       norm=[v.tolist() for v in norm],calibration=store.calibration)
    def batch(self,ids):
        vals=self.store.transform(ids,self.norm)
        return tuple(torch.as_tensor(v,device='cuda') for v in (*vals,np.asarray(self.store.feature_mask[ids])))

class BatteryPool:
    def __init__(self,name):
        self.domain,self.name='battery',name
        self.c=battery.Corpus(name)
        self.groups={str(i):ids for i,ids in enumerate(self.c.train_groups)}
        self.tw,self.vw=self.c.windows['train'],self.c.windows['val']
        self.vr=self.c.val_rows
        self.audit=dict(domain='battery',name=name,**self.c.audit,norm=self.c.norm.tolist())
    def batch(self,ids): return self.c.batch(ids)

def windows(s,units):
    blocks=[np.lib.stride_tricks.sliding_window_view(s.groups[u],7) for u in units if len(s.groups[u])>=7]
    return np.concatenate(blocks) if blocks else np.empty((0,7),dtype=int)

def prepare():
    pools={d:[] for d in DOMAINS}
    for domain,pkg in [('milling','milling_val200')]:
        rt=MILLING_RUNTIME
        # Frozen source caches and their historical train-only scalers.
        gl.OUT=fo.OUT=rt
        audit=json.loads((rt/'upstream_splits_scalers.json').read_text())
        for item in audit:
            name=item['dataset']
            if domain=='bearing':
                path=rt/'all_bearing_temporal_cache_v4'/f'{name}_bearing'
            else:
                assert name in MILLING
                path=rt/'cache'/name
            local=fo.observed_store(data.Store(path))
            source=Path(local.info['signature']['path'])
            assert source.is_relative_to(ROOT/'code/data_phm/processed_health_tokens')
            assert source.stat().st_size==local.info['signature']['bytes'], source
            if domain=='milling': local=select_sensors(local,'all')
            store=gl.GlobalLocalStore(local,calibrate=domain=='bearing',channels=local.x.shape[1])
            train,val=store.split()
            assert train==item['train'] and val==item.get('val',item.get('validation'))
            norm=tuple(np.asarray(v,np.float32) for v in item['center_scale'])
            wf=bearing.temporal_windows if domain=='bearing' else windows
            pool=Pool(domain,name,store,train,val,norm,wf(store,train),wf(store,val))
            assert len(pool.tw)>0
            pools[domain].append(pool)
            print('DATA_READY',domain,name,len(store.x),len(pool.tw),flush=True)
    for domain in DOMAINS:
        assert any(len(p.vw) and len(p.vr) for p in pools[domain]),domain
    write(OUT/'data_audit.json',[p.audit for d in DOMAINS for p in pools[d]])
    return pools

def losses(model,p,ids,seq,fixed=None):
    lm=model.reconstruction(p.domain,*p.batch(ids),fixed=fixed)
    vals=p.batch(seq.reshape(-1))
    ld=model.dynamics(p.domain,*vals[:4])
    return lm,ld

@torch.no_grad()
def validate(model,pools):
    model.eval(); result={}
    for d in DOMAINS:
        scores={}
        for p in pools[d]:
            if not len(p.vr) or not len(p.vw): continue
            n=min(64,len(p.vr),len(p.vw))
            ids=p.vr[np.linspace(0,len(p.vr)-1,n,dtype=int)]
            seq=p.vw[np.linspace(0,len(p.vw)-1,n,dtype=int)]
            total=np.zeros(2)
            for i in range(0,n,8):
                lm,ld=losses(model,p,ids[i:i+8],seq[i:i+8],9000+i)
                total+=np.array([lm.item(),ld.item()])*len(ids[i:i+8])/n
            scores[p.name]=dict(mask=total[0],temporal=total[1],total=float(total[0]+.2*total[1]))
        result[d]=dict(datasets=scores,total=float(np.mean([v['total'] for v in scores.values()])))
    return float(np.mean([v['total'] for v in result.values()])),result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'last.pt').exists() and not args.resume: raise RuntimeError('Existing run; use --resume')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    write(OUT/'status.json',dict(state='preparing',pid=os.getpid()))
    write(BASE/'protocol.json',dict(seed=42,max_epochs=500,patience=30,min_delta=1e-4,
        updates_per_epoch=20,batch_per_domain=32,joint_batch=32,microbatch=8,
        lr=1e-4,weight_decay=1e-4,dropout=.1,gradient_clip=5,scheduler=None,
        architecture='original joint model plus domain residual LN96-Linear96x24-GELU-Linear24x96 Adapter after each shared Transformer block; zero output initialization; shared GRU96 retained; Milling reconstruction query capacity expanded 3 to 20 for canonical all-channel inputs',
        adapter_count=2,adapter_parameters=9840,reference='joint_upstream500_adapter_seed42',
        objective='macro mean domains (masked observed-feature MSE + 0.2 * 6-to-1 stop-gradient MSE)',
        sampling='dataset round robin per domain; uniform unit/row and temporal window; no R2F weighting; full eligible pool',
        validation='same deterministic unit-disjoint rule; fixed windows/masks; macro dataset then macro domain',
        excluded=['XJTU bearing downstream','CALCE CS2','PHM2010','ISU-ILCC'],
        initialization='identical joint initialization then prune bearing/battery modules',
        sampling_domain_rng_index=2,
        sources=dict(milling=list(MILLING)),
        only_protocol_change='canonical seven-source all-channel Milling input; all optimization and objective settings unchanged'))
    pools=prepare()
    model=JointModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    best=float('inf');stale=0;start=1;history=[];best_epoch=0
    if args.resume:
        ck=torch.load(OUT/'last.pt',weights_only=False,map_location='cpu')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1;best=ck['best'];stale=ck['stale'];best_epoch=ck['best_epoch'];history=ck['history']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    began=time.time()
    for epoch in range(start,501):
        model.train(); totals={d:[] for d in DOMAINS}
        for update in range(20):
            opt.zero_grad(set_to_none=True)
            for di,d in enumerate(DOMAINS):
                di=2  # Original milling domain index: preserve sampled rows/windows/masks.
                p=pools[d][((epoch-1)*20+update)%len(pools[d])]
                rng=np.random.default_rng(SEED+epoch*10000+update*10+di)
                groups=list(p.groups.values())
                ids=np.array([rng.choice(groups[rng.integers(len(groups))]) for _ in range(32)])
                seq=p.tw[rng.integers(len(p.tw),size=32)]
                torch.manual_seed(SEED+epoch*10000+update*10+di)
                value=np.zeros(2)
                for i in range(0,32,8):
                    with torch.autocast('cuda',dtype=torch.bfloat16): lm,ld=losses(model,p,ids[i:i+8],seq[i:i+8])
                    loss=(lm+.2*ld)/4
                    assert torch.isfinite(loss),(d,p.name)
                    loss.backward();value+=np.array([lm.item(),ld.item()])/4
                totals[d].append(value.tolist())
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            assert torch.isfinite(grad)
            opt.step()
            write(OUT/'status.json',dict(state='training',pid=os.getpid(),epoch=epoch,joint_update=(epoch-1)*20+update+1))
            if update==0: print('JOINT_UPDATE',epoch,(epoch-1)*20+1,flush=True)
        monitor,detail=validate(model,pools)
        assert np.isfinite(monitor)
        improved=monitor<best-1e-4
        if improved: best=monitor;best_epoch=epoch;stale=0
        else: stale+=1
        history.append(dict(epoch=epoch,train={d:np.mean(totals[d],axis=0).tolist() for d in DOMAINS},validation=detail,monitor=monitor,seconds=time.time()-began))
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        ck=dict(model=state,optimizer=opt.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,history=history,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        save(OUT/'last.pt',ck)
        if improved:
            save(OUT/'best.pt',ck);save(OUT/'encoder.pt',{k:v.cpu() for k,v in model.encoder.state_dict().items()})
        write(OUT/'history.json',history)
        print('EPOCH',epoch,'VAL',monitor,'BEST',best_epoch,'STALE',stale,flush=True)
        if stale>=30: break
    write(OUT/'status.json',dict(state='complete',epoch=epoch,best_epoch=best_epoch,best_validation=best,reason='early_stopping' if stale>=30 else 'max_epochs'))

if __name__=='__main__': main()
