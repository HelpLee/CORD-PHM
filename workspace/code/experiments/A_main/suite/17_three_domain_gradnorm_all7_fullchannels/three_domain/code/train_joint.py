"""All-channel seven-source joint pretraining with domain-level GradNorm."""
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
ALL_DOMAINS = tuple(DOMAINS)

BASE=Path(__file__).resolve().parents[1]
ROOT=data.ROOT
SUITE=Path(__file__).resolve().parents[2]
EXPERIMENT=BASE.parent
A_SUITE=EXPERIMENT.parent
SHARED=A_SUITE/'01_shared_dependencies'
MILLING_RUNTIME=EXPERIMENT/'runtime/milling'
OUT=BASE/'training'
SEED=42
MAX_EPOCHS=2000
PATIENCE=30
WARMUP_EPOCHS=5
GRADNORM_ALPHA=.5
GRADNORM_INTERVAL=5
GRADNORM_LR=.025
WEIGHT_MIN=.3
WEIGHT_MAX=3.
MILLING=('luh_milling','matwi_milling','nonastreda_milling','qit_cemc_milling',
         'piecuch_milling','hmotp_milling','nasa_milling')

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
    for attempt in range(40):
        try:
            os.replace(tmp,path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.25)

def save(path,obj):
    tmp=path.with_name(f'{path.name}.{os.getpid()}.tmp')
    torch.save(obj,tmp)
    for attempt in range(40):
        try:
            os.replace(tmp,path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.25)

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
    for domain,pkg in [('bearing','bearing_fixed500_paused'),('milling','milling_val200')]:
        if domain not in DOMAINS:
            continue
        rt=MILLING_RUNTIME if domain=='milling' else SHARED/f'{domain}_runtime'
        # Frozen source caches and their historical train-only scalers.
        gl.OUT=fo.OUT=rt
        audit_path=(rt/'upstream_splits_scalers.json' if domain=='milling' else
                    rt/'attention65_upstream_seed42/upstream_splits_scalers.json')
        audit=json.loads(audit_path.read_text())
        for item in audit:
            name=item['dataset']
            if domain=='bearing':
                path=rt/'all_bearing_temporal_cache_v4'/f'{name}_bearing'
            else:
                assert name in MILLING
                path=rt/'cache'/name
            local=fo.observed_store(data.Store(path))
            if domain=='bearing':
                source=ROOT/'code/data_phm/processed_health_tokens/bearing'/f'{name}_bearing_health_tokens.npz'
            else:
                source=Path(local.info['signature']['path'])
                assert source.is_relative_to(ROOT/'code/data_phm/processed_health_tokens')
            assert source.stat().st_size==local.info['signature']['bytes'], source
            if domain=='milling': local=select_sensors(local,'all')
            store=gl.GlobalLocalStore(local,calibrate=domain=='bearing',channels=local.x.shape[1])
            # The runtime audit is the immutable split contract.  Do not
            # recompute and compare it here: a cache can enumerate units in a
            # different (but valid) order across hosts, while its recorded
            # split and normalizer remain the intended reproducible protocol.
            train = list(item['train'])
            val = list(item.get('val', item.get('validation', [])))
            missing = (set(train) | set(val)) - set(store.groups)
            if missing or set(train) & set(val):
                raise RuntimeError(
                    f'Invalid audited split for {domain}/{name}: '
                    f'missing={sorted(missing)}, overlap={sorted(set(train) & set(val))}')
            print('AUDITED_SPLIT', domain, name, len(train), len(val), flush=True)
            norm=tuple(np.asarray(v,np.float32) for v in item['center_scale'])
            wf=bearing.temporal_windows if domain=='bearing' else windows
            pool=Pool(domain,name,store,train,val,norm,wf(store,train),wf(store,val))
            assert len(pool.tw)>0
            pools[domain].append(pool)
            print('DATA_READY',domain,name,len(store.x),len(pool.tw),flush=True)
    # Battery caches are rebuilt portably from the canonical processed NPZs on
    # each machine; no workstation-only runtime files are required.
    battery.BASE=BASE/'runtime/battery_data'
    if 'battery' in DOMAINS:
        for name in battery.SOURCES: pools['battery'].append(BatteryPool(name))
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

def normalized_monitor(detail,reference):
    ratios={d:float(detail[d]['total']/reference[d]) for d in DOMAINS}
    return float(np.mean(list(ratios.values()))),ratios

def main():
    global DOMAINS, BASE, OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--domain',choices=ALL_DOMAINS,help='Single-domain matched-protocol control; weight fixed to one')
    args=parser.parse_args()
    DOMAINS=(args.domain,) if args.domain else ALL_DOMAINS
    BASE=EXPERIMENT/'single_domain'/args.domain if args.domain else EXPERIMENT/'three_domain'
    OUT=BASE/'training'
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'last.pt').exists() and not args.resume: raise RuntimeError('Existing run; use --resume')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    write(OUT/'status.json',dict(state='preparing',pid=os.getpid()))
    write(BASE/'protocol.json',dict(seed=42,max_epochs=MAX_EPOCHS,patience=PATIENCE,min_delta=1e-4,
        updates_per_epoch=20,batch_per_domain=32,joint_batch=32*len(DOMAINS),microbatch=8,
        lr=1e-4,weight_decay=1e-4,dropout=.1,gradient_clip=5,scheduler=None,
        architecture='original joint model plus domain residual LN96-Linear96x24-GELU-Linear24x96 Adapter after each shared Transformer block; zero output initialization; shared GRU96 retained; Milling reconstruction query capacity expanded 3 to 20 for canonical all-channel inputs',
        adapter_count=6,adapter_parameters=29520,reference='joint_upstream500_seed42',
        objective='domain-normalized weighted mean of (masked observed-feature MSE + 0.2 * 6-to-1 stop-gradient MSE)',
        active_domains=list(DOMAINS),minimum_update_budget=None,
        domain_balancing=dict(method='GradNorm' if len(DOMAINS)>1 else 'fixed_single_weight_1',alpha=GRADNORM_ALPHA,warmup_epochs=WARMUP_EPOCHS,
            update_interval=GRADNORM_INTERVAL,weight_lr=GRADNORM_LR,
            weight_bounds=[WEIGHT_MIN,WEIGHT_MAX],weight_sum=len(DOMAINS),
            gradient_reference='last shared Transformer block',
            loss_scale='divide each domain loss by its initial frozen validation loss'),
        sampling='dataset round robin per domain; uniform unit/row and temporal window; no R2F weighting; full eligible pool',
        validation='fixed windows/masks; macro dataset within domain; early stopping on macro relative-to-initial domain loss',
        excluded=['XJTU bearing downstream','CALCE CS2','PHM2010','ISU-ILCC'],
        sources={d:list(dict(bearing=bearing.SOURCES,battery=battery.SOURCES,milling=MILLING)[d]) for d in DOMAINS},
        only_protocol_change='GradNorm domain balancing and normalized macro validation; data, architecture, sampling, and SSL tasks unchanged'))
    pools=prepare()
    model=JointModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    named_params=[(name,param) for name,param in model.named_parameters() if param.requires_grad]
    params=[param for _,param in named_params]
    shared_indices=[i for i,(name,_) in enumerate(named_params)
                    if name.startswith('encoder.transformer.layers.1.')]
    assert shared_indices, 'No parameters found in the last shared Transformer block'
    task_weights=torch.nn.Parameter(torch.ones(len(DOMAINS),device='cuda'))
    weight_opt=torch.optim.Adam([task_weights],lr=GRADNORM_LR)
    best=float('inf');stale=0;start=1;history=[];best_epoch=0;train_reference=None
    if args.resume:
        ck=torch.load(OUT/'last.pt',weights_only=False,map_location='cpu')
        if set(ck['validation_reference']) != set(DOMAINS):
            raise RuntimeError('Checkpoint domains do not match this training arm')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1;best=ck['best'];stale=ck['stale'];best_epoch=ck['best_epoch'];history=ck['history']
        validation_reference=ck['validation_reference'];train_reference=ck['train_reference']
        with torch.no_grad(): task_weights.copy_(torch.as_tensor(ck['task_weights'],device='cuda'))
        weight_opt.load_state_dict(ck['weight_optimizer'])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    else:
        raw_initial,initial_detail=validate(model,pools)
        validation_reference={d:float(initial_detail[d]['total']) for d in DOMAINS}
        assert all(np.isfinite(v) and v>0 for v in validation_reference.values())
        write(OUT/'initial_validation.json',dict(raw_macro=raw_initial,
            domains=validation_reference,detail=initial_detail))
    began=time.time()
    for epoch in range(start,MAX_EPOCHS+1):
        model.train(); totals={d:[] for d in DOMAINS};normalized_totals={d:[] for d in DOMAINS}
        epoch_gradnorm=[]
        for update in range(20):
            domain_grads={};domain_normalized_losses=[]
            for di,d in enumerate(DOMAINS):
                # Keep each domain's original stream, even when trained alone.
                domain_seed_index=ALL_DOMAINS.index(d)
                p=pools[d][((epoch-1)*20+update)%len(pools[d])]
                rng=np.random.default_rng(SEED+epoch*10000+update*10+domain_seed_index)
                groups=list(p.groups.values())
                ids=np.array([rng.choice(groups[rng.integers(len(groups))]) for _ in range(32)])
                seq=p.tw[rng.integers(len(p.tw),size=32)]
                torch.manual_seed(SEED+epoch*10000+update*10+domain_seed_index)
                value=np.zeros(2);accumulated=[None]*len(params);normalized_value=0.
                for i in range(0,32,8):
                    with torch.autocast('cuda',dtype=torch.bfloat16): lm,ld=losses(model,p,ids[i:i+8],seq[i:i+8])
                    raw=lm+.2*ld
                    normalized=raw/(4*validation_reference[d])
                    assert torch.isfinite(normalized),(d,p.name)
                    grads=torch.autograd.grad(normalized,params,allow_unused=True)
                    for pi,g in enumerate(grads):
                        if g is not None:
                            accumulated[pi]=g.detach() if accumulated[pi] is None else accumulated[pi]+g.detach()
                    value+=np.array([lm.item(),ld.item()])/4
                    normalized_value+=float(normalized.detach())
                totals[d].append(value.tolist())
                normalized_totals[d].append(normalized_value)
                domain_normalized_losses.append(normalized_value)
                domain_grads[d]=accumulated
            base_norms=[]
            for d in DOMAINS:
                squares=[domain_grads[d][pi].float().square().sum() for pi in shared_indices
                         if domain_grads[d][pi] is not None]
                base_norms.append(torch.sqrt(torch.stack(squares).sum()).detach())
            base_norms=torch.stack(base_norms)
            global_update=(epoch-1)*20+update+1
            gradnorm_loss=None
            if len(DOMAINS)>1 and epoch>WARMUP_EPOCHS and global_update%GRADNORM_INTERVAL==0:
                assert train_reference is not None
                rates=torch.as_tensor(domain_normalized_losses,device='cuda')/torch.as_tensor(train_reference,device='cuda')
                rates=rates/rates.mean().clamp_min(1e-12)
                weighted_norms=task_weights*base_norms
                targets=weighted_norms.mean().detach()*rates.pow(GRADNORM_ALPHA)
                gradnorm_loss=(weighted_norms-targets).abs().sum()
                weight_opt.zero_grad(set_to_none=True);gradnorm_loss.backward();weight_opt.step()
                with torch.no_grad():
                    task_weights.clamp_(WEIGHT_MIN,WEIGHT_MAX)
                    task_weights.mul_(len(DOMAINS)/task_weights.sum())
            opt.zero_grad(set_to_none=True)
            detached_weights=task_weights.detach()/len(DOMAINS)
            for pi,param in enumerate(params):
                pieces=[domain_grads[d][pi]*detached_weights[di] for di,d in enumerate(DOMAINS)
                        if domain_grads[d][pi] is not None]
                param.grad=torch.stack(pieces).sum(0) if pieces else None
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            assert torch.isfinite(grad)
            opt.step()
            epoch_gradnorm.append(dict(update=global_update,weights=task_weights.detach().cpu().tolist(),
                base_gradient_norms=base_norms.cpu().tolist(),normalized_losses=domain_normalized_losses,
                gradnorm_loss=(None if gradnorm_loss is None else float(gradnorm_loss.detach()))))
            write(OUT/'status.json',dict(state='training',pid=os.getpid(),epoch=epoch,joint_update=global_update,
                domain_weights={d:float(task_weights[di].detach()) for di,d in enumerate(DOMAINS)}))
            if update==0: print('JOINT_UPDATE',epoch,global_update,'WEIGHTS',task_weights.detach().cpu().tolist(),flush=True)
        epoch_normalized=[float(np.mean(normalized_totals[d])) for d in DOMAINS]
        if train_reference is None:
            train_reference=epoch_normalized
        raw_monitor,detail=validate(model,pools)
        monitor,validation_ratios=normalized_monitor(detail,validation_reference)
        assert np.isfinite(monitor)
        improved=monitor<best-1e-4
        if improved: best=monitor;best_epoch=epoch;stale=0
        else: stale+=1
        history.append(dict(epoch=epoch,train={d:np.mean(totals[d],axis=0).tolist() for d in DOMAINS},
            normalized_train={d:epoch_normalized[di] for di,d in enumerate(DOMAINS)},
            task_weights={d:float(task_weights[di].detach()) for di,d in enumerate(DOMAINS)},
            gradnorm=epoch_gradnorm,validation=detail,validation_ratios=validation_ratios,
            raw_macro_validation=raw_monitor,monitor=monitor,seconds=time.time()-began))
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        ck=dict(model=state,optimizer=opt.state_dict(),weight_optimizer=weight_opt.state_dict(),
            task_weights=task_weights.detach().cpu().tolist(),validation_reference=validation_reference,
            train_reference=train_reference,epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,history=history,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        save(OUT/'last.pt',ck)
        if improved:
            save(OUT/'best.pt',ck);save(OUT/'encoder.pt',{k:v.cpu() for k,v in model.encoder.state_dict().items()})
        write(OUT/'history.json',history)
        print('EPOCH',epoch,'VAL',monitor,'BEST',best_epoch,'STALE',stale,flush=True)
        if stale>=PATIENCE: break
    write(OUT/'status.json',dict(state='complete',epoch=epoch,best_epoch=best_epoch,best_validation=best,
        final_domain_weights={d:float(task_weights[di].detach()) for di,d in enumerate(DOMAINS)},
        reason='early_stopping' if stale>=PATIENCE else 'max_epochs'))

if __name__=='__main__': main()
