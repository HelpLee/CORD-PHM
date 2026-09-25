"""Bearing Adapter transfer: 200 epochs, interval source validation, no warmup."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import global_local_data as gl
import run_bearing_delta6_no_b24_fewshot as old
from compression_model import Attention65RUL
from adapter_encoder import DomainAdapterEncoder
from helpers import st,hh,score
from data import write
from joint_model import JointModel

SUITE=Path(__file__).resolve().parents[2]
TRAIN=('Bearing2_2','Bearing2_3','Bearing2_4','Bearing2_5')
TEST='Bearing2_1'
SEEDS=(42,43,44,45,46)
FRACTIONS=(.1,.2,1.)
PARTIAL=('transformer.layers.1.','final_norm.','adapters.1.')

class BearingAdapterRUL(Attention65RUL):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder=DomainAdapterEncoder(channels=2,variable_channels=True,dropout=.05)
        self.delta_projection=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,96),nn.GELU())
        self.gru=nn.GRU(96,96,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(.05),nn.Linear(64,1))
    # Inherit the exact historical six-state forward and masking behavior.

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def convert(path):
    source=torch.load(path,map_location='cpu',weights_only=True)
    converted={k:v for k,v in source.items() if not k.startswith(('stems.','adapters.'))}
    for k,v in source.items():
        if k.startswith('adapters.bearing.'):
            converted[k.replace('adapters.bearing.','adapters.',1)]=v
    for target,origin in [('local_norm','local.0'),('local_proj','local.1'),('global_norm','global.0'),('global_proj','global.1')]:
        for field in ('weight','bias'):converted[f'{target}.{field}']=source[f'stems.bearing.{origin}.{field}']
    test=BearingAdapterRUL().encoder
    test.load_state_dict(converted,strict=True)
    assert len(converted)==58
    return source,converted

def configure(model,arm,stage):
    for name,p in model.named_parameters():
        active=not name.startswith('encoder.') or arm in ('scratch','full_finetune')
        if name.startswith('encoder.') and arm=='partial_finetune' and stage==2:
            active=name.removeprefix('encoder.').startswith(PARTIAL)
        p.requires_grad_(active)

def optimizer_for(model,arm,stage):
    configure(model,arm,stage)
    if arm=='scratch':return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    enc=[p for p in model.encoder.parameters() if p.requires_grad]
    head=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]
    groups=([dict(params=enc,lr=.0001)] if enc else [])+[dict(params=head,lr=.001)]
    return torch.optim.AdamW(groups,weight_decay=1e-4)

def set_mode(model,arm,stage):
    model.train()
    if arm in ('partial_finetune','frozen_probe'):
        model.encoder.eval()
        if arm=='partial_finetune' and stage==2:
            model.encoder.transformer.layers[1].train()
            model.encoder.final_norm.train()
            model.encoder.adapters[1].train()

def make_model(seed,arm,checkpoint):
    seed_all(seed);model=BearingAdapterRUL()
    if arm!='scratch':model.encoder.load_state_dict(checkpoint,strict=True)
    configure(model,arm,1)
    return model

def prepare():
    from portable_inputs import load_store
    store=load_store('bearing')
    norm=store.normalize(TRAIN)
    sequences,masks=store.sequences_with_mask(TRAIN,6)
    splits={}; validations={}
    for fraction in FRACTIONS:
        trs=[]; tms=[]; vas=[]; vms=[]; counts={}
        for unit in TRAIN:
            seq,mask=store.sequences_with_mask((unit,),6)
            n=max(2,int(np.ceil(len(seq)*fraction)))
            selected=np.unique(np.rint(np.linspace(0,len(seq)-1,n)).astype(int))
            nv=max(1,int(np.ceil(len(selected)*.2)))
            vi=np.unique(np.rint(np.linspace(0,len(selected)-1,nv)).astype(int))
            ti=np.setdiff1d(np.arange(len(selected)),vi)
            assert len(ti)>0
            trs.append(seq[selected[ti]]);tms.append(mask[selected[ti]])
            vas.append(seq[selected[vi]]);vms.append(mask[selected[vi]])
            counts[unit]=dict(total=len(seq),selected=len(selected),train=len(ti),validation=len(vi))
        splits[fraction]=(np.concatenate(trs),np.concatenate(tms),counts)
        validations[fraction]=(np.concatenate(vas),np.concatenate(vms))
        assert set(splits[fraction][0][:,-1]).isdisjoint(validations[fraction][0][:,-1])
    test,tm=store.sequences_with_mask((TEST,),6)
    assert set(test[:,-1]).isdisjoint(splits[max(FRACTIONS)][0][:,-1])
    return store,norm,splits,validations,test,tm

def verify(store,norm,splits,source,checkpoint):
    seed_all(42);original=Attention65RUL(channels=2,variable_channels=True,dropout=.05,head_dropout=.05)
    seed_all(42);model=BearingAdapterRUL().eval()
    for k,v in original.state_dict().items():assert torch.equal(v,model.state_dict()[k]),k
    if checkpoint is not None:model.encoder.load_state_dict(checkpoint,strict=True)
    # Use source-only real snapshots, not held-out labels, for a gradient check.
    seq,mask,_=splits[.1]
    arrays=tuple(torch.as_tensor(v) for v in store.transform(seq[:2].reshape(-1),norm))
    with torch.no_grad():
        if source is not None:
            ref=JointModel().encoder.eval()
            for d in tuple(ref.stems):
                if f'stems.{d}.local.0.weight' not in source:del ref.stems[d];del ref.adapters[d]
            ref.load_state_dict(source,strict=True)
            expected,local=ref('bearing',*arrays)
            actual=model.encoder(*arrays)
            torch.testing.assert_close(expected,actual['snapshot'],rtol=0,atol=0)
            torch.testing.assert_close(local,actual['local_hidden'],rtol=0,atol=0)
    batch=[v.reshape(2,6,*v.shape[1:]) for v in arrays]
    loss=F.smooth_l1_loss(model(*batch,torch.as_tensor(mask[:2])),torch.as_tensor(store.y[seq[:2,-1]]),beta=.05)
    assert torch.isfinite(loss);loss.backward()
    for arm in ('scratch','frozen_probe','partial_finetune','full_finetune'):
        for stage in (1,2):
            configure(model,arm,stage);set_mode(model,arm,stage)
            for n,p in model.encoder.named_parameters():
                want=arm in ('scratch','full_finetune') or (arm=='partial_finetune' and stage==2 and n.startswith(PARTIAL))
                assert p.requires_grad==want
            if arm=='frozen_probe':assert not model.encoder.training
    schedule={str(f):dict(labeled=len(s[0]),updates_per_epoch=int(np.ceil(len(s[0])/32)),
                          frozen_epochs=0,max_epochs=200) for f,s in splits.items()}
    optimizer_for(model,'partial_finetune',2)
    assert all(p.requires_grad==n.startswith(PARTIAL) for n,p in model.encoder.named_parameters())
    return dict(passed=True,stage_at_epoch1=2,encoder_exact=source is not None,original_initial_weights_exact=True,
                freeze_masks=True,nested=False,validation_fraction=.2,schedule=schedule)

def train(model,arm,gpu,store,seq,masks,validation,seed,package):
    stage=2 if arm=='partial_finetune' else 1
    opt=optimizer_for(model,arm,stage);epoch=updates=0;history=[];order_hash=hashlib.sha256();unfrozen=0
    best=float('inf');best_epoch=0;weights=None
    began=time.time()
    sequence_tensor = (torch.as_tensor(seq,device='cuda')
                       if os.environ.get('HEALTHTOKEN_PRELOAD_INDICES','0') == '1' else None)
    mask_tensor = (torch.as_tensor(masks,device='cuda')
                   if os.environ.get('HEALTHTOKEN_PRELOAD_INDICES','0') == '1' else None)
    target_tensor = (torch.as_tensor(store.y[seq[:,-1]],device='cuda')
                     if os.environ.get('HEALTHTOKEN_PRELOAD_INDICES','0') == '1' else None)
    for epoch in range(1,201):
        torch.manual_seed(seed+epoch);set_mode(model,arm,stage)
        order=np.random.default_rng(seed+epoch).permutation(len(seq));order_hash.update(order.tobytes())
        losses=[]
        for start in range(0,len(order),32):
            selected=order[start:start+32];opt.zero_grad(set_to_none=True)
            for micro in range(0,len(selected),8):
                take=selected[micro:micro+8]
                if sequence_tensor is None:
                    rows=torch.as_tensor(seq[take],device='cuda')
                    hm=torch.as_tensor(masks[take],device='cuda')
                    target=torch.as_tensor(store.y[seq[take,-1]],device='cuda')
                else:
                    take_tensor=torch.as_tensor(take,device='cuda')
                    rows=sequence_tensor[take_tensor]
                    hm=mask_tensor[take_tensor]
                    target=target_tensor[take_tensor]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=F.smooth_l1_loss(model(*(v[rows] for v in gpu),hm),target,beta=.05)
                assert torch.isfinite(loss);(loss*len(take)/len(selected)).backward();losses.append(loss.item())
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),5))
            opt.step();updates+=1
            if arm=='partial_finetune' and stage==2:unfrozen+=1
        history.append(dict(epoch=epoch,stage=stage,updates=updates,loss=float(np.mean(losses))))
        write(package/'progress.json',dict(seed=seed,arm=arm,labels=len(seq),epoch=epoch,updates=updates,pid=os.getpid()))
        if epoch%2==0:
            metrics,_,_=score(model,gpu,*validation,store.y)
            history[-1]['validation']=metrics
            if metrics['rmse']<best:
                best=metrics['rmse'];best_epoch=epoch;weights=st(model)
            if epoch-best_epoch>=15:break
    assert weights is not None
    model.load_state_dict(weights,strict=True)
    return dict(epochs=epoch,best_epoch=best_epoch,validation_rmse=best,optimizer_updates=updates,unfrozen_updates=unfrozen,
                partial_reached_stage2=(unfrozen>0) if arm=='partial_finetune' else None,
                sample_order_hash=order_hash.hexdigest(),seconds=time.time()-began),history

def main(package):
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    package=Path(package);cfg=json.loads((package/'config.json').read_text());torch.set_num_threads(4)
    store,norm,splits,validations,test,tm=prepare()
    source=checkpoint=None
    cp=(SUITE/cfg['checkpoint'] if cfg.get('checkpoint') else
        (SUITE/cfg['upstream']/'training/encoder.pt' if cfg.get('upstream') else None))
    if cp:
        assert json.loads((cp.parent/'status.json').read_text())['state']=='complete'
        source,checkpoint=convert(cp)
    labels={str(f):dict(rows=s[:,-1].tolist(),validation=validations[f][0][:,-1].tolist(),counts=counts) for f,(s,m,counts) in splits.items()}
    protocol=dict(train=list(TRAIN),test=TEST,seeds=list(SEEDS),fractions=list(FRACTIONS),arms=cfg['arms'],
                  max_epochs=200,validation='20% interval selection within independently selected source labels',
                  validation_interval=2,patience_epochs=15,early_stopping=True,selection='lowest validation RMSE',nested=False,
                  batch=32,microbatch=8,history=6,dropout=.05,loss='SmoothL1 beta0.05',weight_decay=.0001,
                  scratch_lr=.001,encoder_lr=.0001,head_lr=.001,clip=5,scheduler=None,swa=False,precision='BF16',
                  frozen_epochs=0,partial_unfreeze=list(PARTIAL),partial_optimizer_reset=False,
                  checkpoint=str(cp) if cp else None,checkpoint_sha256=file_hash(cp) if cp else None,
                  labels_sha256=hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest(),
                  npz=store.info['signature'])
    if (package/'runs').exists():assert json.loads((package/'protocol.json').read_text())==protocol,'Protocol differs from existing runs'
    write(package/'protocol.json',protocol);write(package/'labels.json',labels)
    write(package/'scalers.json',dict(train=list(TRAIN),center_scale=[v.tolist() for v in norm]))
    write(package/'verification.json',verify(store,norm,splits,source,checkpoint))
    if args.verify_only:return
    allowed=store.rows((TEST,)+TRAIN);gpu=[]
    for v in store.transform(allowed,norm):
        tensor=torch.zeros((len(store.x),*v.shape[1:]),dtype=torch.from_numpy(v).dtype,device='cuda')
        tensor[torch.as_tensor(allowed,device='cuda')]=torch.as_tensor(v,device='cuda');gpu.append(tensor)
    results=[]
    for seed in SEEDS:
        for f,(seq,masks,counts) in splits.items():
            for arm in cfg['arms']:
                dest=package/'runs'/f'seed{seed}'/f'fraction{int(f*100)}'/arm
                if (dest/'metrics.json').exists():results.append(json.loads((dest/'metrics.json').read_text()));continue
                dest.mkdir(parents=True,exist_ok=True)
                write(package/'status.json',dict(state='running',seed=seed,fraction=f,arm=arm,pid=os.getpid()))
                model=make_model(seed,arm,checkpoint).cuda();head_hash=hh({k:v for k,v in st(model).items() if not k.startswith('encoder.')})
                audit,history=train(model,arm,gpu,store,seq,masks,validations[f],seed,package)
                metrics,pred,y=score(model,gpu,test,tm,store.y)
                row=dict(seed=seed,fraction=f,arm=arm,metrics=metrics,head_hash=head_hash,**audit)
                torch.save(st(model),dest/'model.pt');np.savez(dest/'predictions.npz',pred=pred,y=y,rows=test[:,-1])
                write(dest/'history.json',history);write(dest/'metrics.json',row);results.append(row)
                write(package/'results.json',dict(complete=False,rows=results));print(json.dumps(row),flush=True)
                del model;torch.cuda.empty_cache()
    summary=[]
    for f in FRACTIONS:
        for arm in cfg['arms']:
            group=[r for r in results if r['fraction']==f and r['arm']==arm];assert len(group)==len(SEEDS)
            summary.append(dict(fraction=f,arm=arm,n=len(group),metrics={m:dict(mean=float(np.mean([r['metrics'][m] for r in group])),std=(float(np.std([r['metrics'][m] for r in group],ddof=1)) if len(group)>1 else None)) for m in ('rmse','mae','r2','bias')}))
    write(package/'results.json',dict(complete=True,rows=results,summary=summary))
    write(package/'status.json',dict(state='complete',runs=len(results)))
