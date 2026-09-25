"""Battery Adapter study: immutable source protocol, paired initialization."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import battery_state_trend as architecture
from adapter_encoder import DomainAdapterEncoder as BatteryAdapterEncoder
architecture.GlobalLocalEncoder = BatteryAdapterEncoder
import run_battery_state_trend as original
from queue_battery_val200 import split, score
from run_battery_fullrows_65token import state, write
from reference_joint_model import JointModel

SUITE=Path(__file__).resolve().parents[2]
PREFIXES=('transformer.layers.1.','final_norm.','adapters.1.')
SEEDS=(42,43,44,45,46)
FRACTIONS=(.1,.2,1.)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def convert(path):
    source=torch.load(path,map_location='cpu',weights_only=True)
    converted={k:v for k,v in source.items() if not k.startswith(('stems.','adapters.'))}
    for k,v in source.items():
        if k.startswith('adapters.battery.'):
            converted[k.replace('adapters.battery.','adapters.',1)]=v
    for target,origin in [('local_norm','local.0'),('local_proj','local.1'),
                          ('global_norm','global.0'),('global_proj','global.1')]:
        for suffix in ('weight','bias'):
            converted[f'{target}.{suffix}']=source[f'stems.battery.{origin}.{suffix}']
    encoder=BatteryAdapterEncoder(channels=1,dropout=.05)
    encoder.load_state_dict(converted,strict=True)
    assert len(converted)==58
    return source,converted

def make_model(seed,arm,checkpoint):
    original.seed_all(seed)
    model=architecture.StateTrendRUL(dropout=.05)
    if arm!='scratch':model.encoder.load_state_dict(checkpoint,strict=True)
    configure(model,arm,1)
    return model

def configure(model,arm,stage):
    for name,p in model.named_parameters():
        p.requires_grad_(not name.startswith('encoder.') or arm in ('scratch','full_finetune'))
    if arm=='partial_finetune' and stage==2:
        for name,p in model.encoder.named_parameters():p.requires_grad_(name.startswith(PREFIXES))

def optimizer(model,arm,stage):
    configure(model,arm,stage)
    head=[p for n,p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
    encoder=[p for p in model.encoder.parameters() if p.requires_grad]
    groups=[dict(params=head,lr=1e-3)]
    if encoder:groups.append(dict(params=encoder,lr=1e-3 if arm=='scratch' else 1e-4))
    return torch.optim.AdamW(groups,weight_decay=1e-4)

def train_mode(model,arm,stage):
    model.train()
    if arm in ('frozen_probe','partial_finetune'):
        model.encoder.eval()
        if arm=='partial_finetune' and stage==2:
            model.encoder.transformer.layers[1].train()
            model.encoder.final_norm.train()
            model.encoder.adapters[1].train()

def verify(data,checkpoint,source):
    scratch=make_model(42,'scratch',None).eval()
    ft=make_model(42,'full_finetune',checkpoint).eval() if checkpoint is not None else make_model(42,'scratch',None).eval()
    for k,v in scratch.state_dict().items():
        if not k.startswith('encoder.'):assert torch.equal(v,ft.state_dict()[k]),k
    batch=data.batch(data.groups[original.TEST][-2:],'cpu')
    with torch.no_grad():assert torch.isfinite(ft(*batch)).all()
    if source is not None:
        ref=JointModel().encoder.eval()
        for d in tuple(ref.stems):
            if f'stems.{d}.local.0.weight' not in source:
                del ref.stems[d];del ref.adapters[d]
        ref.load_state_dict(source,strict=True)
        x,g,cm,tm,_=batch
        vals=[a.reshape(-1,*a.shape[2:]) for a in (x,g,cm,tm)]
        with torch.no_grad():
            expected,local=ref('battery',*vals)
            actual=ft.encoder(*vals)
            torch.testing.assert_close(expected,actual['snapshot'],rtol=0,atol=0)
            torch.testing.assert_close(local,actual['local_hidden'],rtol=0,atol=0)
    for arm in ('scratch','frozen_probe','partial_finetune','full_finetune'):
        for stage in (1,2):
            configure(ft,arm,stage)
            for name,p in ft.encoder.named_parameters():
                want=arm in ('scratch','full_finetune') or (arm=='partial_finetune' and stage==2 and name.startswith(PREFIXES))
                assert p.requires_grad==want,(arm,stage,name)
    configure(ft,'full_finetune',1);ft.train()
    loss=F.smooth_l1_loss(ft(*batch),torch.tensor(data.y[data.groups[original.TEST][-2:]]))
    loss.backward()
    for adapter in ft.encoder.adapters:
        assert adapter.net[-1].weight.grad is not None
        assert torch.isfinite(adapter.net[-1].weight.grad).all()
    # Verification is CPU-only; no optimizer step or saved trained parameters.
    optimizer(ft,'partial_finetune',2)
    assert all(p.requires_grad==n.startswith(PREFIXES) for n,p in ft.encoder.named_parameters())
    return dict(passed=True,stage_at_epoch1=2,strict_load=source is not None,paired_heads=True,
                encoder_forward_exact=source is not None,freeze_masks=True,
                adapter_gradients=True,loss_beta=1.)

def main(package):
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    package=Path(package)
    cfg=json.loads((package/'config.json').read_text())
    torch.set_num_threads(4)
    original.OUT=package/'support'
    data=original.Downstream()
    if os.environ.get('HEALTHTOKEN_PRELOAD_INDICES', '0') == '1':
        data.enable_cuda_cache()
    cp=(SUITE/cfg['checkpoint'] if cfg.get('checkpoint') else
        (SUITE/cfg['upstream']/'training/encoder.pt' if cfg.get('upstream') else None))
    source=checkpoint=None
    if cp:
        assert json.loads((cp.parent/'status.json').read_text())['state']=='complete','Upstream not complete'
        source,checkpoint=convert(cp)
    splits={f:split(data,f) for f in FRACTIONS}
    audit={str(f):dict(train=tr.tolist(),validation=va.tolist()) for f,(tr,va) in splits.items()}
    digest=hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest()
    protocol=dict(seeds=list(SEEDS),fractions=list(FRACTIONS),arms=cfg['arms'],train=list(original.TRAIN),test=original.TEST,
                  max_epochs=200,validation_interval=2,patience_epochs=15,selection='lowest validation RMSE',
                  validation='20% interval selection within independently selected source labels',nested=False,
                  batch=32,microbatch=32,dropout=.05,loss='SmoothL1 beta=1',weight_decay=1e-4,
                  scratch_lr=.001,head_lr=.001,pretrained_lr=.0001,clip=5,scheduler=None,precision='BF16',
                  frozen_epochs=0,
                  partial='epoch1+ block2/finalnorm/adapter2; other encoder parameters frozen',
                  partial_early_stop='active from first validation; no warmup or stage transition',
                  upstream=str(cp) if cp else None,upstream_sha256=sha(cp) if cp else None,
                  npz=str(data.path),npz_sha256=sha(data.path),labels_sha256=digest)
    previous=package/'protocol.json'
    if (package/'runs').exists():
        assert json.loads(previous.read_text())==protocol,'Resume protocol/source mismatch'
    write(previous,protocol);write(package/'labels.json',audit)
    write(package/'scalers.json',dict(norm=data.norm.tolist(),train=list(original.TRAIN)))
    write(package/'verification.json',verify(data,checkpoint,source))
    if args.verify_only:return
    write(package/'status.json',dict(state='running',pid=os.getpid()))
    rows=[]
    for seed in SEEDS:
        for f,(tr,va) in splits.items():
            for arm in cfg['arms']:
                dest=package/'runs'/f'seed{seed}'/f'fraction{int(f*100)}'/arm
                if (dest/'metrics.json').exists():
                    rows.append(json.loads((dest/'metrics.json').read_text()));continue
                dest.mkdir(parents=True,exist_ok=True)
                model=make_model(seed,arm,checkpoint).cuda()
                stage=2 if arm=='partial_finetune' else 1;opt=optimizer(model,arm,stage)
                rng=np.random.default_rng(seed)
                best=float('inf');best_epoch=0;weights=None;history=[];updates=0
                began=time.time()
                for epoch in range(1,201):
                    train_mode(model,arm,stage)
                    order=rng.permutation(tr)
                    for start in range(0,len(order),32):
                        ids=order[start:start+32];opt.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            pred=model(*data.batch(ids,'cuda'))
                            target = (data.cuda_y[torch.as_tensor(ids, device='cuda')]
                                      if data.cuda_y is not None else
                                      torch.as_tensor(data.y[ids],device='cuda'))
                            loss=F.smooth_l1_loss(pred.float(),target,beta=1.)
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),5))
                        opt.step();updates+=1
                    write(package/'status.json',dict(state='running',seed=seed,fraction=f,arm=arm,epoch=epoch,completed=len(rows),pid=os.getpid()))
                    if epoch%2==0:
                        metrics,_,_=score(model,data,va)
                        history.append(dict(epoch=epoch,stage=stage,updates=updates,validation=metrics))
                        if metrics['rmse']<best:
                            best=metrics['rmse'];best_epoch=epoch;weights=state(model)
                        if epoch-best_epoch>=15:break
                model.load_state_dict(weights,strict=True)
                test=data.groups[original.TEST];metrics,pred,y=score(model,data,test)
                metrics['bias']=float(np.mean(pred-y))
                row=dict(seed=seed,fraction=f,arm=arm,best_epoch=best_epoch,stopped_epoch=epoch,updates=updates,
                         validation_rmse=best,metrics=metrics,seconds=time.time()-began)
                torch.save(weights,dest/'model.pt')
                np.savez(dest/'predictions.npz',pred=pred,y=y,rows=test)
                write(dest/'history.json',history);write(dest/'metrics.json',row)
                rows.append(row);write(package/'results.json',dict(complete=False,rows=rows))
                print(json.dumps(row),flush=True)
                del model,opt,weights;torch.cuda.empty_cache()
    summary=[]
    for f in FRACTIONS:
        for arm in cfg['arms']:
            group=[r for r in rows if r['fraction']==f and r['arm']==arm]
            assert len(group)==len(SEEDS)
            summary.append(dict(fraction=f,arm=arm,n=len(group),metrics={m:dict(mean=float(np.mean([r['metrics'][m] for r in group])),std=(float(np.std([r['metrics'][m] for r in group],ddof=1)) if len(group)>1 else None)) for m in ('rmse','mae','r2','bias')}))
    write(package/'results.json',dict(complete=True,rows=rows,summary=summary))
    write(package/'status.json',dict(state='complete',runs=len(rows)))
