"""Finish corrected bearing experiment, then battery 30-run validation study."""
import json,os,sys,subprocess,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import run_battery_state_trend as b
from run_battery_fullrows_65token import ROOT,write,state
PACKAGE=Path(__file__).resolve().parents[1]
OUT=Path('\\\\?\\'+str(PACKAGE/'outputs'))
UP=PACKAGE.parent/'01_fullrows_65token_state_trend_fixed500_nested/outputs/upstream/run_seed42/encoder.pt'
BEARING=ROOT/'outputs/local_transfer_research/bearing_reproduction_packages/04_attention65_fixed500_nested'

def split(data,f):
    train=[];val=[]
    for u in b.TRAIN:
        ids=data.groups[u];n=int(np.ceil(len(ids)*f))
        chosen=ids[np.unique(np.rint(np.linspace(0,len(ids)-1,n)).astype(int))]
        nv=int(np.ceil(.2*len(chosen)))
        vi=np.unique(np.rint(np.linspace(0,len(chosen)-1,nv)).astype(int))
        train.extend(np.delete(chosen,vi));val.extend(chosen[vi])
    train=np.array(train);val=np.array(val)
    assert len(train) and len(val) and not set(train)&set(val)
    assert not np.any(data.units[np.r_[train,val]]==b.TEST)
    return train,val

@torch.no_grad()
def score(model,data,ids):
    model.eval();pred=[]
    # Larger evaluation batches do not change the validation set or selection
    # protocol, but avoid Python/launch overhead for this small model.
    for i in range(0,len(ids),32):pred.extend(model(*data.batch(ids[i:i+32],'cuda')).float().cpu().tolist())
    pred=np.asarray(pred);y=data.y[ids];e=pred-y
    return dict(rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),r2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),n=len(ids)),pred,y

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    data=b.Downstream();checkpoint,sha=b.load_checkpoint(UP)
    OUT.mkdir(parents=True,exist_ok=True)
    splits={f:split(data,f) for f in (.1,.2,1.)}
    write(OUT/'labels.json',{str(f):dict(train=tr.tolist(),validation=va.tolist()) for f,(tr,va) in splits.items()})
    write(OUT/'protocol.json',dict(train=b.TRAIN,test=b.TEST,fractions=[.1,.2,1.],seeds=[42,43,44,45,46],
        nested=False,validation='20% of independently selected source labels, evenly spaced',max_epochs=200,
        validation_interval=2,patience_epochs=15,selection='validation RMSE minimum',dropout=.05,
        upstream=str(UP),upstream_sha256=sha,full_finetune=True,scaling='original all-source-input-only scaler',
        loss='original smooth_l1 beta=1',batch=32,microbatch=32,lr=dict(scratch=.001,encoder=.0001,head=.001)))
    model,_=b.make_model(42,'finetune',checkpoint)
    with torch.no_grad():assert torch.isfinite(model(*data.batch(splits[.1][0][:2],'cpu'))).all()
    del model
    if args.verify_only:return
    write(PACKAGE/'status.json',dict(state='starting',pid=os.getpid()))
    results=[]
    for seed in (42,43,44,45,46):
        for f,(train,val) in splits.items():
            for arm in ('scratch','finetune'):
                dest=OUT/f'seed{seed}'/f'fraction{int(f*100)}'/arm
                if (dest/'metrics.json').exists():
                    results.append(json.loads((dest/'metrics.json').read_text()))
                    continue
                dest.mkdir(parents=True,exist_ok=True)
                write(PACKAGE/'status.json',dict(state='running',seed=seed,fraction=f,arm=arm,pid=os.getpid()))
                model,groups=b.make_model(seed,arm,checkpoint,'cuda')
                optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
                rng=np.random.default_rng(seed);best=float('inf');best_epoch=0;updates=0;history=[];weights=None
                for epoch in range(1,201):
                    model.train();order=rng.permutation(train)
                    for start in range(0,len(order),32):
                        selected=order[start:start+32];optimizer.zero_grad(set_to_none=True)
                        # Full effective batch is now processed in one forward;
                        # this is the requested microbatch=32 setting.
                        ids=selected
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            pred=model(*data.batch(ids,'cuda'))
                            loss=F.smooth_l1_loss(pred.float(),torch.as_tensor(data.y[ids],device='cuda'))
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),5))
                        optimizer.step();updates+=1
                    if epoch%2==0:
                        metrics,_,_=score(model,data,val)
                        history.append(dict(epoch=epoch,updates=updates,validation=metrics))
                        if metrics['rmse']<best:best=metrics['rmse'];best_epoch=epoch;weights=state(model)
                        if epoch-best_epoch>=15:break
                model.load_state_dict(weights,strict=True)
                test=data.groups[b.TEST];metrics,pred,y=score(model,data,test)
                row=dict(seed=seed,fraction=f,arm=arm,best_epoch=best_epoch,stopped_epoch=epoch,updates=updates,validation_rmse=best,metrics=metrics)
                torch.save(weights,dest/'model.pt');np.savez(dest/'predictions.npz',pred=pred,y=y,rows=test)
                write(dest/'history.json',history);write(dest/'metrics.json',row)
                results.append(row);write(OUT/'results.json',dict(complete=False,rows=results))
                print(json.dumps(row),flush=True)
                del model,optimizer,groups,weights;torch.cuda.empty_cache()
    write(OUT/'results.json',dict(complete=True,rows=results))
    write(PACKAGE/'status.json',dict(state='completed',runs=30))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(PACKAGE/'status.json',dict(state='failed',error=repr(e)))
        raise
