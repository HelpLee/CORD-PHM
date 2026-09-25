import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from compression_model import Attention65MaskedModel, Attention65RUL
from pipeline import PACKAGE, OLD, FIXED
from data import OUT,Store,write
from feature_observations import observed_store
from global_local_data import GlobalLocalStore,build_condition2
import run_bearing_all_temporal_variable_channels_upstream as up
import run_bearing_delta6_no_b24_fewshot as down

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    model=Attention65MaskedModel(channels=8).eval()
    seen=[]
    hook=model.encoder.transformer.register_forward_pre_hook(lambda m,args:seen.append(tuple(args[0].shape)))
    for c in (1,2,4,8):
        x=torch.randn(2,c,64,26);g=torch.randn(2,c,26)
        cm=torch.ones(2,c,dtype=torch.bool);tm=torch.ones(2,c,64,dtype=torch.bool)
        tm[:, :, -3:]=False
        pred,mask,out=model(x,g,cm,tm,42)
        assert pred.shape==x.shape and out['snapshot'].shape==(2,96) and seen[-1]==(2,65,96)
        (pred[mask].square().mean()+out['snapshot'].square().mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            baseline=model.encoder(x,g,cm,tm)['snapshot']
            pad=model.encoder(torch.cat((x,torch.randn(2,2,64,26)),1),torch.cat((g,torch.randn(2,2,26)),1),
                              torch.cat((cm,torch.zeros(2,2,dtype=torch.bool)),1),torch.cat((tm,torch.zeros(2,2,64,dtype=torch.bool)),1))['snapshot']
            assert torch.allclose(baseline,pad,atol=1e-5)
            perm=torch.arange(c-1,-1,-1)
            swapped=model.encoder(x[:,perm],g[:,perm],cm[:,perm],tm[:,perm])['snapshot']
            assert torch.allclose(baseline,swapped,atol=1e-5)
    hook.remove()
    downstream=Attention65RUL(channels=2,dropout=.05,head_dropout=.05)
    downstream.encoder.load_state_dict(model.encoder.state_dict(),strict=True)
    assert all(p.requires_grad for p in downstream.parameters())
    assert all(m.p==.05 for m in downstream.modules() if isinstance(m,nn.Dropout))
    assert down.MAX_UPDATES==500 and down.NESTED_LABELS and down.DETERMINISTIC and down.MODEL_DROPOUT==.05
    store=GlobalLocalStore(build_condition2(),channels=2)
    train=tuple(u for u in down.UNITS if u!='Bearing2_1')
    seq,hm=store.sequences_with_mask(train,6)
    counts={}; subsets={}
    for f in (.1,.2):
        s,m,cnt=down.evenly_spaced_labeled_sequences(store,seq,hm,train,f)
        subsets[f]=set(s[:,-1]);counts[str(f)]=cnt
        assert not any(store.units[r]=='Bearing2_1' for r in subsets[f])
        reference_path=FIXED/f'results/bearing_mask_dyn_b21_fixed500_lr1e4_1e3_drop005_v1/seed42/fraction_{down.fraction_key(f)}/Bearing2_1_finetune_seed42/summary.json'
        reference=json.loads(Path('\\\\?\\'+str(reference_path)).read_text())
        assert cnt==reference['label_counts']
    assert subsets[.1] < subsets[.2]
    norm=store.normalize(train)
    batch=[torch.as_tensor(v) for v in store.transform(seq[0],norm)]
    pred=downstream(*(v[None] for v in batch),torch.as_tensor(hm[:1]))
    pred.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in downstream.parameters() if p.grad is not None)
    original=json.loads((OLD/'results/bearing_all_temporal_variable_channels_seed42_v1/upstream_splits_scalers.json').read_text())
    audit=[]
    for stem in up.SOURCES:
        local=observed_store(Store(up.build_temporal_cache(stem)))
        current=GlobalLocalStore(local,channels=local.x.shape[1])
        tr,va=current.split();sc=current.normalize(tr)
        old=next(r for r in original if r['dataset']==stem)
        assert list(tr)==old['train'] and list(va)==old['val']
        assert all(np.allclose(a,b,rtol=1e-6,atol=1e-6) for a,b in zip(sc,old['center_scale'])),stem
        assert len(up.temporal_windows(current,tr))==old['train_windows']
        assert len(up.temporal_windows(current,va))==old['val_windows']
        audit.append(dict(dataset=stem,rows=len(local.x),channels=local.x.shape[1],split_scalers_windows_match=True))
    report=dict(passed=True,token_shape='65x96',native_channels=[1,2,4,8],padding_invariance=True,
                permutation_invariance=True,strict_upstream_to_downstream_load=True,full_finetune=True,
                max_updates=down.MAX_UPDATES,nested=True,deterministic=True,dropout=down.MODEL_DROPOUT,
                labels=counts,upstream=audit)
    write(PACKAGE/'verification.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
