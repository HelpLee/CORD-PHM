"""Unseen N-CMAPSS domain calibration and paired downstream evaluation."""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'code/data_phm').is_dir())
UP = ROOT/'code/experiments/source_pretraining_and_transfer/10_upstream_three_domain_adapter'
UP_CODE, UP_TRAIN = UP/'code', UP/'training'
DATA = ROOT/'code/data_phm/processed_health_tokens/engine/ncmapss_ds02_downstream_health_tokens.npz'
RAW = ROOT/'code/data_phm/raw/Engine/N-CMAPSS_DS02-006.h5'
LEGACY_CODE = ROOT/'outputs/shared_stem_engine_transfer_v1/code'
RESULTS = HERE/'results'
TRAIN_UNITS = ('U2','U5','U10','U16','U18','U20')
TEST_UNIT = 'U11'
SEEDS, FRACTIONS = (42,43,44,45,46), (.1,.2,1.)
ARMS = ('scratch','engine_only_ssl','pretrained_random_interface','pretrained_calibrated_interface','partial_finetune_calibrated')

sys.path.insert(0, str(UP_CODE)); joint_model = importlib.import_module('joint_model')
sys.path.insert(0, str(LEGACY_CODE)); ref = importlib.import_module('engine_reference')
ref.PACKAGE=RESULTS; ref.ROOT=ROOT; ref.DATA=DATA; ref.RAW=RAW; ref.TRAIN=TRAIN_UNITS; ref.TEST=TEST_UNIT

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8'); os.replace(temp,path)

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def add_engine_interface(model):
    model.encoder.stems['engine']=nn.ModuleDict({'local':nn.Sequential(nn.LayerNorm(26),nn.Linear(26,96)),'global':nn.Sequential(nn.LayerNorm(26),nn.Linear(26,96))})
    model.encoder.adapters['engine']=nn.ModuleList([joint_model.DomainAdapter(),joint_model.DomainAdapter()])
    model.channels['engine']=nn.Embedding(14,96)
    model.decoders['engine']=nn.Sequential(nn.LayerNorm(288),nn.Linear(288,192),nn.GELU(),nn.Linear(192,26))
    return model

def make_upstream(pretrained=True):
    model=joint_model.JointModel()
    if pretrained:
        checkpoint=torch.load(UP_TRAIN/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'],strict=True)
    return add_engine_interface(model)

def engine_keys(name):
    return name.startswith(('encoder.stems.engine.','encoder.adapters.engine.','channels.engine.','decoders.engine.'))

def freeze_for_calibration(model, train_all=False):
    if train_all:
        for p in model.parameters(): p.requires_grad_(True)
        return
    for name,p in model.named_parameters():p.requires_grad_(engine_keys(name))
    assert all(p.requires_grad==engine_keys(n) for n,p in model.named_parameters())

def normalize_unlabeled():
    """Load no target key; fit statistics on source engines only."""
    with np.load(DATA,allow_pickle=False) as z:
        keys=('x_health','x_global','feature_mask','global_feature_mask','c_mask','token_mask','sample_unit_id','sample_cycle_index')
        a={k:z[k] for k in keys}
    units=a['sample_unit_id'].astype(str);cycles=a['sample_cycle_index'];source=np.isin(units,TRAIN_UNITS)
    cm=a['c_mask'];tm=a['token_mask']&cm[...,None]
    def norm(x,mask):
        center=np.zeros((x.shape[1],x.shape[-1]),np.float32);scale=np.ones_like(center)
        for c in range(x.shape[1]):
            for f in range(x.shape[-1]):
                v=x[source,c,:,f][mask[source,c,:,f]] if x.ndim==4 else x[source,c,f][mask[source,c,f]]
                if len(v):
                    q,med,r=np.percentile(v,[25,50,75]);center[c,f]=med;scale[c,f]=max(r-q,1e-6)
        center,scale=((center[:,None],scale[:,None]) if x.ndim==4 else (center,scale))
        return np.where(mask,np.clip(np.arcsinh((x-center)/scale),-20,20),0).astype('float32')
    x=norm(a['x_health'],a['feature_mask']&tm[...,None]);g=norm(a['x_global'],a['global_feature_mask']&cm[...,None]);observed=a['feature_mask']&tm[...,None]
    tr=[];va=[];tw=[];vw=[]
    for unit in TRAIN_UNITS:
        rows=np.flatnonzero(units==unit);rows=rows[np.argsort(cycles[rows])];assert np.all(np.diff(cycles[rows])==1)
        cut=max(7,int(np.floor(.8*len(rows))));tr.extend(rows[:cut]);va.extend(rows[cut:])
        windows=np.asarray([rows[i-6:i+1] for i in range(6,len(rows))])
        tw.extend(windows[windows[:,-1]<rows[cut]]);vw.extend(windows[windows[:,-1]>=rows[cut]])
    audit=dict(label_keys_loaded=[],train_units=list(TRAIN_UNITS),test_unit=TEST_UNIT,test_rows_seen=0,train_rows=len(tr),validation_rows=len(va),train_windows=len(tw),validation_windows=len(vw),split='first 80% per source engine train; last 20% unlabeled validation')
    return tuple(torch.from_numpy(v) for v in (x,g,cm,tm,observed)),np.asarray(tr),np.asarray(va),np.asarray(tw),np.asarray(vw),audit

def calibration_loss(model,arrays,rows,windows,device):
    x,g,cm,tm,observed=arrays;rr=torch.as_tensor(rows);ww=torch.as_tensor(windows.reshape(-1))
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        mask=model.reconstruction('engine',*(v[rr].to(device) for v in (x,g,cm,tm)),observed[rr].to(device))
        dyn=model.dynamics('engine',*(v[ww].to(device) for v in (x,g,cm,tm)))
        return mask+.2*dyn,mask.detach(),dyn.detach()

@torch.no_grad()
def calibration_score(model,arrays,rows,windows,device):
    model.eval();torch.manual_seed(99173);values=[];rng=np.random.default_rng(99173)
    for start in range(0,min(len(rows),256),32):
        r=rows[start:start+32];w=windows[rng.integers(0,len(windows),max(1,len(r)//4))]
        loss,mask,dyn=calibration_loss(model,arrays,r,w,device);values.append((loss.item(),mask.item(),dyn.item()))
    return dict(total=float(np.mean([v[0] for v in values])),mask=float(np.mean([v[1] for v in values])),dynamics=float(np.mean([v[2] for v in values])))

def calibrate(device, pretrained=True, output_name='calibration'):
    out=RESULTS/output_name;out.mkdir(parents=True,exist_ok=True);target=out/('engine_calibrated.pt' if output_name=='calibration' else 'engine_only_ssl.pt')
    if target.exists():return target
    arrays,tr,va,tw,vw,audit=normalize_unlabeled();write(out/'data_audit.json',audit)
    seed_all(42);model=make_upstream(pretrained);freeze_for_calibration(model,train_all=not pretrained);model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    best=float('inf');best_epoch=0;history=[];updates=0;rng=np.random.default_rng(42)
    for epoch in range(1,501):
        model.train();losses=[]
        for _ in range(20):
            rows=rng.choice(tr,32,replace=len(tr)<32);windows=tw[rng.integers(0,len(tw),8)]
            optimizer.zero_grad(set_to_none=True);loss,_,_=calibration_loss(model,arrays,rows,windows,device);loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5);optimizer.step();losses.append(loss.item());updates+=1
        val=calibration_score(model,arrays,va,vw,device);history.append(dict(epoch=epoch,updates=updates,train=float(np.mean(losses)),validation=val))
        if val['total']<best-1e-4:
            best=val['total'];best_epoch=epoch;torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,validation=val),target)
        write(out/'history.json',history);write(RESULTS/'status.json',dict(state='calibrating',epoch=epoch,best_epoch=best_epoch,pid=os.getpid()))
        if epoch-best_epoch>=30:break
    write(out/'summary.json',dict(best_epoch=best_epoch,best_validation=best,updates=updates,objective='L_mask + 0.2 L_dyn',shared_frozen=pretrained,initialization='three-domain checkpoint' if pretrained else 'random initialization'));return target

class EngineRUL(nn.Module):
    def __init__(self,upstream):
        super().__init__();self.upstream=upstream;self.sensor=nn.Sequential(nn.Linear(28,32),nn.GELU());self.context=nn.Sequential(nn.Linear(8,16),nn.GELU());self.fuse=nn.Linear(144,96);self.gru=nn.GRU(96,64,batch_first=True);self.head=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,32),nn.GELU(),nn.Dropout(.05),nn.Linear(32,1))
    def forward(self,x,g,cm,tm,context):
        b,s,c,t,f=x.shape;e=self.upstream.encoder('engine',x.reshape(b*s,c,t,f),g.reshape(b*s,c,f),cm.reshape(b*s,c),tm.reshape(b*s,c,t))[0].reshape(b,s,96)
        z=self.fuse(torch.cat((e,self.sensor(context[...,:28]),self.context(context[...,28:])),-1));return self.head(self.gru(z)[0][:,-1]).squeeze(-1).sigmoid()

def build_arm(arm,calibrated,engine_ssl):
    up=make_upstream(arm not in ('scratch','engine_only_ssl'))
    if arm=='engine_only_ssl':up.load_state_dict(torch.load(engine_ssl,map_location='cpu',weights_only=True)['model'],strict=True)
    elif arm=='pretrained_calibrated_interface':up.load_state_dict(torch.load(calibrated,map_location='cpu',weights_only=True)['model'],strict=True)
    return EngineRUL(up)

def configure(model,arm):
    partial=('upstream.encoder.transformer.layers.1.','upstream.encoder.final_norm.','upstream.encoder.fusion.')
    for n,p in model.named_parameters():
        active=(not n.startswith('upstream.')) or arm=='scratch' or engine_keys(n.removeprefix('upstream.')) or (arm=='partial_finetune_calibrated' and n.startswith(partial));p.requires_grad_(active)
    high=[];low=[]
    for n,p in model.named_parameters():
        if p.requires_grad:(low if n.startswith(partial) else high).append(p)
    groups=[dict(params=high,lr=1e-3)]
    if low:groups.append(dict(params=low,lr=1e-4))
    return torch.optim.AdamW(groups,weight_decay=1e-4)

def train_mode(model,arm):
    model.train()
    if arm!='scratch':
        model.upstream.eval();model.upstream.encoder.stems['engine'].train();model.upstream.encoder.adapters['engine'].train()
        if arm=='partial_finetune_calibrated':model.upstream.encoder.transformer.layers[1].train();model.upstream.encoder.final_norm.train();model.upstream.encoder.fusion.train()

def downstream(device,calibrated,engine_ssl):
    arrays,y,units,cycles=ref.prepare();arrays=ref.shortcut(arrays,units);rows=[]
    for seed in SEEDS:
      for fraction in FRACTIONS:
        train,val,test,audit=ref.splits(units,cycles,fraction);write(RESULTS/f'splits/{int(100*fraction)}pct.json',audit)
        for arm in ARMS:
          folder=RESULTS/f'downstream/seed{seed}_{int(100*fraction)}pct_{arm}';folder.mkdir(parents=True,exist_ok=True)
          if (folder/'result.json').exists():rows.append(json.loads((folder/'result.json').read_text()));continue
          seed_all(seed);model=build_arm(arm,calibrated,engine_ssl).to(device);optimizer=configure(model,arm)
          write(folder/'load_report.json',dict(arm=arm,source=str(UP_TRAIN/'best.pt') if arm!='scratch' else None,calibrated=str(calibrated) if 'calibrated' in arm else None,trainable=[n for n,p in model.named_parameters() if p.requires_grad],frozen_epochs=0))
          best=float('inf');best_epoch=0;history=[];updates=0;began=time.time();start_epoch=1
          # Resume an interrupted run from its last verified best checkpoint.
          # Completed runs are skipped above; no random reinitialization occurs
          # for the one partially written run.
          if (folder/'best.pt').exists() and (folder/'history.json').exists():
            resume=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(resume['model'],strict=True)
            best_epoch=int(resume['epoch']);best=float(resume['validation']['rmse'])
            history=json.loads((folder/'history.json').read_text())
            updates=int(history[-1].get('updates',0)) if history else 0
            start_epoch=best_epoch+1
            del resume
          for epoch in range(start_epoch,201):
            torch.manual_seed(seed+epoch);train_mode(model,arm);order=np.random.default_rng(seed+epoch).permutation(len(train));losses=[]
            for start in range(0,len(order),32):
              selected=order[start:start+32];optimizer.zero_grad(set_to_none=True)
              for micro in range(0,len(selected),8):
                idx=train[selected[micro:micro+8]];target=torch.as_tensor(y[idx[:,-1]],device=device)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):loss=F.smooth_l1_loss(model(*(v[idx].to(device) for v in arrays)),target,beta=.05)
                (loss*len(idx)/len(selected)).backward();losses.append(loss.item())
              nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();updates+=1
            entry=dict(epoch=epoch,updates=updates,loss=float(np.mean(losses)))
            if epoch%2==0:
              metrics,_,_=ref.score(model,arrays,y,val,device);entry['validation']=metrics
              if metrics['rmse']<best:best=metrics['rmse'];best_epoch=epoch;torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,validation=metrics),folder/'best.pt')
            history.append(entry);write(folder/'history.json',history);write(RESULTS/'status.json',dict(state='downstream',seed=seed,fraction=fraction,arm=arm,epoch=epoch,completed=len(rows),pid=os.getpid()))
            if epoch-best_epoch>=15 and epoch%2==0:break
          ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);metrics,pred,target=ref.score(model,arrays,y,test,device)
          np.savez_compressed(folder/'predictions.npz',prediction=pred,target=target,rows=test[:,-1],cycles=cycles[test[:,-1]])
          row=dict(seed=seed,fraction=fraction,arm=arm,best_epoch=ck['epoch'],last_epoch=epoch,updates=updates,metrics=metrics,seconds=time.time()-began);write(folder/'result.json',row);rows.append(row);write(RESULTS/'results.json',dict(complete=False,rows=rows));del model,optimizer,ck;torch.cuda.empty_cache()
    summary=[]
    for fraction in FRACTIONS:
      for arm in ARMS:
        group=[r for r in rows if r['fraction']==fraction and r['arm']==arm];summary.append(dict(fraction=fraction,arm=arm,n=len(group),metrics={m:dict(mean=float(np.mean([r['metrics'][m] for r in group])),std=float(np.std([r['metrics'][m] for r in group],ddof=1))) for m in ('rmse','mae','r2','bias')}))
    write(RESULTS/'results.json',dict(complete=True,rows=rows,summary=summary));write(RESULTS/'status.json',dict(state='complete',runs=len(rows)))

def verify():
    status=json.loads((UP_TRAIN/'status.json').read_text());assert status['state']=='complete' and status['best_epoch']==283
    model=make_upstream(True);freeze_for_calibration(model);assert all(engine_keys(n) for n,p in model.named_parameters() if p.requires_grad)
    arrays,tr,va,tw,vw,audit=normalize_unlabeled();assert audit['test_rows_seen']==0 and audit['label_keys_loaded']==[]
    model.eval();idx=tr[:2]
    with torch.no_grad():e,_=model.encoder('engine',*(v[idx] for v in arrays[:4]))
    assert e.shape==(2,96) and torch.isfinite(e).all()
    report=dict(passed=True,upstream_best_epoch=283,source_sha256=digest(UP_TRAIN/'best.pt'),engine_embedding_shape=list(e.shape),calibration_trainable_only_engine=True,no_rul_loaded=True,no_u11_calibration=True);write(RESULTS/'preflight.json',report);print(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if args.verify_only:return verify()
    assert torch.cuda.is_available();device=torch.device('cuda');calibrated=calibrate(device,pretrained=True,output_name='calibration');engine_ssl=calibrate(device,pretrained=False,output_name='engine_only_ssl');downstream(device,calibrated,engine_ssl)

if __name__=='__main__':
    try:main()
    except BaseException as error:write(RESULTS/'status.json',dict(state='failed',error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()));raise
