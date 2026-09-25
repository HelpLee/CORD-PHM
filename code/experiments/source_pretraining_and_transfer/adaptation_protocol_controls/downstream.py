"""SELECTED_MODEL data/splits/heads, with explicit scratch and bearing ADAPTATION_CONTROLS."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
import torch
from config import HERE,SUITE,DOMAINS,SEEDS,source_for

def load(name,path,source=None):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    if source is None:spec.loader.exec_module(m)
    else:exec(compile(source,str(path),'exec'),m.__dict__)
    return m

def replace_once(source,old,new):
    if source.count(old)!=1:raise RuntimeError('Inherited code drift: '+old)
    return source.replace(old,new)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('baseline','adaptive','bearing10'),required=True)
    p.add_argument('--model',required=True);p.add_argument('--domain',choices=DOMAINS,required=True)
    p.add_argument('--fraction',type=float,choices=(.1,.2,1.),required=True)
    p.add_argument('--mode',choices=('full','partial','l2sp1','l2sp2'),default='full')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--smoke-train',action='store_true')
    args=p.parse_args()
    if args.mode!='full' and (args.domain!='bearing' or args.fraction!=.1):raise ValueError('Bearing10 adaptations only')
    scratch=args.model=='scratch';effective_arm='scratch' if scratch else 'partial_finetune' if args.mode=='partial' else 'full_finetune'
    alpha={'l2sp1':.001,'l2sp2':.01}.get(args.mode,0.)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.autocast=lambda *a,**kw:contextlib.nullcontext()
    basepath=SUITE/'full_channel_source_models/run_downstream.py';source=basepath.read_text()
    # Use the existing seven-channel milling splitter/normalizer/trainer verbatim.
    if scratch:
        start=source.index('    state = torch.load(checkpoint,',source.index('def run_milling('))
        end=source.index('    store = load_store("milling")',start)
        source=source[:start]+'    converted_path = None\n'+source[end:]
        source=replace_once(source,'down.ARMS = ("full_finetune",)','down.ARMS = ("scratch",)')
    # Recover incomplete milling seeds by keeping the previous seed folder as an archive.
    source=replace_once(source,'        if not summary.is_file():\n            down.main()',
        '        if not summary.is_file():\n            if summary.parent.exists():\n                archive=package / ("interrupted_"+down.NAME+"_"+str(time.time_ns()))\n                shutil.move(str(summary.parent), str(archive))\n            down.main()')
    d=load('cord_control_downstream',basepath,source);d.time=time
    # Keep inherited milling result metadata consistent with its actual scratch optimizer.
    if scratch:d.ENCODER_LR=.001
    package=HERE/('smoke_downstream' if args.verify_only or args.smoke_train else 'downstream')/args.phase/args.model/args.mode/args.domain/f'p{round(args.fraction*100)}'
    package.mkdir(parents=True,exist_ok=True)
    done=package/'results.json'
    if done.exists() and json.loads(done.read_text()).get('complete'):
        print('ALREADY_COMPLETE',package);return
    cp=None
    origin=source_for(args.model,args.domain)
    if origin is not None:
        d.HERE=origin.parents[3]
        cp=d.upstream(origin.parents[1].name,args.domain,package)
    if args.domain=='milling':
        sys.path[:0]=[str(d.WORKSPACE),str(d.EXP19),str(d.EXP19/'milling_code'),str(d.EXP15/'downstream_milling_10pct/code')]
        path=Path(importlib.util.find_spec('downstream_interval_val200').origin)
        ms=path.read_text()
        if args.smoke_train:
            ms=replace_once(ms,'range(1, epoch_limit + 1)','range(1, 3)')
        milling=load('downstream_interval_val200',path,ms)
        milling.CHANNELS=7
        d.SEEDS=() if args.verify_only else (42,) if args.smoke_train else SEEDS
        d.run_milling(args.model,args.fraction,cp,package)
    else:
        sys.path[:0]=[str(d.WORKSPACE),str(d.WORKSPACE/'data_phm'),str(SUITE/'shared_domain_components'/f'{args.domain}_code'),str(d.EXP19/f'{args.domain}_code'),str(d.EXP19)]
        name='bearing_adapter_study' if args.domain=='bearing' else 'adapter_downstream'
        path=Path(importlib.util.find_spec(name).origin);ds=path.read_text()
        ds=ds.replace("precision='BF16'","precision='FP32'").replace('encoder_lr=.0001','encoder_lr=.0003').replace('pretrained_lr=.0001','pretrained_lr=.0003')
        # Old verifier indexed .1 even when testing p20/p100.
        ds=ds.replace('splits[.1]','splits[next(iter(splits))]')
        if alpha:
            old="loss=F.smooth_l1_loss(model(*(v[rows] for v in gpu),hm),target,beta=.05)"
            ds=replace_once(ds,old,old+'+anchor_penalty(model)')
        if args.smoke_train:
            ds=ds.replace('range(1,201)','range(1,3)')
        driver=load(name,path,ds);driver.FRACTIONS=(args.fraction,)
        if args.smoke_train:driver.SEEDS=(42,)
        if alpha:
            maker=driver.make_model
            def make(*a,**kw):
                model=maker(*a,**kw)
                model._anchor={n:v.detach().clone() for n,v in model.encoder.named_parameters()}
                return model
            driver.make_model=make
            def penalty(model):
                return .5*alpha*sum((v-model._anchor[n].to(v.device)).square().sum() for n,v in model.encoder.named_parameters() if v.requires_grad)
            driver.anchor_penalty=penalty
        def optimizer(network,arm,stage):
            driver.configure(network,arm,stage)
            enc=[v for v in network.encoder.parameters() if v.requires_grad]
            head=[v for n,v in network.named_parameters() if not n.startswith('encoder.') and v.requires_grad]
            groups=[dict(params=head,lr=.001)]
            if enc:groups.append(dict(params=enc,lr=.001 if scratch else .0003))
            return torch.optim.AdamW(groups,weight_decay=.0001)
        if args.domain=='bearing':driver.optimizer_for=optimizer
        else:driver.optimizer=optimizer
        d.write(package/'config.json',dict(checkpoint=str(cp) if cp else None,upstream=None,arms=[effective_arm]))
        sys.argv=[sys.argv[0]]+(['--verify-only'] if args.verify_only else [])
        driver.main(package)
    d.write(package/'effective_protocol.json',dict(model=args.model,domain=args.domain,fraction=args.fraction,
        mode=effective_arm,anchor_alpha=alpha,precision='FP32',encoder_lr=.001 if scratch else .0003,head_lr=.001,
        seeds=[42] if args.smoke_train else list(SEEDS),smoke=args.smoke_train or args.verify_only,
        checkpoint=str(origin) if origin else None,base_sha256=hashlib.sha256(source.encode()).hexdigest(),
        note='Scratch uses identical inputs, heads, labels and stopping with scratch LR1e-3; pretrained encoder LR3e-4'))
    if not args.verify_only:
        result=json.loads(done.read_text());expected={42} if args.smoke_train else set(SEEDS)
        assert result['complete'] and len(result['rows'])==len(expected) and {r['seed'] for r in result['rows']}==expected
    print('DOWNSTREAM_OK',args.phase,args.model,args.domain,args.fraction,args.mode,flush=True)

if __name__=='__main__':
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    main()
