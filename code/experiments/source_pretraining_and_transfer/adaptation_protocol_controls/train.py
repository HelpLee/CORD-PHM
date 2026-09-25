import hashlib
import importlib.util
import sys
import numpy as np
import torch
from config import COMPONENT_ROUTING,SELECTED_MODEL,HERE,settings
from routing import Router

def main():
    arm=sys.argv[sys.argv.index('--arm')+1];cfg=settings(arm);router=Router(cfg)
    source=(COMPONENT_ROUTING/'train.py').read_text()
    replacements=[('max_epochs=500,','max_epochs=2000,'),
        ('(1 if args.smoke else 500) + 1','(1 if args.smoke else 2000) + 1'),
        ('epoch < 500:','epoch < 2000:'),
        ('    shared_params = [named[i][1] for i in shared_index]',
         '    router.initialize(named, shared_index, active)\n    shared_params = [named[i][1] for i in shared_index]'),
        ('        model.load_state_dict(ck["model"]);',
         '        router.load_state_dict(ck["routing_state"])\n        model.load_state_dict(ck["model"]);')]
    for old,new in replacements:
        if source.count(old)!=1:raise RuntimeError('Base trainer changed: '+old)
        source=source.replace(old,new)
    spec=importlib.util.spec_from_file_location('cord_control_trainbase',COMPONENT_ROUTING/'train.py')
    base=importlib.util.module_from_spec(spec);base.router=router
    exec(compile(source,str(COMPONENT_ROUTING/'train.py'),'exec'),base.__dict__)
    fs=importlib.util.spec_from_file_location('cord_bearing_floor',SELECTED_MODEL/'methods.py');fm=importlib.util.module_from_spec(fs);fs.loader.exec_module(fm)
    original_grad=base.component_gradients;original_combine=base.combine;original_trainer=base.trainer
    calls={d:0 for d in cfg['domains']}
    def gradients(t,model,pool,ids,seq,scales,named,index):
        rec,dyn,private,values=original_grad(t,model,pool,ids,seq,scales,named,index)
        step=calls[pool.domain];calls[pool.domain]+=1
        update=True;reference=dyn
        if cfg['method']!='cosine':
            update=step%cfg['reference_interval']==0
            if update:
                # Fresh TRAIN minibatch. Preserve main training dropout RNG.
                rng=np.random.default_rng(390000+step*10+base.DOMAINS.index(pool.domain))
                rid,rseq=base.previous.sample(pool,rng,8)
                params=[named[i][1] for i in index]
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(730000+step)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        _,ld=t.losses(model,pool,rid,rseq)
                        loss=.5*ld/scales['dynamics']
                    ref=torch.autograd.grad(loss,params,allow_unused=True)
                reference=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(ref,params)])
        routed=router.apply(pool.domain,rec,reference,step,update)
        return routed,dyn,private,values
    def combine(gs,method,step=0):
        direction,info=original_combine(gs,method,step)
        if len(gs)>1:
            direction,info['bearing_floor']=fm.bearing_floor(direction,gs[cfg['domains'].index('bearing')],len(gs))
        info['routing']=router.records[-len(gs):]
        return direction,info
    def trainer(name):
        t=original_trainer(name);save=t.save;validate=t.validate
        def saved(path,payload):
            if path.name=='last.pt':payload=dict(payload,routing_state=router.state_dict())
            save(path,payload)
        def validated(model,pools):
            result=validate(model,pools)
            if router.records:
                last=router.records[-1]['step'];t.write(t.OUT/'routing'/f'epoch_{last//20+1:04d}.json',router.records)
                router.records.clear()
            return result
        t.save=saved;t.validate=validated
        # Restore the update counter together with gate/optimizer/RNG state.
        cp=t.OUT/'last.pt'
        if cp.exists():
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            for d in calls:calls[d]=ck['epoch']*20
        t.write(t.BASE/'routing_implementation.json',dict(settings=cfg,
            sha256=hashlib.sha256((HERE/'routing.py').read_bytes()).hexdigest()))
        return t
    base.component_gradients=gradients;base.combine=combine;base.trainer=trainer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required')
    base.main()

if __name__=='__main__':main()
