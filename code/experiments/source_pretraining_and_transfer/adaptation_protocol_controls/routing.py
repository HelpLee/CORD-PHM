"""Adaptive auxiliary gradients; learned gates minimize a first-order surrogate."""
import math
import torch

class Router:
    def __init__(self,cfg):
        self.cfg=cfg;self.groups=[];self.theta={};self.ema={};self.records=[]

    def initialize(self,named,index,active):
        groups={};offset=0
        for i in index:
            name,p=named[i];end=offset+p.numel()
            key='all'
            if self.cfg['method']=='layerwise':
                if name.startswith('encoder.transformer.layers.'):
                    key='block'+name.split('.')[3]
                elif name.startswith('encoder.'): key='encoder_other'
                else: key='dynamics_head'
            groups.setdefault(key,[]).append((offset,end));offset=end
        self.groups=list(groups.items())
        # All domains/layers have the same rho=.5 initialization.
        initial=math.log((.5-.05)/(1.-.5))
        for d in active:
            self.theta[d]=torch.full((len(self.groups),),initial,dtype=torch.float64)
            self.ema[d]=torch.full((len(self.groups),),(.5-.05)/.95 if self.cfg['method']=='cosine' else 0.,dtype=torch.float64)

    def state_dict(self):
        return dict(groups=self.groups,theta=self.theta,ema=self.ema)

    def load_state_dict(self,state):
        if state['groups']!=self.groups: raise RuntimeError('Routing parameter layout drift')
        self.theta=state['theta'];self.ema=state['ema']

    def apply(self,domain,rec,reference,step,update=True):
        values=[];alignment=[]
        for _,spans in self.groups:
            a=torch.cat([rec[s:e] for s,e in spans]);b=torch.cat([reference[s:e] for s,e in spans])
            alignment.append(float((a@b)/(a.norm()*b.norm()).clamp_min(1e-20)))
        evidence=torch.tensor(alignment,dtype=torch.float64)
        if update:
            self.ema[domain]=self.cfg['ema']*self.ema[domain]+(1-self.cfg['ema'])*evidence
        if self.cfg['method']=='cosine':
            rho=(.05+.95*self.ema[domain].clamp(0,1)).clamp(.05,1.)
        else:
            z=self.theta[domain].detach().requires_grad_(True)
            q=z.sigmoid();rho=.05+.95*q
            # Maximize estimated dynamic-loss decrease, with a weak common prior
            # against gate saturation. This is not the weighted training loss.
            kl=q*torch.log((q/.5).clamp_min(1e-12))+(1-q)*torch.log(((1-q)/.5).clamp_min(1e-12))
            surrogate=-(rho*self.ema[domain]).mean()+self.cfg['entropy_weight']*kl.mean()
            if update:
                gradient=torch.autograd.grad(surrogate,z)[0]
                self.theta[domain]=(z-self.cfg['gate_lr']*gradient).detach().clamp(-8,8)
                rho=.05+.95*self.theta[domain].sigmoid()
        result=rec.clone()
        for value,(_,spans) in zip(rho,self.groups):
            values.append(float(value))
            for s,e in spans: result[s:e]*=float(value)
        self.records.append(dict(domain=domain,step=step,rho=values,alignment=alignment,
                                 groups=[g[0] for g in self.groups],gate_updated=update))
        return result
