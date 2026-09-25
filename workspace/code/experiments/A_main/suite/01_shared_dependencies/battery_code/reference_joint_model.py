"""Three independent stems/decoders with one shared attention65 backbone."""
import torch
from torch import nn
from compression_model import Attention65Encoder
from masking import structured_mask

DOMAINS = ('bearing', 'battery', 'milling')

class DomainAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(96), nn.Linear(96,24), nn.GELU(), nn.Linear(24,96))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return h + self.net(h)

class JointEncoder(Attention65Encoder):
    def __init__(self):
        super().__init__(channels=8, dropout=.1)
        del self.local_norm, self.local_proj, self.global_norm, self.global_proj
        self.stems = nn.ModuleDict({d: nn.ModuleDict({
            'local': nn.Sequential(nn.LayerNorm(26), nn.Linear(26,96)),
            'global': nn.Sequential(nn.LayerNorm(26), nn.Linear(26,96))}) for d in DOMAINS})

    def forward(self, domain, x, g, cm, tm, masked=None):
        self.active_domain = domain
        b,c,t,f=x.shape
        assert (t,f)==(64,26)
        valid=tm.bool() & cm.bool()[...,None]
        local=self.stems[domain]['local'](x)
        if masked is not None:
            local=torch.where(masked[...,None],self.mask_embedding.expand_as(local),local)
        local=self.local_pool(local.transpose(1,2),valid.transpose(1,2))+self.position[None]
        global_token=self.global_pool(self.stems[domain]['global'](g),cm.bool())[:,None]+self.global_type
        local_valid=valid.any(1)
        padding=torch.cat((torch.zeros(b,1,dtype=torch.bool,device=x.device),~local_valid),1)
        hidden=self.final_norm(self.transformer(torch.cat((global_token,local),1),src_key_padding_mask=padding))
        hg,hl=hidden[:,0],hidden[:,1:]
        pooled=(hl*local_valid[...,None]).sum(1)/local_valid.sum(1,keepdim=True).clamp_min(1)
        return self.fusion(torch.cat((hg,pooled),-1)), hl

    def install_adapters(self):
        # Called after the entire original model is initialized; preserve its RNG.
        with torch.random.fork_rng(devices=[]):
            self.adapters = nn.ModuleDict({d: nn.ModuleList([DomainAdapter(), DomainAdapter()]) for d in DOMAINS})
        for index, layer in enumerate(self.transformer.layers):
            layer.register_forward_hook(self.adapter_hook(index))

    def adapter_hook(self, index):
        def apply_adapter(module, args, output):
            return self.adapters[self.active_domain][index](output)
        return apply_adapter

class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=JointEncoder()
        self.channels=nn.ModuleDict({d:nn.Embedding(c,96) for d,c in zip(DOMAINS,(8,1,3))})
        self.decoders=nn.ModuleDict({d:nn.Sequential(nn.LayerNorm(288),nn.Linear(288,192),nn.GELU(),nn.Linear(192,26)) for d in DOMAINS})
        self.gru=nn.GRU(96,96,batch_first=True)
        self.projection=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,96))
        self.encoder.install_adapters()

    def reconstruction(self, domain, x,g,cm,tm,observed,fixed=None):
        b,c,t,_=x.shape
        masked=structured_mask(tm.bool() & cm.bool()[...,None],.3,'random',fixed)
        e,h=self.encoder(domain,x,g,cm,tm,masked)
        channel=self.channels[domain](torch.arange(c,device=x.device))[None,:,None].expand(b,c,t,96)
        pred=self.decoders[domain](torch.cat((h[:,None].expand(b,c,t,96),e[:,None,None].expand(b,c,t,96),channel),-1))
        chosen=masked[...,None] & observed.bool()
        return ((pred.float()-x.float()).square()*chosen).sum()/chosen.sum().clamp_min(1)

    def dynamics(self,domain,x,g,cm,tm):
        e,_=self.encoder(domain,x,g,cm,tm)
        e=e.reshape(-1,7,96)
        pred=self.projection(self.gru(e[:,:6])[1][-1])
        return (pred.float()-e[:,6].detach().float()).square().mean()
