"""One attention-pooled token per position, preserving all native input channels."""
import math
import torch
from torch import nn
from global_local_model import GlobalLocalEncoder, DeltaGlobalLocalRULModel as OriginalDelta
from masking import structured_mask

class ChannelPool(nn.Module):
    def __init__(self,d=96,k=24):
        super().__init__()
        self.key=nn.Linear(d,k,bias=False)
        self.query=nn.Parameter(torch.randn(k)*.02)

    def forward(self,h,valid):
        # Channel axis is -2: [batch,window,channel,dim] or [batch,channel,dim].
        logits=(self.key(h)*self.query).sum(-1)/math.sqrt(len(self.query))
        logits=logits.float().masked_fill(~valid,float('-inf'))
        any_valid=valid.any(-1,keepdim=True)
        logits=torch.where(any_valid,logits,torch.zeros_like(logits))
        weights=logits.softmax(-1)*valid
        weights=weights.to(h.dtype)
        return (torch.where(valid[...,None],h,torch.zeros_like(h))*weights[...,None]).sum(-2)

class Attention65Encoder(GlobalLocalEncoder):
    def __init__(self,**kwargs):
        kwargs['variable_channels']=True
        super().__init__(**kwargs)
        self.local_pool=ChannelPool(self.d)
        self.global_pool=ChannelPool(self.d)

    def forward(self,x,g,cm,tm,masked=None):
        b,c,t,f=x.shape
        assert (t,f)==(64,26)
        valid=tm.bool() & cm.bool()[...,None]
        local=self.local_proj(self.local_norm(x))
        if masked is not None:
            local=torch.where(masked[...,None],self.mask_embedding.expand_as(local),local)
        fused=self.local_pool(local.transpose(1,2),valid.transpose(1,2))+self.position[None]
        global_token=self.global_pool(self.global_proj(self.global_norm(g)),cm.bool())[:,None]+self.global_type
        seq=torch.cat((global_token,fused),1)
        local_valid=valid.any(1)
        padding=torch.cat((torch.zeros(b,1,dtype=torch.bool,device=x.device),~local_valid),1)
        hidden=self.final_norm(self.transformer(seq,src_key_padding_mask=padding))
        hg,hl=hidden[:,0],hidden[:,1:]
        pooled=(hl*local_valid[...,None]).sum(1)/local_valid.sum(1,keepdim=True).clamp_min(1)
        snapshot=self.fusion(torch.cat((hg,pooled),-1))
        return dict(snapshot=snapshot,global_hidden=hg,local_hidden=hl,valid=local_valid)

class Attention65MaskedModel(nn.Module):
    def __init__(self,channels=8,**kwargs):
        super().__init__()
        self.encoder=Attention65Encoder(channels=channels,**kwargs)
        self.decoder_channel=nn.Embedding(channels,96)
        self.decoder=nn.Sequential(nn.LayerNorm(288),nn.Linear(288,192),nn.GELU(),nn.Linear(192,26))

    def forward(self,x,g,cm,tm,fixed=None):
        b,c,t,_=x.shape
        valid=tm.bool()&cm.bool()[...,None]
        mask=structured_mask(valid,.3,'random',fixed)
        out=self.encoder(x,g,cm,tm,mask)
        local=out['local_hidden'][:,None].expand(b,c,t,96)
        snap=out['snapshot'][:,None,None].expand_as(local)
        channel=self.decoder_channel(torch.arange(c,device=x.device))[None,:,None].expand_as(local)
        pred=self.decoder(torch.cat((local,snap,channel),-1))
        return pred,mask,out

class Attention65RUL(OriginalDelta):
    def __init__(self,head_dropout=None,independent_channels=False,**kwargs):
        nn.Module.__init__(self)
        assert not independent_channels
        self.encoder=Attention65Encoder(**kwargs)
        dropout=kwargs.get('dropout',.1) if head_dropout is None else head_dropout
        self.delta_projection=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,96),nn.GELU())
        self.gru=nn.GRU(96,96,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1))
