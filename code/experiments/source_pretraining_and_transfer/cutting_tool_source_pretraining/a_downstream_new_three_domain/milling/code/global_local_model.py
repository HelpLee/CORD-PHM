"""One global raw-snapshot token plus all local HealthTokens."""
import torch
from torch import nn
from masking import structured_mask


class GlobalLocalEncoder(nn.Module):
    def __init__(self,features=26,channels=2,tokens=64,d=96,heads=4,layers=2,ff=192,dropout=.1,
                 input_norm=True,variable_channels=False):
        super().__init__();self.features=features;self.channels=channels;self.tokens=tokens;self.d=d
        self.variable_channels=bool(variable_channels)
        self.local_norm=nn.LayerNorm(features) if input_norm else nn.Identity()
        global_features=features if self.variable_channels else channels*features
        self.global_norm=nn.LayerNorm(global_features) if input_norm else nn.Identity()
        self.local_proj=nn.Linear(features,d);self.global_proj=nn.Linear(global_features,d)
        self.mask_embedding=nn.Parameter(torch.randn(1,1,1,d)*.02)
        self.global_type=nn.Parameter(torch.randn(1,1,d)*.02)
        self.channel_embedding=None if self.variable_channels else nn.Embedding(channels,d)
        self.register_buffer('position',self._sin(tokens,d),persistent=False)
        layer=nn.TransformerEncoderLayer(d,heads,ff,dropout,'gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,layers);self.final_norm=nn.LayerNorm(d)
        self.fusion=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,d))

    @staticmethod
    def _sin(n,d):
        pos=torch.arange(n).float().unsqueeze(1);div=torch.exp(torch.arange(0,d,2).float()*(-torch.log(torch.tensor(10000.))/d))
        pe=torch.zeros(n,d);pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div);return pe

    def forward(self,x,g,cm,tm,masked=None):
        b,c,t,f=x.shape
        assert (t,f)==(self.tokens,self.features) and c>0
        if not self.variable_channels: assert c==self.channels
        local=self.local_proj(self.local_norm(x))+self.position.view(1,1,t,-1)
        if self.channel_embedding is not None:
            local=local+self.channel_embedding.weight.view(1,c,1,-1)
        if masked is not None:local=torch.where(masked[...,None],self.mask_embedding.expand_as(local),local)
        if self.variable_channels:
            # Encode every physical channel independently, then pool only the
            # channels that really exist.  Padding therefore cannot alter the
            # Global Token and datasets may retain their native channel count.
            per_channel=self.global_proj(self.global_norm(g))
            weights=cm.bool().unsqueeze(-1)
            global_hidden=(per_channel*weights).sum(1)/weights.sum(1).clamp_min(1)
            global_token=global_hidden.unsqueeze(1)+self.global_type
        else:
            global_token=self.global_proj(self.global_norm(g.reshape(b,-1))).unsqueeze(1)+self.global_type
        sequence=torch.cat((global_token,local.reshape(b,c*t,-1)),1)
        valid=tm.bool()&cm.bool().unsqueeze(-1);padding=torch.cat((torch.zeros((b,1),dtype=torch.bool,device=x.device),~valid.reshape(b,c*t)),1)
        hidden=self.final_norm(self.transformer(sequence,src_key_padding_mask=padding))
        hg=hidden[:,0];hl=hidden[:,1:].reshape(b,c,t,-1);weights=valid[...,None]
        pooled=(hl*weights).sum((1,2))/weights.sum((1,2)).clamp_min(1)
        snapshot=self.fusion(torch.cat((hg,pooled),-1))
        return dict(global_hidden=hg,local_hidden=hl,snapshot=snapshot,valid=valid)


class GlobalLocalMaskedModel(nn.Module):
    def __init__(self,**encoder_kwargs):
        super().__init__();self.encoder=GlobalLocalEncoder(**encoder_kwargs)
        self.decoder=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,192),nn.GELU(),nn.Linear(192,26))

    def forward(self,x,g,cm,tm,fixed=None):
        valid=tm.bool()&cm.bool().unsqueeze(-1);mask=structured_mask(valid,.3,'random',fixed)
        out=self.encoder(x,g,cm,tm,mask);state=out['snapshot'][:,None,None,:].expand_as(out['local_hidden'])
        pred=self.decoder(torch.cat((out['local_hidden'],state),-1))
        return pred,mask,out


class GlobalLocalRULModel(nn.Module):
    def __init__(self,**encoder_kwargs):
        super().__init__();self.encoder=GlobalLocalEncoder(**encoder_kwargs);self.gru=nn.GRU(96,96,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self,x,g,cm,tm):
        b,s,c,t,f=x.shape
        state=self.encoder(x.reshape(b*s,c,t,f),g.reshape(b*s,c,f),cm.reshape(b*s,c),tm.reshape(b*s,c,t))['snapshot'].reshape(b,s,-1)
        h,_=self.gru(state);return self.head(h[:,-1]).squeeze(-1)


class RelativeGlobalLocalRULModel(nn.Module):
    """Causal trajectory head using absolute state and three relative degradation signals."""
    def __init__(self):
        super().__init__();self.encoder=GlobalLocalEncoder()
        self.trajectory_projection=nn.Sequential(
            nn.LayerNorm(4*96),nn.Linear(4*96,96),nn.GELU(),nn.Linear(96,96))
        self.gru=nn.GRU(96,96,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self,x,g,cm,tm,history_mask,return_sequence=False):
        b,s,c,t,f=x.shape
        state=self.encoder(x.reshape(b*s,c,t,f),g.reshape(b*s,c,f),cm.reshape(b*s,c),tm.reshape(b*s,c,t))['snapshot'].reshape(b,s,-1)
        lengths=history_mask.sum(1).long();compact=state.new_zeros(state.shape)
        for i in range(b):compact[i,:lengths[i]]=state[i,history_mask[i]]
        baseline=compact[:,:1];relative=compact-baseline
        step=compact.new_zeros(compact.shape);step[:,1:]=compact[:,1:]-compact[:,:-1]
        elapsed=torch.arange(s,device=compact.device,dtype=compact.dtype).clamp_min(1).view(1,s,1)
        rate=relative/elapsed
        z=self.trajectory_projection(torch.cat((compact,relative,step,rate),-1))
        valid=torch.arange(s,device=compact.device)[None,:]<lengths[:,None];z=z*valid[...,None]
        packed=nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed_h,h_last=self.gru(packed)
        h,_=nn.utils.rnn.pad_packed_sequence(packed_h,batch_first=True,total_length=s)
        pred_seq=self.head(h).squeeze(-1);pred_last=self.head(h_last[-1]).squeeze(-1)
        if return_sequence:return pred_last,pred_seq,lengths
        return pred_last


class DeltaGlobalLocalRULModel(nn.Module):
    """Six-state trajectory head using only current state and one-step change."""
    def __init__(self,head_dropout=None,independent_channels=False,**encoder_kwargs):
        super().__init__()
        if independent_channels:
            from channel_independent_model import ChannelIndependentEncoder
            self.encoder=ChannelIndependentEncoder(**encoder_kwargs)
        else:
            self.encoder=GlobalLocalEncoder(**encoder_kwargs)
        if head_dropout is None:head_dropout=encoder_kwargs.get('dropout',.1)
        self.delta_projection=nn.Sequential(nn.LayerNorm(2*96),nn.Linear(2*96,96),nn.GELU())
        self.gru=nn.GRU(96,96,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(head_dropout),nn.Linear(64,1))

    def forward(self,x,g,cm,tm,history_mask):
        b,s,c,t,f=x.shape
        state=self.encoder(x.reshape(b*s,c,t,f),g.reshape(b*s,c,f),cm.reshape(b*s,c),tm.reshape(b*s,c,t))['snapshot'].reshape(b,s,-1)
        lengths=history_mask.sum(1).long();compact=state.new_zeros(state.shape)
        for i in range(b):compact[i,:lengths[i]]=state[i,history_mask[i]]
        delta=compact.new_zeros(compact.shape);delta[:,1:]=compact[:,1:]-compact[:,:-1]
        z=self.delta_projection(torch.cat((compact,delta),-1))
        valid=torch.arange(s,device=z.device)[None,:]<lengths[:,None];z=z*valid[...,None]
        packed=nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed);return self.head(h[-1]).squeeze(-1)


class ReferenceDeltaGlobalLocalRULModel(nn.Module):
    """Delta trajectory relative to a causal, unlabeled early-life reference."""
    def __init__(self,**encoder_kwargs):
        super().__init__();self.encoder=GlobalLocalEncoder(**encoder_kwargs)
        self.delta_projection=nn.Sequential(nn.LayerNorm(2*96),nn.Linear(2*96,96),nn.GELU())
        self.gru=nn.GRU(96,96,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self,x,g,cm,tm,history_mask,reference_x,reference_g,reference_cm,reference_tm):
        b,s,c,t,f=x.shape;_,k,_,_,_=reference_x.shape
        state=self.encoder(x.reshape(b*s,c,t,f),g.reshape(b*s,c,f),cm.reshape(b*s,c),tm.reshape(b*s,c,t))['snapshot'].reshape(b,s,-1)
        reference=self.encoder(reference_x.reshape(b*k,c,t,f),reference_g.reshape(b*k,c,f),reference_cm.reshape(b*k,c),reference_tm.reshape(b*k,c,t))['snapshot'].reshape(b,k,-1).mean(1)
        lengths=history_mask.sum(1).long();compact=state.new_zeros(state.shape)
        for i in range(b):compact[i,:lengths[i]]=state[i,history_mask[i]]
        relative=compact-reference[:,None,:]
        delta=relative.new_zeros(relative.shape);delta[:,1:]=relative[:,1:]-relative[:,:-1]
        z=self.delta_projection(torch.cat((relative,delta),-1))
        valid=torch.arange(s,device=z.device)[None,:]<lengths[:,None];z=z*valid[...,None]
        packed=nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed);return self.head(h[-1]).squeeze(-1)


class TemporalPriorDeltaGlobalLocalRULModel(DeltaGlobalLocalRULModel):
    """Delta model with a source-scale, monotone elapsed-time correction."""
    def __init__(self, **encoder_kwargs):
        super().__init__(**encoder_kwargs)
        self.temporal_a0=nn.Parameter(torch.zeros(()))
        self.temporal_b=nn.Parameter(torch.zeros(()))

    def forward(self, x, g, cm, tm, history_mask, tau):
        raw = super().forward(x, g, cm, tm, history_mask)
        return torch.sigmoid(raw - nn.functional.softplus(self.temporal_a0) * tau + self.temporal_b)


class DeltaProgressGlobalLocalRULModel(nn.Module):
    """Six-state head: current state, one-step change, and causal elapsed progress."""
    def __init__(self):
        super().__init__();self.encoder=GlobalLocalEncoder()
        self.input_projection=nn.Sequential(nn.LayerNorm(2*96+1),nn.Linear(2*96+1,96),nn.GELU())
        self.gru=nn.GRU(96,96,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self,x,g,cm,tm,history_mask,progress):
        b,s,c,t,f=x.shape
        state=self.encoder(x.reshape(b*s,c,t,f),g.reshape(b*s,c,f),cm.reshape(b*s,c),tm.reshape(b*s,c,t))['snapshot'].reshape(b,s,-1)
        lengths=history_mask.sum(1).long();compact=state.new_zeros(state.shape);compact_progress=state.new_zeros((b,s,1))
        for i in range(b):
            compact[i,:lengths[i]]=state[i,history_mask[i]]
            compact_progress[i,:lengths[i],0]=progress[i,history_mask[i]]
        delta=compact.new_zeros(compact.shape);delta[:,1:]=compact[:,1:]-compact[:,:-1]
        z=self.input_projection(torch.cat((compact,delta,compact_progress),-1))
        valid=torch.arange(s,device=z.device)[None,:]<lengths[:,None];z=z*valid[...,None]
        packed=nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed);return self.head(h[-1]).squeeze(-1)


class StateTrendGlobalLocalRULModel(nn.Module):
    """Current-state branch gated with a GRU-aggregated local trend branch."""
    def __init__(self):
        super().__init__();self.encoder=GlobalLocalEncoder()
        self.trend_gru=nn.GRU(96,96,batch_first=True)
        self.gate=nn.Sequential(nn.LayerNorm(2*96),nn.Linear(2*96,96),nn.Sigmoid())
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self,x,g,cm,tm,history_mask):
        b,s,c,t,f=x.shape
        state=self.encoder(x.reshape(b*s,c,t,f),g.reshape(b*s,c,f),cm.reshape(b*s,c),tm.reshape(b*s,c,t))['snapshot'].reshape(b,s,-1)
        lengths=history_mask.sum(1).long();compact=state.new_zeros(state.shape)
        for i in range(b):compact[i,:lengths[i]]=state[i,history_mask[i]]
        current=compact[torch.arange(b,device=compact.device),lengths-1]
        differences=compact[:,1:]-compact[:,:-1];trend_lengths=(lengths-1).clamp_min(1)
        trend_valid=torch.arange(s-1,device=compact.device)[None,:]<trend_lengths[:,None]
        differences=differences*trend_valid[...,None]
        packed=nn.utils.rnn.pack_padded_sequence(differences,trend_lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,trend=self.trend_gru(packed);trend=trend[-1]
        trend=torch.where((lengths>1)[:,None],trend,torch.zeros_like(trend))
        alpha=self.gate(torch.cat((current,trend),-1));fused=alpha*current+(1-alpha)*trend
        return self.head(fused).squeeze(-1)


class MultiTaskDeltaGlobalLocalRULModel(nn.Module):
    """State+velocity GRU with independent continuous-RUL and 3-stage heads."""
    def __init__(self):
        super().__init__();self.encoder=GlobalLocalEncoder()
        self.delta_projection=nn.Sequential(nn.LayerNorm(2*96),nn.Linear(2*96,96),nn.GELU())
        self.gru=nn.GRU(96,96,batch_first=True)
        self.rul_head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
        self.stage_head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,3))

    def forward(self,x,g,cm,tm,history_mask):
        b,s,c,t,f=x.shape
        state=self.encoder(x.reshape(b*s,c,t,f),g.reshape(b*s,c,f),cm.reshape(b*s,c),tm.reshape(b*s,c,t))['snapshot'].reshape(b,s,-1)
        lengths=history_mask.sum(1).long();compact=state.new_zeros(state.shape)
        for i in range(b):compact[i,:lengths[i]]=state[i,history_mask[i]]
        delta=compact.new_zeros(compact.shape);delta[:,1:]=compact[:,1:]-compact[:,:-1]
        z=self.delta_projection(torch.cat((compact,delta),-1))
        valid=torch.arange(s,device=z.device)[None,:]<lengths[:,None];z=z*valid[...,None]
        packed=nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed);q=h[-1]
        return self.rul_head(q).squeeze(-1),self.stage_head(q)


class FPTBaselineDeltaGlobalLocalRULModel(nn.Module):
    """Use state relative to the unit's observed FPT baseline plus local velocity."""
    def __init__(self):
        super().__init__();self.encoder=GlobalLocalEncoder()
        self.input_projection=nn.Sequential(nn.LayerNorm(2*96),nn.Linear(2*96,96),nn.GELU())
        self.gru=nn.GRU(96,96,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x,g,cm,tm,history_mask,baseline_x,baseline_g,baseline_cm,baseline_tm):
        b,s,c,t,f=x.shape
        state=self.encoder(x.reshape(b*s,c,t,f),g.reshape(b*s,c,f),cm.reshape(b*s,c),tm.reshape(b*s,c,t))['snapshot'].reshape(b,s,-1)
        baseline=self.encoder(baseline_x,baseline_g,baseline_cm,baseline_tm)['snapshot']
        lengths=history_mask.sum(1).long();compact=state.new_zeros(state.shape)
        for i in range(b):compact[i,:lengths[i]]=state[i,history_mask[i]]
        relative=compact-baseline[:,None,:]
        delta=compact.new_zeros(compact.shape);delta[:,1:]=compact[:,1:]-compact[:,:-1]
        z=self.input_projection(torch.cat((relative,delta),-1))
        valid=torch.arange(s,device=z.device)[None,:]<lengths[:,None];z=z*valid[...,None]
        packed=nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed);return self.head(h[-1]).squeeze(-1)
