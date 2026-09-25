"""Battery State-Trend downstream, with no extra physical bypass features."""
import torch
from torch import nn
from global_local_model import GlobalLocalEncoder


class StateTrendRUL(nn.Module):
    def __init__(self, dropout=.05):
        super().__init__()
        self.encoder = GlobalLocalEncoder(channels=1, dropout=dropout)
        layer = nn.TransformerEncoderLayer(96, 4, 192, dropout, 'gelu',
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.temporal_norm = nn.LayerNorm(96)
        self.register_buffer('position', GlobalLocalEncoder._sin(20, 96), persistent=False)
        self.head = nn.Sequential(nn.LayerNorm(288), nn.Linear(288, 96), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(96, 1), nn.Sigmoid())

    @staticmethod
    def trend(e, lengths):
        # No fabricated earlier cycles: trend is zero until ten real states exist.
        index = lengths[:, None]-10+torch.arange(10, device=e.device)[None, :]
        history = e.gather(1, index.clamp_min(0)[..., None].expand(-1, -1, 96))
        d = history[:, 5:].mean(1)-history[:, :5].mean(1)
        return torch.where((lengths >= 10)[:, None], d, torch.zeros_like(d))

    def temporal_states(self, e, history_mask):
        b, s, _ = e.shape
        lengths = history_mask.sum(1).long()
        assert torch.all(lengths > 0)
        assert torch.equal(history_mask, torch.arange(s, device=e.device)[None, :] < lengths[:, None])
        causal = torch.ones((s, s), dtype=torch.bool, device=e.device).triu(1)
        h = self.temporal(e+self.position[:s], mask=causal, src_key_padding_mask=~history_mask)
        return self.temporal_norm(h)

    def forward(self, x, g, cm, tm, history_mask):
        b, s, c, t, f = x.shape
        valid = history_mask.reshape(-1)
        # Encode only real history; padded cycles never enter the snapshot encoder.
        e_valid = self.encoder(x.reshape(-1, c, t, f)[valid], g.reshape(-1, c, f)[valid],
                               cm.reshape(-1, c)[valid], tm.reshape(-1, c, t)[valid])['snapshot']
        e = e_valid.new_zeros((b*s, 96))
        e[valid] = e_valid
        e = e.reshape(b, s, 96)
        lengths = history_mask.sum(1).long()
        h = self.temporal_states(e, history_mask)
        ix = torch.arange(b, device=e.device)
        current = e[ix, lengths-1]
        latest = h[ix, lengths-1]
        return self.head(torch.cat((latest, self.trend(e, lengths), current), -1)).squeeze(-1)
