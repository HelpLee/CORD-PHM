"""Single-domain view of the upstream Domain Adapter encoder."""
import torch
from torch import nn
from compression_model import Attention65Encoder


class DomainAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(96), nn.Linear(96,24),
                                 nn.GELU(), nn.Linear(24,96))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return h + self.net(h)


class DomainAdapterEncoder(Attention65Encoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.adapters = nn.ModuleList([DomainAdapter(), DomainAdapter()])
        for index, layer in enumerate(self.transformer.layers):
            layer.register_forward_hook(self.adapter_hook(index))

    def adapter_hook(self, index):
        def apply_adapter(module, args, output):
            return self.adapters[index](output)
        return apply_adapter
