"""Auxiliary-head isolation and FAMO equations (Liu et al., NeurIPS 2023).

The transferable encoder is unmodified. Extra heads are discarded downstream.
FAMO reference: https://github.com/Cranial-XIX/FAMO/blob/main/famo.py
"""
import copy
import torch
from torch import nn


class LossRateBalancer:
    """Equivalent FAMO gradient coefficients for separately accumulated losses.

    Before/after losses MUST use identical examples, masks and dropout seeds.
    The logits update uses training losses only, never validation or test labels.
    """
    def __init__(self, tasks, device):
        self.logits = nn.Parameter(torch.zeros(tasks, device=device))
        self.optimizer = torch.optim.Adam([self.logits], lr=.025, weight_decay=.01)

    @torch.no_grad()
    def coefficients(self, losses):
        ratios = self.logits.softmax(0) / (losses.detach() + 1e-8)
        return ratios / ratios.sum()

    def update(self, before, after):
        improvement = (before.detach() + 1e-8).log() - (after.detach() + 1e-8).log()
        with torch.enable_grad():
            objective = (self.logits.softmax(0) * improvement).sum()
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward()
            self.optimizer.step()

    def state_dict(self):
        return dict(logits=self.logits.detach(), optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state):
        with torch.no_grad():
            self.logits.copy_(state['logits'])
        self.optimizer.load_state_dict(state['optimizer'])


def build_model(base_class, sources, private_dynamics=False, dataset_heads=False):
    class Model(base_class):
        def __init__(self):
            super().__init__()
            self.private_dynamics = private_dynamics
            self.dataset_heads = dataset_heads
            # Clone identical initialization; do not consume another RNG draw.
            if private_dynamics:
                self.domain_grus = nn.ModuleDict({d: copy.deepcopy(self.gru) for d in sources})
                self.domain_projections = nn.ModuleDict({d: copy.deepcopy(self.projection) for d in sources})
                del self.gru, self.projection
            if dataset_heads:
                self.source_channels = nn.ModuleDict()
                self.source_decoders = nn.ModuleDict()
                for domain, names in sources.items():
                    for name in names:
                        key = domain + '__' + name
                        self.source_channels[key] = copy.deepcopy(self.channels[domain])
                        self.source_decoders[key] = copy.deepcopy(self.decoders[domain])
                del self.channels, self.decoders

        def reconstruction_source(self, domain, name, x, g, cm, tm, observed, fixed=None):
            if not self.dataset_heads:
                return super().reconstruction(domain, x, g, cm, tm, observed, fixed=fixed)
            from masking import structured_mask
            b, c, t, _ = x.shape
            masked = structured_mask(tm.bool() & cm.bool()[..., None], .3, 'random', fixed)
            embedding, local = self.encoder(domain, x, g, cm, tm, masked)
            key = domain + '__' + name
            query = self.source_channels[key](torch.arange(c, device=x.device))
            query = query[None, :, None].expand(b, c, t, 96)
            prediction = self.source_decoders[key](torch.cat((
                local[:, None].expand(b, c, t, 96),
                embedding[:, None, None].expand(b, c, t, 96), query), -1))
            chosen = masked[..., None] & observed.bool()
            return ((prediction.float() - x.float()).square() * chosen).sum() / chosen.sum().clamp_min(1)

        def dynamics(self, domain, x, g, cm, tm):
            if not self.private_dynamics:
                return super().dynamics(domain, x, g, cm, tm)
            embedding, _ = self.encoder(domain, x, g, cm, tm)
            embedding = embedding.reshape(-1, 7, 96)
            prediction = self.domain_projections[domain](self.domain_grus[domain](embedding[:, :6])[1][-1])
            return (prediction.float() - embedding[:, 6].detach().float()).square().mean()

    return Model()


def losses(model, pool, ids, sequence, fixed=None):
    masked = model.reconstruction_source(pool.domain, pool.name, *pool.batch(ids), fixed=fixed)
    values = pool.batch(sequence.reshape(-1))
    temporal = model.dynamics(pool.domain, *values[:4])
    return masked, temporal


def gradient_info(grads, weights):
    gram = grads @ grads.T
    norm = gram.diag().clamp_min(0).sqrt()
    applied = weights @ grads
    return dict(norms=norm.tolist(),
        cosine=(gram / (norm[:, None] * norm[None, :]).clamp_min(1e-20)).tolist(),
        direction_cosine=((grads @ applied) / (norm * applied.norm()).clamp_min(1e-20)).tolist())
