"""Local masked-token experiments; original paper_v3 model stays untouched."""
from types import MethodType
import torch


def structured_mask(valid, ratio, kind, fixed_seed=None):
    assert 0 < ratio < 1
    assert kind in ('random', 'span_sync')
    valid = valid.bool()
    generator = None
    if fixed_seed is not None:
        generator = torch.Generator(device=valid.device).manual_seed(int(fixed_seed))
    counts = valid.sum(-1)
    # Preserve one visible token when possible. A singleton has no contextual
    # reconstruction task and is left visible (excluded from loss).
    selected = torch.minimum(torch.round(counts.float()*ratio).long().clamp_min(1),
                             (counts-1).clamp_min(0))
    if kind == 'random':
        scores = torch.rand(valid.shape, device=valid.device, generator=generator)
        order = scores.masked_fill(~valid, 2.).argsort(-1)
        ranks = torch.empty_like(order)
        ranks.scatter_(-1, order, torch.arange(valid.shape[-1],device=valid.device).expand_as(order))
        return valid & (ranks < selected[...,None])
    # One continuous interval of valid positions, with shared relative start
    # across sensors. Equal valid layouts therefore hide identical positions,
    # preventing an unmasked parallel sensor from supplying the same patch.
    u = torch.rand((valid.shape[0],1),device=valid.device,generator=generator)
    start = torch.floor(u*(counts-selected+1)).long()
    rank = valid.long().cumsum(-1)-1
    return valid & (rank>=start[...,None]) & (rank<(start+selected)[...,None])


def configure_mask(model, kind='random', ratio=.3):
    if kind == 'random' and ratio == .3:
        return model  # Preserve the exact historical RNG/count contract.
    def draw(backbone, valid, fixed_seed=None, mask_ratio=None):
        return structured_mask(valid, ratio if mask_ratio is None else mask_ratio,kind,fixed_seed)
    model.backbone.random_token_mask = MethodType(draw,model.backbone)
    return model
def masked_regression(prediction,target,mask,kind='mse',feature_mask=None):
    """Equal feature weighting; Huber bounds the influence of extreme targets."""
    import torch.nn.functional as F
    assert kind in ('mse','huber')
    error=(prediction.float()-target.float()).square() if kind=='mse' else F.smooth_l1_loss(prediction.float(),target.float(),beta=1.,reduction='none')
    if feature_mask is None:return (error*mask[...,None]).sum()/(mask.sum()*target.shape[-1]).clamp_min(1)
    valid=mask[...,None]&feature_mask.bool()
    return (error*valid).sum()/valid.sum().clamp_min(1)
