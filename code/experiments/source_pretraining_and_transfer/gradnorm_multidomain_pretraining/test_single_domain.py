"""Bounded synthetic smoke test; never trains on research data or changes results."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'three_domain/code'))
import numpy as np
import torch
from torch import nn
import train_joint as trainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.transformer = nn.Module()
        self.encoder.transformer.layers = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])
        self.heads = nn.ParameterDict({d: nn.Parameter(torch.tensor(.5)) for d in trainer.ALL_DOMAINS})


class TinyPool:
    def __init__(self, domain):
        self.domain = self.name = domain
        self.groups = {'unit': np.arange(8)}
        self.tw = np.arange(7)[None, :]
        self.vr = np.arange(8)
        self.vw = self.tw


def fake_losses(model, pool, ids, seq, fixed=None):
    pred = model.encoder.transformer.layers[1](
        model.encoder.transformer.layers[0](torch.ones(1, 1, device='cuda')))
    pred = pred + model.heads[pool.domain]
    return (pred - .25).square().mean(), (pred + .3).square().mean()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is needed for this bounded trainer smoke test')
    original_experiment = trainer.EXPERIMENT
    with tempfile.TemporaryDirectory(prefix='single_norm_smoke_', dir=HERE.parents[4]) as folder:
        trainer.EXPERIMENT = Path(folder)
        try:
            with patch.object(trainer, 'JointModel', TinyModel), \
                 patch.object(trainer, 'prepare', lambda: {d: [TinyPool(d)] for d in trainer.DOMAINS}), \
                 patch.object(trainer, 'losses', fake_losses):
                for domain in trainer.ALL_DOMAINS:
                    trainer.MAX_EPOCHS = 1
                    with patch.object(sys, 'argv', ['train_joint.py', '--domain', domain]):
                        trainer.main()
                    trainer.MAX_EPOCHS = 6
                    with patch.object(sys, 'argv', ['train_joint.py', '--domain', domain, '--resume']):
                        trainer.main()
                    root = Path(folder) / 'single_domain' / domain
                    history = json.loads((root / 'training/history.json').read_text())
                    protocol = json.loads((root / 'protocol.json').read_text())
                    assert len(history) == 6 and history[-1]['gradnorm'][-1]['update'] == 120
                    assert protocol['joint_batch'] == 32 and protocol['active_domains'] == [domain]
                    assert protocol['domain_balancing']['method'] == 'fixed_single_weight_1'
                    assert all(r['task_weights'] == {domain: 1.0} for r in history)
                    assert all(g['gradnorm_loss'] is None for r in history for g in r['gradnorm'])
                    ck = torch.load(root / 'training/last.pt', weights_only=False, map_location='cpu')
                    for inactive in set(trainer.ALL_DOMAINS) - {domain}:
                        assert ck['model']['heads.' + inactive].item() == .5
                    print('PASS single-domain, isolated output, inactive params, resume:', domain)
                # Default joint path still uses all three domains and adaptive weights.
                trainer.MAX_EPOCHS = 6
                with patch.object(sys, 'argv', ['train_joint.py']):
                    trainer.main()
                history = json.loads((Path(folder) / 'three_domain/training/history.json').read_text())
                assert len(history[-1]['task_weights']) == 3
                assert any(g['gradnorm_loss'] is not None for g in history[-1]['gradnorm'])
                assert abs(sum(history[-1]['task_weights'].values()) - 3) < 1e-5
                print('PASS unchanged joint mode and active GradNorm')
        finally:
            trainer.EXPERIMENT = original_experiment


if __name__ == '__main__':
    main()
