"""E33: fair FP32 controls on V100; dataset-private auxiliary heads/FAMO."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import contextlib
import hashlib
import importlib.util
import json
import random
import signal
import sys
import time
import traceback
import numpy as np
import torch
from config import ARMS, DOMAINS, HERE, SUITE, settings
from methods import LossRateBalancer, build_model, gradient_info, losses

sys.path.insert(0, str(SUITE / '29_joint_optimization_comparison'))
spec = importlib.util.spec_from_file_location('e29_train', SUITE / '29_joint_optimization_comparison/train.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def load_trainer(arm):
    # Reuse frozen verified caches and historical splits, but isolate all outputs.
    previous.HERE = HERE
    trainer = previous.load_trainer(arm)
    trainer.losses = losses
    return trainer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_batch(pools, domain, epoch, update):
    pool = pools[domain][((epoch - 1) * 20 + update) % len(pools[domain])]
    seed = 42 + epoch * 10000 + update * 10 + DOMAINS.index(domain)
    ids, seq = previous.sample(pool, np.random.default_rng(seed))
    return pool, ids, seq, seed


def domain_gradient(model, batch, scale, backward):
    pool, ids, seq, seed = batch
    torch.manual_seed(seed)
    values = np.zeros(2)
    context = contextlib.nullcontext() if backward else torch.no_grad()
    with context:
        for i in range(0, 32, 8):
            lm, ld = losses(model, pool, ids[i:i+8], seq[i:i+8])
            objective = (lm + .2 * ld) / (4 * scale)
            if not torch.isfinite(objective):
                raise FloatingPointError((pool.domain, pool.name, 'nonfinite loss'))
            if backward:
                objective.backward()
            values += np.array([lm.item(), ld.item()]) / 4
    return values, float((values[0] + .2 * values[1]) / scale)


def parameter_groups(named):
    groups = {'shared_encoder': [], 'shared_dynamics': [], 'block0': [], 'block1': [], 'pool_fusion': []}
    offset = 0
    for name, p in named:
        indexes = torch.arange(offset, offset + p.numel(), device=p.device)
        if name.startswith('encoder.') and not name.startswith(('encoder.stems.', 'encoder.adapters.')):
            groups['shared_encoder'].append(indexes)
            if name.startswith('encoder.transformer.layers.0.'):
                groups['block0'].append(indexes)
            elif name.startswith('encoder.transformer.layers.1.'):
                groups['block1'].append(indexes)
            else:
                groups['pool_fusion'].append(indexes)
        elif name.startswith(('gru.', 'projection.')):
            groups['shared_dynamics'].append(indexes)
        offset += p.numel()
    return {name: torch.cat(value) for name, value in groups.items() if value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = settings(args.arm)
    active = cfg['domains']
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(42)
    run_name = 'smoke_' + args.arm if args.smoke else args.arm
    t = load_trainer(run_name)
    status = t.OUT / 'status.json'
    if status.exists() and json.loads(status.read_text()).get('state') == 'complete':
        print('ALREADY_COMPLETE', run_name, flush=True)
        return
    t.write(status, dict(state='preparing', arm=args.arm, pid=os.getpid()))
    pools = t.prepare()
    t.DOMAINS = active
    sources = {d: [p.name for p in pools[d]] for d in DOMAINS}
    model = build_model(t.JointModel, sources, cfg['private_dynamics'], cfg['dataset_heads']).cuda()
    named = list(model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    balancer = LossRateBalancer(len(active), 'cuda') if cfg['famo'] else None
    groups = parameter_groups(named)
    # Calibration is identical at initialization because heads are exact clones.
    scales, baseline = previous.calibrate(t, model, pools, active)
    protocol = dict(arm=args.arm, seed=42, settings=cfg, sources=sources,
        max_epochs=500, patience=30, min_delta=1e-4, updates_per_epoch=20,
        batch_per_domain=32, microbatch=8, lr=1e-4, weight_decay=1e-4, clip=5,
        precision='float32; TF32 disabled for every E33 upstream arm',
        train_scales=scales, validation_baseline=baseline,
        selection='macro over datasets then domains of val_total / initial_val_total',
        objective='(masked_MSE + 0.2*latent_dynamics_MSE)/fixed_initial_train_loss',
        train_parameters=sum(p.numel() for p in model.parameters()),
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        famo={'logit_lr': .025, 'logit_weight_decay': .01, 'min_loss': 0,
              'same_batch_replay': True} if balancer else None,
        validation_excluded={d: [p.name for p in pools[d] if not len(p.vr) or not len(p.vw)] for d in active},
        device=torch.cuda.get_device_name(), torch_version=torch.__version__)
    immutable = dict(protocol)
    immutable.pop('device'); immutable.pop('torch_version')
    fingerprint = hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()
    best, best_epoch, stale, start, history = float('inf'), 0, 0, 1, []
    checkpoint = t.OUT / 'last.pt'
    if checkpoint.exists():
        if not args.resume:
            raise RuntimeError('Existing run; use --resume')
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck['fingerprint'] != fingerprint:
            raise RuntimeError('Protocol drift: refusing incompatible resume')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        if balancer:
            balancer.load_state_dict(ck['balancer'])
        best, best_epoch, stale = ck['best'], ck['best_epoch'], ck['stale']
        start, history = ck['epoch'] + 1, ck['history']
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
    t.write(t.BASE / 'protocol.json', dict(protocol, fingerprint=fingerprint))
    requested_stop = [False]
    def stop_after_epoch(*_):
        requested_stop[0] = True
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, stop_after_epoch)
    began = time.monotonic()
    final_epoch = start - 1
    try:
        for epoch in range(start, (1 if args.smoke else 500) + 1):
            if stale >= 30:
                break
            model.train()
            totals = {d: [] for d in active}
            diagnostics = []
            for step in range(1 if args.smoke else 20):
                all_grads, used, batches, before = [], set(), [], []
                for domain in active:
                    optimizer.zero_grad(set_to_none=True)
                    batch = select_batch(pools, domain, epoch, step)
                    values, value = domain_gradient(model, batch, scales[domain][batch[0].name], True)
                    totals[domain].append(values.tolist())
                    batches.append(batch); before.append(value)
                    used.update(i for i, (_, p) in enumerate(named) if p.grad is not None)
                    all_grads.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for _, p in named]))
                grads = torch.stack(all_grads)
                before = torch.as_tensor(before, device='cuda', dtype=torch.float32)
                weights = balancer.coefficients(before) if balancer else torch.full_like(before, 1 / len(active))
                direction = weights @ grads
                if step == 0:
                    diagnostics.append(dict(step=step, weights=weights.tolist(),
                        losses=before.tolist(), datasets=[b[0].name for b in batches],
                        groups={key: gradient_info(grads[:, ids], weights) for key, ids in groups.items()}))
                optimizer.zero_grad(set_to_none=True)
                offset = 0
                for i, (_, p) in enumerate(named):
                    if i in used:
                        p.grad = direction[offset:offset + p.numel()].view_as(p).clone()
                    offset += p.numel()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if not torch.isfinite(norm):
                    raise FloatingPointError('nonfinite gradients')
                optimizer.step()
                if balancer:
                    # Replay stochastic masks and dropout on the same TRAIN batches.
                    after = [domain_gradient(model, b, scales[b[0].domain][b[0].name], False)[1] for b in batches]
                    balancer.update(before, torch.as_tensor(after, device='cuda'))
                t.write(status, dict(state='training', arm=args.arm, epoch=epoch,
                    update=(epoch - 1) * 20 + step + 1, pid=os.getpid()))
            raw, detail = t.validate(model, pools)
            score = previous.normalized_score(detail, baseline)
            if not np.isfinite(score):
                raise FloatingPointError('nonfinite validation')
            improved = score < best - 1e-4
            if improved:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            history.append(dict(epoch=epoch, monitor=score, raw=raw, validation=detail,
                train={d: np.mean(v, axis=0).tolist() for d, v in totals.items()},
                gradients=diagnostics, session_seconds=time.monotonic() - began))
            ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                balancer=balancer.state_dict() if balancer else None, epoch=epoch,
                best=best, best_epoch=best_epoch, stale=stale, history=history,
                fingerprint=fingerprint, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            t.save(checkpoint, ck)
            if improved:
                t.save(t.OUT / 'encoder.pt', {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()})
            t.write(t.OUT / 'history.json', history)
            final_epoch = epoch
            print('EPOCH', args.arm, epoch, 'VAL', score, 'BEST', best_epoch, 'STALE', stale, flush=True)
            if requested_stop[0] and stale < 30 and epoch < 500:
                t.write(status, dict(state='interrupted', arm=args.arm, epoch=epoch, best_epoch=best_epoch,
                                    reason='Slurm time signal; saved full checkpoint'))
                raise SystemExit(75)
        t.write(status, dict(state='complete', arm=args.arm, epoch=final_epoch,
            best_epoch=best_epoch, best_validation=best, fingerprint=fingerprint,
            reason='smoke' if args.smoke else 'early_stopping' if stale >= 30 else 'max_epochs'))
    except Exception as error:
        t.write(status, dict(state='failed', arm=args.arm, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    main()
