"""Reuse COMPONENT_ROUTING component training; alter only bearing routing/gradient aggregation."""
import contextlib
import hashlib
import importlib.util
import json
import sys
import torch
from config import COMPONENT_ROUTING, HERE, MAX_EPOCHS, settings
from methods import bearing_floor, route
from continuation import import_checkpoint


def main():
    path = COMPONENT_ROUTING/'train.py'
    source = path.read_text(encoding='utf-8')
    smoke = '--smoke' in sys.argv
    effective = source
    for old, new in (
        ('max_epochs=500,', f'max_epochs={MAX_EPOCHS},'),
        ('(1 if args.smoke else 500) + 1', f'(1 if args.smoke else {MAX_EPOCHS}) + 1'),
        ('epoch < 500:', f'epoch < {MAX_EPOCHS}:'),
        ('    if checkpoint.exists():', '    if not args.smoke: import_checkpoint(t, args.arm, protocol, fingerprint)\n    if checkpoint.exists():'),
    ):
        if effective.count(old) != 1: raise RuntimeError('Base trainer changed: '+old)
        effective = effective.replace(old, new)
    if smoke:
        effective = effective.replace('BF16 autocast; FP32 gradients', 'FP32 smoke only; excluded from results')
        torch.autocast = lambda *a, **kw: contextlib.nullcontext()
    spec = importlib.util.spec_from_file_location('cord_selected_base_train', path)
    base = importlib.util.module_from_spec(spec)
    base.import_checkpoint = import_checkpoint
    exec(compile(effective, str(path), 'exec'), base.__dict__)
    arm = sys.argv[sys.argv.index('--arm')+1]; cfg = settings(arm)
    original_gradients, original_combine, original_trainer = base.component_gradients, base.combine, base.trainer
    components = []; epoch_diagnostics = []

    def routed(t, model, pool, ids, sequence, scales, named, shared_index):
        rec, dyn, private, values = original_gradients(t, model, pool, ids, sequence, scales, named, shared_index)
        rho = cfg['reconstruction_routing'][pool.domain]
        components.append(dict(domain=pool.domain, dataset=pool.name, rho=rho,
            reconstruction_norm=float(rec.norm()), dynamics_norm=float(dyn.norm())))
        return route(rec, dyn, private, values, rho)

    def aggregate(grads, method, step=0):
        direction, info = original_combine(grads, method, step)
        if cfg['bearing_floor']:
            index = cfg['domains'].index('bearing')
            direction, floor_info = bearing_floor(direction, grads[index], len(grads))
            info['bearing_floor'] = floor_info
        norms = torch.stack([g.norm() for g in grads])
        dots = torch.stack([g @ direction for g in grads])
        info['final_direction_cosine'] = (dots/(norms*direction.norm()).clamp_min(1e-20)).tolist()
        info['relative_domain_progress'] = (dots/norms.square().clamp_min(1e-20)).tolist()
        info['components'] = list(components); components.clear()
        epoch_diagnostics.append(dict(step=step, **info))
        return direction, info

    def trainer(run_name):
        t = original_trainer(run_name)
        validate, save = t.validate, t.save
        def recorded_validation(model, pools):
            result = validate(model, pools)
            if epoch_diagnostics:
                epoch = epoch_diagnostics[-1]['step']//20 + 1
                t.write(t.OUT/'gradient_audit'/f'epoch_{epoch:04d}.json', list(epoch_diagnostics))
                epoch_diagnostics.clear()
            return result
        def recorded_save(path, payload):
            save(path, payload)
            if path.name == 'last.pt' and payload['epoch'] in (140, 250, 500, 1000, 1500, 2000) and not smoke:
                snapshot = {k.removeprefix('encoder.'):v.detach().cpu() for k,v in payload['model'].items() if k.startswith('encoder.')}
                save(t.OUT/f'diagnostic_encoder_epoch{payload["epoch"]}.pt', snapshot)
        t.validate = recorded_validation; t.save = recorded_save
        t.write(t.BASE/'implementation.json', dict(settings=cfg, smoke=smoke,
            trainer_sha256=hashlib.sha256(source.encode()).hexdigest(),
            wrapper_sha256=hashlib.sha256((HERE/'train.py').read_bytes()).hexdigest(),
            precision='FP32 smoke only' if smoke else 'BF16 autocast, FP32 gradients; identical to CONDITIONAL_ROUTING'))
        return t

    base.trainer = trainer; base.component_gradients = routed; base.combine = aggregate
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if not smoke and not torch.cuda.is_bf16_supported(): raise RuntimeError('Production must run on BF16-capable gpua100')
    base.main()


if __name__ == '__main__': main()
