"""Reuse the fixed E34 trainer; only shared reconstruction routing changes."""
import importlib.util
import sys
import torch
from config import E34, settings


def route(rec, dyn, private, values, rho):
    # Private gradients and the dynamics gradient are preserved exactly.
    return rho * rec, dyn, private, values


def main():
    spec = importlib.util.spec_from_file_location('e35_base_train', E34 / 'train.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    arm = sys.argv[sys.argv.index('--arm') + 1]
    cfg = settings(arm)
    original = base.component_gradients
    original_combine = base.combine
    component_info = []

    def routed(t, model, pool, ids, sequence, scales, named, shared_index):
        rec, dyn, private, values = original(t, model, pool, ids, sequence, scales, named, shared_index)
        rho = cfg['reconstruction_routing'][pool.domain]
        component_info.append(dict(domain=pool.domain, dataset=pool.name, rho=rho,
            reconstruction_norm=float(rec.norm()), dynamics_norm=float(dyn.norm()),
            rec_dyn_cosine=float((rec @ dyn) / (rec.norm() * dyn.norm()).clamp_min(1e-20))))
        return route(rec, dyn, private, values, rho)

    def aggregate(grads, method, step=0):
        direction, info = original_combine(grads, method, step)
        info['components'] = list(component_info)
        component_info.clear()
        return direction, info

    base.component_gradients = routed
    base.combine = aggregate
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    base.main()


if __name__ == '__main__':
    main()
