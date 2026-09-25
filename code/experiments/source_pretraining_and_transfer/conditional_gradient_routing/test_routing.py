"""Verify routing, single-domain degeneration, and control matching on CPU."""
import importlib.util
import torch
from config import ARMS, COMPONENT_ROUTING, DOMAINS, settings, control_for
from train import route


def main():
    r = torch.tensor([2., -3.]); d = torch.tensor([4., 5.])
    private = {0: torch.tensor([9.])}; values = [1., 2.]
    rr, dd, pp, vv = route(r, d, private, values, .3)
    assert torch.allclose(rr, r * .3) and dd is d and pp is private and vv is values
    assert torch.equal(route(r, d, private, values, 1.)[0], r)
    spec = importlib.util.spec_from_file_location('gm', COMPONENT_ROUTING.parent / 'gradient_aggregation_comparison/gradient_methods.py')
    gm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
    gm.self_test()
    for arm in ARMS:
        cfg = settings(arm)
        for domain in cfg['domains']:
            assert cfg['reconstruction_routing']['milling'] == 1.
            if arm.startswith('joint_'):
                root, control = control_for(arm, domain)
                if root != COMPONENT_ROUTING:
                    assert settings(control)['reconstruction_routing'][domain] == cfg['reconstruction_routing'][domain]
                else:
                    assert cfg['reconstruction_routing'][domain] == 1.
    for method in ('mean', 'pcgrad', 'cagrad'):
        out, _ = gm.combine([r + d], method)
        assert torch.equal(out, r + d)
    print('ROUTING_TEST_OK', flush=True)


if __name__ == '__main__': main()
