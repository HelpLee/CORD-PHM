"""CPU tests: FAMO math, cloned initialization, routing and encoder compatibility."""
import copy
import importlib.util
import sys
from pathlib import Path
import torch
from methods import LossRateBalancer, build_model, gradient_info
from config import SUITE, ARMS, settings


def main():
    torch.set_num_threads(2)
    # Test the accumulated-gradient form against differentiating the FAMO loss.
    x = torch.tensor([.8, -.4], requires_grad=True)
    losses = torch.stack(((x[0] - 1).square() + .5, (x[1] + 2).square() + .2))
    controller = LossRateBalancer(2, 'cpu')
    weights = controller.coefficients(losses)
    reference_weights = controller.logits.softmax(0)
    scalar = (reference_weights * (losses + 1e-8).log()).sum() / (reference_weights / (losses + 1e-8)).sum().detach()
    expected = torch.autograd.grad(scalar, x, retain_graph=True)[0]
    actual = sum(weights[i] * torch.autograd.grad(losses[i], x, retain_graph=True)[0] for i in range(2))
    assert torch.allclose(actual, expected, atol=1e-7)
    before = controller.logits.detach().clone()
    controller.update(torch.tensor([1., 1.]), torch.tensor([.5, .99]))
    assert controller.logits[1] > controller.logits[0], (before, controller.logits)
    restored = LossRateBalancer(2, 'cpu')
    restored.load_state_dict(copy.deepcopy(controller.state_dict()))
    controller.update(torch.tensor([1., 1.]), torch.tensor([.8, .9]))
    restored.update(torch.tensor([1., 1.]), torch.tensor([.8, .9]))
    assert torch.equal(controller.logits, restored.logits)
    assert torch.equal(LossRateBalancer(1, 'cpu').coefficients(torch.tensor([.2])), torch.ones(1))

    source = SUITE / '15_three_domain_original4_milling_all_channels/three_domain/code'
    sys.path.insert(0, str(source))
    # The architecture itself needs no raw preprocessing or runtime caches.
    from joint_model import JointModel
    sources = {'bearing': ['a', 'b'], 'battery': ['a', 'b'], 'milling': ['a', 'b']}
    torch.manual_seed(42)
    reference = JointModel().eval()
    for private, dataset in ((False, False), (True, False), (False, True), (True, True)):
        torch.manual_seed(42)
        candidate = build_model(JointModel, sources, private, dataset).eval()
        assert set(candidate.encoder.state_dict()) == set(reference.encoder.state_dict())
        assert all(torch.equal(v, reference.encoder.state_dict()[k]) for k, v in candidate.encoder.state_dict().items())
        for domain, channels in (('bearing', 2), ('battery', 1), ('milling', 7)):
            x = torch.randn(7, channels, 64, 26)
            g = torch.randn(7, channels, 26)
            cm = torch.ones(7, channels, dtype=torch.bool)
            tm = torch.ones(7, channels, 64, dtype=torch.bool)
            observed = torch.ones_like(x, dtype=torch.bool)
            lm = candidate.reconstruction_source(domain, 'a', x[:1], g[:1], cm[:1], tm[:1], observed[:1], 9)
            expected = reference.reconstruction(domain, x[:1], g[:1], cm[:1], tm[:1], observed[:1], 9)
            assert torch.equal(lm, expected), (private, dataset, domain, lm, expected)
            ld = candidate.dynamics(domain, x, g, cm, tm)
            assert torch.equal(ld, reference.dynamics(domain, x, g, cm, tm))
            candidate.zero_grad(set_to_none=True)
            (lm + .2 * ld).backward()
            assert candidate.encoder.fusion[1].weight.grad is not None
            if private:
                assert candidate.domain_grus[domain].weight_ih_l0.grad is not None
                assert all(m.weight_ih_l0.grad is None for d, m in candidate.domain_grus.items() if d != domain)
            if dataset:
                assert candidate.source_channels[domain + '__a'].weight.grad is not None
                assert candidate.source_channels[domain + '__b'].weight.grad is None
        reference.encoder.load_state_dict(candidate.encoder.state_dict(), strict=True)
    for arm in ARMS:
        assert settings(arm)['domains']
    info = gradient_info(torch.tensor([[1., 0.], [-1., 1.]]), torch.tensor([.5, .5]))
    assert info['cosine'][0][1] < 0
    print('TEST_METHODS_PASSED: FAMO equivalence/resume; all branches; exact initial outputs; gradient routing; unchanged encoder')


if __name__ == '__main__':
    main()
