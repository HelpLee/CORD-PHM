"""Shared-parameter gradient aggregation; private gradients stay task-specific."""
import numpy as np
import torch


def combine(grads, method, step=0, alpha=.4):
    g = torch.stack(grads)
    mean = g.mean(0)
    gram = g @ g.T
    norms = gram.diag().clamp_min(0).sqrt()
    cosine = gram / (norms[:, None] * norms[None, :]).clamp_min(1e-20)
    info = {"norms": norms.tolist(), "cosine": cosine.tolist()}
    if method == "pcgrad" and len(grads) > 1:
        rng = np.random.default_rng(1729 + step)
        projected = g.clone()
        for i in range(len(grads)):
            for j in rng.permutation(len(grads)):
                if j == i:
                    continue
                dot = projected[i] @ g[j]
                if dot < 0:
                    projected[i] -= dot / gram[j, j].clamp_min(1e-20) * g[j]
        result = projected.mean(0)
    elif method == "cagrad" and len(grads) > 1:
        from scipy.optimize import minimize
        # Solve the original CAGrad simplex objective on a 3 x 3 Gram matrix.
        A = gram.detach().double().cpu().numpy()
        A /= max(float(np.max(np.diag(A))), 1e-20)
        b = np.ones(len(grads)) / len(grads)
        c = alpha * np.sqrt(max(b @ A @ b, 0) + 1e-20)
        def objective(w):
            return w @ A @ b + c * np.sqrt(max(w @ A @ w, 0) + 1e-20)
        solution = minimize(objective, b, bounds=[(0., 1.)] * len(b),
                            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                            method="SLSQP", options={"ftol": 1e-10, "maxiter": 100})
        if not solution.success:
            raise RuntimeError(f"CAGrad simplex solver failed: {solution.message}")
        weights = torch.as_tensor(solution.x, dtype=g.dtype, device=g.device)
        weighted = weights @ g
        # Official rescale=1 convention. It is fixed, not selected on test data.
        result = (mean + alpha * mean.norm() / weighted.norm().clamp_min(1e-20) * weighted) / (1 + alpha ** 2)
        info["weights"] = solution.x.tolist()
    else:
        result = mean
    info["direction_cosine"] = ((g @ result) / (norms * result.norm()).clamp_min(1e-20)).tolist()
    return result, info


def self_test():
    gs = [torch.tensor([1., 0.]), torch.tensor([-1., 1.]), torch.tensor([0., 1.])]
    for name in ("mean", "pcgrad", "cagrad"):
        result, _ = combine(gs, name)
        assert torch.isfinite(result).all()
        one, _ = combine(gs[:1], name)
        assert torch.equal(one, gs[0])
    result, _ = combine([gs[0], gs[0]], "pcgrad")
    assert torch.equal(result, gs[0])
    result, _ = combine(gs, "mean")
    assert torch.allclose(result, torch.stack(gs).mean(0))
    result, _ = combine(gs[:2], "pcgrad")
    # Analytic two-task projection: [1,0] -> [.5,.5], [-1,1] -> [0,1].
    assert torch.allclose(result, torch.tensor([.25, .75]))
