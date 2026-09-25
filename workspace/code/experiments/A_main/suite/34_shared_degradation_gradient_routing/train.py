"""Separate observation reconstruction from transferable degradation dynamics.

Both losses are independently normalized at initialization. Observation
reconstruction keeps full gradients in private interfaces/heads, while only a
prespecified fraction enters the shared backbone. Dynamics keeps full gradients.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
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

sys.path.insert(0, str(SUITE / "29_joint_optimization_comparison"))
from gradient_methods import combine, self_test

spec = importlib.util.spec_from_file_location("e29_train", SUITE / "29_joint_optimization_comparison/train.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def trainer(arm):
    previous.HERE = HERE
    return previous.load_trainer(arm)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def calibrate_components(t, model, pools, active):
    model.eval(); scales = {}
    for domain in active:
        scales[domain] = {}
        for index, pool in enumerate(pools[domain]):
            ids, sequence = previous.sample(pool, np.random.default_rng(88000 + 100 * DOMAINS.index(domain) + index), 64)
            values = []
            for start in range(0, 64, 8):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lm, ld = t.losses(model, pool, ids[start:start+8], sequence[start:start+8], fixed=98000 + start)
                values.append((lm.item(), ld.item()))
            mean = np.mean(values, axis=0)
            scales[domain][pool.name] = dict(reconstruction=max(float(mean[0]), 1e-8),
                                                    dynamics=max(float(mean[1]), 1e-8))
    _, detail = t.validate(model, pools)
    baseline = {d: {name: max(value["total"], 1e-8) for name, value in info["datasets"].items()}
                for d, info in detail.items()}
    return scales, baseline


def select_batch(pools, domain, epoch, update):
    pool = pools[domain][((epoch - 1) * 20 + update) % len(pools[domain])]
    rng = np.random.default_rng(42 + epoch * 10000 + update * 10 + DOMAINS.index(domain))
    ids, sequence = previous.sample(pool, rng)
    return pool, ids, sequence


def component_gradients(t, model, pool, ids, sequence, scales, named, shared_index):
    rec_shared = None; dyn_shared = None; private = {}; values = np.zeros(2)
    parameters = [p for _, p in named]
    for start in range(0, 32, 8):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lm, ld = t.losses(model, pool, ids[start:start+8], sequence[start:start+8])
            rec = .5 * lm / scales["reconstruction"] / 4
            dyn = .5 * ld / scales["dynamics"] / 4
        if not torch.isfinite(rec + dyn):
            raise FloatingPointError((pool.domain, pool.name, lm.item(), ld.item()))
        rg = torch.autograd.grad(rec, parameters, retain_graph=True, allow_unused=True)
        dg = torch.autograd.grad(dyn, parameters, allow_unused=True)
        rs = torch.cat([(rg[i] if rg[i] is not None else torch.zeros_like(parameters[i])).flatten()
                        for i in shared_index])
        ds = torch.cat([(dg[i] if dg[i] is not None else torch.zeros_like(parameters[i])).flatten()
                        for i in shared_index])
        rec_shared = rs if rec_shared is None else rec_shared + rs
        dyn_shared = ds if dyn_shared is None else dyn_shared + ds
        for i, parameter in enumerate(parameters):
            if i in shared_index:
                continue
            total = (rg[i] if rg[i] is not None else torch.zeros_like(parameter)) + \
                    (dg[i] if dg[i] is not None else torch.zeros_like(parameter))
            if rg[i] is not None or dg[i] is not None:
                private[i] = private.get(i, torch.zeros_like(parameter)) + total
        values += np.array([lm.item(), ld.item()]) / 4
    return rec_shared, dyn_shared, private, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); cfg = settings(args.arm); active = cfg["domains"]
    torch.set_num_threads(8); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    seed_all(42); self_test()
    run_name = "smoke_" + args.arm if args.smoke else args.arm
    t = trainer(run_name)
    status = t.OUT / "status.json"
    if status.exists() and json.loads(status.read_text()).get("state") == "complete":
        print("ALREADY_COMPLETE", run_name); return
    t.write(status, dict(state="preparing", arm=args.arm, pid=os.getpid()))
    # The inherited preparation routine always materializes all three immutable
    # source pools. Restricting DOMAINS before this call makes its pool dictionary
    # omit keys that the routine still fills (and caused the failed single runs).
    pools = t.prepare()
    t.DOMAINS = active
    model = t.JointModel().cuda(); named = list(model.named_parameters())
    shared_index = tuple(i for i, (name, _) in enumerate(named) if previous.shared_parameter(name))
    shared_params = [named[i][1] for i in shared_index]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scales, baseline = calibrate_components(t, model, pools, active)
    protocol = dict(arm=args.arm, seed=42, settings=cfg, sources={d: [p.name for p in pools[d]] for d in active},
        max_epochs=500, patience=30, min_delta=1e-4, updates_per_epoch=20, batch_per_domain=32,
        microbatch=8, lr=1e-4, weight_decay=1e-4, clip=5, precision="BF16 autocast; FP32 gradients",
        component_scales=scales, validation_baseline=baseline,
        objective="0.5*reconstruction/initial_reconstruction + 0.5*dynamics/initial_dynamics",
        routing="private observation parameters receive 100% reconstruction; shared parameters receive rho",
        selection="macro datasets then domains of legacy val_total / initial_val_total",
        private_definition="encoder.stems, encoder.adapters, channels, decoders",
        device=torch.cuda.get_device_name(), torch_version=torch.__version__)
    immutable = dict(protocol); immutable.pop("device"); immutable.pop("torch_version")
    fingerprint = hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()
    checkpoint = t.OUT / "last.pt"; best = float("inf"); best_epoch = 0; stale = 0; start = 1; history = []
    if checkpoint.exists():
        if not args.resume: raise RuntimeError("Existing run; use --resume")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck["fingerprint"] != fingerprint: raise RuntimeError("Protocol drift")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        best, best_epoch, stale, start, history = ck["best"], ck["best_epoch"], ck["stale"], ck["epoch"] + 1, ck["history"]
        random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"])
    t.write(t.BASE / "protocol.json", dict(protocol, fingerprint=fingerprint))
    requested_stop = [False]
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: requested_stop.__setitem__(0, True))
    began = time.monotonic(); final_epoch = start - 1
    try:
        for epoch in range(start, (1 if args.smoke else 500) + 1):
            if stale >= 30: break
            model.train(); totals = {d: [] for d in active}; diagnostics = []
            for update in range(1 if args.smoke else 20):
                domain_shared = []; private_sum = {}; selected = []
                for domain in active:
                    pool, ids, sequence = select_batch(pools, domain, epoch, update)
                    rec, dyn, private, values = component_gradients(
                        t, model, pool, ids, sequence, scales[domain][pool.name], named, shared_index)
                    routed = cfg["shared_reconstruction_fraction"] * rec + dyn
                    domain_shared.append(routed); selected.append(pool.name); totals[domain].append(values.tolist())
                    for index, gradient in private.items():
                        private_sum[index] = private_sum.get(index, torch.zeros_like(gradient)) + gradient / len(active)
                direction, info = combine(domain_shared, cfg["aggregation"], (epoch - 1) * 20 + update)
                info.update(datasets=selected, shared_reconstruction_fraction=cfg["shared_reconstruction_fraction"])
                if update == 0: diagnostics.append(info)
                optimizer.zero_grad(set_to_none=True); offset = 0
                for parameter in shared_params:
                    parameter.grad = direction[offset:offset + parameter.numel()].view_as(parameter).clone()
                    offset += parameter.numel()
                for index, gradient in private_sum.items(): named[index][1].grad = gradient
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if not torch.isfinite(norm): raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                t.write(status, dict(state="training", arm=args.arm, epoch=epoch,
                                     update=(epoch - 1) * 20 + update + 1, pid=os.getpid()))
            raw, detail = t.validate(model, pools); score = previous.normalized_score(detail, baseline)
            if not np.isfinite(score): raise FloatingPointError("nonfinite validation")
            improved = score < best - 1e-4
            if improved: best, best_epoch, stale = score, epoch, 0
            else: stale += 1
            history.append(dict(epoch=epoch, monitor=score, raw=raw, validation=detail,
                train={d: np.mean(v, axis=0).tolist() for d, v in totals.items()},
                gradients=diagnostics, session_seconds=time.monotonic() - began))
            ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, best=best,
                best_epoch=best_epoch, stale=stale, history=history, fingerprint=fingerprint,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all())
            t.save(checkpoint, ck)
            if improved: t.save(t.OUT / "encoder.pt", {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()})
            t.write(t.OUT / "history.json", history); final_epoch = epoch
            print("EPOCH", args.arm, epoch, "VAL", score, "BEST", best_epoch, "STALE", stale, flush=True)
            if requested_stop[0] and stale < 30 and epoch < 500:
                t.write(status, dict(state="interrupted", arm=args.arm, epoch=epoch, best_epoch=best_epoch,
                                     reason="Slurm time signal; checkpoint saved"))
                raise SystemExit(75)
        t.write(status, dict(state="complete", arm=args.arm, epoch=final_epoch, best_epoch=best_epoch,
            best_validation=best, fingerprint=fingerprint,
            reason="smoke" if args.smoke else "early_stopping" if stale >= 30 else "max_epochs"))
    except Exception as error:
        t.write(status, dict(state="failed", arm=args.arm, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    main()
