"""Isolated, controlled multi-domain optimization study (no architecture changes)."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from gradient_methods import combine, self_test

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
DOMAINS = ("bearing", "battery", "milling")
ARMS = ("joint_raw", "joint_norm", "joint_pcgrad", "joint_cagrad") + tuple(
    f"single_{d}_{mode}" for d in DOMAINS for mode in ("raw", "norm"))


def load_trainer(arm):
    source = SUITE / "multidomain_original_cutting_tool_sources/three_domain/code"
    sys.path.insert(0, str(source))
    __import__("data")
    spec = importlib.util.spec_from_file_location("global_local_data", SUITE / "gradnorm_downstream_evaluation/bearing_code/global_local_data.py")
    gl = importlib.util.module_from_spec(spec)
    sys.modules["global_local_data"] = gl
    spec.loader.exec_module(gl)
    spec = importlib.util.spec_from_file_location("original_joint", source / "train_joint.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    t.BASE = HERE / "models" / arm
    t.OUT = t.BASE / "training"
    t.OUT.mkdir(parents=True, exist_ok=True)
    t.SHARED = SUITE / "shared_domain_components"
    t.MILLING_RUNTIME = SUITE / "full_channel_source_models/runtime/milling"
    t.MILLING = ("luh_milling", "matwi_milling", "nonastreda_milling", "qit_cemc_milling", "hmotp_milling")
    audits = {}
    for domain, path in (("bearing", t.SHARED / "bearing_runtime/attention65_upstream_seed42/upstream_splits_scalers.json"),
                         ("milling", t.MILLING_RUNTIME / "upstream_splits_scalers.json")):
        for record in json.loads(path.read_text()):
            audits[(domain, record["dataset"])] = (record["train"], record.get("val", record.get("validation")))
    original = gl.GlobalLocalStore.split
    def split(store):
        name = store.name[:-8] if store.domain == "bearing" and store.name.endswith("_bearing") else store.name
        return audits.get((store.domain, name), None) or original(store)
    gl.GlobalLocalStore.split = split
    return t


def sample(pool, rng, n=32):
    groups = list(pool.groups.values())
    ids = np.array([rng.choice(groups[rng.integers(len(groups))]) for _ in range(n)])
    return ids, pool.tw[rng.integers(len(pool.tw), size=n)]


@torch.no_grad()
def calibrate(t, model, pools, active):
    model.eval()
    scales = {}
    for d in active:
        scales[d] = {}
        for j, p in enumerate(pools[d]):
            ids, seq = sample(p, np.random.default_rng(88000 + 100 * DOMAINS.index(d) + j), 64)
            values = []
            for i in range(0, 64, 8):
                lm, ld = t.losses(model, p, ids[i:i+8], seq[i:i+8], fixed=98000 + i)
                values.append((lm + .2 * ld).item())
            scales[d][p.name] = max(float(np.mean(values)), 1e-8)
    _, val = t.validate(model, pools)
    baseline = {d: {name: max(v["total"], 1e-8) for name, v in val[d]["datasets"].items()} for d in active}
    return scales, baseline


def normalized_score(detail, baseline):
    return float(np.mean([np.mean([v["total"] / baseline[d][name]
                                  for name, v in info["datasets"].items()])
                          for d, info in detail.items()]))


def shared_parameter(name):
    return not name.startswith(("encoder.stems.", "encoder.adapters.", "channels.", "decoders."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    active = DOMAINS if args.arm.startswith("joint_") else (args.arm.split("_")[1],)
    normalize = args.arm != "joint_raw" and not args.arm.endswith("_raw")
    method = args.arm.removeprefix("joint_") if args.arm in ("joint_pcgrad", "joint_cagrad") else "mean"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    self_test()
    t = load_trainer("smoke" if args.smoke else args.arm)
    status_path = t.OUT / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("state") == "complete":
        print("ALREADY_COMPLETE", args.arm, flush=True)
        return
    t.write(status_path, {"state": "preparing", "arm": args.arm})
    pools = t.prepare()
    t.DOMAINS = active
    model = t.JointModel().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    named = list(model.named_parameters())
    shared = [p for name, p in named if shared_parameter(name)]
    private = [p for name, p in named if not shared_parameter(name)]
    scales, baseline = calibrate(t, model, pools, active)
    t.write(t.BASE / "protocol.json", dict(arm=args.arm, seed=42, domains=active, normalize=normalize,
        gradient_method=method, cagrad_alpha=.4, max_epochs=500, patience=30, min_delta=1e-4,
        updates_per_epoch=20, batch_per_domain=32, microbatch=8, lr=1e-4, weight_decay=1e-4,
        gradient_clip=5, architecture="unchanged experiment15 joint encoder and adapters",
        sources={d: [p.name for p in pools[d]] for d in active},
        train_scales=scales, validation_baseline=baseline,
        validation_excluded={d: [p.name for p in pools[d] if not len(p.vr) or not len(p.vw)] for d in active},
        shared_parameter_names=[n for n, p in named if shared_parameter(n)],
        selection="raw macro loss" if not normalize else "macro dataset val loss / initial dataset val loss",
        normalization="fixed TRAIN-only initial combined SSL loss per dataset; validation never used as training weight"))
    best, selection_best, stale, best_epoch, selection_epoch = float("inf"), float("inf"), 0, 0, 0
    history, start = [], 1
    if (t.OUT / "last.pt").exists():
        if not args.resume:
            raise RuntimeError("Existing checkpoint; explicitly use --resume")
        ck = torch.load(t.OUT / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        best, selection_best, stale = ck["best"], ck["selection_best"], ck["stale"]
        best_epoch, selection_epoch = ck["best_epoch"], ck["selection_epoch"]
        history, start = ck["history"], ck["epoch"] + 1
        scales, baseline = ck["scales"], ck["baseline"]
        random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"])
    began = time.time()
    final_epoch = start - 1
    for epoch in range(start, (1 if args.smoke else 500) + 1):
        if stale >= 30:
            break
        model.train()
        totals, diagnostics = {d: [] for d in active}, []
        for update in range(1 if args.smoke else 20):
            shared_grads, private_grads, shared_used = [], {}, set()
            for d in active:
                optimizer.zero_grad(set_to_none=True)
                p = pools[d][((epoch - 1) * 20 + update) % len(pools[d])]
                seed = 42 + epoch * 10000 + update * 10 + DOMAINS.index(d)
                ids, seq = sample(p, np.random.default_rng(seed))
                torch.manual_seed(seed)
                values = np.zeros(2)
                for i in range(0, 32, 8):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lm, ld = t.losses(model, p, ids[i:i+8], seq[i:i+8])
                        loss = (lm + .2 * ld) / 4 / (scales[d][p.name] if normalize else 1.)
                    if not torch.isfinite(loss):
                        raise FloatingPointError((d, p.name))
                    loss.backward()
                    values += np.array([lm.item(), ld.item()]) / 4
                shared_used.update(j for j, parameter in enumerate(shared) if parameter.grad is not None)
                shared_grads.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in shared]))
                for j, parameter in enumerate(private):
                    if parameter.grad is not None:
                        if j not in private_grads:
                            private_grads[j] = parameter.grad.detach().clone() / len(active)
                        else:
                            private_grads[j].add_(parameter.grad / len(active))
                totals[d].append(values.tolist())
            direction, diagnostic = combine(shared_grads, method, (epoch - 1) * 20 + update)
            if args.smoke:
                for check_method in ("pcgrad", "cagrad"):
                    check, _ = combine(shared_grads, check_method)
                    assert torch.isfinite(check).all()
            diagnostics.append(diagnostic)
            optimizer.zero_grad(set_to_none=True)
            offset = 0
            for j, parameter in enumerate(shared):
                if j in shared_used:
                    parameter.grad = direction[offset:offset + parameter.numel()].view_as(parameter).clone()
                offset += parameter.numel()
            for j, gradient in private_grads.items():
                private[j].grad = gradient
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step()
            t.write(status_path, dict(state="training", arm=args.arm, epoch=epoch, update=(epoch-1)*20+update+1))
            print("UPDATE", args.arm, epoch, update + 1, flush=True)
        raw, detail = t.validate(model, pools)
        normalized = normalized_score(detail, baseline)
        score = normalized if normalize else raw
        if not np.isfinite(score):
            raise FloatingPointError("nonfinite validation")
        improved = score < best - 1e-4
        if improved:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        selected = normalized < selection_best - 1e-4
        if selected:
            selection_best, selection_epoch = normalized, epoch
        history.append(dict(epoch=epoch, validation=detail, monitor=score, raw=raw, normalized=normalized,
            train={d: np.mean(v, axis=0).tolist() for d, v in totals.items()}, gradients=diagnostics,
            session_seconds=time.time()-began))
        ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, best=best,
                  best_epoch=best_epoch, selection_best=selection_best, selection_epoch=selection_epoch,
                  stale=stale, history=history, scales=scales, baseline=baseline,
                  python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                  torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        t.save(t.OUT / "last.pt", ck)
        encoder = {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
        if improved:
            t.save(t.OUT / "encoder.pt", encoder)
        if args.arm == "joint_raw" and selected:
            selection_dir = HERE / "models/joint_selection/training"
            selection_dir.mkdir(parents=True, exist_ok=True)
            t.save(selection_dir / "encoder.pt", encoder)
        t.write(t.OUT / "history.json", history)
        final_epoch = epoch
        print("EPOCH", args.arm, epoch, "VAL", score, "BEST", best_epoch, "STALE", stale, flush=True)
    state = dict(state="complete", arm=args.arm, epoch=final_epoch, best_epoch=best_epoch, best_validation=best,
                 reason="smoke" if args.smoke else "early_stopping" if stale >= 30 else "max_epochs")
    t.write(status_path, state)
    if args.arm == "joint_raw" and not args.smoke:
        t.write(HERE / "models/joint_selection/training/status.json", dict(state, arm="joint_selection",
                best_epoch=selection_epoch, best_validation=selection_best, trajectory="joint_raw"))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    main()
