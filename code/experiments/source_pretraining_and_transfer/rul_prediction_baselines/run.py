"""Run one model/domain/fraction/seed fair-baseline job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import time
from pathlib import Path

# CuBLAS reads this at process startup.  It must be set before importing
# torch; setting it after torch.use_deterministic_algorithms() is too late.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from models import InputShape, build_model


HERE = Path(__file__).resolve().parent
_canonical = HERE / "prepared"
_default_prepared = HERE / "prepared"
if not _default_prepared.exists() and _canonical.exists():
    _default_prepared = _canonical
PREPARED = Path(os.environ.get("HEALTHTOKEN_BASELINE_PREPARED", _default_prepared))
RESULTS = HERE / "results"
MODELS = (
    "native_xgboost", "native_random_forest", "native_cnn_lstm_attention",
    "native_tcn", "native_patchtst", "native_itransformer",
    "native_moment_frozen", "native_moment_full",
    "controlled_mlp", "controlled_tcn", "controlled_cnn_lstm_attention",
    "controlled_patchtst", "controlled_itransformer",
    "controlled_moment_frozen", "controlled_moment_full")
DOMAINS = ("milling", "battery", "bearing")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cuBLAS kernels used by PatchTST are not covered by deterministic
    # implementations on RTX 1000/CUDA 12.8.  Keep deterministic algorithms
    # where available and warn (rather than aborting an entire baseline cell)
    # for those kernels; seed, splits and protocol remain fixed.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Dataset:
    def __init__(self, domain: str, fraction: float):
        root = PREPARED / domain
        if not (root / "manifest.json").exists():
            raise FileNotFoundError("Prepared inputs are absent. Run: python prepare_splits.py")
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        with np.load(root / "snapshots.npz") as values:
            self.local = values["local"]
            self.global_x = values["global_x"]
            self.channel_mask = values["channel_mask"]
            self.token_mask = values["token_mask"]
            self.y = values["y"]
        with np.load(root / f"split_{int(fraction * 100)}.npz") as values:
            self.train = (values["train_sequence"], values["train_history_mask"])
            self.validation = (values["validation_sequence"], values["validation_history_mask"])
            self.test = (values["test_sequence"], values["test_history_mask"])
        self.shape = InputShape(self.train[0].shape[1], self.local.shape[1],
                                self.local.shape[2], self.local.shape[3])
        # Optional exact input cache.  The cache stores the same float32/bool
        # arrays on CUDA once and changes only data-transfer overhead.
        self._device_cache = {}

    def prepare_device(self, device: str) -> None:
        if not str(device).startswith("cuda"):
            return
        if device in self._device_cache:
            return
        self._device_cache[device] = (
            torch.as_tensor(self.local, device=device),
            torch.as_tensor(self.global_x, device=device),
            torch.as_tensor(self.channel_mask, device=device),
            torch.as_tensor(self.token_mask, device=device),
        )

    def batch(self, sample_set, indices, device):
        sequence, history_mask = sample_set
        sequence = sequence[indices]
        cached = self._device_cache.get(device)
        if cached is not None:
            local, global_x, channel_mask, token_mask = cached
            sequence_device = torch.as_tensor(sequence, device=device)
            return (
                local[sequence_device], global_x[sequence_device],
                channel_mask[sequence_device], token_mask[sequence_device],
                torch.as_tensor(history_mask[indices], device=device),
            )
        return (
            torch.as_tensor(self.local[sequence], device=device),
            torch.as_tensor(self.global_x[sequence], device=device),
            torch.as_tensor(self.channel_mask[sequence], device=device),
            torch.as_tensor(self.token_mask[sequence], device=device),
            torch.as_tensor(history_mask[indices], device=device),
        )

    def target(self, sample_set, indices):
        sequence = sample_set[0][indices]
        return self.y[sequence[:, -1]]

    def flatten(self, sample_set):
        sequence, history = sample_set
        local = self.local[sequence].copy()
        global_x = self.global_x[sequence].copy()
        channel = self.channel_mask[sequence]
        token = self.token_mask[sequence]
        valid_history = history[:, :, None, None, None]
        local *= valid_history
        global_x *= history[:, :, None, None]
        return np.concatenate([
            local.reshape(len(sequence), -1), global_x.reshape(len(sequence), -1),
            channel.reshape(len(sequence), -1).astype(np.float32),
            token.reshape(len(sequence), -1).astype(np.float32),
            history.astype(np.float32),
        ], axis=1)


def metrics(prediction, target):
    prediction = np.asarray(prediction, np.float64)
    target = np.asarray(target, np.float64)
    error = prediction - target
    denominator = np.sum((target - target.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / denominator),
        "bias": float(error.mean()),
        "n": int(len(target)),
    }


@torch.no_grad()
def predict(model, data: Dataset, sample_set, device, batch_size=32):
    model.eval()
    values = []
    for start in range(0, len(sample_set[0]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(sample_set[0])))
        values.append(model(*data.batch(sample_set, indices, device)).float().cpu().numpy())
    return np.concatenate(values)


def state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def run_svr(data: Dataset, config: dict, destination: Path):
    """RBF support-vector regression on the complete flattened input."""
    try:
        from sklearn.svm import SVR
    except ImportError as error:
        raise RuntimeError("Install requirements-baselines.txt before running SVR") from error
    train_x, validation_x, test_x = (data.flatten(value) for value in
                                     (data.train, data.validation, data.test))
    train_y = data.target(data.train, np.arange(len(data.train[0])))
    validation_y = data.target(data.validation, np.arange(len(data.validation[0])))
    test_y = data.target(data.test, np.arange(len(data.test[0])))
    model = SVR(
        kernel=config["kernel"], C=config["C"], epsilon=config["epsilon"],
        gamma=config["gamma"], cache_size=config["cache_size_mb"])
    began = time.time()
    model.fit(train_x, train_y)
    validation_prediction = np.clip(model.predict(validation_x), 0, 1)
    prediction = np.clip(model.predict(test_x), 0, 1)
    with (destination / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream)
    np.savez(destination / "predictions.npz", pred=prediction, y=test_y)
    return {
        "metrics": metrics(prediction, test_y), "seconds": time.time() - began,
        "validation_rmse": metrics(validation_prediction, validation_y)["rmse"],
        "feature_count": int(train_x.shape[1]),
        "support_vector_count": int(len(model.support_)),
    }


def run_tree(name: str, data: Dataset, config: dict, destination: Path):
    if name == "native_xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=config["n_estimators"], max_depth=config["max_depth"],
            learning_rate=config["learning_rate"], subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"], n_jobs=config["n_jobs"],
            random_state=config["seed"], tree_method="hist")
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"], max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"], n_jobs=config["n_jobs"],
            random_state=config["seed"])
    train_x, val_x, test_x = (data.flatten(split) for split in
                              (data.train, data.validation, data.test))
    train_y = data.target(data.train, np.arange(len(data.train[0])))
    val_y = data.target(data.validation, np.arange(len(data.validation[0])))
    test_y = data.target(data.test, np.arange(len(data.test[0])))
    began = time.time()
    model.fit(train_x, train_y)
    val_pred = np.clip(model.predict(val_x), 0, 1)
    prediction = np.clip(model.predict(test_x), 0, 1)
    with (destination / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream)
    np.savez(destination / "predictions.npz", pred=prediction, y=test_y)
    return {"metrics": metrics(prediction, test_y), "seconds": time.time() - began,
            "validation_rmse": metrics(val_pred, val_y)["rmse"],
            "feature_count": int(train_x.shape[1])}


def run_neural(name: str, data: Dataset, config: dict, protocol: dict, seed: int,
               destination: Path, device: str):
    seed_all(seed)
    if name.startswith("native_"):
        from native_models import build_native
        model = build_native(name, data.shape, config).to(device)
    else:
        model = build_model(name, data.shape, config).to(device)
    if os.environ.get("HEALTHTOKEN_GPU_CACHE", "1") == "1":
        data.prepare_device(device)
    # Materialize MOMENT's lazy RUL head before optimizer construction.
    with torch.no_grad():
        model(*data.batch(data.train, np.array([0]), device))
    if "moment" in name:
        foundation = [p for p in model.foundation.parameters() if p.requires_grad]
        other = [p for parameter_name, p in model.named_parameters()
                 if not parameter_name.startswith("foundation.") and p.requires_grad]
        groups = [{"params": other, "lr": config["interface_learning_rate"]}]
        if foundation:
            groups.append({"params": foundation, "lr": config["backbone_learning_rate"]})
        microbatch = config["microbatch"]
    else:
        groups = [{"params": model.parameters(), "lr": config["learning_rate"]}]
        microbatch = config["microbatch"]
    optimizer = torch.optim.AdamW(groups, weight_decay=protocol["neural_training"]["weight_decay"])
    beta = protocol["domains"][data.manifest["domain"]]["loss_beta"]
    effective = protocol["neural_training"]["effective_batch"]
    best = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    updates = 0
    began = time.time()
    for epoch in range(1, protocol["neural_training"]["max_epochs"] + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(data.train[0]))
        losses = []
        for start in range(0, len(order), effective):
            selected = order[start:start + effective]
            optimizer.zero_grad(set_to_none=True)
            for begin in range(0, len(selected), microbatch):
                indices = selected[begin:begin + microbatch]
                target = torch.as_tensor(data.target(data.train, indices), device=device)
                # MOMENT full fine-tuning can overflow in BF16 on the long
                # 512-token path.  Keep the established BF16 path for the
                # other neural baselines, but run this baseline in FP32 so a
                # finite loss is not dependent on accelerator rounding.
                amp_enabled = str(device).startswith("cuda") and not name.endswith("moment_full")
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=amp_enabled):
                    prediction = model(*data.batch(data.train, indices, device))
                    loss = F.smooth_l1_loss(prediction.float(), target, beta=beta)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                (loss * len(indices) / len(selected)).backward()
                losses.append(float(loss.detach().cpu()))
            if name.endswith("moment_full"):
                # MOMENT's checkpointed foundation occasionally emits NaN/Inf
                # gradients for padded battery windows.  Sanitise only this
                # baseline's gradients; the finite interface/head gradients
                # still train and the protocol records the intervention.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad = torch.nan_to_num(parameter.grad, nan=0.0,
                                                          posinf=0.0, neginf=0.0)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  protocol["neural_training"]["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            updates += 1
        record = {"epoch": epoch, "updates": updates, "loss": float(np.mean(losses))}
        if epoch % protocol["neural_training"]["validation_interval"] == 0:
            validation_prediction = predict(model, data, data.validation, device,
                                            batch_size=max(1, microbatch))
            validation_y = data.target(data.validation, np.arange(len(data.validation[0])))
            validation = metrics(validation_prediction, validation_y)
            record["validation"] = validation
            if validation["rmse"] < best:
                best = validation["rmse"]
                best_epoch = epoch
                best_state = state(model)
            if epoch - best_epoch >= protocol["neural_training"]["patience_epochs"]:
                history.append(record)
                break
        history.append(record)
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    prediction = predict(model, data, data.test, device, batch_size=max(1, microbatch))
    target = data.target(data.test, np.arange(len(data.test[0])))
    if "moment" not in name:
        torch.save(best_state, destination / "model.pt")
        np.savez(destination / "predictions.npz", pred=prediction, y=target)
    write_json(destination / "history.json", history)
    return {
        "metrics": metrics(prediction, target), "seconds": time.time() - began,
        "best_epoch": best_epoch, "stopped_epoch": epoch, "updates": updates,
        "validation_rmse": best,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--fraction", required=True, type=float, choices=(0.1, 0.2, 1.0))
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44, 45, 46))
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    # Slurm already restricts visible devices to the allocated GPU.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    configs = json.loads((HERE / "model_configs.json").read_text(encoding="utf-8"))
    if args.model.startswith("native_"):
        from native_data import NativeDataset
        native_view = "moment_raw" if "moment" in args.model else "source_scaled"
        data = NativeDataset(args.domain, args.fraction, view=native_view)
    else:
        data = Dataset(args.domain, args.fraction)
    destination = RESULTS / args.model / args.domain / f"fraction{int(args.fraction * 100)}" / f"seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "metrics.json").exists():
        return
    write_json(destination / "status.json", {"state": "running", "pid": os.getpid()})
    try:
        if args.model in ("native_xgboost", "native_random_forest"):
            tree_config = dict(configs[args.model], seed=args.seed)
            result = run_tree(args.model, data, tree_config, destination)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA logical device {args.gpu} is unavailable")
            result = run_neural(args.model, data, configs[args.model], protocol, args.seed,
                                destination, "cuda")
        result.update({"model": args.model, "domain": args.domain,
                       "fraction": args.fraction, "seed": args.seed,
                       "input_family": "native_raw" if args.model.startswith("native_")
                       else "controlled_healthtoken",
                       "split_sha256": data.manifest["splits"][str(args.fraction)]["sha256"],
                       "snapshot_sha256": data.manifest["snapshot_sha256"]})
        if args.model.startswith("native_"):
            result.update({"raw_sha256": data.raw_sha256,
                           "raw_signal_name": data.raw_signal_name,
                           "native_view": data.view,
                           "normalization": data.normalization})
        write_json(destination / "metrics.json", result)
        write_json(destination / "status.json", {"state": "complete"})
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write_json(destination / "status.json", {"state": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
