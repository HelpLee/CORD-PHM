"""Evaluate the refreshed three-domain encoder on seven-channel PHM2010."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PACKAGE = Path(__file__).resolve().parent
EXPERIMENT = PACKAGE.parents[1]
WORKSPACE = next(parent for parent in PACKAGE.parents if (parent / "code/data_phm").is_dir())
CODE = PACKAGE / "code"
sys.path.insert(0, str(WORKSPACE / "code"))
sys.path.insert(0, str(CODE))

import downstream_interval_val200 as down  # noqa: E402
from adapter_encoder import MillingAdapterEncoder  # noqa: E402
import run_all_channels_scratch as input_condition  # noqa: E402


SEEDS = (42, 43, 44, 45, 46)
FRACTIONS = (.1, .2, 1.)
ARMS = ("frozen_probe", "partial_finetune", "full_finetune")
SOURCE = EXPERIMENT / "three_domain/training/encoder.pt"
SCRATCH_RESULTS = WORKSPACE / "outputs/milling_all_7channel_complete_transfer_matrix/results.json"
RUNTIME = PACKAGE / "runtime"
RESULTS = PACKAGE / "results.json"
STATUS = PACKAGE / "status.json"


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def convert_checkpoint() -> Path:
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    converted = {key: value for key, value in source.items()
                 if not key.startswith(("stems.", "adapters."))}
    for key, value in source.items():
        if key.startswith("adapters.milling."):
            converted[key.replace("adapters.milling.", "adapters.", 1)] = value
    mapping = {
        "local_norm.weight": "stems.milling.local.0.weight",
        "local_norm.bias": "stems.milling.local.0.bias",
        "local_proj.weight": "stems.milling.local.1.weight",
        "local_proj.bias": "stems.milling.local.1.bias",
        "global_norm.weight": "stems.milling.global.0.weight",
        "global_norm.bias": "stems.milling.global.0.bias",
        "global_proj.weight": "stems.milling.global.1.weight",
        "global_proj.bias": "stems.milling.global.1.bias",
    }
    for target, origin in mapping.items():
        converted[target] = source[origin]
    verifier = MillingAdapterEncoder(channels=7, dropout=.05)
    verifier.load_state_dict(converted, strict=True)
    destination = PACKAGE / "checkpoint/three_domain_refreshed_encoder.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, destination)
    write(PACKAGE / "checkpoint/provenance.json", {
        "source": str(SOURCE), "source_sha256": sha256(SOURCE),
        "converted": str(destination), "converted_sha256": sha256(destination),
        "strict_load": True, "mapping": mapping,
        "note": "Channel count is not encoded in stem or channel-attention parameters.",
    })
    return destination


def configure(store, checkpoint: Path) -> None:
    down.prepare_store = lambda _path: store
    down.OUT = RUNTIME
    down.SOURCE = checkpoint
    down.MAX_UPDATES = 0
    down.FROZEN_EPOCHS = 0
    down.MAX_EPOCHS = 200
    down.PATIENCE = 15
    down.VAL_INTERVAL = 2
    down.TRAIN_VALIDATION_FRACTION = .2
    down.TEST_VALIDATION_FRACTION = 0
    down.NESTED_LABELS = False
    down.source_split = input_condition.interval_split


def recover(output: Path) -> None:
    if output.exists() and not (output / "development_summary.json").exists():
        destination = PACKAGE / "interrupted_runs" / f"{output.name}_{datetime.now():%Y%m%d_%H%M%S}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(destination))


def aggregate(rows):
    summary = []
    for initialization, arms in (("scratch", ("scratch",)), ("three_domain_refreshed", ARMS)):
        for arm in arms:
            for fraction in FRACTIONS:
                group = [row for row in rows if row["initialization"] == initialization
                         and row["arm"] == arm and row["fraction"] == fraction]
                if not group:
                    continue
                summary.append({
                    "initialization": initialization, "arm": arm, "fraction": fraction,
                    "n": len(group), "metrics": {
                        metric: {"mean": float(np.mean([row["metrics"][metric] for row in group])),
                                 "std": (float(np.std([row["metrics"][metric] for row in group], ddof=1))
                                         if len(group) > 1 else None)}
                        for metric in ("rmse", "mae", "r2", "bias")},
                })
    return summary


def scratch_rows():
    payload = json.loads(SCRATCH_RESULTS.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload["rows"] if row.get("initialization") == "scratch"]
    if len(rows) != 15:
        raise ValueError(f"Expected 15 completed seven-channel Scratch rows, found {len(rows)}")
    for row in rows:
        row.update(imported=True, comparison_source=str(SCRATCH_RESULTS))
    return rows


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    checkpoint = convert_checkpoint()
    store = input_condition.load_store()
    if store.x.shape[1:] != (7, 64, 26):
        raise ValueError(f"Unexpected downstream tensor: {store.x.shape}")
    configure(store, checkpoint)
    rows = scratch_rows()
    if RESULTS.exists():
        previous = json.loads(RESULTS.read_text(encoding="utf-8"))
        rows.extend(row for row in previous.get("rows", [])
                    if row.get("initialization") == "three_domain_refreshed")
    completed = {(row["fraction"], row["seed"], row["arm"])
                 for row in rows if row["initialization"] == "three_domain_refreshed"}
    write(PACKAGE / "protocol.json", {
        "comparison": "seven-channel Scratch vs refreshed three-domain upstream",
        "source_checkpoint": str(SOURCE), "source_checkpoint_sha256": sha256(SOURCE),
        "train": ["C1", "C4"], "test": "C6", "channels": 7,
        "fractions": list(FRACTIONS), "seeds": list(SEEDS), "arms": list(ARMS),
        "max_epochs": 200, "patience": 15, "validation_interval": 2,
        "validation": "20% uniform interval subset of independently selected labels per source unit",
        "batch": 32, "microbatch": 8, "dropout": .05,
        "head_lr": 1e-3, "encoder_lr": 1e-4, "weight_decay": 1e-4,
        "loss": "SmoothL1 beta=0.05", "conv1d_stem": False,
        "partial": "Block2 + FinalNorm + Adapter2 + downstream head from epoch 1",
    })
    for fraction in FRACTIONS:
        for seed in SEEDS:
            missing = [arm for arm in ARMS if (fraction, seed, arm) not in completed]
            if not missing:
                continue
            name = f"three_domain_refresh_seed{seed}_{int(fraction * 100)}pct"
            output = RUNTIME / name
            if (output / "development_summary.json").exists():
                payload = json.loads((output / "development_summary.json").read_text(encoding="utf-8"))
            else:
                recover(output)
                down.SEED, down.LABEL_FRACTION = seed, fraction
                down.NAME, down.ARMS = name, tuple(missing)
                write(STATUS, {"state": "running", "fraction": fraction, "seed": seed,
                               "arms": missing, "completed_transfer_rows": len(completed),
                               "target_transfer_rows": 45, "pid": os.getpid()})
                down.main()
                payload = json.loads((output / "development_summary.json").read_text(encoding="utf-8"))
            for original in payload["rows"]:
                identity = (fraction, seed, original["arm"])
                if identity in completed:
                    continue
                row = dict(original)
                row.update(initialization="three_domain_refreshed", fraction=fraction,
                           seed=seed, imported=False, source_checkpoint=str(SOURCE))
                rows.append(row)
                completed.add(identity)
            write(RESULTS, {"complete": False, "rows": rows, "summary": aggregate(rows)})
    if len(completed) != 45:
        raise RuntimeError(f"Incomplete transfer grid: {len(completed)}/45")
    write(RESULTS, {"complete": True, "rows": rows, "summary": aggregate(rows)})
    write(STATUS, {"state": "complete", "scratch_rows": 15, "transfer_rows": 45})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(STATUS, {"state": "failed", "error": repr(error),
                       "traceback": traceback.format_exc(), "pid": os.getpid()})
        raise
