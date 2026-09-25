"""One full-finetuning job for a model, target domain, and label fraction."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
WORKSPACE = next(path for path in HERE.parents if (path / "data_phm").is_dir())
EXP15 = SUITE / "15_three_domain_original4_milling_all_channels"
EXP19 = SUITE / "19_gradnorm_downstream_cluster"
SEEDS = (42, 43, 44, 45, 46)
ENCODER_LR = 3e-4
HEAD_LR = 1e-3


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream(model: str, domain: str, package: Path) -> Path:
    source = HERE / "models" / model / "training/encoder.pt"
    status = json.loads((source.parent / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError(f"Upstream incomplete: {source}")
    snapshot = package / "source_snapshot/encoder.pt"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.is_file() and sha(snapshot) != sha(source):
        raise RuntimeError("Refusing to mix downstream results from different upstream checkpoints")
    if not snapshot.is_file():
        shutil.copy2(source, snapshot)
    # The immutable downstream drivers verify completion from a status file
    # adjacent to the selected checkpoint.  Preserve that provenance when the
    # checkpoint is snapshotted into the downstream package.
    write(snapshot.parent / "status.json", status)

    effective = snapshot
    interface = "pretrained"
    if model == "single5" and domain in ("bearing", "battery"):
        # The Milling-only encoder has no target-domain observation interface.
        # Create it deterministically without borrowing any trained joint-model weights.
        code = EXP15 / "three_domain/code"
        sys.path.insert(0, str(code))
        from joint_model import JointModel
        torch.manual_seed(42)
        complete = JointModel().encoder.state_dict()
        learned = torch.load(snapshot, map_location="cpu", weights_only=True)
        for key, value in learned.items():
            if key in complete and complete[key].shape == value.shape:
                complete[key] = value
        effective = package / "source_snapshot/encoder_with_seed42_target_interfaces.pt"
        torch.save(complete, effective)
        interface = "target stem/adapter initialized deterministically at seed 42; shared and Milling weights pretrained"
    write(package / "source.json", {
        "model": model, "source": str(source), "source_sha256": sha(source),
        "effective_checkpoint": str(effective), "effective_sha256": sha(effective),
        "selected_epoch": status.get("best_epoch"), "final_epoch": status.get("epoch"),
        "target_interface": interface,
    })
    return effective


def run_bearing_or_battery(domain: str, model: str, fraction: float,
                           checkpoint: Path, package: Path) -> None:
    # Experiment 19 keeps the domain-specific driver files, while the shared
    # adapter/model modules live in suite/01_shared_dependencies.  Include
    # both explicitly so a clean cluster checkout does not depend on an
    # accidental local PYTHONPATH.
    shared = SUITE / "01_shared_dependencies" / f"{domain}_code"
    sys.path[:0] = [str(WORKSPACE), str(WORKSPACE / "data_phm"), str(shared),
                    str(EXP19 / f"{domain}_code"), str(EXP19)]
    module = importlib.import_module(
        "bearing_adapter_study" if domain == "bearing" else "adapter_downstream")

    if domain == "bearing":
        def optimizer_for(network, arm, stage):
            module.configure(network, arm, stage)
            encoder = [p for p in network.encoder.parameters() if p.requires_grad]
            head = [p for name, p in network.named_parameters() if not name.startswith("encoder.")]
            return torch.optim.AdamW([{"params": encoder, "lr": ENCODER_LR},
                                      {"params": head, "lr": HEAD_LR}], weight_decay=1e-4)
        module.optimizer_for = optimizer_for
    else:
        def optimizer(network, arm, stage):
            module.configure(network, arm, stage)
            head = [p for name, p in network.named_parameters()
                    if not name.startswith("encoder.") and p.requires_grad]
            encoder = [p for p in network.encoder.parameters() if p.requires_grad]
            return torch.optim.AdamW([{"params": head, "lr": HEAD_LR},
                                      {"params": encoder, "lr": ENCODER_LR}], weight_decay=1e-4)
        module.optimizer = optimizer

    module.FRACTIONS = (fraction,)
    write(package / "config.json", {
        "checkpoint": str(checkpoint.resolve()), "upstream": None,
        "arms": ["full_finetune"],
    })
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        module.main(package)
    finally:
        sys.argv = old_argv
    write(package / "lr_override.json", {
        "effective_encoder_lr": ENCODER_LR, "effective_head_lr": HEAD_LR,
        "arm": "full_finetune", "fraction": fraction,
    })


def run_milling(model: str, fraction: float, checkpoint: Path, package: Path) -> None:
    sys.path[:0] = [str(WORKSPACE), str(EXP19), str(EXP19 / "milling_code"),
                    str(EXP15 / "downstream_milling_10pct/code")]
    import downstream_interval_val200 as down
    from adapter_encoder import MillingAdapterEncoder
    from portable_inputs import load_store

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    converted = {key: value for key, value in state.items()
                 if not key.startswith(("stems.", "adapters."))}
    for key, value in state.items():
        if key.startswith("adapters.milling."):
            converted[key.replace("adapters.milling.", "adapters.", 1)] = value
    for target, origin in (("local_norm", "local.0"), ("local_proj", "local.1"),
                           ("global_norm", "global.0"), ("global_proj", "global.1")):
        for field in ("weight", "bias"):
            converted[f"{target}.{field}"] = state[f"stems.milling.{origin}.{field}"]
    MillingAdapterEncoder(channels=7, dropout=.05).load_state_dict(converted, strict=True)
    converted_path = package / "converted_encoder.pt"
    torch.save(converted, converted_path)
    store = load_store("milling")
    if store.x.shape[1:] != (7, 64, 26):
        raise RuntimeError(f"Unexpected PHM2010 shape: {store.x.shape}")

    def interval_split(current):
        trains, validations, counts = [], [], {}
        for unit in down.TRAIN:
            sequences = down.full_sequences(current, (unit,))
            count = max(1, int(np.ceil(len(sequences) * fraction)))
            selected = np.unique(np.rint(np.linspace(0, len(sequences) - 1, count)).astype(np.int64))
            chosen = sequences[selected]
            number_validation = max(1, int(np.ceil(len(chosen) * .2)))
            validation_indices = np.unique(
                np.rint(np.linspace(0, len(chosen) - 1, number_validation)).astype(np.int64))
            train_indices = np.setdiff1d(np.arange(len(chosen)), validation_indices)
            trains.append(chosen[train_indices])
            validations.append(chosen[validation_indices])
            counts[unit] = {"total": len(sequences), "selected": len(chosen),
                            "train": len(train_indices), "validation": len(validation_indices)}
        return np.concatenate(trains), np.concatenate(validations), counts

    down.prepare_store = lambda _path: store
    down.source_split = interval_split
    down.SOURCE = converted_path
    down.OUT = package / "runtime"
    down.MAX_EPOCHS, down.MAX_UPDATES = 200, 0
    down.FROZEN_EPOCHS, down.PATIENCE, down.VAL_INTERVAL = 0, 15, 2
    down.TRAIN_VALIDATION_FRACTION, down.TEST_VALIDATION_FRACTION = .2, 0
    down.NESTED_LABELS, down.LABEL_FRACTION = False, fraction
    down.HEAD_LR, down.LAST_BLOCK_LR = HEAD_LR, ENCODER_LR
    down.ARMS = ("full_finetune",)
    rows = []
    for seed in SEEDS:
        down.SEED = seed
        down.NAME = f"seed{seed}_p{round(fraction * 100)}"
        summary = down.OUT / down.NAME / "development_summary.json"
        write(package / "status.json", {"state": "running", "model": model,
              "seed": seed, "fraction": fraction, "completed": len(rows), "pid": os.getpid()})
        if not summary.is_file():
            down.main()
        for row in json.loads(summary.read_text(encoding="utf-8"))["rows"]:
            rows.append(dict(row, seed=seed, fraction=fraction, model=model,
                             encoder_lr=ENCODER_LR, head_lr=HEAD_LR))
        write(package / "results.json", {"complete": False, "rows": rows})
    write(package / "results.json", {"complete": True, "rows": rows})
    write(package / "status.json", {"state": "complete", "runs": len(rows)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("three4", "three5", "single5"))
    parser.add_argument("--domain", required=True, choices=("bearing", "battery", "milling"))
    parser.add_argument("--fraction", required=True, type=float, choices=(.1, .2, 1.0))
    args = parser.parse_args()
    package = HERE / "downstream" / args.model / args.domain / f"p{round(args.fraction * 100)}"
    package.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint = upstream(args.model, args.domain, package)
        if args.domain == "milling":
            run_milling(args.model, args.fraction, checkpoint, package)
        else:
            run_bearing_or_battery(args.domain, args.model, args.fraction, checkpoint, package)
    except BaseException as error:
        write(package / "status.json", {"state": "failed", "error": repr(error),
              "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    main()
