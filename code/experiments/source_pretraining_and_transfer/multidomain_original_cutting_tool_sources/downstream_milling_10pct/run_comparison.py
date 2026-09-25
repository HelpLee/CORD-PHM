"""10%-label frozen-probe comparison for original-four-source all-channel upstreams."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


PACKAGE = Path(__file__).resolve().parent
EXPERIMENT = PACKAGE.parent
WORKSPACE = next(parent for parent in PACKAGE.parents if (parent / "code/data_phm").is_dir())
CODE = PACKAGE / "code"
sys.path.insert(0, str(WORKSPACE / "code"))
sys.path.insert(0, str(CODE))

import downstream_interval_val200 as down  # noqa: E402
from adapter_encoder import MillingAdapterEncoder  # noqa: E402
import run_all_channels_scratch as input_condition  # noqa: E402


SEEDS = (42, 43, 44, 45, 46)
FRACTION = .1
ARM = "frozen_probe"
SOURCES = {
    "milling_only_original4_all_channels": EXPERIMENT / "milling_only/training/encoder.pt",
    "three_domain_original4_all_channels": EXPERIMENT / "three_domain/training/encoder.pt",
}
CONTROL_RESULTS = {
    "milling_only_original4_3channel": WORKSPACE / "outputs/reproduction_final/milling_only_upstream500_adapter_seed42/downstream_milling/results.json",
    "three_domain_original4_3channel": WORKSPACE / "outputs/reproduction_final/joint_upstream500_adapter_seed42/downstream_milling/results.json",
}
SCRATCH_RESULTS = WORKSPACE / "outputs/milling_all_7channel_complete_transfer_matrix/results.json"
RESULTS = PACKAGE / "results.json"
STATUS = PACKAGE / "status.json"


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for attempt in range(40):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.25)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def convert_checkpoint(name: str, source_path: Path) -> Path:
    source = torch.load(source_path, map_location="cpu", weights_only=True)
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
    destination = PACKAGE / "checkpoints" / f"{name}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, destination)
    write(destination.with_suffix(".provenance.json"), {
        "source": str(source_path), "source_sha256": sha256(source_path),
        "converted": str(destination), "converted_sha256": sha256(destination),
        "strict_load": True, "mapping": mapping,
    })
    return destination


def aggregate(rows):
    summary = []
    for initialization in ("scratch", *CONTROL_RESULTS, *SOURCES):
        group = [row for row in rows if row["initialization"] == initialization]
        if not group:
            continue
        def spread(metric):
            values = [row["metrics"][metric] for row in group]
            # Intermediate checkpoints can contain one completed seed; its
            # sample standard deviation is conventionally reported as 0.0.
            return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append({
            "initialization": initialization, "arm": group[0]["arm"],
            "fraction": FRACTION, "n": len(group),
            "metrics": {metric: {
                "mean": float(np.mean([row["metrics"][metric] for row in group])),
                "std": spread(metric),
            } for metric in ("rmse", "mae", "r2", "bias")},
        })
    return summary


def scratch_rows():
    payload = json.loads(SCRATCH_RESULTS.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload["rows"]
            if row.get("initialization") == "scratch" and row.get("fraction") == FRACTION]
    if len(rows) != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} seven-channel Scratch rows, found {len(rows)}")
    for row in rows:
        row.update(initialization="scratch", arm="scratch", imported=True,
                   comparison_source=str(SCRATCH_RESULTS))
    return rows


def control_rows():
    rows = []
    for initialization, result_path in CONTROL_RESULTS.items():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        selected = [dict(row) for row in payload["rows"]
                    if row.get("fraction") == FRACTION and row.get("arm") == ARM]
        if len(selected) != len(SEEDS):
            raise ValueError(f"Expected {len(SEEDS)} rows in {result_path}, found {len(selected)}")
        for row in selected:
            row.update(initialization=initialization, imported=True,
                       comparison_source=str(result_path), input_channels=3)
        rows.extend(selected)
    return rows


def configure(store, checkpoint: Path, runtime: Path) -> None:
    down.prepare_store = lambda _path: store
    down.OUT = runtime
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=tuple(SOURCES), action="append",
                        help="run a selected pretrained initialization only")
    args = parser.parse_args()
    selected_sources = {name: SOURCES[name] for name in (args.only or tuple(SOURCES))}
    missing_sources = [str(path) for path in selected_sources.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(missing_sources)
    store = input_condition.load_store()
    if store.x.shape[1:] != (7, 64, 26):
        raise ValueError(f"Unexpected downstream tensor: {store.x.shape}")
    # The focused Scratch-versus-new-pretrained run should not import the
    # historical 3-channel controls.  Some legacy rows are intentionally
    # incomplete and can contain NaN metrics unrelated to this comparison.
    rows = scratch_rows()
    if not args.only:
        rows.extend(control_rows())
    if RESULTS.exists():
        prior = json.loads(RESULTS.read_text(encoding="utf-8"))
        rows.extend(row for row in prior.get("rows", []) if row["initialization"] in SOURCES)
    completed = {(row["initialization"], row["seed"]) for row in rows if row["initialization"] in SOURCES}
    write(PACKAGE / "protocol.json", {
        "question": "original three-channel versus all available channels with the original four Milling upstream datasets",
        "upstreams": {name: {"path": str(path), "sha256": sha256(path)} for name, path in SOURCES.items()},
        "imported_three_channel_controls": {name: str(path) for name, path in CONTROL_RESULTS.items()},
        "milling_sources": ["luh_milling", "matwi_milling", "nonastreda_milling", "qit_cemc_milling"],
        "downstream": {"train": ["C1", "C4"], "test": "C6", "channels": 7,
                       "fraction": FRACTION, "seeds": list(SEEDS), "arm": ARM,
                       "max_epochs": 200, "patience": 15, "validation_interval": 2,
                       "validation": "20% uniform interval subset of independently selected labels per source unit"},
    })
    for initialization, source in selected_sources.items():
        checkpoint = convert_checkpoint(initialization, source)
        runtime = PACKAGE / "runtime" / initialization
        if os.name == "nt":
            runtime = Path("\\\\?\\" + str(runtime))
        configure(store, checkpoint, runtime)
        for seed in SEEDS:
            if (initialization, seed) in completed:
                continue
            down.SEED, down.LABEL_FRACTION = seed, FRACTION
            down.NAME = f"{initialization}_seed{seed}_10pct"
            down.ARMS = (ARM,)
            output = runtime / down.NAME / "development_summary.json"
            write(STATUS, {"state": "running", "initialization": initialization,
                           "seed": seed, "completed": len(completed),
                           "target": len(selected_sources) * len(SEEDS),
                           "pid": os.getpid()})
            # A seed can finish just before an interruption while the summary
            # table is being written. Reuse that completed seed safely.
            if not output.is_file():
                down.main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            if len(payload["rows"]) != 1 or payload["rows"][0]["arm"] != ARM:
                raise RuntimeError(f"Unexpected downstream payload: {output}")
            row = dict(payload["rows"][0])
            row.update(initialization=initialization, fraction=FRACTION, seed=seed,
                       imported=False, source_checkpoint=str(source))
            rows.append(row)
            completed.add((initialization, seed))
            write(RESULTS, {"complete": False, "rows": rows, "summary": aggregate(rows)})
    selected_completed = {(name, seed) for name, seed in completed if name in selected_sources}
    target = len(selected_sources) * len(SEEDS)
    if len(selected_completed) != target:
        raise RuntimeError(f"Incomplete grid: {len(selected_completed)}/{target}")
    write(RESULTS, {"complete": True, "rows": rows, "summary": aggregate(rows)})
    write(STATUS, {"state": "complete", "scratch_rows": 5, "transfer_rows": 10})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(STATUS, {"state": "failed", "error": repr(error),
                       "traceback": traceback.format_exc(), "pid": os.getpid()})
        raise
