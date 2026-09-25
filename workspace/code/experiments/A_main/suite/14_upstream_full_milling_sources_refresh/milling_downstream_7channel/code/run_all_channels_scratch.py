"""Seven-channel PHM2010 scratch ablation.

The only experimental change from the canonical 3-force-channel scratch run
is the input sensor selection: all seven released PHM2010 streams are retained.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np

# The standalone package reuses the canonical HealthToken feature extractor.
WORKSPACE = next(parent for parent in Path(__file__).resolve().parents
                 if (parent / "code" / "preprocess_health_tokens").is_dir())
sys.path.insert(0, str(WORKSPACE / "code"))

import data as data_module
import downstream_interval_val200 as down
from data import build_one
from feature_observations import observed_store
import global_local_data as gl
import feature_observations as fo
from run_milling_global_local_multiscale12 import prepare_store


PACKAGE = Path(__file__).resolve().parents[1]
RUNTIME = PACKAGE / "runtime"
SEEDS = (42, 43)
FRACTIONS = (0.1, 0.2)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_store():
    # New cache guarantees that the all-channel tensors cannot be confused with
    # the immutable force-only cache used by prior experiments.
    data_module.OUT = RUNTIME
    gl.OUT = fo.OUT = RUNTIME
    cache = build_one("milling", data_module.DOWN["milling"], downstream=True)
    store = prepare_store(cache)
    expected = ["force_x", "force_y", "force_z", "vibration_x", "vibration_y", "vibration_z", "ae_rms"]
    assert store.x.shape[1] == 7, store.x.shape
    write(PACKAGE / "data_provenance.json", {
        "source_npz": str((data_module.DATA / "milling/phm2010_milling_downstream_health_tokens.npz").resolve()),
        "source_sha256": sha256(data_module.DATA / "milling/phm2010_milling_downstream_health_tokens.npz"),
        "sensor_selection": "all",
        "sensor_channels": expected,
        "channel_count": 7,
        "train_units": ["C1", "C4"], "test_unit": "C6",
        "control": "canonical scratch is identical except force-only Fx/Fy/Fz selection",
    })
    return store


def interval_split(store):
    trains, validations, counts = [], [], {}
    for unit in down.TRAIN:
        sequences = down.full_sequences(store, (unit,))
        count = max(1, int(np.ceil(len(sequences) * down.LABEL_FRACTION)))
        selected = np.unique(np.rint(np.linspace(0, len(sequences) - 1, count)).astype(np.int64))
        chosen = sequences[selected]
        n_validation = max(1, int(np.ceil(len(chosen) * 0.2)))
        validation_index = np.unique(np.rint(
            np.linspace(0, len(chosen) - 1, n_validation)).astype(np.int64))
        train_index = np.setdiff1d(np.arange(len(chosen)), validation_index)
        assert len(train_index) and len(validation_index)
        trains.append(chosen[train_index]); validations.append(chosen[validation_index])
        counts[unit] = {"total": len(sequences), "selected": len(chosen),
                        "train": len(train_index), "validation": len(validation_index)}
    return np.concatenate(trains), np.concatenate(validations), counts


def main():
    store = load_store()
    down.prepare_store = lambda _path: store
    down.OUT = RUNTIME
    down.ARMS = ("scratch",)
    down.MAX_UPDATES = 0
    down.MAX_EPOCHS = 200
    down.PATIENCE = 15
    down.VAL_INTERVAL = 2
    down.TRAIN_VALIDATION_FRACTION = .2
    down.TEST_VALIDATION_FRACTION = 0
    down.NESTED_LABELS = False
    down.source_split = interval_split
    write(PACKAGE / "protocol.json", {
        "condition": "all_7channel_scratch", "control": "canonical_force_3channel_scratch",
        "changed_variable": "PHM2010 input channels only", "channels": 7,
        "seeds": list(SEEDS), "fractions": list(FRACTIONS), "arms": ["scratch"],
        "max_epochs": 200, "validation": "20% interval sample within selected C1/C4 labels",
        "test": "C6", "loss": "SmoothL1 beta=0.05", "batch": 32, "microbatch": 8,
        "dropout": .05, "scratch_lr": 1e-3,
    })
    rows = []
    for fraction in FRACTIONS:
        for seed in SEEDS:
            down.SEED, down.LABEL_FRACTION = seed, fraction
            down.NAME = f"all7_scratch_seed{seed}_{int(fraction * 100)}pct"
            write(PACKAGE / "status.json", {"state": "running", "seed": seed, "fraction": fraction})
            down.main()
            payload = json.loads((RUNTIME / down.NAME / "development_summary.json").read_text(encoding="utf-8"))
            row = payload["rows"][0]
            row["fraction"] = fraction
            rows.append(row)
            write(PACKAGE / "results.json", {"complete": False, "rows": rows})
    summary = []
    for fraction in FRACTIONS:
        group = [row for row in rows if row["fraction"] == fraction]
        summary.append({"fraction": fraction, "n": len(group), "metrics": {
            metric: {"mean": float(np.mean([row["metrics"][metric] for row in group])),
                     "std": float(np.std([row["metrics"][metric] for row in group], ddof=1))}
            for metric in ("rmse", "mae", "r2", "bias")}})
    write(PACKAGE / "results.json", {"complete": True, "rows": rows, "summary": summary})
    subprocess.run([sys.executable, str(PACKAGE / "summarize_comparison.py")], check=True)
    write(PACKAGE / "status.json", {"state": "complete", "runs": len(rows)})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(PACKAGE / "status.json", {"state": "failed", "error": repr(error)})
        raise
