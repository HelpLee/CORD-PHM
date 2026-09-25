"""Export immutable A-main inputs and splits for all fair baselines.

Each domain is prepared in a fresh process because the preserved domain code
uses short module names such as ``data`` and ``global_local_data``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[2] / "experiments" / "A_main" / "suite"
PREPARED = HERE / "prepared"
FRACTIONS = (0.1, 0.2, 1.0)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def digest_arrays(*arrays: np.ndarray) -> str:
    result = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        result.update(str(value.dtype).encode())
        result.update(str(value.shape).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def save_domain(domain: str, snapshots, splits, metadata) -> None:
    destination = PREPARED / domain
    destination.mkdir(parents=True, exist_ok=True)
    local, global_x, channel_mask, token_mask, y = snapshots
    np.savez_compressed(
        destination / "snapshots.npz",
        local=np.asarray(local, np.float32),
        global_x=np.asarray(global_x, np.float32),
        channel_mask=np.asarray(channel_mask, bool),
        token_mask=np.asarray(token_mask, bool),
        y=np.asarray(y, np.float32),
    )
    manifest_splits = {}
    for fraction, values in splits.items():
        train_sequence, train_history, validation_sequence, validation_history, test_sequence, test_history = values
        target = destination / f"split_{int(fraction * 100)}.npz"
        np.savez_compressed(
            target,
            train_sequence=np.asarray(train_sequence, np.int64),
            train_history_mask=np.asarray(train_history, bool),
            validation_sequence=np.asarray(validation_sequence, np.int64),
            validation_history_mask=np.asarray(validation_history, bool),
            test_sequence=np.asarray(test_sequence, np.int64),
            test_history_mask=np.asarray(test_history, bool),
        )
        manifest_splits[str(fraction)] = {
            "train": len(train_sequence),
            "validation": len(validation_sequence),
            "test": len(test_sequence),
            "sha256": digest_arrays(*values),
        }
    metadata = dict(metadata)
    metadata.update({
        "domain": domain,
        "snapshot_shape": list(np.asarray(local).shape),
        "snapshot_sha256": digest_arrays(local, global_x, channel_mask, token_mask, y),
        "splits": manifest_splits,
        "normalization": "source devices only; exact canonical A_main transform",
        "test_exclusion": "test labels never used for scaling, fitting, validation, stopping, or selection",
    })
    write_json(destination / "manifest.json", metadata)


def bearing() -> None:
    code = SUITE / "01_shared_dependencies" / "bearing_code"
    sys.path.insert(0, str(code))
    import bearing_adapter_study as canonical

    store, norm, source_splits, validations, test_sequence, test_history = canonical.prepare()
    rows = np.arange(len(store.x), dtype=np.int64)
    local, global_x, channel_mask, token_mask = store.transform(rows, norm)
    split_values = {}
    for fraction in FRACTIONS:
        train_sequence, train_history, counts = source_splits[fraction]
        validation_sequence, validation_history = validations[fraction]
        split_values[fraction] = (
            train_sequence, train_history,
            validation_sequence, validation_history,
            test_sequence, test_history,
        )
    save_domain("bearing", (local, global_x, channel_mask, token_mask, store.y), split_values, {
        "train_units": list(canonical.TRAIN), "test_unit": canonical.TEST,
        "history": 6, "source": str(store.info["signature"]),
    })


def battery() -> None:
    code = SUITE / "01_shared_dependencies" / "battery_code"
    sys.path.insert(0, str(code))
    import run_battery_state_trend as canonical
    from queue_battery_val200 import split

    data = canonical.Downstream()
    test_rows = data.groups[canonical.TEST]
    split_values = {}
    for fraction in FRACTIONS:
        train_rows, validation_rows = split(data, fraction)
        split_values[fraction] = (
            data.sequences[train_rows], data.hm[train_rows],
            data.sequences[validation_rows], data.hm[validation_rows],
            data.sequences[test_rows], data.hm[test_rows],
        )
    save_domain("battery", (data.xn, data.gn, data.cm, data.tm, data.y), split_values, {
        "train_units": list(canonical.TRAIN), "test_unit": canonical.TEST,
        "history": 20, "source": str(data.path),
    })


def milling() -> None:
    code = SUITE / "20_downstream_milling_three_domain_adapter" / "code"
    sys.path.insert(0, str(code))
    import data as data_module
    import downstream_interval_val200 as canonical
    import feature_observations as observations
    import global_local_data as global_local
    from run_milling_global_local_multiscale12 import prepare_store

    runtime = SUITE / "01_shared_dependencies" / "milling_runtime"
    cache = runtime / "raw_cache" / data_module.DOWN["milling"]
    if not (cache / "manifest.json").exists():
        raise FileNotFoundError(
            f"Canonical milling cache missing: {cache}. Run the A_main preprocessing package first.")
    global_local.OUT = observations.OUT = runtime
    store = prepare_store(cache)
    norm = store.normalize(canonical.TRAIN)
    rows = np.arange(len(store.x), dtype=np.int64)
    local, global_x, channel_mask, token_mask = store.transform(rows, norm)
    test_sequence = canonical.full_sequences(store, (canonical.HELD,))
    test_history = np.ones(test_sequence.shape, dtype=bool)
    split_values = {}
    for fraction in FRACTIONS:
        trains, validations = [], []
        for unit in canonical.TRAIN:
            candidates = canonical.full_sequences(store, (unit,))
            count = max(1, int(np.ceil(len(candidates) * fraction)))
            selected = np.unique(np.rint(np.linspace(0, len(candidates) - 1, count)).astype(np.int64))
            chosen = candidates[selected]
            n_validation = max(1, int(np.ceil(len(chosen) * 0.2)))
            validation_index = np.unique(
                np.rint(np.linspace(0, len(chosen) - 1, n_validation)).astype(np.int64))
            train_index = np.setdiff1d(np.arange(len(chosen)), validation_index)
            if not len(train_index):
                raise RuntimeError(f"No milling training labels for {unit}, fraction={fraction}")
            trains.append(chosen[train_index])
            validations.append(chosen[validation_index])
        train_sequence = np.concatenate(trains)
        validation_sequence = np.concatenate(validations)
        split_values[fraction] = (
            train_sequence, np.ones(train_sequence.shape, bool),
            validation_sequence, np.ones(validation_sequence.shape, bool),
            test_sequence, test_history,
        )
    save_domain("milling", (local, global_x, channel_mask, token_mask, store.y), split_values, {
        "train_units": list(canonical.TRAIN), "test_unit": canonical.HELD,
        "history": canonical.LENGTH, "source": str(store.info["signature"]),
    })


def worker(domain: str) -> None:
    {"bearing": bearing, "battery": battery, "milling": milling}[domain]()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("bearing", "battery", "milling"))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    for domain in ("milling", "battery", "bearing"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", domain],
                       check=True, cwd=HERE)
    write_json(PREPARED / "status.json", {"state": "complete", "domains": ["milling", "battery", "bearing"]})


if __name__ == "__main__":
    main()
