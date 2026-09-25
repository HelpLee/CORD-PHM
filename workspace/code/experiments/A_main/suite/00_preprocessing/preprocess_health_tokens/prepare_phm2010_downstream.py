#!/usr/bin/env python3
"""Build the strict PHM 2010 milling downstream HealthToken artifact.

Only the three officially labelled cutters (C1, C4 and C6) are admitted.  One
published cut CSV is one physical observation and therefore exactly one NPZ
row.  Wear measurements are used only to locate the published 0.16 mm
average-flute-wear crossing; they are never model inputs.  The final usable cut
immediately preceding that crossing is RUL zero, matching the published
306/278/238-cut protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from preprocess_health_tokens.suite_paths import data_code_root
CODE_ROOT = data_code_root()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from preprocess_health_tokens.common.bearing_health_token_utils import (
    FEATURE_NAMES,
    extract_local_feature_sequence,
)


LABELLED_CUTTERS = (1, 4, 6)
EXPECTED_CUTS = 315
FAILURE_THRESHOLD_UM = 160.0
EXPECTED_EOL_CUT = {1: 306, 4: 278, 6: 238}
CHANNELS = ("force_x", "force_y", "force_z", "vibration_x", "vibration_y", "vibration_z", "ae_rms")
CONDITION = "rpm=10400|feed_mm_min=1555|radial_doc_mm=0.125|axial_doc_mm=0.2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_blocks(signal: np.ndarray, fs: float = 50_000.0) -> np.ndarray:
    """Represent the complete cut by 32 contiguous non-overlapping windows."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] != len(CHANNELS):
        raise ValueError(f"Expected [7,time] PHM2010 signal, got {signal.shape}")
    if signal.shape[1] < 32 * 8 or not np.isfinite(signal).all():
        raise ValueError("PHM2010 cut is short or contains non-finite sensor values")
    edges = np.linspace(0, signal.shape[1], 33, dtype=np.int64)
    output = np.zeros((len(CHANNELS), 64, len(FEATURE_NAMES)), dtype=np.float32)
    for token, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        features, _ = extract_local_feature_sequence(
            signal[:, start:stop], fs=fs, window_points=int(stop - start), stride=int(stop - start)
        )
        output[:, token] = features[:, 0]
    return output


def cutter_root(raw_root: Path, cutter: int) -> Path:
    candidates = []
    for directory in raw_root.rglob(f"c{cutter}"):
        if directory.is_dir():
            candidates.append(directory)
    candidates.extend(p.parent for p in raw_root.rglob(f"c{cutter}_wear.csv"))
    unique = sorted(set(p.resolve() for p in candidates), key=lambda p: (len(p.parts), str(p).lower()))
    for candidate in unique:
        if list(candidate.rglob(f"c{cutter}_wear.csv")) and sensor_files(candidate, cutter, strict=False):
            return candidate
    raise FileNotFoundError(f"Cannot locate official C{cutter} sensor and wear files under {raw_root}")


def sensor_files(root: Path, cutter: int, *, strict: bool = True) -> Dict[int, Path]:
    pattern = re.compile(rf"^c_?{cutter}_(\d{{3}})\.csv$", re.IGNORECASE)
    result: Dict[int, Path] = {}
    for path in root.rglob("*.csv"):
        match = pattern.match(path.name)
        if match:
            cut = int(match.group(1))
            if cut in result:
                raise ValueError(f"Duplicate C{cutter} cut {cut}: {result[cut]} and {path}")
            result[cut] = path
    if strict and sorted(result) != list(range(1, EXPECTED_CUTS + 1)):
        missing = sorted(set(range(1, EXPECTED_CUTS + 1)) - set(result))
        raise ValueError(f"C{cutter} must contain cuts 1..315 exactly; missing={missing[:20]}, count={len(result)}")
    return result


def wear_table(root: Path, cutter: int) -> np.ndarray:
    import pandas as pd

    matches = list(root.rglob(f"c{cutter}_wear.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one C{cutter} wear file, found {matches}")
    frame = pd.read_csv(matches[0], header=None)
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
    if len(numeric) != EXPECTED_CUTS:
        raise ValueError(f"C{cutter} wear table must have 315 rows, got {len(numeric)}")
    values = numeric.to_numpy(dtype=np.float64)
    # Official files contain cut index followed by the three flute measurements.
    if values.shape[1] >= 4 and np.array_equal(values[:, 0].astype(int), np.arange(1, EXPECTED_CUTS + 1)):
        values = values[:, 1:4]
    elif values.shape[1] >= 3:
        values = values[:, -3:]
    else:
        raise ValueError(f"C{cutter} wear file does not contain three flute columns")
    if values.shape != (EXPECTED_CUTS, 3) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Invalid C{cutter} flute-wear measurements")
    return values.astype(np.float32)


def read_signal(path: Path) -> np.ndarray:
    signal = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if signal.ndim != 2 or signal.shape[1] != len(CHANNELS):
        raise ValueError(f"{path}: expected seven numeric sensor columns, got {signal.shape}")
    return signal.T


def build(raw_root: Path, output: Path, *, limit_cuts: int | None = None) -> dict:
    records: List[dict] = []
    source_audit = []
    started = time.perf_counter()
    for cutter in LABELLED_CUTTERS:
        root = cutter_root(raw_root, cutter)
        files = sensor_files(root, cutter)
        wear = wear_table(root, cutter)
        wear_mean = np.mean(wear, axis=1)
        crossings = np.flatnonzero(wear_mean >= FAILURE_THRESHOLD_UM)
        if not crossings.size:
            raise ValueError(f"C{cutter} never reaches the {FAILURE_THRESHOLD_UM:g} um mean-wear endpoint")
        crossing_cut = int(crossings[0] + 1)
        eol_cut = crossing_cut - 1
        if eol_cut != EXPECTED_EOL_CUT[cutter]:
            raise ValueError(
                f"C{cutter} 0.16 mm endpoint mismatch: expected cut {EXPECTED_EOL_CUT[cutter]}, got {eol_cut}"
            )
        source_audit.append({
            "unit": f"C{cutter}", "cuts": len(files), "wear_rows": len(wear),
            "eol_cut": eol_cut, "threshold_crossing_cut": crossing_cut,
            "failure_threshold_um": FAILURE_THRESHOLD_UM,
            "root": str(root), "wear_first_um": wear[0].tolist(), "wear_at_eol_um": wear[eol_cut - 1].tolist(),
        })
        for cut in range(1, eol_cut + 1):
            if limit_cuts is not None and len(records) >= limit_cuts:
                break
            path = files[cut]
            signal = read_signal(path)
            records.append({
                "x": feature_blocks(signal),
                "sample_dataset": "phm2010_milling_downstream",
                "sample_unit_id": f"C{cutter}",
                "sample_group_id": f"C{cutter}",
                "sample_run_id": f"C{cutter}",
                "sample_condition_id": CONDITION,
                "sample_snapshot_index": cut,
                "sample_cycle_index": cut,
                "sample_segment_id": path.stem,
                "sample_file_id": path.stem,
                "sample_filename": path.name,
                "sample_file_index": cut,
                "sample_source_split": "labelled",
                "sample_source_relpath": path.relative_to(raw_root).as_posix(),
                "sample_chunk_id": 0,
                "order_value": float(cut),
                "eol_order": float(eol_cut),
                "endpoint_observed": True,
                "y_rul": float(eol_cut - cut),
                "y_rul_norm": float(eol_cut - cut) / float(eol_cut - 1),
                "wear_flute_um": wear[cut - 1],
                "wear_mean_um": float(np.mean(wear[cut - 1])),
                "wear_max_um": float(np.max(wear[cut - 1])),
                "fs": 50_000.0,
                "window_points": int(signal.shape[1] // 32),
                "window_points_max": int(np.ceil(signal.shape[1] / 32)),
            })
            if len(records) % 25 == 0:
                elapsed = time.perf_counter() - started
                print(f"processed={len(records)} cutter=C{cutter} cut={cut} elapsed_sec={elapsed:.1f}", flush=True)
        if limit_cuts is not None and len(records) >= limit_cuts:
            break
    expected_samples = sum(EXPECTED_EOL_CUT.values())
    if limit_cuts is None and len(records) != expected_samples:
        raise ValueError(f"Expected {expected_samples} observations, got {len(records)}")

    x = np.stack([record.pop("x") for record in records])
    token_mask = np.zeros(x.shape[:3], dtype=bool)
    token_mask[:, :, :32] = True
    c_mask = np.ones(x.shape[:2], dtype=bool)
    if not np.isfinite(x[token_mask]).all() or np.count_nonzero(x[~token_mask]):
        raise ValueError("Invalid PHM2010 HealthToken values or non-zero padding")
    metadata = {
        "dataset": "phm2010_milling_downstream",
        "dataset_role": "downstream_only",
        "source": "PHM Society 2010 Data Challenge; Kaggle public mirror rabahba/phm-data-challenge-2010",
        "official_url": "https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/",
        "mirror_url": "https://www.kaggle.com/datasets/rabahba/phm-data-challenge-2010",
        "rul_protocol_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12251633/",
        "labelled_cutters_only": ["C1", "C4", "C6"],
        "one_npz_row": "one complete published cut acquisition at or before the final usable cut",
        "sensor_channels": list(CHANNELS),
        "sampling_hz": 50_000,
        "window_definition": "32 contiguous equal-sample windows covering the complete cut; no overlap or resampling",
        "target": "remaining cut acquisitions to the final usable cut immediately before the 0.16 mm crossing",
        "failure_definition": "RUL zero is the cut immediately before mean three-flute wear first reaches 160 um",
        "eol_cuts": {f"C{k}": v for k, v in EXPECTED_EOL_CUT.items()},
        "target_caveat": "The 0.16 mm rule and pre-crossing RUL-zero convention are a published PHM2010 protocol, not official labels in the archive.",
        "wear_is_model_input": False,
        "scaling_deferred_to_downstream_train_fold": True,
        "debug_partial": limit_cuts is not None,
        "source_audit": source_audit,
    }
    payload = {
        "x_health": x,
        "token_mask": token_mask,
        "c_mask": c_mask,
        "feature_names": np.asarray(FEATURE_NAMES),
        "feature_median": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        "feature_iqr": np.ones(len(FEATURE_NAMES), dtype=np.float32),
        "meta_json": np.asarray(json.dumps(metadata)),
    }
    for key in records[0]:
        payload[key] = np.asarray([record[key] for record in records])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    audit_path = output.with_suffix(".lifecycle.json")
    audit_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"output": str(output), "sha256": sha256(output), "samples": len(records), "units": list(map(lambda x: f"C{x}", LABELLED_CUTTERS))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data_phm/raw/Milling/phm2010"))
    parser.add_argument("--output", type=Path, default=Path("data_phm/processed_health_tokens/milling/phm2010_milling_downstream_health_tokens.npz"))
    parser.add_argument("--limit-cuts", type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.raw_root.resolve(), args.output.resolve(), limit_cuts=args.limit_cuts), indent=2))


if __name__ == "__main__":
    main()
