from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from preprocess_health_tokens.common.battery_health_token_utils import BatteryCycle, detect_project_root
from preprocess_health_tokens.datasets.battery._adapter_utils import complete_discharge_cycle, run_adapter


DATASET_NAME = "mich_exp_battery"
PROJECT_ROOT = detect_project_root()
RAW_ROOT = PROJECT_ROOT / "data_phm" / "raw" / "Battery" / "MICH_EXP"
AGING_ROOT = RAW_ROOT / "data" / "2020-10-aging-test-timeseries"
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens" / "battery"


def _main_discharge_mask(current: np.ndarray, discharge_capacity: np.ndarray) -> np.ndarray:
    """Select Michigan's capacity-bearing main discharge operation.

    Formation/diagnostic cycle numbers can contain many short negative-current
    pulses spread over roughly a day, followed by one full discharge.  Treating
    the first through last negative point as one segment would incorrectly fold
    those diagnostic pulses, intervening charges, and long rests into a single
    snapshot.  The full discharge is the contiguous negative-current run with
    the largest increase in the dataset's discharge-capacity channel.
    """

    current = np.asarray(current, dtype=np.float64).reshape(-1)
    capacity = np.asarray(discharge_capacity, dtype=np.float64).reshape(-1)
    n = min(current.size, capacity.size)
    mask = np.zeros((n,), dtype=bool)
    active_indices = np.flatnonzero(np.isfinite(current[:n]) & (current[:n] < -0.005))
    if active_indices.size < 8:
        return mask
    runs = np.split(active_indices, np.flatnonzero(np.diff(active_indices) > 1) + 1)

    def capacity_gain(run: np.ndarray) -> float:
        q = capacity[run]
        q = q[np.isfinite(q)]
        return float(np.max(q) - np.min(q)) if q.size >= 2 else 0.0

    valid_runs = [run for run in runs if run.size >= 8]
    if not valid_runs:
        return mask
    main_run = max(valid_runs, key=lambda run: (capacity_gain(run), run.size))
    mask[int(main_run[0]) : int(main_run[-1]) + 1] = True
    return mask


def _ordered_cycle_groups(path: Path):
    import pandas as pd

    usecols = [
        "Cycle Number",
        "Test Time (s)",
        "Potential (V)",
        "Current (A)",
        "Discharge Capacity (Ah)",
        "Auxiliary Temperature (°C) 0 (°C)",
    ]
    carry = None
    for chunk in pd.read_csv(path, usecols=lambda name: name in usecols, chunksize=200_000):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        if "Cycle Number" not in chunk.columns or chunk.empty:
            continue
        cycle_values = chunk["Cycle Number"].to_numpy()
        last_cycle = cycle_values[-1]
        completed = chunk[cycle_values != last_cycle]
        carry = chunk[cycle_values == last_cycle].copy()
        for cycle_index, group in completed.groupby("Cycle Number", sort=False):
            yield int(cycle_index), group
    if carry is not None and not carry.empty:
        for cycle_index, group in carry.groupby("Cycle Number", sort=False):
            yield int(cycle_index), group


def _full_voltage_excursion(voltage, mask):
    selected = np.asarray(voltage)[mask]
    # Native charge ceiling is 4.2 V and discharge cutoff is 3.0 V.
    # Diagnostic pulse tails (e.g. cycle 53 starts at 3.43 V) are NOT capacity tests.
    return (selected.size >= 8 and np.isfinite(selected[[0, -1]]).all()
            and selected[0] >= 4.0 and selected[-1] <= 3.05)


def iter_cycles(raw_root: Path = RAW_ROOT, limit_files: Optional[int] = None) -> Iterator[BatteryCycle]:
    files = sorted(AGING_ROOT.glob("*.csv"))
    if limit_files is not None:
        files = files[: int(limit_files)]
    for file_index, path in enumerate(files):
        unit_id = path.stem
        for cycle_index, frame in _ordered_cycle_groups(path):
            if cycle_index <= 0:
                continue
            current = frame["Current (A)"].to_numpy(dtype=np.float64)
            temperature_col = "Auxiliary Temperature (°C) 0 (°C)"
            capacity_col = "Discharge Capacity (Ah)"
            capacity = frame[capacity_col].to_numpy(dtype=np.float64)
            mask = _main_discharge_mask(current, capacity)
            if int(mask.sum()) < 8:
                continue
            voltage = frame['Potential (V)'].to_numpy(dtype=np.float64)
            if not _full_voltage_excursion(voltage, mask):
                print(f'[WARN] {unit_id} cycle {cycle_index}: incomplete voltage excursion; excluded', flush=True)
                continue
            yield complete_discharge_cycle(
                dataset=DATASET_NAME,
                unit_id=unit_id,
                cycle_index=cycle_index,
                time_s=frame["Test Time (s)"].to_numpy(dtype=np.float64),
                voltage_v=voltage,
                current_a=current,
                capacity_ah=capacity,
                temperature_c=frame[temperature_col].to_numpy(dtype=np.float64) if temperature_col in frame else None,
                discharge_mask=mask,
                group_id="Michigan_fast_formation",
                condition_id="aging_test",
                segment_id=f"{unit_id}_cycle_{cycle_index}",
                file_id=path.stem,
                filename=path.name,
                source_relpath=str(path.relative_to(raw_root)),
                source_split="2020-10-aging-test-timeseries",
                file_index=file_index,
            )


def process_health_tokens(cfg=None, *, out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return run_adapter(
        cfg=cfg,
        out_root=out_root,
        dataset=DATASET_NAME,
        cycles=iter_cycles(limit_files=limit_segments),
        notes={
            "raw_root": str(RAW_ROOT),
            "selection": "included: aging-test timeseries provide native t/V/I/Q/T; within diagnostic cycle numbers, the contiguous negative-current run with the largest discharge-capacity gain is the main full discharge, so pulse trains and long intervening rests are excluded",
        },
    )


def run(out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return process_health_tokens(out_root=out_root, limit_segments=limit_segments)


if __name__ == "__main__":
    print(run())
