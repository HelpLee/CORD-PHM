from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from preprocess_health_tokens.common.battery_health_token_utils import BatteryCycle, detect_project_root
from preprocess_health_tokens.datasets.battery._adapter_utils import complete_discharge_cycle, run_adapter


DATASET_NAME = "rwth_battery"
PROJECT_ROOT = detect_project_root()
RAW_ROOT = PROJECT_ROOT / "data_phm" / "raw" / "Battery" / "RWTH_Drive_Cycle_Aging"
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens" / "battery"
START_VOLTAGE_V = 4.05
CUTOFF_VOLTAGE_V = 2.75
START_CURRENT_A = -0.05
LOW_VOLTAGE_CONFIRM_POINTS = 6


def _iter_file_cycles(path: Path, file_index: int, raw_root: Path) -> Iterator[BatteryCycle]:
    import pandas as pd

    collecting = False
    low_count = 0
    cycle_index = 0
    t_buf, v_buf, i_buf, temp_buf = [], [], [], []

    def reset() -> None:
        nonlocal t_buf, v_buf, i_buf, temp_buf, low_count
        t_buf, v_buf, i_buf, temp_buf = [], [], [], []
        low_count = 0

    for chunk in pd.read_csv(
        path,
        usecols=["Time", "Voltage", "Current", "Temperature"],
        chunksize=250_000,
    ):
        timestamps = pd.to_datetime(chunk["Time"], errors="coerce")
        valid_timestamp = timestamps.notna().to_numpy()
        time_values = timestamps.astype("int64").to_numpy(dtype=np.float64) / 1e9
        voltage = chunk["Voltage"].to_numpy(dtype=np.float64)
        current = chunk["Current"].to_numpy(dtype=np.float64)
        temperature = chunk["Temperature"].to_numpy(dtype=np.float64)
        for valid_t, t, v, current_a, temp in zip(valid_timestamp, time_values, voltage, current, temperature):
            if not valid_t or not np.isfinite(t + v + current_a):
                continue
            if not collecting:
                # A discharge is armed only at a charged voltage and negative
                # traction current. Partial cycles at the beginning of a file
                # are intentionally skipped.
                if v >= START_VOLTAGE_V and current_a <= START_CURRENT_A:
                    collecting = True
                    reset()
                else:
                    continue
            t_buf.append(t)
            v_buf.append(v)
            i_buf.append(current_a)
            temp_buf.append(temp)
            low_count = low_count + 1 if v <= CUTOFF_VOLTAGE_V else 0
            if low_count < LOW_VOLTAGE_CONFIRM_POINTS:
                continue
            cycle_index += 1
            keep = max(len(t_buf) - LOW_VOLTAGE_CONFIRM_POINTS + 1, 1)
            yield complete_discharge_cycle(
                dataset=DATASET_NAME,
                unit_id=path.stem,
                cycle_index=cycle_index,
                time_s=np.asarray(t_buf[:keep], dtype=np.float64),
                voltage_v=np.asarray(v_buf[:keep], dtype=np.float64),
                current_a=np.asarray(i_buf[:keep], dtype=np.float64),
                capacity_ah=None,
                temperature_c=np.asarray(temp_buf[:keep], dtype=np.float64),
                already_discharge=True,
                group_id="RWTH_Everlast_35E",
                condition_id="dynamic_drive_cycle",
                segment_id=f"{path.stem}_cycle_{cycle_index}",
                file_id=path.stem,
                filename=path.name,
                source_relpath=str(path.relative_to(raw_root)),
                file_index=file_index,
            )
            collecting = False
            reset()


def iter_cycles(raw_root: Path = RAW_ROOT, limit_files: Optional[int] = None) -> Iterator[BatteryCycle]:
    files = sorted(raw_root.glob("Everlast_35E_*.csv"))
    if limit_files is not None:
        files = files[: int(limit_files)]
    for file_index, path in enumerate(files):
        yield from _iter_file_cycles(path, file_index, raw_root)


def process_health_tokens(cfg=None, *, out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return run_adapter(
        cfg=cfg,
        out_root=out_root,
        dataset=DATASET_NAME,
        cycles=iter_cycles(limit_files=limit_segments),
        notes={
            "raw_root": str(RAW_ROOT),
            "selection": "included: continuous files expose timestamp/V/I/T; complete drive-cycle discharges are segmented from charged voltage to confirmed cutoff and Q is integrated",
            "segmentation": {
                "start_voltage_v": START_VOLTAGE_V,
                "start_current_a": START_CURRENT_A,
                "cutoff_voltage_v": CUTOFF_VOLTAGE_V,
                "low_voltage_confirm_points": LOW_VOLTAGE_CONFIRM_POINTS,
            },
        },
    )


def run(out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return process_health_tokens(out_root=out_root, limit_segments=limit_segments)


if __name__ == "__main__":
    print(run())
