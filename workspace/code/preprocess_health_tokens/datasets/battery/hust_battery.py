from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from preprocess_health_tokens.common.battery_health_token_utils import BatteryCycle, detect_project_root
from preprocess_health_tokens.datasets.battery._adapter_utils import (
    complete_discharge_cycle,
    continuous_discharge_span_mask,
    run_adapter,
)


DATASET_NAME = "hust_battery"
PROJECT_ROOT = detect_project_root()
RAW_ROOT = PROJECT_ROOT / "data_phm" / "raw" / "Battery" / "HUST_Battery"
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens" / "battery"


def iter_cycles(raw_root: Path = RAW_ROOT, limit_files: Optional[int] = None) -> Iterator[BatteryCycle]:
    files = sorted(raw_root.glob("*.pkl"))
    if limit_files is not None:
        files = files[: int(limit_files)]
    for file_index, path in enumerate(files):
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        for unit_id, record in loaded.items():
            cycle_map = record.get("data", {}) if isinstance(record, dict) else {}
            for cycle_index, frame in sorted(cycle_map.items(), key=lambda item: int(item[0])):
                required = {"Status", "Current (mA)", "Voltage (V)", "Time (s)"}
                if not required.issubset(frame.columns):
                    continue
                status = frame["Status"].astype(str).str.lower().to_numpy()
                active_mask = np.asarray(["discharge" in value for value in status], dtype=bool)
                current = frame["Current (mA)"].to_numpy(dtype=np.float64) / 1000.0
                mask = continuous_discharge_span_mask(current, activity_mask=active_mask)
                yield complete_discharge_cycle(
                    dataset=DATASET_NAME,
                    unit_id=str(unit_id),
                    cycle_index=int(cycle_index),
                    time_s=frame["Time (s)"].to_numpy(dtype=np.float64),
                    voltage_v=frame["Voltage (V)"].to_numpy(dtype=np.float64),
                    current_a=current,
                    # HUST Capacity resets between protocol substeps; integrate
                    # discharge current instead of treating resets as real Q.
                    capacity_ah=None,
                    temperature_c=None,
                    discharge_mask=mask,
                    group_id=str(unit_id).split("-")[0],
                    condition_id=str(unit_id).split("-")[0],
                    segment_id=f"{unit_id}_cycle_{cycle_index}",
                    file_id=path.stem,
                    filename=path.name,
                    source_relpath=str(path.relative_to(raw_root)),
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
            "selection": "included: each pickle has explicit cycle DataFrames with discharge status, t/V/I; Q is integrated and temperature features are masked",
        },
    )


def run(out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return process_health_tokens(out_root=out_root, limit_segments=limit_segments)


if __name__ == "__main__":
    print(run())
