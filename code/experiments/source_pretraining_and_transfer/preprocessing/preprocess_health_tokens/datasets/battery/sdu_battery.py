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


DATASET_NAME = "sdu_battery"
PROJECT_ROOT = detect_project_root()
RAW_ROOT = PROJECT_ROOT / "data_phm" / "raw" / "Battery" / "SDU"
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens" / "battery"


def iter_cycles(raw_root: Path = RAW_ROOT, limit_files: Optional[int] = None) -> Iterator[BatteryCycle]:
    files = sorted(raw_root.glob("processed_*_phase/*.pkl"))
    if limit_files is not None:
        files = files[: int(limit_files)]
    for file_index, path in enumerate(files):
        with path.open("rb") as handle:
            battery = pickle.load(handle)
        phase = path.parent.name.replace("processed_", "").replace("_phase", "")
        raw_id = str(battery.get("cell_id", path.stem))
        unit_id = f"{phase}_{raw_id}"
        for record in battery.get("cycle_data", []):
            cycle_index = int(record.get("cycle_number", 0))
            if cycle_index <= 0:
                continue
            current = np.asarray(record.get("current_in_A", []), dtype=np.float64)
            voltage = np.asarray(record.get("voltage_in_V", []), dtype=np.float64)
            time_s = np.asarray(record.get("time_in_s", []), dtype=np.float64)
            capacity = np.asarray(record.get("discharge_capacity_in_Ah", []), dtype=np.float64)
            temperature = record.get("temperature_in_C")
            mask = continuous_discharge_span_mask(current)
            yield complete_discharge_cycle(
                dataset=DATASET_NAME,
                unit_id=unit_id,
                cycle_index=cycle_index,
                time_s=time_s,
                voltage_v=voltage,
                current_a=current,
                capacity_ah=capacity if capacity.size == current.size else None,
                temperature_c=temperature,
                discharge_mask=mask,
                group_id=raw_id,
                condition_id=phase,
                segment_id=f"{unit_id}_cycle_{cycle_index}",
                file_id=path.stem,
                filename=path.name,
                source_relpath=str(path.relative_to(raw_root)),
                source_split=phase,
                file_index=file_index,
            )


def process_health_tokens(cfg=None, *, out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    raise ValueError('SDU is excluded from the R2F/EOL-only pipeline; stage-only processing is disabled')


def run(out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return process_health_tokens(out_root=out_root, limit_segments=limit_segments)


if __name__ == "__main__":
    print(run())
