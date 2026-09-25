from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from preprocess_health_tokens.common.battery_health_token_utils import BatteryCycle, detect_project_root
from preprocess_health_tokens.datasets._archive_battery_cycle_trajectory_20260824.nasa_battery import (
    _arr,
    _loadmat_v5_simplified,
    _read_cycles,
)
from preprocess_health_tokens.datasets.battery._adapter_utils import complete_discharge_cycle, run_adapter


DATASET_NAME = "nasa_battery"
PROJECT_ROOT = detect_project_root()
RAW_ROOT = PROJECT_ROOT / "data_phm" / "raw" / "Battery" / "NASA_Battery"
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens" / "battery"


def iter_cycles(raw_root: Path = RAW_ROOT, limit_files: Optional[int] = None) -> Iterator[BatteryCycle]:
    # Several NASA folders contain byte-identical copies with the same battery
    # ID. Keep the first ID so upstream pretraining does not duplicate cells.
    mats = []
    seen_ids = set()
    for path in sorted(raw_root.rglob("B*.mat")):
        unit_id = path.stem.upper()
        if unit_id in seen_ids:
            continue
        seen_ids.add(unit_id)
        mats.append(path)
    if limit_files is not None:
        mats = mats[: int(limit_files)]

    for file_index, path in enumerate(mats):
        discharge_index = 0
        for record in _read_cycles(path):
            if str(record.get("type", "")).lower() != "discharge":
                continue
            discharge_index += 1
            data = record.get("data", {}) or {}
            t = _arr(data, "Time")
            v = _arr(data, "Voltage_measured")
            current = _arr(data, "Current_measured")
            temperature = _arr(data, "Temperature_measured")
            yield complete_discharge_cycle(
                dataset=DATASET_NAME,
                unit_id=path.stem.upper(),
                cycle_index=discharge_index,
                time_s=t,
                voltage_v=v,
                current_a=current,
                temperature_c=temperature if temperature.size == t.size else None,
                already_discharge=True,
                group_id=path.parent.name,
                condition_id=path.parent.name,
                segment_id=f"{path.stem}_discharge_{discharge_index}",
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
            "selection": "included: explicit discharge records contain time, voltage, current, and temperature; Q is integrated from current",
        },
    )


def run(out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return process_health_tokens(out_root=out_root, limit_segments=limit_segments)


if __name__ == "__main__":
    print(run())
