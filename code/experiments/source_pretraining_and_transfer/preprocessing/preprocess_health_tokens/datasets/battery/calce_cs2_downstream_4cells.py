from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from preprocess_health_tokens.common.battery_health_token_utils import BatteryCycle, detect_project_root
from preprocess_health_tokens.datasets._archive_battery_cycle_trajectory_20260824.calce_cs2_battery import (
    _first_existing_numeric,
    _numeric,
    _read_excel_sheets,
    sorted_txt_logs,
    sorted_workbooks,
)
from preprocess_health_tokens.datasets.battery._adapter_utils import (
    complete_discharge_cycle,
    continuous_discharge_span_mask,
    run_adapter,
)


DATASET_NAME = "calce_cs2_downstream_battery"
TARGET_CELLS: Sequence[str] = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")
PROJECT_ROOT = detect_project_root()
RAW_ROOT = PROJECT_ROOT / "data_phm" / "raw" / "Battery" / "CALCE_CS2"
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens" / "battery"


def _excel_groups(path: Path):
    import pandas as pd

    frame = _read_excel_sheets(path)
    required = {"Cycle_Index", "Test_Time(s)", "Current(A)", "Voltage(V)"}
    if not required.issubset(frame.columns):
        return
    frame["Cycle_Index"] = pd.to_numeric(frame["Cycle_Index"], errors="coerce")
    frame = frame[frame["Cycle_Index"].notna()].copy()
    for local_cycle, group in frame.groupby("Cycle_Index", sort=True):
        yield int(local_cycle), group


def _txt_groups(path: Path):
    import pandas as pd

    frame = pd.read_csv(path, sep="\t", engine="python")
    frame = frame.rename(columns={column: str(column).strip() for column in frame.columns})
    required = {"Time", "Pgm cycle", "mV", "mA"}
    if not required.issubset(frame.columns):
        return
    frame["Pgm cycle"] = pd.to_numeric(frame["Pgm cycle"], errors="coerce")
    frame = frame[frame["Pgm cycle"].notna()].copy()
    for local_cycle, group in frame.groupby("Pgm cycle", sort=True):
        yield int(local_cycle), group


def iter_cycles(
    raw_root: Path = RAW_ROOT,
    limit_files: Optional[int] = None,
    target_cells: Sequence[str] = TARGET_CELLS,
) -> Iterator[BatteryCycle]:
    cells = list(target_cells)
    if limit_files is not None:
        cells = cells[: int(limit_files)]
    file_index = 0
    for cell_id in cells:
        cell_dir = raw_root / cell_id
        if not cell_dir.exists():
            raise FileNotFoundError(f"Missing CALCE downstream cell directory: {cell_dir}")
        global_cycle = 0
        source_files = [*sorted_workbooks(cell_dir), *sorted_txt_logs(cell_dir)]
        for path in source_files:
            if path.suffix.lower() in {".xlsx", ".xls"}:
                groups = _excel_groups(path)
                source_type = "Arbin_channel"
            else:
                groups = _txt_groups(path)
                source_type = "Cadex_txt"
            if groups is None:
                continue
            for local_cycle, frame in groups:
                if source_type == "Arbin_channel":
                    time_s = _numeric(frame, "Test_Time(s)")
                    voltage = _numeric(frame, "Voltage(V)")
                    current = _numeric(frame, "Current(A)")
                    capacity = _numeric(frame, "Discharge_Capacity(Ah)")
                    temperature = _first_existing_numeric(
                        frame,
                        ("Aux_Temperature_1(C)", "Temperature(C)", "Temp(C)", "Temperature"),
                    )
                else:
                    time_s = _numeric(frame, "Time")
                    voltage = _numeric(frame, "mV") / 1000.0
                    current = _numeric(frame, "mA") / 1000.0
                    capacity = np.asarray([], dtype=np.float64)
                    temperature = np.asarray([], dtype=np.float64)
                mask = continuous_discharge_span_mask(current)
                if int(mask.sum()) < 8:
                    continue
                global_cycle += 1
                yield complete_discharge_cycle(
                    dataset=DATASET_NAME,
                    unit_id=cell_id,
                    cycle_index=global_cycle,
                    time_s=time_s,
                    voltage_v=voltage,
                    current_a=current,
                    capacity_ah=capacity if capacity.size == current.size else None,
                    temperature_c=temperature if temperature.size == current.size else None,
                    discharge_mask=mask,
                    group_id="CALCE_CS2",
                    condition_id=cell_id,
                    segment_id=f"{cell_id}_cycle_{global_cycle}",
                    file_id=path.stem,
                    filename=path.name,
                    source_relpath=str(path.relative_to(raw_root)),
                    source_split=source_type,
                    file_index=file_index,
                    metadata={"local_cycle_index": local_cycle},
                )
            file_index += 1


def process_health_tokens(cfg=None, *, out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return run_adapter(
        cfg=cfg,
        out_root=out_root,
        dataset=DATASET_NAME,
        cycles=iter_cycles(limit_files=limit_segments),
        output_name="calce_cs2_downstream_battery_health_tokens.npz",
        notes={
            "raw_root": str(RAW_ROOT),
            "role": "downstream_only",
            "target_cells": list(TARGET_CELLS),
            "selection": "only CS2_35/36/37/38; statistics-only sheets are rejected because they lack native discharge samples",
        },
    )


def run(out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return process_health_tokens(out_root=out_root, limit_segments=limit_segments)


if __name__ == "__main__":
    print(run())
