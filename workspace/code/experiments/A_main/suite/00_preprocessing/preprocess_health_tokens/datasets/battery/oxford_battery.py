from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from preprocess_health_tokens.common.battery_health_token_utils import BatteryCycle, detect_project_root
from preprocess_health_tokens.datasets._archive_battery_cycle_trajectory_20260824.oxford_battery import (
    _extract_cycle_index,
    _loadmat,
)
from preprocess_health_tokens.datasets.battery._adapter_utils import complete_discharge_cycle, run_adapter


DATASET_NAME = "oxford_battery"
PROJECT_ROOT = detect_project_root()
RAW_ROOT = PROJECT_ROOT / "data_phm" / "raw" / "Battery" / "Oxford_Battery"
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens" / "battery"


def _arr(data, name: str) -> np.ndarray:
    return np.asarray(data.get(name, []), dtype=np.float64).reshape(-1)


def iter_cycles(raw_root: Path = RAW_ROOT, limit_files: Optional[int] = None) -> Iterator[BatteryCycle]:
    mats = sorted(raw_root.glob("*.mat"))
    if limit_files is not None:
        mats = mats[: int(limit_files)]
    for file_index, path in enumerate(mats):
        loaded = _loadmat(path)
        for cell_name in sorted((k for k in loaded if str(k).startswith("Cell")), key=_extract_cycle_index):
            cell = loaded[cell_name]
            if not isinstance(cell, dict):
                continue
            for cycle_name in sorted((k for k in cell if str(k).startswith("cyc")), key=_extract_cycle_index):
                record = cell[cycle_name]
                discharge = record.get("C1dc", {}) if isinstance(record, dict) else {}
                if not isinstance(discharge, dict):
                    continue
                t = _arr(discharge, "t") * 86400.0
                v = _arr(discharge, "v")
                q = _arr(discharge, "q") / 1000.0
                temperature = _arr(discharge, "T")
                n = min(t.size, q.size)
                if n < 2:
                    continue
                q_oriented = np.abs(q[:n] - q[0])
                dq_dt = np.gradient(q_oriented, t[:n], edge_order=1)
                current = -np.maximum(dq_dt * 3600.0, 0.0)
                yield complete_discharge_cycle(
                    dataset=DATASET_NAME,
                    unit_id=str(cell_name),
                    cycle_index=_extract_cycle_index(cycle_name),
                    time_s=t[:n],
                    voltage_v=v[:n],
                    current_a=current,
                    capacity_ah=q_oriented,
                    temperature_c=temperature[:n] if temperature.size >= n else None,
                    already_discharge=True,
                    group_id=path.stem,
                    condition_id="40C_urban_artemis_characterization",
                    segment_id=f"{cell_name}_{cycle_name}_C1dc",
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
            "selection": "included: C1dc contains native t/V/Q/T characterization traces; I is physically derived as -dQ/dt",
        },
    )


def run(out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return process_health_tokens(out_root=out_root, limit_segments=limit_segments)


if __name__ == "__main__":
    print(run())

