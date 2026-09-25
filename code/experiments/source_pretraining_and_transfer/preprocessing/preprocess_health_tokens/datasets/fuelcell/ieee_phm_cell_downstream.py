"""IEEE PHM 2014 PEMFC data for strict cross-stack downstream RUL.

FC1 and FC2 are two independent five-cell stacks under constant-current and
dynamic-ripple ageing respectively.  Each dense four-hour operating segment is
one physical snapshot.  For every stack, the five measured cell voltages are
kept as five cell-level trajectories, but ``sample_group_id`` remains the stack
ID so downstream splitting must hold out the *entire stack*.

Each cell snapshot has three physically aligned channels:
cell voltage, stack current, and cell electrical power.  Local HealthTokens are
then extracted by the common 26-feature implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np


DATASET_NAME = "ieee_phm_cell_downstream_fuelcell"
RAW_ROOT = Path(__file__).resolve().parents[3] / "data_phm" / "raw" / "FuelCell" / "ieee_phm_2014" / "IEEE 2014 Data Challenge Data"
SNAPSHOT_HOURS = 4.0
MAX_TIME_GAP_HOURS = 0.05


@dataclass
class Segment:
    file_path: Path
    segment_id: str
    sampling_rate: float
    metadata: dict[str, object] = field(default_factory=dict)


@lru_cache(maxsize=2)
def _frame(path_text: str) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(path_text, encoding="latin1").iloc[:, :25]
    frame.columns = [
        "time", "u1", "u2", "u3", "u4", "u5", "utot", "j", "i",
        "tinh2", "touth2", "tinair", "toutair", "tinwat", "toutwat",
        "pinair", "poutair", "pouth2", "pinh2", "dinh2", "douth2",
        "dinair", "doutair", "dwat", "rh",
    ]
    numeric = frame[["time", "u1", "u2", "u3", "u4", "u5", "i"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    numeric = numeric[numeric["i"].between(60.0, 80.0)]
    return numeric.to_numpy(dtype=np.float64)


def _intervals(data: np.ndarray) -> list[tuple[int, int]]:
    if len(data) < 2:
        return []
    time = data[:, 0]
    # Files may contain both acquisition gaps and clock resets.  Either event
    # starts a new continuous physical operating segment.
    delta = np.diff(time)
    gaps = np.flatnonzero((delta > MAX_TIME_GAP_HOURS) | (delta <= 0.0)) + 1
    edges = np.r_[0, gaps, len(data)]
    result: list[tuple[int, int]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        start = int(lo)
        while start < hi:
            # Search only inside this monotonic continuous segment.  Searching
            # the whole file is invalid after a clock reset.
            local = time[start:hi]
            stop = start + int(np.searchsorted(local, time[start] + SNAPSHOT_HOURS, side="left"))
            if stop >= hi:
                break
            current = data[start:stop, 6]
            if stop - start >= 256 and np.std(current) / max(abs(np.mean(current)), 1e-6) <= 0.10:
                result.append((start, stop))
            start = stop
    return result


def build_segments() -> list[Segment]:
    output: list[Segment] = []
    folders = [
        ("FC1", "FC1_Without_Ripples", "constant_70A"),
        ("FC2", "Full_FC2_With_Ripples", "dynamic_70A_plusminus_7A_5Hz"),
    ]
    for stack, folder, condition in folders:
        stack_snapshot = 0
        for file_index, path in enumerate(sorted((RAW_ROOT / folder).glob(f"{stack}_Ageing_part*.csv"))):
            data = _frame(str(path))
            for start, stop in _intervals(data):
                dt_h = float(np.median(np.diff(data[start:stop, 0])))
                fs = 1.0 / max(dt_h * 3600.0, 1e-6)
                for cell_index in range(5):
                    cell = f"{stack}_Cell{cell_index + 1}"
                    output.append(Segment(
                        path,
                        f"ieee__{cell}__snapshot_{stack_snapshot:04d}",
                        fs,
                        {
                            "group_id": stack,
                            "unit_id": cell,
                            "run_id": stack,
                            "condition_id": condition,
                            "file_id": path.stem,
                            "filename": path.name,
                            "source_relpath": str(path.relative_to(RAW_ROOT)),
                            "source_split": "downstream_cross_stack_only",
                            "file_index": file_index,
                            "snapshot_index": stack_snapshot,
                            "cycle_index": stack_snapshot,
                            "cell_index": cell_index,
                            "snapshot_start_hours": float(data[start, 0]),
                            "snapshot_end_hours": float(data[stop - 1, 0]),
                            "start_index": start,
                            "stop_index": stop,
                        },
                    ))
                stack_snapshot += 1
    if not output:
        raise RuntimeError(f"No IEEE cell-level downstream snapshots found under {RAW_ROOT}")
    return output


def read_signal(segment: Segment):
    data = _frame(str(segment.file_path))
    start, stop = int(segment.metadata["start_index"]), int(segment.metadata["stop_index"])
    cell_col = 1 + int(segment.metadata["cell_index"])
    voltage = data[start:stop, cell_col]
    current = data[start:stop, 6]
    return np.stack([voltage, current, voltage * current]).astype(np.float32), float(segment.sampling_rate)
