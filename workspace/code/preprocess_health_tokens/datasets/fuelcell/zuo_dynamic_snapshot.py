"""Zuo et al. dynamic PEMFC durability data as physical FC-DLC snapshots.

Each contiguous non-zero-current FC-DLC interval is one snapshot.  Its native
1-Hz sensor records are never mixed with another cycle; main.py subsequently
creates local 26-feature tokens within that snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re

import numpy as np


DATASET_NAME = "zuo_dynamic_snapshot_fuelcell"
RAW_ROOT = Path(__file__).resolve().parents[3] / "data_phm" / "raw" / "FuelCell" / "Durability_test_dataset"


@dataclass
class Segment:
    file_path: Path
    segment_id: str
    sampling_rate: float
    metadata: dict[str, object] = field(default_factory=dict)


def _sort_key(path: Path) -> int:
    match = re.search(r"(\d+)_h", path.stem)
    return int(match.group(1)) if match else 10**9


@lru_cache(maxsize=1)
def _frame(path_text: str) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(path_text, skiprows=2)
    columns = [c for c in frame.columns if c not in {"1", "Elapsed time"}]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    # The first column is current.  Drop rows that cannot represent a sensor
    # snapshot rather than interpolating across acquisition gaps.
    return values[np.isfinite(values[:, 0])]


def _active_intervals(current: np.ndarray) -> list[tuple[int, int]]:
    active = np.asarray(current > 0.1, dtype=np.int8)
    edges = np.diff(np.pad(active, (1, 1)))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    # A complete FC-DLC lasts about 1145 samples.  256 accepts intact cycles
    # while rejecting short turn-on/off fragments at file boundaries.
    return [(int(s), int(e)) for s, e in zip(starts, stops) if e - s >= 256]


def build_segments() -> list[Segment]:
    output: list[Segment] = []
    for file_index, path in enumerate(sorted(RAW_ROOT.glob("*_h.csv"), key=_sort_key)):
        data = _frame(str(path))
        for cycle_index, (start, stop) in enumerate(_active_intervals(data[:, 0])):
            output.append(Segment(
                path, f"zuo__{path.stem}__fcdlc_{cycle_index:03d}", 1.0,
                {"group_id": "zuo_dynamic_durability", "unit_id": "Zuo_PEMFC_01",
                 "run_id": path.stem, "condition_id": "FC_DLC", "file_id": path.stem,
                 "filename": path.name, "source_relpath": path.name,
                 "source_split": "durability_test_dataset", "file_index": file_index,
                 "snapshot_index": cycle_index, "start_index": start, "stop_index": stop},
            ))
    if not output:
        raise RuntimeError(f"No complete FC-DLC snapshots found under {RAW_ROOT}")
    return output


def read_signal(segment: Segment):
    data = _frame(str(segment.file_path))
    start, stop = int(segment.metadata["start_index"]), int(segment.metadata["stop_index"])
    # The three physical quantities common to all strict PEMFC sources are
    # current, voltage, and power.  Keeping this channel contract fixed avoids
    # treating a source-specific auxiliary sensor as a different modality.
    return data[start:stop, :3].T.astype(np.float32, copy=False), 1.0
