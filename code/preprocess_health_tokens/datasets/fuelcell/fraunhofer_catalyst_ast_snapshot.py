"""Fraunhofer catalyst AST records as dense physical HealthToken snapshots.

The archive contains uninterrupted ``pot_cycles`` acquisitions at successive
ageing checkpoints.  Each acquisition is split only at its native 10-cycle
boundaries: one snapshot therefore contains ten *real* AST potential cycles
(620 samples at 10 Hz), never a synthetic interpolation across checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
import zipfile

import numpy as np


DATASET_NAME = "fraunhofer_catalyst_ast_snapshot_fuelcell"
RAW_ROOT = Path(__file__).resolve().parents[3] / "data_phm" / "raw" / "FuelCell" / "fraunhofer_catalyst_ast"
CYCLES_PER_SNAPSHOT = 10
POINTS_PER_AST_CYCLE = 62


@dataclass
class Segment:
    file_path: Path
    segment_id: str
    sampling_rate: float
    metadata: dict[str, object] = field(default_factory=dict)


def _archive_key(path: Path) -> int:
    match = re.search(r"Cell_(\d+)", path.stem)
    return int(match.group(1)) if match else 10**9


def _ast_members(archive: Path) -> list[tuple[int, str]]:
    """Return potential-cycle acquisitions in their physical test order."""
    result: list[tuple[int, str]] = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            match = re.fullmatch(r"12-AST/(\d+)cycles/pot_cycles_(\d+)\.csv", name)
            if match:
                # A split acquisition (e.g. 20000cycles/pot_cycles_0,_1)
                # remains sequential in the same ageing checkpoint.
                result.append((int(match.group(1)) * 10 + int(match.group(2)), name))
    return sorted(result)


@lru_cache(maxsize=1)
def _load_archive(path_text: str) -> np.ndarray:
    """Load a single cell's AST trace once; callers process its snapshots in order."""
    import pandas as pd

    archive = Path(path_text)
    blocks: list[np.ndarray] = []
    with zipfile.ZipFile(archive) as zf:
        for _, member in _ast_members(archive):
            frame = pd.read_csv(zf.open(member), sep="\t")
            if frame.shape[1] < 3:
                raise RuntimeError(f"Expected time/voltage/current columns in {archive.name}::{member}")
            physical = frame.iloc[:, 1:3].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            if not np.isfinite(physical).all():
                raise RuntimeError(
                    f"Non-finite voltage/current would break native cycle alignment in "
                    f"{archive.name}::{member}"
                )
            # A few archived acquisitions end with a partial native cycle.
            # Drop only that member-local remainder.  Crucially, do this before
            # concatenation so it cannot shift the offsets of later ageing
            # checkpoints or be joined to the next acquisition.
            complete_points = (len(physical) // POINTS_PER_AST_CYCLE) * POINTS_PER_AST_CYCLE
            if complete_points == 0:
                raise RuntimeError(f"No complete native AST cycle in {archive.name}::{member}")
            physical = physical[:complete_points]
            voltage = physical[:, 0]
            current = physical[:, 1]
            blocks.append(np.stack([voltage, current, voltage * current]))
    if not blocks:
        raise RuntimeError(f"No readable AST potential records in {archive}")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def build_segments() -> list[Segment]:
    output: list[Segment] = []
    for file_index, archive in enumerate(sorted(RAW_ROOT.glob("Cell_*.zip"), key=_archive_key)):
        cell = archive.stem.replace("Cell_", "Cell")
        # Every AST record uses 62 points per native potential cycle.  The
        # archived raw files are exact multiples; incomplete tails are dropped
        # rather than padded or joined to a subsequent checkpoint.
        with zipfile.ZipFile(archive) as zf:
            member_info = [(key, name, zf.getinfo(name).file_size) for key, name in _ast_members(archive)]
        global_snapshot = 0
        point_offset = 0
        for member_index, (_, member, _) in enumerate(member_info):
            with zipfile.ZipFile(archive) as zf:
                raw_lines = sum(1 for _ in zf.open(member)) - 1
            complete_cycles = max(0, raw_lines // POINTS_PER_AST_CYCLE)
            complete_snapshots = complete_cycles // CYCLES_PER_SNAPSHOT
            for within_member in range(complete_snapshots):
                start = point_offset + within_member * CYCLES_PER_SNAPSHOT * POINTS_PER_AST_CYCLE
                stop = start + CYCLES_PER_SNAPSHOT * POINTS_PER_AST_CYCLE
                output.append(Segment(
                    archive,
                    f"fraunhofer__{cell}__ast_{global_snapshot:05d}",
                    10.0,
                    {
                        "group_id": cell,
                        "unit_id": cell,
                        # Match the bearing/battery metadata contract: one
                        # independent physical trajectory is one run. Each
                        # Fraunhofer cell is a separate AST experiment.
                        "run_id": cell,
                        "condition_id": "catalyst_AST_potential_cycles",
                        "file_id": archive.stem,
                        "filename": archive.name,
                        "source_relpath": archive.name + "::" + member,
                        "source_split": "fraunhofer_catalyst_ast",
                        "file_index": file_index,
                        "snapshot_index": global_snapshot,
                        "cycle_index": global_snapshot * CYCLES_PER_SNAPSHOT,
                        "member_index": member_index,
                        "start_index": start,
                        "stop_index": stop,
                    },
                ))
                global_snapshot += 1
            point_offset += complete_cycles * POINTS_PER_AST_CYCLE
    if not output:
        raise RuntimeError(f"No complete 10-cycle AST snapshots found under {RAW_ROOT}")
    return output


def read_signal(segment: Segment):
    data = _load_archive(str(segment.file_path))
    start, stop = int(segment.metadata["start_index"]), int(segment.metadata["stop_index"])
    return data[:, start:stop], 10.0
