from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from preprocess_health_tokens.common.battery_health_token_utils import BatteryCycle, detect_project_root
from preprocess_health_tokens.datasets._archive_battery_cycle_trajectory_20260824.xjtu_battery import (
    _arr_any,
    _as_list,
    _battery_root,
    _loadmat,
)
from preprocess_health_tokens.datasets.battery._adapter_utils import complete_discharge_cycle, run_adapter


DATASET_NAME = "xjtu_battery"
PROJECT_ROOT = detect_project_root()
RAW_ROOT = PROJECT_ROOT / "data_phm" / "raw" / "Battery" / "XJTU_battery"
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens" / "battery"


def _discharge_segment_mask(current: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Locate the discharge phase inside one XJTU charge+discharge record.

    XJTU ``capacity_Ah`` resets at protocol boundaries. In random-walk batches,
    filtering only negative-current points can retain a pulse before the real
    discharge reset and create an artificial jump from u=0 to u~=0.6. A reset
    followed by substantial negative current and increasing capacity marks the
    real discharge start; the following reset marks its end.
    """

    current = np.asarray(current, dtype=np.float64).reshape(-1)
    capacity = np.asarray(capacity, dtype=np.float64).reshape(-1)
    n = min(current.size, capacity.size)
    mask = np.zeros((n,), dtype=bool)
    if n == 0:
        return mask

    finite_q = capacity[:n][np.isfinite(capacity[:n])]
    q_scale = float(np.nanmax(finite_q)) if finite_q.size else 0.0
    reset_threshold = max(0.05, 0.20 * q_scale)
    resets = np.flatnonzero(np.diff(capacity[:n]) < -reset_threshold) + 1

    valid_starts = []
    for reset_index, start in enumerate(resets):
        stop = int(resets[reset_index + 1]) if reset_index + 1 < resets.size else n
        post_current = current[start:stop]
        post_capacity = capacity[start:stop]
        finite_capacity = post_capacity[np.isfinite(post_capacity)]
        capacity_span = (
            float(np.nanmax(finite_capacity) - np.nanmin(finite_capacity))
            if finite_capacity.size
            else 0.0
        )
        if int(np.sum(post_current < -0.005)) >= 8 and capacity_span > 0.05:
            valid_starts.append(int(start))

    if valid_starts:
        start = valid_starts[-1]
    else:
        negative = np.flatnonzero(np.isfinite(current[:n]) & (current[:n] < -0.005))
        if negative.size == 0:
            return mask
        start = int(negative[0])

    following_resets = resets[resets > start]
    stop = int(following_resets[0]) if following_resets.size else n
    if stop - start < 8:
        return mask
    mask[start:stop] = True
    return mask


def iter_cycles(raw_root: Path = RAW_ROOT, limit_files: Optional[int] = None) -> Iterator[BatteryCycle]:
    files = sorted(p for p in raw_root.rglob("*.mat") if p.name != "Temperature_Compensation_Data.mat")
    if limit_files is not None:
        files = files[: int(limit_files)]
    for file_index, path in enumerate(files):
        if path.parent.name == 'Batch-5':
            print(f'[EXCLUDE] {path.name}: Batch-5 begins with 20 partial random-walk discharges; no initial full-capacity reference',flush=True)
            continue
        root = _battery_root(_loadmat(path))
        unit_id = f"{path.parent.name}_{path.stem}"
        for cycle_index, record in enumerate(_as_list(root.get("data", [])), start=1):
            if not isinstance(record, dict):
                continue
            t = _arr_any(record, ["relative_time_min", "relative_time", "time_min", "time"]) * 60.0
            v = _arr_any(record, ["voltage_V", "Voltage_V", "voltage", "Voltage", "V"])
            current = _arr_any(record, ["current_A", "Current_A", "current", "Current", "I"])
            q = _arr_any(record, ["capacity_Ah", "Capacity_Ah", "capacity", "Capacity", "Q", "Qd"])
            temperature = _arr_any(
                record,
                ["temperature_C", "Temperature_C", "temperature", "Temperature", "T"],
            )
            n = min(t.size, v.size, current.size)
            if n < 8:
                continue
            discharge_mask = (
                _discharge_segment_mask(current[:n], q[:n])
                if q.size >= n
                else np.isfinite(current[:n]) & (current[:n] < -0.005)
            )
            cycle = complete_discharge_cycle(
                dataset=DATASET_NAME,
                unit_id=unit_id,
                cycle_index=cycle_index,
                time_s=t[:n],
                voltage_v=v[:n],
                current_a=current[:n],
                # XJTU capacity_Ah resets at random-profile substeps. It is
                # used above only to locate protocol boundaries; the common
                # tokenizer reconstructs monotone Q_d(t) by integrating the
                # native discharge current over the selected segment.
                capacity_ah=None,
                temperature_c=temperature[:n] if temperature.size >= n else None,
                discharge_mask=discharge_mask,
                group_id=path.parent.name,
                condition_id=str(record.get("description", path.parent.name)),
                segment_id=f"{unit_id}_cycle_{cycle_index}",
                file_id=path.stem,
                filename=path.name,
                source_relpath=str(path.relative_to(raw_root)),
                file_index=file_index,
            )
            if cycle.voltage_v.size < 8 or cycle.voltage_v[-1] > 2.55:
                continue  # Retain full-capacity discharges, not timed/partial-DoD operations.
            active = np.abs(cycle.current_a[cycle.current_a < -.005])
            if path.parent.name == 'Batch-3' and cycle_index > 1:
                if not active.size or not np.isclose(np.median(active),2.,atol=.1):
                    continue  # Fixed 1C comparison in the varying-rate batch.
            yield cycle


def process_health_tokens(cfg=None, *, out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return run_adapter(
        cfg=cfg,
        out_root=out_root,
        dataset=DATASET_NAME,
        cycles=iter_cycles(limit_files=limit_segments),
        notes={
            "raw_root": str(RAW_ROOT),
            "selection": "included: per-cycle raw records expose time/V/I/Q/T; source capacity resets define the complete discharge boundary, then monotone Q_d(t) is reconstructed from native time/current so random-walk substep resets cannot corrupt the capacity axis; capacity-only fallback cycles are rejected",
            "excluded_source_cohorts": {'Batch-5': 'no initial full-capacity calibration in first 20 partial cycles'},
            "capacity_comparability": 'Discharge cutoff <=2.55 V; Batch-3 uses 1C observations plus initial reference; original cycle indices preserved',
        },
    )


def run(out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return process_health_tokens(out_root=out_root, limit_segments=limit_segments)


if __name__ == "__main__":
    print(run())
