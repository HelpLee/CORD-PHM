from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from preprocess_health_tokens.common.battery_health_token_utils import BatteryCycle, detect_project_root
from preprocess_health_tokens.datasets._archive_battery_cycle_trajectory_20260824.isu_ilcc_battery import (
    _cell_group,
    _condition_id,
    _object_list,
    _read_json_dict,
    _read_valid_cells,
    _safe_float_array,
    _zip_json_name_map,
)
from preprocess_health_tokens.datasets.battery._adapter_utils import complete_discharge_cycle, run_adapter


DATASET_NAME = "isu_ilcc_battery"
PROJECT_ROOT = detect_project_root()
RAW_ROOT = PROJECT_ROOT / "data_phm" / "raw" / "Battery" / "ISU-ILCC"
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens" / "battery"


def _time_seconds(values):
    """Require physical source timestamps; never substitute sample indices."""
    values = np.asarray(values).reshape(-1)
    if not values.size:
        return np.asarray([], dtype=float)
    if np.issubdtype(values.dtype, np.number):
        seconds = values.astype(float)
    else:
        dt = values.astype('datetime64[ns]')
        if np.isnat(dt).any():
            raise ValueError('Missing native RPT timestamp')
        seconds = np.asarray((dt-dt[0])/np.timedelta64(1,'s'),dtype=float)
    if not np.isfinite(seconds).all():
        raise ValueError('Invalid native RPT time')
    return seconds-seconds[0]


def iter_cycles(raw_root: Path = RAW_ROOT, limit_files: Optional[int] = None) -> Iterator[BatteryCycle]:
    cells = _read_valid_cells(raw_root)
    if limit_files is not None:
        cells = cells[: int(limit_files)]
    archive = raw_root / "RPT_json.zip"
    name_map = _zip_json_name_map(archive)
    for file_index, cell_id in enumerate(cells):
        member = name_map.get(cell_id)
        if not member:
            raise FileNotFoundError(f'Missing published valid cell {cell_id} in {archive}')
        data = _read_json_dict(archive, member)
        qv = data.get("QV_discharge_C_5", {})
        if not isinstance(qv, dict):
            raise ValueError(f'{cell_id}: missing native C/5 RPT waveform')
        q_list = _object_list(qv.get("Q", []))
        v_list = _object_list(qv.get("V", []))
        i_list = _object_list(qv.get("I", []))
        t_list = _object_list(qv.get("t", []))
        temp_list = _object_list(qv.get("T", qv.get("temperature", [])))
        lengths = [len(q_list),len(v_list),len(i_list),len(t_list)]
        if len(set(lengths)) != 1 or not lengths[0]:
            raise ValueError(f'{cell_id}: unaligned C/5 RPT record arrays: {lengths}')
        count = lengths[0]
        origin = np.asarray(t_list[0]).reshape(-1)[0]
        origin = np.datetime64(str(origin), 'ns')
        for index in range(count):
            q = _safe_float_array(q_list[index])
            v = _safe_float_array(v_list[index])
            current = _safe_float_array(i_list[index])
            t = _time_seconds(t_list[index])
            temp = _safe_float_array(temp_list[index]) if index < len(temp_list) else np.asarray([])
            if len({q.size,v.size,current.size,t.size}) != 1:
                raise ValueError(f'{cell_id} RPT {index+1}: unaligned native channels')
            n = q.size
            if n < 8:
                print(f'[WARN] {cell_id} RPT {index+1}: only {n} native samples; excluded, not interpolated', flush=True)
                continue
            yield complete_discharge_cycle(
                dataset=DATASET_NAME,
                unit_id=cell_id,
                cycle_index=index + 1,
                time_s=t[:n],
                voltage_v=v[:n],
                current_a=current[:n],
                capacity_ah=q[:n],
                temperature_c=temp[:n] if temp.size >= n else None,
                already_discharge=True,
                group_id=f"G{_cell_group(cell_id)}",
                condition_id=_condition_id(cell_id),
                segment_id=f"{cell_id}_cycle_{index + 1}",
                file_id=cell_id,
                filename=Path(member).name,
                source_relpath=member,
                source_split="RPT_json.zip/C_5",
                file_index=file_index,
                metadata={'elapsed_days': float((np.datetime64(str(np.asarray(t_list[index]).reshape(-1)[0]), 'ns')-origin)/np.timedelta64(1,'D'))},
            )


def process_health_tokens(cfg=None, *, out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return run_adapter(
        cfg=cfg,
        out_root=out_root,
        dataset=DATASET_NAME,
        cycles=iter_cycles(limit_files=limit_segments),
        notes={
            "raw_root": str(RAW_ROOT),
            "selection": "Native C/5 RPT discharge t/V/I/Q only; partial-DoD cycling cannot measure full-capacity EOL. Sampling is periodic, not dense in aging time.",
            "rul_unit": "elapsed_days (native C/5 RPT timestamps; endpoint interval-censored between RPTs)",
        },
    )


def run(out_root: Path = OUT_ROOT, limit_segments: Optional[int] = None):
    return process_health_tokens(out_root=out_root, limit_segments=limit_segments)


if __name__ == "__main__":
    print(run())
