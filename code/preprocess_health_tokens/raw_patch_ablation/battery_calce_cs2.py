from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from preprocess_health_tokens.common.battery_health_token_utils import sanitize_cycle
from preprocess_health_tokens.datasets.battery.calce_cs2_downstream_4cells import iter_cycles
from preprocess_health_tokens.raw_patch_ablation.common import (
    GLOBAL_POINTS,
    LOCAL_POINTS,
    limit_payload,
    load_reference,
    resample_1d,
    save_raw_npz,
)


VARIABLES = ("voltage_v", "current_a", "capacity_fraction", "elapsed_time_fraction", "temperature_c")


def _cycle_variables(cycle) -> Tuple[np.ndarray, np.ndarray]:
    cycle = sanitize_cycle(cycle)
    t = np.asarray(cycle.time_s, dtype=np.float64)
    v = np.asarray(cycle.voltage_v, dtype=np.float64)
    current = np.asarray(cycle.current_a, dtype=np.float64)
    q = np.asarray(cycle.capacity_ah, dtype=np.float64)
    q_fraction = q / max(float(q[-1]), 1e-12)
    t_fraction = t / max(float(t[-1]), 1e-12)
    temperature_available = cycle.temperature_c is not None and np.isfinite(cycle.temperature_c).sum() >= 2
    temp = np.asarray(cycle.temperature_c, dtype=np.float64) if temperature_available else np.zeros_like(t)
    values = np.stack((v, current, q_fraction, t_fraction, temp), axis=0)
    mask = np.asarray((True, True, True, True, temperature_available), dtype=bool)
    return values, mask


def build(
    *,
    raw_root: Path,
    reference: Path,
    output: Path,
    limit_samples: Optional[int] = None,
) -> Dict[str, object]:
    inherited = load_reference(reference)
    total = int(inherited["token_mask"].shape[0])
    count = total if limit_samples is None else min(total, int(limit_samples))
    inherited = limit_payload(inherited, count)
    token_mask = np.asarray(inherited["token_mask"], dtype=bool)
    starts = np.asarray(inherited["token_start_points"], dtype=np.int64)
    point_counts = np.asarray(inherited["token_point_count"], dtype=np.int64)
    units = np.asarray(inherited["sample_unit_id"]).astype(str)
    cycle_indices = np.asarray(inherited["sample_cycle_index"], dtype=np.int64)
    if token_mask.shape != (count, 1, 64) or starts.shape != token_mask.shape:
        raise ValueError(f"Unexpected CALCE reference contract: {token_mask.shape}, {starts.shape}")

    wanted = {(units[row], int(cycle_indices[row])): row for row in range(count)}
    if len(wanted) != count:
        raise ValueError("CALCE reference contains duplicate (cell, cycle) identities")
    local = np.zeros((count, 1, 64, LOCAL_POINTS), dtype=np.float32)
    global_ = np.zeros((count, 1, GLOBAL_POINTS), dtype=np.float32)
    found = np.zeros((count,), dtype=bool)

    for raw_cycle in iter_cycles(raw_root=raw_root):
        key = (str(raw_cycle.unit_id), int(raw_cycle.cycle_index))
        row = wanted.get(key)
        if row is None:
            continue
        values, variable_mask = _cycle_variables(raw_cycle)
        assert variable_mask[0], "CALCE voltage must be observed"
        global_[row, 0] = resample_1d(values[0], GLOBAL_POINTS)
        for token in np.flatnonzero(token_mask[row, 0]):
            start = int(starts[row, 0, token])
            stop = start + int(point_counts[row, 0, token])
            if start < 0 or stop <= start or stop > values.shape[1]:
                raise ValueError(f"Invalid CALCE patch row={row}, token={token}, [{start}:{stop}]")
            local[row, 0, token] = resample_1d(values[0, start:stop], LOCAL_POINTS)
        found[row] = True
        if int(found.sum()) % 100 == 0:
            print(f"[calce_cs2] {int(found.sum())}/{count}", flush=True)
        if found.all():
            break
    if not found.all():
        missing = [(units[i], int(cycle_indices[i])) for i in np.flatnonzero(~found)[:20]]
        raise ValueError(f"Could not reproduce {int((~found).sum())} CALCE cycles; first missing={missing}")

    return save_raw_npz(
        output,
        reference=reference,
        inherited=inherited,
        x_raw_local=local,
        x_raw_global=global_,
        raw_signal_name="voltage_v",
        dataset="calce_cs2_downstream_battery",
        raw_roots=(raw_root,),
        extra_metadata={
            "physical_channels": ["battery_cycle"],
            "window_alignment": "reference 5%-capacity windows at 2.5%-capacity stride; native point indices copied exactly",
            "signal_selection": "voltage_v is the physical degradation waveform; current/capacity/time define the inherited handcrafted window boundaries but are not extra model dimensions",
        },
    )
