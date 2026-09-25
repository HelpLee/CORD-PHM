from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from preprocess_health_tokens.common.battery_health_token_utils import (
    BatteryCycle,
    as_float_1d,
    process_cycle_dataset,
    select_discharge_samples,
    standardize_discharge_current,
)


def continuous_discharge_span_mask(
    current_a: object,
    *,
    activity_mask: object = None,
    threshold_a: float = 0.005,
    min_active_points: int = 8,
) -> np.ndarray:
    """Keep the native interval from discharge start through discharge end.

    The active points identify the boundaries, while intervening rests,
    current pulses, and regenerative samples remain part of the physical
    discharge segment.  Charge samples before/after that interval are removed.
    """

    current = as_float_1d(current_a)
    if activity_mask is None:
        active = np.isfinite(current) & (current < -abs(float(threshold_a)))
    else:
        active = np.asarray(activity_mask, dtype=bool).reshape(-1)
        if active.size != current.size:
            raise ValueError(
                f"activity_mask length {active.size} does not match current length {current.size}"
            )
        active &= np.isfinite(current)
    indices = np.flatnonzero(active)
    span = np.zeros((current.size,), dtype=bool)
    if indices.size >= int(min_active_points):
        span[int(indices[0]) : int(indices[-1]) + 1] = True
    return span


def complete_discharge_cycle(
    *,
    dataset: str,
    unit_id: str,
    cycle_index: int,
    time_s: object,
    voltage_v: object,
    current_a: object,
    capacity_ah: object = None,
    temperature_c: object = None,
    already_discharge: bool = False,
    discharge_mask: object = None,
    **metadata,
) -> BatteryCycle:
    # A source file/record is the acquisition run for datasets without a
    # separate run hierarchy.  This preserves a non-empty provenance key while
    # leaving unit_id as the physical cell identifier.
    metadata.setdefault("run_id", metadata.get("file_id") or unit_id)
    current = as_float_1d(current_a)
    if already_discharge:
        current = standardize_discharge_current(current)
        explicit_mask = np.ones((current.size,), dtype=bool)
    else:
        explicit_mask = discharge_mask
    t, v, current, q, temp = select_discharge_samples(
        time_s=time_s,
        voltage_v=voltage_v,
        current_a=current,
        capacity_ah=capacity_ah,
        temperature_c=temperature_c,
        discharge_mask=explicit_mask,
    )
    return BatteryCycle(
        dataset=dataset,
        unit_id=unit_id,
        cycle_index=int(cycle_index),
        time_s=t,
        voltage_v=v,
        current_a=current,
        capacity_ah=q if q.size else None,
        temperature_c=temp,
        **metadata,
    )


def run_adapter(
    *,
    cfg,
    out_root: Path,
    dataset: str,
    cycles: Iterable[BatteryCycle],
    output_name: Optional[str] = None,
    notes: Optional[Dict[str, object]] = None,
):
    return process_cycle_dataset(
        cfg=cfg,
        out_root=out_root,
        dataset=dataset,
        cycles=cycles,
        output_name=output_name,
        extra_meta=notes,
    )
