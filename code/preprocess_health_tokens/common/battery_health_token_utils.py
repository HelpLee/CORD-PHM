"""Unified battery preprocessing utilities.

This module contains the complete battery preprocessing implementation used by
the project: legacy cycle-trajectory helpers, the current discharge-snapshot
HealthToken pipeline, and the CALCE-CS2 baseline cache helpers.  It consolidates
the previous battery_health_token_utils.py, battery_snapshot_utils.py, and
battery_baseline_utils.py without changing their algorithms or output contracts.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Legacy cycle-trajectory utilities (preserved)
# -----------------------------------------------------------------------------

EPS = 1e-12


DEFAULT_ROBUST_CLIP_ABS = 20.0


BATTERY_FEATURE_NAMES: Tuple[str, ...] = (
    "discharge_capacity_clean",
    "charge_capacity",
    "soh_clean",
    "capacity_delta_prev",
    "capacity_slope_5",
    "capacity_rolling_std_5",
    "charge_duration",
    "cc_charge_time",
    "cv_charge_time",
    "time_to_4p0v_charge",
    "time_to_4p1v_charge",
    "time_to_4p2v_charge",
    "duration_3p8_4p0v_charge",
    "duration_4p0_4p1v_charge",
    "duration_4p1_4p2v_charge",
    "discharge_duration",
    "voltage_drop_discharge",
    "voltage_slope_discharge",
    "voltage_area_discharge",
    "mean_discharge_current",
    "mean_charge_current",
    "discharge_energy",
    "charge_energy",
    "internal_resistance",
    "resistance_slope_5",
    "cycle_index_norm",
)


BASE_FEATURE_KEYS: Tuple[str, ...] = (
    "discharge_capacity",
    "charge_capacity",
    "discharge_energy",
    "charge_energy",
    "discharge_duration",
    "charge_duration",
    "mean_discharge_voltage",
    "std_discharge_voltage",
    "start_discharge_voltage",
    "end_discharge_voltage",
    "min_discharge_voltage",
    "max_discharge_voltage",
    "voltage_slope",
    "voltage_area",
    "mean_discharge_current",
    "std_discharge_current",
    "mean_charge_current",
    "std_charge_current",
    "internal_resistance",
    "impedance_or_resistance_proxy",
    "temperature_mean",
    "temperature_max",
    "temperature_rise",
    "rest_before_seconds",
    "poly_a2",
    "poly_a3",
    "poly_a4",
    "poly_a5",
    "voltage_plateau_fraction",
)


STANDARD_SAMPLE_KEYS: Tuple[str, ...] = (
    "sample_dataset",
    "sample_group_id",
    "sample_unit_id",
    "sample_run_id",
    "sample_condition_id",
    "sample_segment_id",
    "sample_file_id",
    "sample_filename",
    "sample_source_relpath",
    "sample_source_split",
    "sample_file_index",
    "sample_snapshot_index",
    "sample_cycle_index",
    "sample_chunk_id",
)


@dataclass
class BatteryTrajectory:
    unit_id: str
    group_id: str
    condition_id: str
    feature_table: np.ndarray
    cycle_indices: np.ndarray
    source_relpath: str
    source_split: str = "raw_root"
    file_id: str = ""
    filename: str = ""
    file_index: int = -1
    meta: Optional[Dict[str, object]] = None


def as_real_array(values: object, *, complex_mode: str = "real") -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if np.iscomplexobj(arr):
        if complex_mode == "abs":
            arr = np.abs(arr)
        else:
            arr = np.real(arr)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return arr


def finite_array(values: object, *, complex_mode: str = "real") -> np.ndarray:
    arr = as_real_array(values, complex_mode=complex_mode)
    return arr[np.isfinite(arr)]


def finite_or_nan(values: object, *, complex_mode: str = "real") -> np.ndarray:
    arr = as_real_array(values, complex_mode=complex_mode)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def first_finite(values: object, default: float = np.nan) -> float:
    arr = finite_array(values)
    return float(arr[0]) if arr.size else float(default)


def last_finite(values: object, default: float = np.nan) -> float:
    arr = finite_array(values)
    return float(arr[-1]) if arr.size else float(default)


def mean_finite(values: object, default: float = np.nan) -> float:
    arr = finite_array(values)
    return float(np.mean(arr)) if arr.size else float(default)


def std_finite(values: object, default: float = np.nan) -> float:
    arr = finite_array(values)
    return float(np.std(arr)) if arr.size else float(default)


def min_finite(values: object, default: float = np.nan) -> float:
    arr = finite_array(values)
    return float(np.min(arr)) if arr.size else float(default)


def max_finite(values: object, default: float = np.nan) -> float:
    arr = finite_array(values)
    return float(np.max(arr)) if arr.size else float(default)


def median_nonzero(values: object, default: float = np.nan) -> float:
    arr = finite_array(values)
    arr = arr[np.abs(arr) > EPS]
    return float(np.median(arr)) if arr.size else float(default)


def duration_from_time(time_s: object, mask: Optional[np.ndarray] = None) -> float:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if mask is not None and t.size == mask.size:
        t = t[mask]
    t = t[np.isfinite(t)]
    if t.size == 0:
        return float("nan")
    if t.size == 1:
        return float(max(t[0], 0.0))
    return float(max(t[-1] - t[0], 0.0))


def integrate_abs_current_capacity(time_s: object, current_a: object, mask: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    i = np.asarray(current_a, dtype=np.float64).reshape(-1)
    if t.size != i.size or t.size != mask.size:
        return float("nan")
    t = t[mask]
    i = i[mask]
    finite = np.isfinite(t) & np.isfinite(i)
    t = t[finite]
    i = i[finite]
    if t.size < 2:
        return float("nan")
    return float(np.trapz(np.abs(i), t) / 3600.0)


def integrate_abs_power_energy(time_s: object, voltage_v: object, current_a: object, mask: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    v = np.asarray(voltage_v, dtype=np.float64).reshape(-1)
    i = np.asarray(current_a, dtype=np.float64).reshape(-1)
    if t.size != v.size or t.size != i.size or t.size != mask.size:
        return float("nan")
    t = t[mask]
    v = v[mask]
    i = i[mask]
    finite = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    t = t[finite]
    v = v[finite]
    i = i[finite]
    if t.size < 2:
        return float("nan")
    return float(np.trapz(v * np.abs(i), t) / 3600.0)


def integrate_voltage_area(time_s: object, voltage_v: object, mask: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    v = np.asarray(voltage_v, dtype=np.float64).reshape(-1)
    if t.size != v.size or t.size != mask.size:
        return float("nan")
    t = t[mask]
    v = v[mask]
    finite = np.isfinite(t) & np.isfinite(v)
    t = t[finite]
    v = v[finite]
    if t.size < 2:
        return float("nan")
    return float(np.trapz(v, t) / 3600.0)


def voltage_slope(time_s: object, voltage_v: object, mask: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    v = np.asarray(voltage_v, dtype=np.float64).reshape(-1)
    if t.size != v.size or t.size != mask.size:
        return float("nan")
    t = t[mask]
    v = v[mask]
    finite = np.isfinite(t) & np.isfinite(v)
    t = t[finite]
    v = v[finite]
    if t.size < 2:
        return float("nan")
    den = float(t[-1] - t[0])
    if abs(den) <= EPS:
        return float("nan")
    return float((v[-1] - v[0]) / den)


def voltage_poly_coefficients(voltage_v: object, *, degree: int = 5) -> Tuple[float, float, float, float]:
    v = finite_array(voltage_v)
    if v.size < degree + 1:
        return (float("nan"),) * 4
    tau = np.linspace(0.0, 1.0, v.size, dtype=np.float64)
    try:
        coeff_desc = np.polyfit(tau, v, degree)
    except Exception:
        return (float("nan"),) * 4
    coeff_asc = coeff_desc[::-1]
    return tuple(float(coeff_asc[i]) for i in range(2, 6))


def voltage_plateau_fraction(voltage_v: object) -> float:
    v = finite_array(voltage_v)
    if v.size == 0:
        return float("nan")
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    span = vmax - vmin
    if span <= EPS:
        return 1.0
    lo = vmin + 0.40 * span
    hi = vmin + 0.80 * span
    return float(np.mean((v >= lo) & (v <= hi)))


def time_to_voltage(time_s: object, voltage_v: object, mask: np.ndarray, threshold: float) -> float:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    v = np.asarray(voltage_v, dtype=np.float64).reshape(-1)
    if t.size != v.size or t.size != mask.size:
        return float("nan")
    t = t[mask]
    v = v[mask]
    ok = np.isfinite(t) & np.isfinite(v)
    t = t[ok]
    v = v[ok]
    if t.size == 0:
        return float("nan")
    hit = np.where(v >= threshold)[0]
    if hit.size == 0:
        return float("nan")
    return float(max(t[int(hit[0])] - t[0], 0.0))


def duration_between_voltage(time_s: object, voltage_v: object, mask: np.ndarray, lo_v: float, hi_v: float) -> float:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    v = np.asarray(voltage_v, dtype=np.float64).reshape(-1)
    if t.size != v.size or t.size != mask.size:
        return float("nan")
    t = t[mask]
    v = v[mask]
    ok = np.isfinite(t) & np.isfinite(v) & (v >= lo_v) & (v < hi_v)
    if np.count_nonzero(ok) < 2:
        return 0.0
    tt = t[ok]
    return float(max(tt[-1] - tt[0], 0.0))


def value_or_integral_capacity(values: object, time_s: object, current_a: object, mask: np.ndarray) -> float:
    arr = finite_array(values)
    if arr.size:
        direct = float(np.max(arr) - np.min(arr)) if arr.size > 1 else float(arr[0])
        if np.isfinite(direct) and abs(direct) > EPS:
            return abs(direct)
        fallback = max_finite(arr)
        if np.isfinite(fallback) and abs(fallback) > EPS:
            return abs(fallback)
    return integrate_abs_current_capacity(time_s, current_a, mask)


def value_or_integral_energy(values: object, time_s: object, voltage_v: object, current_a: object, mask: np.ndarray) -> float:
    arr = finite_array(values)
    if arr.size:
        direct = float(np.max(arr) - np.min(arr)) if arr.size > 1 else float(arr[0])
        if np.isfinite(direct) and abs(direct) > EPS:
            return abs(direct)
        fallback = max_finite(arr)
        if np.isfinite(fallback) and abs(fallback) > EPS:
            return abs(fallback)
    return integrate_abs_power_energy(time_s, voltage_v, current_a, mask)


def cycle_features_from_arrays(
    *,
    time_s: object,
    voltage_v: object,
    current_a: object,
    charge_capacity: object = (),
    discharge_capacity: object = (),
    charge_energy: object = (),
    discharge_energy: object = (),
    internal_resistance: object = (),
    impedance: object = (),
    temperature_c: object = (),
    rest_before_seconds: float = np.nan,
    current_threshold: float = 1e-5,
) -> Dict[str, float]:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    v = np.asarray(voltage_v, dtype=np.float64).reshape(-1)
    i = np.asarray(current_a, dtype=np.float64).reshape(-1)
    n = min(t.size, v.size, i.size)
    if n == 0:
        return {k: float("nan") for k in BASE_FEATURE_KEYS}
    t = t[:n]
    v = v[:n]
    i = i[:n]
    finite = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    t = t[finite]
    v = v[finite]
    i = i[finite]
    if t.size == 0:
        return {k: float("nan") for k in BASE_FEATURE_KEYS}

    order = np.argsort(t, kind="stable")
    t = t[order]
    v = v[order]
    i = i[order]
    charge_mask = i > current_threshold
    discharge_mask = i < -current_threshold
    if not np.any(discharge_mask):
        discharge_mask = i < -abs(float(np.nanmedian(i))) if np.nanmedian(i) < 0 else np.zeros_like(i, dtype=bool)
    if not np.any(charge_mask):
        charge_mask = i > abs(float(np.nanmedian(i))) if np.nanmedian(i) > 0 else np.zeros_like(i, dtype=bool)
    v_dis = v[discharge_mask] if np.any(discharge_mask) else np.asarray([], dtype=np.float64)
    i_dis = i[discharge_mask] if np.any(discharge_mask) else np.asarray([], dtype=np.float64)
    i_chg = i[charge_mask] if np.any(charge_mask) else np.asarray([], dtype=np.float64)
    temp = np.asarray(temperature_c, dtype=np.float64).reshape(-1)
    if temp.size == n:
        temp = temp[finite][order]
    else:
        temp = np.asarray([], dtype=np.float64)
    temp_seg = temp[discharge_mask] if temp.size == t.size and np.any(discharge_mask) else temp
    poly_a2, poly_a3, poly_a4, poly_a5 = voltage_poly_coefficients(v_dis)
    plateau_fraction = voltage_plateau_fraction(v_dis)

    chg_cap_arr = np.asarray(charge_capacity, dtype=np.float64).reshape(-1)
    dis_cap_arr = np.asarray(discharge_capacity, dtype=np.float64).reshape(-1)
    chg_energy_arr = np.asarray(charge_energy, dtype=np.float64).reshape(-1)
    dis_energy_arr = np.asarray(discharge_energy, dtype=np.float64).reshape(-1)
    if chg_cap_arr.size == n:
        chg_cap_arr = chg_cap_arr[finite][order]
    if dis_cap_arr.size == n:
        dis_cap_arr = dis_cap_arr[finite][order]
    if chg_energy_arr.size == n:
        chg_energy_arr = chg_energy_arr[finite][order]
    if dis_energy_arr.size == n:
        dis_energy_arr = dis_energy_arr[finite][order]

    return {
        "discharge_capacity": value_or_integral_capacity(dis_cap_arr, t, i, discharge_mask),
        "charge_capacity": value_or_integral_capacity(chg_cap_arr, t, i, charge_mask),
        "discharge_energy": value_or_integral_energy(dis_energy_arr, t, v, i, discharge_mask),
        "charge_energy": value_or_integral_energy(chg_energy_arr, t, v, i, charge_mask),
        "discharge_duration": duration_from_time(t, discharge_mask),
        "charge_duration": duration_from_time(t, charge_mask),
        "mean_discharge_voltage": mean_finite(v_dis),
        "std_discharge_voltage": std_finite(v_dis),
        "start_discharge_voltage": first_finite(v_dis),
        "end_discharge_voltage": last_finite(v_dis),
        "min_discharge_voltage": min_finite(v_dis),
        "max_discharge_voltage": max_finite(v_dis),
        "voltage_slope": voltage_slope(t, v, discharge_mask),
        "voltage_area": integrate_voltage_area(t, v, discharge_mask),
        "mean_discharge_current": mean_finite(i_dis),
        "std_discharge_current": std_finite(i_dis),
        "mean_charge_current": mean_finite(i_chg),
        "std_charge_current": std_finite(i_chg),
        "internal_resistance": median_nonzero(internal_resistance),
        "impedance_or_resistance_proxy": median_nonzero(impedance),
        "temperature_mean": mean_finite(temp_seg),
        "temperature_max": max_finite(temp_seg),
        "temperature_rise": last_finite(temp_seg) - first_finite(temp_seg) if finite_array(temp_seg).size else float("nan"),
        "rest_before_seconds": float(rest_before_seconds),
        "poly_a2": poly_a2,
        "poly_a3": poly_a3,
        "poly_a4": poly_a4,
        "poly_a5": poly_a5,
        "voltage_plateau_fraction": plateau_fraction,
        "cc_charge_time": duration_from_time(t, charge_mask & (v < 4.19)),
        "cv_charge_time": duration_from_time(t, charge_mask & (v >= 4.19)),
        "time_to_4p0v_charge": time_to_voltage(t, v, charge_mask, 4.0),
        "time_to_4p1v_charge": time_to_voltage(t, v, charge_mask, 4.1),
        "time_to_4p2v_charge": time_to_voltage(t, v, charge_mask, 4.2),
        "duration_3p8_4p0v_charge": duration_between_voltage(t, v, charge_mask, 3.8, 4.0),
        "duration_4p0_4p1v_charge": duration_between_voltage(t, v, charge_mask, 4.0, 4.1),
        "duration_4p1_4p2v_charge": duration_between_voltage(t, v, charge_mask, 4.1, 4.2),
    }


def choose_reference_capacity(capacity: np.ndarray, nominal_capacity: Optional[float] = None) -> float:
    if nominal_capacity is not None and np.isfinite(nominal_capacity) and nominal_capacity > EPS:
        return float(nominal_capacity)
    cap = np.asarray(capacity, dtype=np.float64).reshape(-1)
    cap = cap[np.isfinite(cap) & (cap > EPS)]
    if cap.size == 0:
        return 1.0
    first = cap[: min(5, cap.size)]
    return float(np.median(first))


def recent_slope(values: np.ndarray, end_idx: int, window: int = 5) -> float:
    start = max(0, end_idx - window + 1)
    y = np.asarray(values[start : end_idx + 1], dtype=np.float64)
    valid = np.isfinite(y)
    y = y[valid]
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    x = x - np.mean(x)
    den = float(np.sum(x * x))
    if den <= EPS:
        return 0.0
    return float(np.sum(x * (y - np.mean(y))) / den)


def rolling_std(values: np.ndarray, end_idx: int, window: int = 5) -> float:
    start = max(0, end_idx - window + 1)
    y = np.asarray(values[start : end_idx + 1], dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return 0.0
    return float(np.std(y))


def recent_max(values: np.ndarray, end_idx: int, window: int = 10) -> float:
    start = max(0, end_idx - window + 1)
    y = np.asarray(values[start : end_idx + 1], dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    return float(np.max(y))


def recent_curvature(values: np.ndarray, end_idx: int, window: int = 10) -> float:
    start = max(0, end_idx - window + 1)
    y = np.asarray(values[start : end_idx + 1], dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    x = x - np.mean(x)
    try:
        a2, _, _ = np.polyfit(x, y, 2)
    except Exception:
        return 0.0
    return float(2.0 * a2)


def impute_feature_matrix(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    for j in range(out.shape[1]):
        col = out[:, j].astype(np.float64, copy=True)
        valid = np.isfinite(col)
        if not np.any(valid):
            out[:, j] = 0.0
            continue
        idx = np.where(valid, np.arange(col.size), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = col[idx]
        first_valid = int(np.argmax(valid))
        filled[:first_valid] = col[first_valid]
        filled[~np.isfinite(filled)] = 0.0
        out[:, j] = filled.astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def finalize_cycle_feature_table(
    rows: Sequence[Dict[str, float]],
    *,
    cycle_indices: Optional[Sequence[int]] = None,
    nominal_capacity: Optional[float] = None,
    slope_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("No battery cycle rows to finalize.")

    base = np.full((len(rows), len(BASE_FEATURE_KEYS)), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, key in enumerate(BASE_FEATURE_KEYS):
            try:
                base[i, j] = float(row.get(key, np.nan))
            except Exception:
                base[i, j] = np.nan

    tmp = impute_feature_matrix(base)
    col = {name: idx for idx, name in enumerate(BASE_FEATURE_KEYS)}
    discharge_capacity = tmp[:, col["discharge_capacity"]].astype(np.float64)
    charge_capacity = tmp[:, col["charge_capacity"]].astype(np.float64)
    discharge_energy = tmp[:, col["discharge_energy"]].astype(np.float64)
    charge_energy = tmp[:, col["charge_energy"]].astype(np.float64)
    discharge_duration = tmp[:, col["discharge_duration"]].astype(np.float64)
    charge_duration = tmp[:, col["charge_duration"]].astype(np.float64)
    start_v = tmp[:, col["start_discharge_voltage"]].astype(np.float64)
    end_v = tmp[:, col["end_discharge_voltage"]].astype(np.float64)
    voltage_drop = start_v - end_v
    ref_capacity = choose_reference_capacity(discharge_capacity, nominal_capacity)
    prev_cap = np.r_[discharge_capacity[0], discharge_capacity[:-1]]

    def optional_series(name: str, fallback: float = np.nan) -> np.ndarray:
        vals = []
        for row in rows:
            try:
                vals.append(float(row.get(name, fallback)))
            except Exception:
                vals.append(float(fallback))
        return impute_feature_matrix(np.asarray(vals, dtype=np.float32).reshape(-1, 1))[:, 0].astype(np.float64)

    internal_resistance = tmp[:, col["internal_resistance"]].astype(np.float64)
    if not np.isfinite(internal_resistance).any() or np.nanmax(np.abs(internal_resistance)) <= EPS:
        internal_resistance = tmp[:, col["impedance_or_resistance_proxy"]].astype(np.float64)

    cc_charge_time = optional_series("cc_charge_time")
    cv_charge_time = optional_series("cv_charge_time")
    time_to_4p0 = optional_series("time_to_4p0v_charge")
    time_to_4p1 = optional_series("time_to_4p1v_charge")
    time_to_4p2 = optional_series("time_to_4p2v_charge")
    dur_38_40 = optional_series("duration_3p8_4p0v_charge", 0.0)
    dur_40_41 = optional_series("duration_4p0_4p1v_charge", 0.0)
    dur_41_42 = optional_series("duration_4p1_4p2v_charge", 0.0)

    # Fallbacks for datasets without detailed charge traces. They keep the feature
    # positions consistent without inventing dataset-specific alternate meanings.
    cc_charge_time = np.where(np.isfinite(cc_charge_time), cc_charge_time, charge_duration)
    cv_charge_time = np.where(np.isfinite(cv_charge_time), cv_charge_time, 0.0)
    time_to_4p0 = np.where(np.isfinite(time_to_4p0), time_to_4p0, 0.0)
    time_to_4p1 = np.where(np.isfinite(time_to_4p1), time_to_4p1, 0.0)
    time_to_4p2 = np.where(np.isfinite(time_to_4p2), time_to_4p2, charge_duration)

    if cycle_indices is None:
        cyc = np.arange(1, len(rows) + 1, dtype=np.int64)
    else:
        cyc = np.asarray(cycle_indices, dtype=np.int64).reshape(-1)
        if cyc.shape[0] != len(rows):
            cyc = np.arange(1, len(rows) + 1, dtype=np.int64)
    max_cycle = max(float(cyc[-1]) if cyc.size else 1.0, 1.0)

    out = np.zeros((len(rows), len(BATTERY_FEATURE_NAMES)), dtype=np.float32)
    out[:, 0] = discharge_capacity.astype(np.float32)
    out[:, 1] = charge_capacity.astype(np.float32)
    out[:, 2] = (discharge_capacity / max(ref_capacity, EPS)).astype(np.float32)
    out[:, 3] = (discharge_capacity - prev_cap).astype(np.float32)
    out[:, 4] = np.asarray([recent_slope(discharge_capacity, i, 5) for i in range(len(rows))], dtype=np.float32)
    out[:, 5] = np.asarray([rolling_std(discharge_capacity, i, 5) for i in range(len(rows))], dtype=np.float32)
    out[:, 6] = charge_duration.astype(np.float32)
    out[:, 7] = cc_charge_time.astype(np.float32)
    out[:, 8] = cv_charge_time.astype(np.float32)
    out[:, 9] = time_to_4p0.astype(np.float32)
    out[:, 10] = time_to_4p1.astype(np.float32)
    out[:, 11] = time_to_4p2.astype(np.float32)
    out[:, 12] = dur_38_40.astype(np.float32)
    out[:, 13] = dur_40_41.astype(np.float32)
    out[:, 14] = dur_41_42.astype(np.float32)
    out[:, 15] = discharge_duration.astype(np.float32)
    out[:, 16] = voltage_drop.astype(np.float32)
    out[:, 17] = tmp[:, col["voltage_slope"]]
    out[:, 18] = tmp[:, col["voltage_area"]]
    out[:, 19] = tmp[:, col["mean_discharge_current"]]
    out[:, 20] = tmp[:, col["mean_charge_current"]]
    out[:, 21] = discharge_energy.astype(np.float32)
    out[:, 22] = charge_energy.astype(np.float32)
    out[:, 23] = internal_resistance.astype(np.float32)
    out[:, 24] = np.asarray([recent_slope(internal_resistance, i, 5) for i in range(len(rows))], dtype=np.float32)
    out[:, 25] = (cyc.astype(np.float64) / max_cycle).astype(np.float32)

    out = impute_feature_matrix(out)
    return out, cyc


def make_window_starts(num_tokens: int, max_tokens: int, stride: int) -> List[int]:
    if num_tokens <= 0:
        return []
    if num_tokens <= max_tokens:
        return [0]
    stride = max(1, int(stride))
    starts = list(range(0, num_tokens - max_tokens + 1, stride))
    final = num_tokens - max_tokens
    if starts[-1] != final:
        starts.append(final)
    return starts


def finalize_sample_meta(acc: Dict[str, List[object]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    int_keys = {"sample_file_index", "sample_snapshot_index", "sample_cycle_index", "sample_chunk_id"}
    for key in STANDARD_SAMPLE_KEYS:
        vals = acc.get(key, [])
        if key in int_keys:
            out[key] = np.asarray(vals, dtype=np.int64)
        else:
            out[key] = np.asarray(["" if v is None else str(v) for v in vals], dtype=object)
    return out


def save_battery_health_tokens(
    *,
    out_path: Path,
    dataset: str,
    trajectories: Sequence[BatteryTrajectory],
    max_tokens: int = 64,
    stride: int = 16,
    apply_robust_scaling: bool = True,
    robust_clip_abs: float = DEFAULT_ROBUST_CLIP_ABS,
    extra_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    x_chunks: List[np.ndarray] = []
    token_masks: List[np.ndarray] = []
    c_masks: List[np.ndarray] = []
    token_starts: List[np.ndarray] = []
    fs_values: List[float] = []
    y_rul_values: List[float] = []
    y_rul_norm_values: List[float] = []
    meta_acc: Dict[str, List[object]] = {key: [] for key in STANDARD_SAMPLE_KEYS}

    for traj in trajectories:
        table = impute_feature_matrix(np.asarray(traj.feature_table, dtype=np.float32))
        cycles = np.asarray(traj.cycle_indices, dtype=np.int64).reshape(-1)
        if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] != len(BATTERY_FEATURE_NAMES):
            raise ValueError(f"{traj.unit_id}: expected feature table [T,{len(BATTERY_FEATURE_NAMES)}], got {table.shape}")
        if cycles.shape[0] != table.shape[0]:
            cycles = np.arange(1, table.shape[0] + 1, dtype=np.int64)

        for chunk_id, start in enumerate(make_window_starts(table.shape[0], max_tokens, stride)):
            end = min(start + max_tokens, table.shape[0])
            n_tok = end - start
            x = np.zeros((1, max_tokens, len(BATTERY_FEATURE_NAMES)), dtype=np.float32)
            tm = np.zeros((1, max_tokens), dtype=bool)
            starts = np.full((max_tokens,), -1, dtype=np.int64)
            # Left-pad so the last valid token is always the current prediction/reconstruction cycle.
            x[0, -n_tok:] = table[start:end]
            tm[0, -n_tok:] = True
            starts[-n_tok:] = cycles[start:end]

            x_chunks.append(x)
            token_masks.append(tm)
            c_masks.append(np.ones((1,), dtype=bool))
            token_starts.append(starts)
            fs_values.append(float("nan"))
            last_cycle = float(cycles[end - 1])
            eol_cycle = float(cycles[-1])
            y_rul_values.append(max(eol_cycle - last_cycle, 0.0))
            y_rul_norm_values.append(max(eol_cycle - last_cycle, 0.0) / max(eol_cycle, EPS))

            file_id = traj.file_id or traj.unit_id
            filename = traj.filename or file_id
            meta_acc["sample_dataset"].append(dataset)
            meta_acc["sample_group_id"].append(traj.group_id)
            meta_acc["sample_unit_id"].append(traj.unit_id)
            meta_acc["sample_run_id"].append(traj.unit_id)
            meta_acc["sample_condition_id"].append(traj.condition_id)
            meta_acc["sample_segment_id"].append(traj.unit_id)
            meta_acc["sample_file_id"].append(file_id)
            meta_acc["sample_filename"].append(filename)
            meta_acc["sample_source_relpath"].append(traj.source_relpath)
            meta_acc["sample_source_split"].append(traj.source_split)
            meta_acc["sample_file_index"].append(int(traj.file_index))
            meta_acc["sample_snapshot_index"].append(int(start))
            meta_acc["sample_cycle_index"].append(int(cycles[end - 1]))
            meta_acc["sample_chunk_id"].append(int(chunk_id))

    if not x_chunks:
        raise RuntimeError(f"No valid battery health-token chunks produced for dataset={dataset}")

    x_all = np.stack(x_chunks, axis=0).astype(np.float32, copy=False)
    token_mask_all = np.stack(token_masks, axis=0).astype(bool, copy=False)
    c_mask_all = np.stack(c_masks, axis=0).astype(bool, copy=False)
    if apply_robust_scaling:
        x_all, feature_median, feature_iqr = robust_fit_transform(
            x_all,
            token_mask_all,
            clip_value=robust_clip_abs,
        )
    else:
        feature_median = np.zeros((len(BATTERY_FEATURE_NAMES),), dtype=np.float32)
        feature_iqr = np.ones((len(BATTERY_FEATURE_NAMES),), dtype=np.float32)

    sample_meta = finalize_sample_meta(meta_acc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": dataset,
        "max_tokens": int(max_tokens),
        "sequence_stride": int(stride),
        "feature_dim": len(BATTERY_FEATURE_NAMES),
        "feature_names": list(BATTERY_FEATURE_NAMES),
        "num_trajectories": int(len(trajectories)),
        "num_samples": int(x_all.shape[0]),
        "x_shape_semantics": "[N, 1, Mmax, 26], where one channel is one battery cycle-level HI sequence.",
        "token_mask_semantics": "True means this cycle-level health token is valid.",
        "label_semantics": "y_rul is EOL_cycle minus the last valid cycle index in the 64-token window; y_rul_norm divides this by EOL_cycle.",
        "pretraining_note": "Aligned with bearing health-token pretraining: mask valid cycle-level HI tokens and reconstruct 26-dimensional battery HI features.",
        "robust_clip_abs": float(robust_clip_abs) if apply_robust_scaling and robust_clip_abs > 0 else None,
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {
        "dataset": np.asarray(dataset),
        "x_health": x_all,
        "token_mask": token_mask_all,
        "c_mask": c_mask_all,
        "y_rul": np.asarray(y_rul_values, dtype=np.float32),
        "y_rul_norm": np.asarray(y_rul_norm_values, dtype=np.float32),
        "fs": np.asarray(fs_values, dtype=np.float32),
        "rotation_hz": np.full((x_all.shape[0],), np.nan, dtype=np.float32),
        "load_value": np.full((x_all.shape[0],), np.nan, dtype=np.float32),
        "window_points": np.asarray(token_mask_all.sum(axis=(1, 2)), dtype=np.int64),
        "window_stride": np.full((x_all.shape[0],), int(stride), dtype=np.int64),
        "window_seconds": np.full((x_all.shape[0],), np.nan, dtype=np.float32),
        "window_revolutions": np.full((x_all.shape[0],), np.nan, dtype=np.float32),
        "token_start_points": np.stack(token_starts, axis=0).astype(np.int64),
        "feature_names": np.asarray(BATTERY_FEATURE_NAMES, dtype=object),
        "feature_median": feature_median.astype(np.float32, copy=False),
        "feature_iqr": feature_iqr.astype(np.float32, copy=False),
        "meta_json": np.asarray(json.dumps(meta, ensure_ascii=False, indent=2)),
    }
    payload.update(sample_meta)
    np.savez_compressed(out_path, **payload)
    return {
        "dataset": dataset,
        "out_path": str(out_path),
        "num_samples": int(x_all.shape[0]),
        "num_segments_total": int(len(trajectories)),
        "num_segments_ok": int(len(trajectories)),
        "num_segments_skipped": 0,
        "num_local_windows_total": int(token_mask_all.sum()),
        "cmax": 1,
        "max_tokens": int(max_tokens),
        "feature_dim": len(BATTERY_FEATURE_NAMES),
    }


# -----------------------------------------------------------------------------
# Self-contained robust scaling helper (same implementation used previously)
# -----------------------------------------------------------------------------

def robust_fit_transform(
    x_all: np.ndarray,
    token_mask: np.ndarray,
    clip_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = token_mask.astype(bool)
    vals = x_all[valid]
    if vals.size == 0:
        median = np.zeros((x_all.shape[-1],), dtype=np.float32)
        iqr = np.ones((x_all.shape[-1],), dtype=np.float32)
        return x_all, median, iqr

    median = np.median(vals, axis=0).astype(np.float32)
    q25 = np.percentile(vals, 25, axis=0).astype(np.float32)
    q75 = np.percentile(vals, 75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    iqr[np.abs(iqr) < 1e-6] = 1.0

    out = x_all.copy()
    out[valid] = (out[valid] - median) / iqr
    out[~np.isfinite(out)] = 0.0
    if clip_value is not None and clip_value > 0:
        out[valid] = np.clip(out[valid], -float(clip_value), float(clip_value))
    return out.astype(np.float32, copy=False), median, iqr


# -----------------------------------------------------------------------------
# Current discharge-snapshot HealthToken pipeline (preserved)
# -----------------------------------------------------------------------------

EPS = 1e-12


WINDOW_CAPACITY_FRACTION = 0.05


WINDOW_STRIDE_FRACTION = 0.025


MAX_TOKENS = 64


MIN_VALID_TOKENS = 30


MIN_CYCLE_POINTS = 8


MIN_WINDOW_POINTS = 3


BATTERY_LOCAL_FEATURE_NAMES: Tuple[str, ...] = (
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_max",
    "voltage_ptp",
    "voltage_skewness",
    "voltage_kurtosis",
    "voltage_slope",
    "voltage_derivative_mean_abs",
    "voltage_curvature_mean_abs",
    "crate_mean",
    "crate_std",
    "crate_min",
    "crate_max",
    "crate_mean_abs",
    "crate_slope",
    "temperature_mean",
    "temperature_std",
    "temperature_min",
    "temperature_max",
    "temperature_delta",
    "temperature_slope",
    "local_duration",
    "local_capacity",
    "local_energy",
    "dvdq_mean",
)


TEMPERATURE_FEATURE_SLICE = slice(16, 22)


STANDARD_SAMPLE_KEYS: Tuple[str, ...] = (
    "sample_dataset",
    "sample_group_id",
    "sample_unit_id",
    "sample_run_id",
    "sample_condition_id",
    "sample_segment_id",
    "sample_file_id",
    "sample_filename",
    "sample_source_relpath",
    "sample_source_split",
    "sample_file_index",
    "sample_snapshot_index",
    "sample_cycle_index",
    "sample_chunk_id",
)


@dataclass
class BatteryCycle:
    """Native samples for one complete discharge segment."""

    dataset: str
    unit_id: str
    cycle_index: int
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    capacity_ah: Optional[np.ndarray] = None
    temperature_c: Optional[np.ndarray] = None
    group_id: str = ""
    run_id: str = ""
    condition_id: str = ""
    segment_id: str = ""
    file_id: str = ""
    filename: str = ""
    source_relpath: str = ""
    source_split: str = "raw_root"
    file_index: int = -1
    metadata: Dict[str, object] = field(default_factory=dict)


def detect_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "data_phm" / "raw").exists():
            return parent
    return here.parent.parent.parent


def progress_iter(items: Iterable[object], desc: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc, leave=False)
    except Exception:
        return items


def as_float_1d(values: object) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(values).reshape(-1)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    try:
        return np.asarray(arr, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        out: List[float] = []
        for value in arr:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return np.asarray(out, dtype=np.float64)


def cumulative_discharge_capacity(time_s: object, current_a: object) -> np.ndarray:
    """Integrate only discharge throughput, assuming discharge current is negative."""

    t = as_float_1d(time_s)
    current = as_float_1d(current_a)
    n = min(t.size, current.size)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    t = t[:n]
    current = current[:n]
    dt = np.diff(t, prepend=t[0])
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    fallback_dt = float(np.median(positive_dt)) if positive_dt.size else 1.0
    dt = np.where(np.isfinite(dt) & (dt >= 0), dt, fallback_dt)
    discharge_a = np.maximum(-current, 0.0)
    dq = np.zeros((n,), dtype=np.float64)
    if n > 1:
        dq[1:] = 0.5 * (discharge_a[1:] + discharge_a[:-1]) * dt[1:] / 3600.0
    return np.cumsum(dq)


def discharge_mask_from_current(current_a: object, *, threshold_a: Optional[float] = None) -> np.ndarray:
    current = as_float_1d(current_a)
    finite_abs = np.abs(current[np.isfinite(current) & (np.abs(current) > 0)])
    if threshold_a is None:
        scale = float(np.median(finite_abs)) if finite_abs.size else 0.0
        threshold_a = max(0.005, 0.01 * scale)
    return np.isfinite(current) & (current < -abs(float(threshold_a)))


def standardize_discharge_current(current_a: object) -> np.ndarray:
    """Use the common convention that discharge current is negative."""

    current = as_float_1d(current_a)
    finite_nonzero = current[np.isfinite(current) & (np.abs(current) > EPS)]
    if finite_nonzero.size and float(np.median(finite_nonzero)) > 0:
        return -current
    return current


def select_discharge_samples(
    *,
    time_s: object,
    voltage_v: object,
    current_a: object,
    capacity_ah: object = None,
    temperature_c: object = None,
    discharge_mask: Optional[object] = None,
    threshold_a: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Select the discharge part of a full cycle without synthesizing samples."""

    t = as_float_1d(time_s)
    v = as_float_1d(voltage_v)
    current = as_float_1d(current_a)
    q = as_float_1d(capacity_ah)
    temp = as_float_1d(temperature_c)
    lengths = [t.size, v.size, current.size]
    if q.size:
        lengths.append(q.size)
    if temp.size:
        lengths.append(temp.size)
    n = min(lengths) if lengths else 0
    if n == 0:
        return t[:0], v[:0], current[:0], q[:0], None

    t, v, current = t[:n], v[:n], current[:n]
    q = q[:n] if q.size else np.asarray([], dtype=np.float64)
    temp = temp[:n] if temp.size else np.asarray([], dtype=np.float64)
    if discharge_mask is None:
        mask = discharge_mask_from_current(current, threshold_a=threshold_a)
    else:
        mask = np.asarray(discharge_mask, dtype=bool).reshape(-1)[:n]
    mask &= np.isfinite(t) & np.isfinite(v) & np.isfinite(current)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return t[:0], v[:0], current[:0], q[:0], None

    # Keep all explicitly selected discharge samples.  Gaps are not filled and
    # the native sample count remains visible in token_point_count.
    t_out = t[indices]
    v_out = v[indices]
    i_out = current[indices]
    q_out = q[indices] if q.size else np.asarray([], dtype=np.float64)
    temp_out = temp[indices] if temp.size else None
    return t_out, v_out, i_out, q_out, temp_out


def _orient_capacity(capacity_ah: np.ndarray) -> np.ndarray:
    q = np.asarray(capacity_ah, dtype=np.float64).reshape(-1)
    finite = np.isfinite(q)
    if finite.sum() < 2:
        return np.full_like(q, np.nan)
    q_fill = q.copy()
    idx = np.arange(q.size)
    q_fill[~finite] = np.interp(idx[~finite], idx[finite], q[finite])
    corr = np.corrcoef(idx.astype(np.float64), q_fill)[0, 1] if q.size > 2 else np.sign(q_fill[-1] - q_fill[0])
    oriented = q_fill[0] - q_fill if np.isfinite(corr) and corr < 0 else q_fill - q_fill[0]
    oriented = np.maximum.accumulate(np.maximum(oriented, 0.0))
    return oriented


def sanitize_cycle(cycle: BatteryCycle) -> BatteryCycle:
    t = as_float_1d(cycle.time_s)
    v = as_float_1d(cycle.voltage_v)
    current = as_float_1d(cycle.current_a)
    q_raw = as_float_1d(cycle.capacity_ah)
    temp_raw = as_float_1d(cycle.temperature_c)
    lengths = [t.size, v.size, current.size]
    if q_raw.size:
        lengths.append(q_raw.size)
    if temp_raw.size:
        lengths.append(temp_raw.size)
    n = min(lengths) if lengths else 0
    if n < MIN_CYCLE_POINTS:
        raise ValueError(f"{cycle.unit_id} cycle {cycle.cycle_index}: only {n} discharge samples")

    t, v, current = t[:n], v[:n], current[:n]
    q_raw = q_raw[:n] if q_raw.size else np.asarray([], dtype=np.float64)
    temp = temp_raw[:n] if temp_raw.size else None
    finite = np.isfinite(t) & np.isfinite(v) & np.isfinite(current)
    t, v, current = t[finite], v[finite], current[finite]
    if q_raw.size:
        q_raw = q_raw[finite]
    if temp is not None:
        temp = temp[finite]
    if t.size < MIN_CYCLE_POINTS:
        raise ValueError(f"{cycle.unit_id} cycle {cycle.cycle_index}: insufficient finite samples")

    order = np.argsort(t, kind="stable")
    t, v, current = t[order], v[order], current[order]
    if q_raw.size:
        q_raw = q_raw[order]
    if temp is not None:
        temp = temp[order]
    t = t - t[0]

    q = _orient_capacity(q_raw) if q_raw.size else np.asarray([], dtype=np.float64)
    q_span = float(q[-1] - q[0]) if q.size and np.all(np.isfinite(q[[0, -1]])) else 0.0
    if q_span <= EPS:
        q = cumulative_discharge_capacity(t, current)
        q_span = float(q[-1]) if q.size else 0.0
    if q_span <= EPS:
        raise ValueError(f"{cycle.unit_id} cycle {cycle.cycle_index}: zero discharged capacity")

    return BatteryCycle(
        **{
            **cycle.__dict__,
            "time_s": t,
            "voltage_v": v,
            "current_a": current,
            "capacity_ah": q,
            "temperature_c": temp,
        }
    )


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan")
    x0 = x - float(np.mean(x))
    denom = float(np.sum(x0 * x0))
    if denom <= EPS:
        return float("nan")
    return float(np.sum(x0 * (y - float(np.mean(y)))) / denom)


def _moments(values: np.ndarray) -> Tuple[float, float]:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std <= EPS:
        return 0.0, 0.0
    z = (vals - mean) / std
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


def _safe_gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if y.size < 2:
        return np.full_like(y, np.nan, dtype=np.float64)
    x_safe = np.asarray(x, dtype=np.float64).copy()
    min_step = max(float(np.nanmax(x_safe) - np.nanmin(x_safe)) * 1e-9, 1e-12)
    for idx in range(1, x_safe.size):
        if not np.isfinite(x_safe[idx]) or x_safe[idx] <= x_safe[idx - 1]:
            x_safe[idx] = x_safe[idx - 1] + min_step
    return np.gradient(np.asarray(y, dtype=np.float64), x_safe, edge_order=1)


def _capacity_unique_indices(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.asarray([], dtype=np.int64)
    tolerance = max(float(np.nanmax(x) - np.nanmin(x)) * 1e-9, 1e-12)
    keep = np.r_[True, np.diff(x) > tolerance]
    return np.flatnonzero(keep)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 2:
        return float("nan")
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def extract_local_features(
    *,
    u: np.ndarray,
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    capacity_ah: np.ndarray,
    temperature_c: Optional[np.ndarray],
    cycle_capacity_ah: float,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(BATTERY_LOCAL_FEATURE_NAMES),), dtype=np.float32)
    observed = np.zeros_like(values, dtype=bool)

    def put(index: int, value: float) -> None:
        if np.isfinite(value):
            values[index] = np.float32(value)
            observed[index] = True

    v = np.asarray(voltage_v, dtype=np.float64)
    current = np.asarray(current_a, dtype=np.float64)
    skew, kurt = _moments(v)
    derivative_indices = _capacity_unique_indices(u)
    u_derivative = u[derivative_indices]
    v_derivative = v[derivative_indices]
    dv_du = _safe_gradient(v_derivative, u_derivative)
    d2v_du2 = _safe_gradient(dv_du, u_derivative)
    voltage_values = (
        np.mean(v),
        np.std(v),
        np.min(v),
        np.max(v),
        np.ptp(v),
        skew,
        kurt,
        _linear_slope(u, v),
        np.mean(np.abs(dv_du)),
        np.mean(np.abs(d2v_du2)),
    )
    for idx, value in enumerate(voltage_values):
        put(idx, float(value))

    crate = current / max(float(cycle_capacity_ah), EPS)
    crate_values = (
        np.mean(crate),
        np.std(crate),
        np.min(crate),
        np.max(crate),
        np.mean(np.abs(crate)),
        _linear_slope(u, crate),
    )
    for idx, value in enumerate(crate_values, start=10):
        put(idx, float(value))

    if temperature_c is not None:
        temp = np.asarray(temperature_c, dtype=np.float64)
        finite_temp = np.isfinite(temp)
        if finite_temp.sum() >= 2:
            temp = temp[finite_temp]
            u_temp = u[finite_temp]
            temp_values = (
                np.mean(temp),
                np.std(temp),
                np.min(temp),
                np.max(temp),
                temp[-1] - temp[0],
                _linear_slope(u_temp, temp),
            )
            for idx, value in enumerate(temp_values, start=16):
                put(idx, float(value))

    put(22, float(time_s[-1] - time_s[0]))
    put(23, float(capacity_ah[-1] - capacity_ah[0]))
    put(24, _trapz(voltage_v * np.abs(current_a), time_s) / 3600.0)
    q_indices = _capacity_unique_indices(capacity_ah)
    dv_dq = _safe_gradient(voltage_v[q_indices], capacity_ah[q_indices])
    put(25, float(np.mean(dv_dq)))
    return values, observed


def capacity_window_starts(
    width: float = WINDOW_CAPACITY_FRACTION,
    stride: float = WINDOW_STRIDE_FRACTION,
) -> np.ndarray:
    if not (0 < width <= 1):
        raise ValueError(f"window width must be in (0,1], got {width}")
    if not (0 < stride <= 1):
        raise ValueError(f"window stride must be in (0,1], got {stride}")
    count = int(np.floor((1.0 - width) / stride + 1e-9)) + 1
    return np.arange(count, dtype=np.float64) * stride


def tokenize_cycle(
    cycle: BatteryCycle,
    *,
    max_tokens: int = MAX_TOKENS,
    min_valid_tokens: int = MIN_VALID_TOKENS,
    window_fraction: float = WINDOW_CAPACITY_FRACTION,
    stride_fraction: float = WINDOW_STRIDE_FRACTION,
) -> Dict[str, np.ndarray]:
    cycle = sanitize_cycle(cycle)
    q = np.asarray(cycle.capacity_ah, dtype=np.float64)
    q_end = float(q[-1])
    u = np.clip(q / max(q_end, EPS), 0.0, 1.0)
    starts = capacity_window_starts(window_fraction, stride_fraction)
    if starts.size > max_tokens:
        raise ValueError(
            f"Capacity-window rule creates {starts.size} tokens, exceeding max_tokens={max_tokens}; "
            "change the explicit window rule rather than silently truncating a cycle."
        )
    if not (1 <= int(min_valid_tokens) <= int(starts.size)):
        raise ValueError(
            f"min_valid_tokens must be in [1,{starts.size}] for the selected capacity-window rule, "
            f"got {min_valid_tokens}"
        )

    feature_dim = len(BATTERY_LOCAL_FEATURE_NAMES)
    x = np.zeros((1, max_tokens, feature_dim), dtype=np.float32)
    token_mask = np.zeros((1, max_tokens), dtype=bool)
    feature_mask = np.zeros((1, max_tokens, feature_dim), dtype=bool)
    start_u = np.full((1, max_tokens), np.nan, dtype=np.float32)
    end_u = np.full((1, max_tokens), np.nan, dtype=np.float32)
    start_points = np.full((1, max_tokens), -1, dtype=np.int64)
    point_counts = np.zeros((1, max_tokens), dtype=np.int64)

    valid_token_count = 0
    for window_idx, start in enumerate(starts):
        end = min(start + window_fraction, 1.0)
        # Slots 0..38 have a fixed physical meaning for the default rule:
        # 0-5%, 2.5-7.5%, ..., 95-100%.  Never compact a missing window into
        # an earlier slot, otherwise positional semantics differ by cycle.
        start_u[0, window_idx] = np.float32(start)
        end_u[0, window_idx] = np.float32(end)
        mask = (u >= start - 1e-12) & (u <= end + 1e-12)
        indices = np.flatnonzero(mask)
        if indices.size < MIN_WINDOW_POINTS:
            continue
        q_local = q[indices]
        if float(q_local[-1] - q_local[0]) <= EPS:
            continue
        temp_local = None if cycle.temperature_c is None else np.asarray(cycle.temperature_c)[indices]
        features, observed = extract_local_features(
            u=u[indices],
            time_s=np.asarray(cycle.time_s)[indices],
            voltage_v=np.asarray(cycle.voltage_v)[indices],
            current_a=np.asarray(cycle.current_a)[indices],
            capacity_ah=q_local,
            temperature_c=temp_local,
            cycle_capacity_ah=q_end,
        )
        x[0, window_idx] = features
        token_mask[0, window_idx] = True
        feature_mask[0, window_idx] = observed
        start_points[0, window_idx] = int(indices[0])
        point_counts[0, window_idx] = int(indices.size)
        valid_token_count += 1

    if valid_token_count < int(min_valid_tokens):
        raise ValueError(
            f"{cycle.unit_id} cycle {cycle.cycle_index}: valid capacity windows "
            f"{valid_token_count}/{starts.size}, require >={int(min_valid_tokens)} for a complete snapshot"
        )
    return {
        "x_health": x,
        "token_mask": token_mask,
        "feature_mask": feature_mask,
        "token_start_capacity_fraction": start_u,
        "token_end_capacity_fraction": end_u,
        "token_start_points": start_points,
        "token_point_count": point_counts,
        "cycle_capacity_ah": np.asarray(q_end, dtype=np.float32),
        "cycle_duration_s": np.asarray(float(cycle.time_s[-1] - cycle.time_s[0]), dtype=np.float32),
        "temperature_available": np.asarray(
            cycle.temperature_c is not None
            and int(np.isfinite(np.asarray(cycle.temperature_c, dtype=np.float64)).sum()) >= 2,
            dtype=bool,
        ),
    }


def _robust_scale_observed(
    x: np.ndarray,
    feature_mask: np.ndarray,
    *,
    enabled: bool,
    clip_abs: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_dim = x.shape[-1]
    median = np.zeros((feature_dim,), dtype=np.float32)
    iqr = np.ones((feature_dim,), dtype=np.float32)
    out = x.copy()
    if not enabled:
        out[~feature_mask] = 0.0
        return out, median, iqr
    for feature_idx in range(feature_dim):
        observed = feature_mask[..., feature_idx]
        vals = x[..., feature_idx][observed]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25, 75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or abs(scale) < 1e-6:
            scale = 1.0
        median[feature_idx] = np.float32(med)
        iqr[feature_idx] = np.float32(scale)
        out[..., feature_idx][observed] = (vals - med) / scale
    out[~feature_mask] = 0.0
    out[~np.isfinite(out)] = 0.0
    if clip_abs > 0:
        out = np.clip(out, -clip_abs, clip_abs)
    return out.astype(np.float32), median, iqr


def _string_array(values: Sequence[object]) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values], dtype=str)


def _tokenize_record(task):
    """Worker result retains provenance, not the now-consumed raw waveform."""
    from dataclasses import replace
    cycle, kwargs = task
    try:
        tokens = tokenize_cycle(cycle, **kwargs)
        empty = np.empty(0,dtype=np.float64)
        identity = replace(cycle,time_s=empty,voltage_v=empty,current_a=empty,
                           capacity_ah=None,temperature_c=None)
        return identity,tokens,None
    except Exception as exc:
        return cycle,None,str(exc)


def _tokenize_records(cycles, kwargs):
    """Bound prefetch to 64 snapshots; worker count is an execution setting."""
    from concurrent.futures import ProcessPoolExecutor
    from itertools import islice
    workers = max(1,int(os.environ.get('HEALTHTOKEN_PREPROCESS_WORKERS','1')))
    if workers == 1:
        for cycle in cycles:
            yield _tokenize_record((cycle,kwargs))
    else:
        source=iter(cycles)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while True:
                batch=list(islice(source,64))
                if not batch: break
                yield from pool.map(_tokenize_record,[(cycle,kwargs) for cycle in batch])


def save_cycle_snapshots(
    *,
    cycles: Iterable[BatteryCycle],
    out_path: Path,
    dataset: str,
    max_tokens: int = MAX_TOKENS,
    min_valid_tokens: int = MIN_VALID_TOKENS,
    window_fraction: float = WINDOW_CAPACITY_FRACTION,
    stride_fraction: float = WINDOW_STRIDE_FRACTION,
    apply_robust_scaling: bool = True,
    extra_meta: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    records: List[Tuple[BatteryCycle, Dict[str, np.ndarray]]] = []
    skipped = 0
    scanned = 0
    kwargs=dict(max_tokens=max_tokens,min_valid_tokens=min_valid_tokens,
                window_fraction=window_fraction,stride_fraction=stride_fraction)
    progress = progress_iter(_tokenize_records(cycles,kwargs), f"battery-snapshots:{dataset}")

    def update_live_counts() -> None:
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                scanned=scanned,
                kept=len(records),
                skipped=skipped,
                refresh=True,
            )

    for cycle, tokens, error in progress:
        scanned += 1
        try:
            if error is not None:
                raise ValueError(error)
            records.append((cycle,tokens))
        except Exception as exc:
            skipped += 1
            warning = (
                f"[WARN] [{dataset}] skip {cycle.unit_id} cycle {cycle.cycle_index}: {exc} "
                f"[scanned={scanned}, kept={len(records)}, skipped={skipped}]"
            )
            if hasattr(progress, "write"):
                progress.write(warning)
            else:
                print(warning, flush=True)
        update_live_counts()
        if scanned % 1000 == 0:
            print(f'[PROGRESS] {dataset}: {scanned} scanned, {len(records)} valid',flush=True)
    if hasattr(progress, "close"):
        progress.close()

    print(
        f"[INFO] [{dataset}] preprocessing totals: "
        f"scanned={scanned}, kept={len(records)}, skipped={skipped}",
        flush=True,
    )
    if not records:
        raise RuntimeError(f"No valid discharge-cycle snapshots produced for {dataset}")

    from preprocess_health_tokens.common.lifecycle import select_battery_records, POLICY
    records, verified_endpoints, lifecycle_audit = select_battery_records(records, dataset)
    audit_path = out_path.with_suffix('.lifecycle.json')
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(lifecycle_audit, indent=2), encoding='utf-8')
    if not records:
        raise RuntimeError(f'No endpoint-verified lifecycles retained for {dataset}; see {audit_path}')

    x = np.stack([record[1]["x_health"] for record in records], axis=0)
    token_mask = np.stack([record[1]["token_mask"] for record in records], axis=0)
    feature_mask = np.stack([record[1]["feature_mask"] for record in records], axis=0)
    x, feature_median, feature_iqr = _robust_scale_observed(
        x, feature_mask, enabled=apply_robust_scaling
    )
    c_mask = np.ones((len(records), 1), dtype=bool)

    units = [cycle.unit_id for cycle, _ in records]
    cycle_indices = np.asarray([cycle.cycle_index for cycle, _ in records], dtype=np.int64)
    unit_eol = {unit: verified_endpoints.get(unit, float('nan')) for unit in sorted(set(units))}
    y_rul = np.asarray(
        [max(unit_eol[cycle.unit_id] - int(cycle.cycle_index), 0) for cycle, _ in records],
        dtype=np.float32,
    )
    y_rul_norm = np.asarray(
        [rul / max(float(unit_eol[cycle.unit_id]), 1.0) for rul, (cycle, _) in zip(y_rul, records)],
        dtype=np.float32,
    )
    order_value = cycle_indices.astype(float)
    eol_value = np.asarray([unit_eol[u] for u in units],dtype=float)
    if dataset == 'isu_ilcc_battery':
        order_value = np.asarray([cycle.metadata['elapsed_days'] for cycle, _ in records])
        endpoint_days = {u: max(order_value[np.asarray(units)==u]) for u in set(units)}
        eol_value = np.asarray([endpoint_days[u] for u in units])
        y_rul = (eol_value-order_value).astype(np.float32)
        y_rul_norm = (y_rul/np.maximum(eol_value,1.)).astype(np.float32)

    meta = {
        "dataset": dataset,
        "preprocessing_contract": "discharge_only_cycle_level_fixed_capacity_slots_min30_v3",
        "snapshot_definition": "One complete discharge segment from discharge start to cutoff.",
        "capacity_coordinate": "u = Q(t) / Q_end in [0,1]",
        "resampling": "None. Every local feature uses native samples inside its capacity window.",
        "window_capacity_fraction": float(window_fraction),
        "window_stride_fraction": float(stride_fraction),
        "theoretical_full_cycle_token_count": int(capacity_window_starts(window_fraction, stride_fraction).size),
        "max_tokens": int(max_tokens),
        "minimum_valid_tokens_per_snapshot": int(min_valid_tokens),
        "minimum_valid_window_coverage": float(min_valid_tokens / max(capacity_window_starts(window_fraction, stride_fraction).size, 1)),
        "feature_dim": len(BATTERY_LOCAL_FEATURE_NAMES),
        "feature_names": list(BATTERY_LOCAL_FEATURE_NAMES),
        "x_shape_semantics": "[N,1,64,26]; N is the number of complete discharge cycles.",
        "token_mask_semantics": "True marks a valid token in its fixed capacity slot; False marks an invalid canonical window or structural padding.",
        "feature_mask_semantics": "True marks an observed/computable feature. Missing temperature dimensions are zero with mask=False.",
        "num_cycles_scanned": int(scanned),
        "num_cycles_ok": int(len(records)),
        "num_cycles_skipped": int(skipped),
        "num_cycles_excluded_by_lifecycle": int(scanned-skipped-len(records)),
        "robust_scaled": bool(apply_robust_scaling),
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    meta.update(lifecycle_policy=POLICY, lifecycle_audit=lifecycle_audit,
                endpoint_rule='See per-unit threshold_capacity and confirmation_cycle in lifecycle_audit; SDU stage-only',
                lifecycle_label_is_input=False)

    payload: Dict[str, np.ndarray] = {
        "dataset": np.asarray(dataset),
        "x_health": x,
        "token_mask": token_mask,
        "feature_mask": feature_mask,
        "c_mask": c_mask,
        "y_rul": y_rul,
        "y_rul_norm": y_rul_norm,
        "endpoint_observed": np.asarray([u in verified_endpoints for u in units]),
        "eol_order": eol_value,
        "order_value": order_value,
        "cycle_capacity_ah": np.asarray([record[1]["cycle_capacity_ah"] for record in records], dtype=np.float32),
        "cycle_duration_s": np.asarray([record[1]["cycle_duration_s"] for record in records], dtype=np.float32),
        "temperature_available": np.asarray([record[1]["temperature_available"] for record in records], dtype=bool),
        "token_start_capacity_fraction": np.stack([record[1]["token_start_capacity_fraction"] for record in records]),
        "token_end_capacity_fraction": np.stack([record[1]["token_end_capacity_fraction"] for record in records]),
        "token_start_points": np.stack([record[1]["token_start_points"] for record in records]),
        "token_point_count": np.stack([record[1]["token_point_count"] for record in records]),
        "feature_names": np.asarray(BATTERY_LOCAL_FEATURE_NAMES, dtype=str),
        "feature_median": feature_median,
        "feature_iqr": feature_iqr,
        "fs": np.full((len(records),), np.nan, dtype=np.float32),
        "rotation_hz": np.full((len(records),), np.nan, dtype=np.float32),
        "load_value": np.full((len(records),), np.nan, dtype=np.float32),
        "window_points": np.asarray(token_mask.sum(axis=(1, 2)), dtype=np.int64),
        "window_stride": np.full((len(records),), np.nan, dtype=np.float32),
        "window_seconds": np.full((len(records),), np.nan, dtype=np.float32),
        "window_revolutions": np.full((len(records),), np.nan, dtype=np.float32),
        "sample_dataset": _string_array([dataset] * len(records)),
        "sample_group_id": _string_array([cycle.group_id for cycle, _ in records]),
        "sample_unit_id": _string_array(units),
        "sample_physical_unit_id": _string_array([cycle.group_id if dataset == 'sdu_battery' else cycle.unit_id for cycle, _ in records]),
        "sample_run_id": _string_array([cycle.run_id for cycle, _ in records]),
        "sample_condition_id": _string_array([cycle.condition_id for cycle, _ in records]),
        "sample_segment_id": _string_array([cycle.segment_id or f"{cycle.unit_id}_cycle_{cycle.cycle_index}" for cycle, _ in records]),
        "sample_file_id": _string_array([cycle.file_id for cycle, _ in records]),
        "sample_filename": _string_array([cycle.filename for cycle, _ in records]),
        "sample_source_relpath": _string_array([cycle.source_relpath for cycle, _ in records]),
        "sample_source_split": _string_array([cycle.source_split for cycle, _ in records]),
        "sample_file_index": np.asarray([cycle.file_index for cycle, _ in records], dtype=np.int64),
        "sample_snapshot_index": cycle_indices.copy(),
        "sample_cycle_index": cycle_indices,
        "sample_chunk_id": np.zeros((len(records),), dtype=np.int64),
        "meta_json": np.asarray(json.dumps(meta, ensure_ascii=False, indent=2)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return {
        "dataset": dataset,
        "out_path": str(out_path),
        "num_samples": int(len(records)),
        "num_segments_total": int(scanned),
        "num_segments_ok": int(len(records)),
        "num_segments_skipped": int(skipped),
        "num_local_windows_total": int(token_mask.sum()),
        "cmax": 1,
        "max_tokens": int(max_tokens),
        "min_valid_tokens": int(min_valid_tokens),
        "feature_dim": len(BATTERY_LOCAL_FEATURE_NAMES),
    }


def process_cycle_dataset(
    *,
    cfg,
    out_root: Path,
    dataset: str,
    cycles: Iterable[BatteryCycle],
    output_name: Optional[str] = None,
    extra_meta: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    max_tokens = int(getattr(cfg, "max_tokens", MAX_TOKENS)) if cfg is not None else MAX_TOKENS
    min_valid_tokens = int(getattr(cfg, "battery_min_valid_tokens", MIN_VALID_TOKENS)) if cfg is not None else MIN_VALID_TOKENS
    width = float(getattr(cfg, "battery_window_fraction", WINDOW_CAPACITY_FRACTION)) if cfg is not None else WINDOW_CAPACITY_FRACTION
    stride = float(getattr(cfg, "battery_stride_fraction", WINDOW_STRIDE_FRACTION)) if cfg is not None else WINDOW_STRIDE_FRACTION
    apply_scaling = bool(getattr(cfg, "apply_robust_scaling", True)) if cfg is not None else True
    name = output_name or getattr(cfg, "output_name", None) or f"{dataset}_health_tokens.npz"
    provenance = dict(extra_meta or {})
    # Keep battery outputs as traceable as bearing outputs.  These describe the
    # active adapter/configuration and do not affect any health feature.
    if cfg is not None:
        provenance.setdefault("source_module", str(getattr(cfg, "source_module", "")))
        if is_dataclass(cfg):
            provenance.setdefault("config", asdict(cfg))
        if bool(getattr(cfg, "scaling_deferred_to_experiment", False)):
            from preprocess_health_tokens.common.raw_token_contract import raw_contract_metadata

            provenance.update(raw_contract_metadata())
    return save_cycle_snapshots(
        cycles=cycles,
        out_path=Path(out_root) / name,
        dataset=dataset,
        max_tokens=max_tokens,
        min_valid_tokens=min_valid_tokens,
        window_fraction=width,
        stride_fraction=stride,
        apply_robust_scaling=apply_scaling,
        extra_meta=provenance,
    )


# -----------------------------------------------------------------------------
# CALCE-CS2 baseline cache helpers (preserved; imports made lazy only)
# -----------------------------------------------------------------------------

TARGET_CALCE_CS2_CELLS = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")


def str_to_bool(value: str, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_cell_list(text: str) -> Tuple[str, ...]:
    cells = tuple(x.strip() for x in str(text).split(",") if x.strip())
    return cells or TARGET_CALCE_CS2_CELLS


def json_dump(obj: Any, path: Path) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_calce_cs2_raw_root(project_root: Path) -> Path:
    raw = os.environ.get("RAW_CALCE_CS2_ROOT", "").strip() or os.environ.get("RAW_BATTERY_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    default = project_root / "data_phm" / "raw" / "Battery" / "CALCE_CS2"
    if default.exists():
        return default
    from preprocess_health_tokens.datasets.battery.calce_cs2_battery import RAW_ROOT as default_calce_cs2_raw_root
    return default_calce_cs2_raw_root


def load_calce_cs2_trajectories(
    *,
    raw_root: Path,
    target_cells: Sequence[str] = TARGET_CALCE_CS2_CELLS,
    dataset_name: str = "calce_cs2_downstream_4cells",
    condition_id: str = "CALCE_CS2_DOWNSTREAM_4CELLS",
) -> List[BatteryTrajectory]:
    missing = [cell for cell in target_cells if not (raw_root / cell).is_dir()]
    if missing:
        available = sorted(p.name for p in raw_root.iterdir() if p.is_dir()) if raw_root.exists() else []
        raise FileNotFoundError(
            f"Missing CALCE CS2 target cells under {raw_root}: {missing}. Available={available}"
        )

    from preprocess_health_tokens.datasets.battery.calce_cs2_battery import load_calce_cell_trajectory

    trajectories: List[BatteryTrajectory] = []
    for idx, cell in enumerate(target_cells):
        cell_dir = raw_root / cell
        print(f"[INFO] parsing {cell_dir}", flush=True)
        traj = load_calce_cell_trajectory(
            cell_dir=cell_dir,
            raw_root=raw_root,
            dataset_name=dataset_name,
            condition_id=condition_id,
            file_index=idx,
        )
        trajectories.append(traj)
        print(f"[OK] {cell}: {traj.feature_table.shape[0]} cycles", flush=True)
    return trajectories


def make_calce_cs2_sequence_cache(
    *,
    trajectories: Sequence[BatteryTrajectory],
    max_tokens: int = 64,
    stride: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if stride is None:
        from preprocess_health_tokens.datasets.battery.calce_cs2_battery import SEQ_STRIDE as _seq_stride
        stride = int(_seq_stride)

    xs: List[np.ndarray] = []
    token_masks: List[np.ndarray] = []
    cells: List[str] = []
    conditions: List[str] = []
    cycle_indices: List[int] = []
    start_cycles: List[int] = []
    end_cycles: List[int] = []
    chunk_ids: List[int] = []
    y_rul: List[float] = []
    y_rul_norm: List[float] = []
    life_cycles: List[float] = []
    file_indices: List[int] = []
    source_relpaths: List[str] = []

    for traj in trajectories:
        table = impute_feature_matrix(np.asarray(traj.feature_table, dtype=np.float32))
        cycles = np.asarray(traj.cycle_indices, dtype=np.int64).reshape(-1)
        if cycles.shape[0] != table.shape[0]:
            cycles = np.arange(1, table.shape[0] + 1, dtype=np.int64)
        eol_cycle = float(cycles[-1])

        for chunk_id, start in enumerate(make_window_starts(table.shape[0], max_tokens, stride)):
            end = min(start + max_tokens, table.shape[0])
            n_tok = int(end - start)
            x = np.zeros((len(BATTERY_FEATURE_NAMES), max_tokens), dtype=np.float32)
            tm = np.zeros((max_tokens,), dtype=bool)
            # Same left-padding convention as the battery health-token downstream NPZ:
            # the last valid token is the current prediction cycle.
            x[:, -n_tok:] = table[start:end].T
            tm[-n_tok:] = True

            last_cycle = float(cycles[end - 1])
            xs.append(x)
            token_masks.append(tm)
            cells.append(str(traj.unit_id))
            conditions.append(str(traj.condition_id))
            cycle_indices.append(int(cycles[end - 1]))
            start_cycles.append(int(cycles[start]))
            end_cycles.append(int(cycles[end - 1]))
            chunk_ids.append(int(chunk_id))
            y = max(eol_cycle - last_cycle, 0.0)
            y_rul.append(float(y))
            y_rul_norm.append(float(y / max(eol_cycle, 1e-8)))
            life_cycles.append(float(eol_cycle))
            file_indices.append(int(traj.file_index))
            source_relpaths.append(str(traj.source_relpath))

    if not xs:
        raise RuntimeError("No CALCE CS2 sequence windows were produced.")

    x_raw = np.stack(xs, axis=0).astype(np.float32)
    token_mask = np.stack(token_masks, axis=0).astype(bool)
    cell_arr = np.asarray(cells, dtype=object)
    return {
        "x_raw": x_raw,
        "token_mask": token_mask,
        "condition": np.asarray(conditions, dtype=object),
        "cell": cell_arr,
        "unit": cell_arr.copy(),
        "battery": cell_arr.copy(),
        "cycle_index": np.asarray(cycle_indices, dtype=np.int64),
        "start_cycle": np.asarray(start_cycles, dtype=np.int64),
        "end_cycle": np.asarray(end_cycles, dtype=np.int64),
        "chunk_id": np.asarray(chunk_ids, dtype=np.int64),
        "file_index": np.asarray(file_indices, dtype=np.int64),
        "source_relpath": np.asarray(source_relpaths, dtype=object),
        "y_rul": np.asarray(y_rul, dtype=np.float32),
        "y_rul_norm": np.asarray(y_rul_norm, dtype=np.float32),
        "life_cycles": np.asarray(life_cycles, dtype=np.float32),
        "feature_names": np.asarray(BATTERY_FEATURE_NAMES, dtype=object),
    }


def make_calce_cs2_feature_cache(*, trajectories: Sequence[BatteryTrajectory]) -> Dict[str, np.ndarray]:
    features: List[np.ndarray] = []
    cells: List[str] = []
    conditions: List[str] = []
    cycle_indices: List[int] = []
    y_rul: List[float] = []
    y_rul_norm: List[float] = []
    life_cycles: List[float] = []
    file_indices: List[int] = []
    source_relpaths: List[str] = []

    for traj in trajectories:
        table = impute_feature_matrix(np.asarray(traj.feature_table, dtype=np.float32))
        cycles = np.asarray(traj.cycle_indices, dtype=np.int64).reshape(-1)
        if cycles.shape[0] != table.shape[0]:
            cycles = np.arange(1, table.shape[0] + 1, dtype=np.int64)
        eol_cycle = float(cycles[-1])
        for i in range(table.shape[0]):
            last_cycle = float(cycles[i])
            features.append(table[i].astype(np.float32, copy=False))
            cells.append(str(traj.unit_id))
            conditions.append(str(traj.condition_id))
            cycle_indices.append(int(cycles[i]))
            y = max(eol_cycle - last_cycle, 0.0)
            y_rul.append(float(y))
            y_rul_norm.append(float(y / max(eol_cycle, 1e-8)))
            life_cycles.append(float(eol_cycle))
            file_indices.append(int(traj.file_index))
            source_relpaths.append(str(traj.source_relpath))

    if not features:
        raise RuntimeError("No CALCE CS2 cycle features were produced.")

    cell_arr = np.asarray(cells, dtype=object)
    return {
        "features": np.vstack(features).astype(np.float32),
        "condition": np.asarray(conditions, dtype=object),
        "cell": cell_arr,
        "unit": cell_arr.copy(),
        "battery": cell_arr.copy(),
        "cycle_index": np.asarray(cycle_indices, dtype=np.int64),
        "file_index": np.asarray(file_indices, dtype=np.int64),
        "source_relpath": np.asarray(source_relpaths, dtype=object),
        "y_rul": np.asarray(y_rul, dtype=np.float32),
        "y_rul_norm": np.asarray(y_rul_norm, dtype=np.float32),
        "life_cycles": np.asarray(life_cycles, dtype=np.float32),
        "feature_names": np.asarray(BATTERY_FEATURE_NAMES, dtype=object),
    }


def save_npz_with_meta(path: Path, payload: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **payload,
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    json_dump(meta, path.with_suffix(".meta.json"))


def default_meta(
    *,
    script: str,
    raw_root: Path,
    out_path: Path,
    target_cells: Sequence[str],
    payload: Dict[str, np.ndarray],
    note: str,
) -> Dict[str, Any]:
    shape_key = "x_raw" if "x_raw" in payload else "features"
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": script,
        "raw_root": str(raw_root),
        "output_npz": str(out_path),
        "target_cells": list(target_cells),
        "num_samples": int(payload[shape_key].shape[0]),
        f"{shape_key}_shape": list(payload[shape_key].shape),
        "feature_names": list(BATTERY_FEATURE_NAMES),
        "label_semantics": "y_rul is remaining cycles: EOL cycle minus current cycle; y_rul_norm divides by EOL cycle.",
        "note": note,
    }
