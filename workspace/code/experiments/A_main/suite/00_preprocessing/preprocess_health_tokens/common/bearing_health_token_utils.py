from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


FEATURE_NAMES = [
    "mean",
    "abs_mean",
    "std",
    "var",
    "rms",
    "energy",
    "peak",
    "ptp",
    "min",
    "max",
    "skewness",
    "kurtosis",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "root_amplitude",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "dominant_frequency",
    "band_energy_low",
    "band_energy_mid",
    "band_energy_high",
    "band_energy_high_low_ratio",
]


STANDARD_SAMPLE_KEYS = [
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
]


@dataclass
class HealthTokenConfig:
    dataset: str
    source_module: str
    modality: str = "bearing"
    output_name: Optional[str] = None
    window_seconds: float = 0.1
    window_revolutions: float = 10.0
    stride_fraction: float = 0.5
    max_tokens: int = 64
    min_window_points: int = 256
    max_window_points: int = 8192
    use_fixed_revolution_when_available: bool = True
    apply_robust_scaling: bool = True
    # Optional post-scaling clipping.  None preserves the historical bearing
    # and battery behavior; strict new-domain datasets opt in explicitly.
    robust_clip_abs: Optional[float] = None
    # Strict datasets must fail instead of silently omitting a malformed
    # physical observation.  The default keeps legacy dataset behavior.
    fail_on_segment_error: bool = False
    # Used only by discharge-only battery custom processors. Bearing and the
    # other modalities ignore these capacity-axis window parameters.
    battery_window_fraction: float = 0.05
    battery_stride_fraction: float = 0.025
    battery_min_valid_tokens: int = 30
    # Versioned opt-in contract.  Defaults preserve every historical dataset
    # path; raw-v2 clones a config and defers scaling until after data splits.
    preprocessing_protocol: str = "legacy_scaled_v1"
    scaling_deferred_to_experiment: bool = False


def clean_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def as_int(value: object, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def parse_first_hz(text: object) -> Optional[float]:
    s = clean_str(text)
    m = re.search(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*Hz", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def infer_rotation_hz(dataset: str, meta: Dict[str, object], file_path: Path) -> Optional[float]:
    for key in ["rotation_hz", "shaft_speed_hz", "speed_hz"]:
        value = meta.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except Exception:
            parsed = parse_first_hz(value)
            if parsed is not None:
                return parsed

    search_fields = [
        meta.get("condition_id"),
        meta.get("condition"),
        meta.get("speed"),
        meta.get("group_id"),
        meta.get("run_id"),
        file_path.stem,
        str(file_path.parent),
    ]
    for item in search_fields:
        parsed = parse_first_hz(item)
        if parsed is not None:
            return parsed

    # CWRU often encodes load rather than speed in filenames, so leave unknown.
    # MFPT/Paderborn also lack a reliable speed field in the local loaders.
    return None


def infer_load_value(meta: Dict[str, object], file_path: Path) -> Optional[float]:
    for key in ["load", "load_kN", "load_N", "load_lbf", "radial_load", "loadCell"]:
        value = meta.get(key)
        if value is None or value == "":
            continue
        try:
            return float(np.asarray(value).reshape(-1)[0])
        except Exception:
            pass

    text = " ".join(clean_str(x) for x in [meta.get("condition_id"), meta.get("condition"), file_path.parent.name])
    m = re.search(r"(\d+(?:\.\d+)?)\s*kN", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def choose_window_points(
    *,
    fs: float,
    rotation_hz: Optional[float],
    cfg: HealthTokenConfig,
) -> Tuple[int, int, str, float, Optional[float]]:
    if (
        cfg.use_fixed_revolution_when_available
        and rotation_hz is not None
        and rotation_hz > 0
    ):
        raw_points = fs * cfg.window_revolutions / rotation_hz
        rule = "fixed_revolution"
        window_seconds = cfg.window_revolutions / rotation_hz
        window_revolutions: Optional[float] = cfg.window_revolutions
    else:
        raw_points = fs * cfg.window_seconds
        rule = "fixed_duration"
        window_seconds = cfg.window_seconds
        window_revolutions = None

    window_points = int(round(raw_points))
    window_points = max(cfg.min_window_points, window_points)
    window_points = min(cfg.max_window_points, window_points)
    stride = max(1, int(round(window_points * cfg.stride_fraction)))
    return window_points, stride, rule, float(window_seconds), window_revolutions


def clean_signal(sig_cn: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig_cn, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[None, :]
    if sig.ndim != 2:
        raise ValueError(f"Expected signal [C,N], got shape={sig.shape}")
    finite_cols = np.isfinite(sig).all(axis=0)
    sig = sig[:, finite_cols]
    if sig.shape[1] == 0:
        raise ValueError("No finite time samples remain after cleaning.")
    return sig.astype(np.float32, copy=False)


def make_window_slices(n: int, window_points: int, stride: int) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    if n <= window_points:
        return [(0, n)]
    slices: List[Tuple[int, int]] = []
    start = 0
    while start + window_points <= n:
        slices.append((start, start + window_points))
        start += stride
    return slices


def safe_moments(x: np.ndarray) -> Tuple[float, float]:
    x64 = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x64))
    xc = x64 - mu
    m2 = float(np.mean(xc ** 2))
    if m2 <= EPS:
        return 0.0, 0.0
    skew = float(np.mean(xc ** 3) / (m2 ** 1.5))
    kurt = float(np.mean(xc ** 4) / (m2 ** 2))
    return skew, kurt


def spectral_features(x: np.ndarray, fs: float) -> Tuple[float, float, float, float, float, float, float, float]:
    x64 = np.asarray(x, dtype=np.float64)
    if x64.size < 4:
        return (0.0,) * 8
    x64 = x64 - np.mean(x64)
    spec = np.fft.rfft(x64)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(x64.size, d=1.0 / max(fs, EPS))
    if power.size > 1:
        power = power[1:]
        freqs = freqs[1:]
    total = float(np.sum(power))
    if total <= EPS:
        return (0.0,) * 8
    p = power / total
    centroid = float(np.sum(freqs * p))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    entropy = float(-np.sum(p * np.log(p + EPS)) / max(math.log(float(p.size)), EPS))
    dominant_frequency = float(freqs[int(np.argmax(power))])

    fmax = max(float(freqs[-1]), EPS)
    low = float(np.sum(power[freqs <= fmax / 3.0]) / total)
    mid = float(np.sum(power[(freqs > fmax / 3.0) & (freqs <= 2.0 * fmax / 3.0)]) / total)
    high = float(np.sum(power[freqs > 2.0 * fmax / 3.0]) / total)
    high_low_ratio = float(high / max(low, EPS))
    return centroid, bandwidth, entropy, dominant_frequency, low, mid, high, high_low_ratio


def extract_health_features_1d(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((len(FEATURE_NAMES),), dtype=np.float32)

    x64 = x.astype(np.float64, copy=False)
    abs_x = np.abs(x64)
    mean = float(np.mean(x64))
    abs_mean = float(np.mean(abs_x))
    std = float(np.std(x64))
    var = float(np.var(x64))
    rms = float(np.sqrt(np.mean(x64 ** 2)))
    energy = float(np.mean(x64 ** 2))
    peak = float(np.max(abs_x))
    ptp = float(np.max(x64) - np.min(x64))
    xmin = float(np.min(x64))
    xmax = float(np.max(x64))
    skew, kurt = safe_moments(x64)
    crest = float(peak / max(rms, EPS))
    shape = float(rms / max(abs_mean, EPS))
    impulse = float(peak / max(abs_mean, EPS))
    root_amp = float(np.square(np.mean(np.sqrt(abs_x))))
    clearance = float(peak / max(root_amp, EPS))
    zcr = float(np.mean(np.diff(np.signbit(x64)) != 0)) if x64.size > 1 else 0.0
    spec = spectral_features(x64, fs)

    out = np.asarray(
        [
            mean,
            abs_mean,
            std,
            var,
            rms,
            energy,
            peak,
            ptp,
            xmin,
            xmax,
            skew,
            kurt,
            crest,
            shape,
            impulse,
            clearance,
            root_amp,
            zcr,
            *spec,
        ],
        dtype=np.float32,
    )
    out[~np.isfinite(out)] = 0.0
    return out


def extract_local_feature_sequence(
    sig_cn: np.ndarray,
    *,
    fs: float,
    window_points: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sig = clean_signal(sig_cn)
    c, n = sig.shape
    slices = make_window_slices(n, window_points, stride)
    if not slices:
        raise ValueError("No window slices produced.")

    feats = np.zeros((c, len(slices), len(FEATURE_NAMES)), dtype=np.float32)
    starts = np.zeros((len(slices),), dtype=np.int64)
    for wi, (s, e) in enumerate(slices):
        starts[wi] = s
        for ch in range(c):
            feats[ch, wi] = extract_health_features_1d(sig[ch, s:e], fs=fs)
    return feats, starts


def extract_observed_feature_sequence(sig_cn, *, fs, window_points, stride):
    """Preserve acquisition positions and mask whole windows touching gaps."""
    sig=np.asarray(sig_cn,dtype=np.float32)
    slices=make_window_slices(sig.shape[1],window_points,stride)
    feats=np.zeros((sig.shape[0],len(slices),len(FEATURE_NAMES)),np.float32)
    valid=np.zeros(feats.shape[:2],bool)
    for wi,(start,end) in enumerate(slices):
        for channel in range(sig.shape[0]):
            window=sig[channel,start:end]
            if np.isfinite(window).all():
                feats[channel,wi]=extract_health_features_1d(window,fs=fs)
                valid[channel,wi]=True
    if not valid.any(axis=1).all():
        raise ValueError('A raw channel has no completely observed windows')
    return feats,np.asarray([a for a,_ in slices],dtype=np.int64),valid


def chunk_feature_sequence(
    feats_cwf: np.ndarray,
    starts_w: np.ndarray,
    *,
    max_tokens: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compatibility wrapper that now returns one sample per observation."""
    x, mask, starts, _ = pack_observation_feature_sequence(
        feats_cwf, starts_w, max_tokens=max_tokens
    )
    return [(x, mask, starts)]


def pack_observation_feature_sequence(
    feats_cwf: np.ndarray,
    starts_w: np.ndarray,
    *,
    max_tokens: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack one physical observation into exactly one padded token sample.

    A lifecycle step is one source observation, irrespective of how many local
    windows can be extracted from its signal.  When more than ``max_tokens``
    windows are available, select positions uniformly across the complete
    observation (including both endpoints).  All channels share the same
    positions, preserving temporal correspondence between sensors.

    Returns ``(x, token_mask, token_start_points, selected_window_indices)``.
    """
    feats = np.asarray(feats_cwf, dtype=np.float32)
    starts = np.asarray(starts_w, dtype=np.int64).reshape(-1)
    if feats.ndim != 3:
        raise ValueError(f"feats_cwf must be [C,W,F], got {feats.shape}")
    c, windows, f = feats.shape
    if windows <= 0 or starts.shape[0] != windows:
        raise ValueError(
            f"Invalid feature sequence: windows={windows}, starts={starts.shape}"
        )
    if int(max_tokens) <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    if windows <= int(max_tokens):
        selected = np.arange(windows, dtype=np.int64)
    else:
        selected = np.rint(
            np.linspace(0, windows - 1, num=int(max_tokens), endpoint=True)
        ).astype(np.int64)
        if np.unique(selected).size != int(max_tokens):
            raise RuntimeError(
                f"Uniform token selection produced duplicates: W={windows}, M={max_tokens}"
            )

    count = int(selected.size)
    x = np.zeros((c, int(max_tokens), f), dtype=np.float32)
    mask = np.zeros((c, int(max_tokens)), dtype=bool)
    packed_starts = np.full((int(max_tokens),), -1, dtype=np.int64)
    x[:, :count, :] = feats[:, selected, :]
    mask[:, :count] = True
    packed_starts[:count] = starts[selected]
    return x, mask, packed_starts, selected


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


def init_meta_accumulator() -> Dict[str, List[object]]:
    return {key: [] for key in STANDARD_SAMPLE_KEYS}


def append_sample_meta(
    acc: Dict[str, List[object]],
    *,
    dataset: str,
    segment_id: str,
    file_path: Path,
    meta: Dict[str, object],
    chunk_id: int,
) -> None:
    file_id = meta.get("file_id", file_path.stem)
    filename = meta.get("filename", file_path.name)
    acc["sample_dataset"].append(dataset)
    acc["sample_group_id"].append(clean_str(meta.get("group_id", meta.get("group", ""))))
    acc["sample_unit_id"].append(clean_str(meta.get("unit_id", meta.get("bearing", meta.get("test", "")))))
    acc["sample_run_id"].append(clean_str(meta.get("run_id", meta.get("run", ""))))
    acc["sample_condition_id"].append(clean_str(meta.get("condition_id", meta.get("condition", ""))))
    acc["sample_segment_id"].append(clean_str(segment_id))
    acc["sample_file_id"].append(clean_str(file_id))
    acc["sample_filename"].append(clean_str(filename))
    acc["sample_source_relpath"].append(clean_str(meta.get("source_relpath", meta.get("relative_path", ""))))
    acc["sample_source_split"].append(clean_str(meta.get("source_split", "")))
    acc["sample_file_index"].append(as_int(meta.get("file_index"), -1))
    acc["sample_snapshot_index"].append(as_int(meta.get("snapshot_index", meta.get("snapshot_idx")), -1))
    acc["sample_cycle_index"].append(as_int(meta.get("cycle_index"), -1))
    acc["sample_chunk_id"].append(int(chunk_id))


def finalize_meta(acc: Dict[str, List[object]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    int_keys = {
        "sample_file_index",
        "sample_snapshot_index",
        "sample_cycle_index",
        "sample_chunk_id",
    }
    for key, vals in acc.items():
        if key in int_keys:
            out[key] = np.asarray(vals, dtype=np.int64)
        else:
            out[key] = np.asarray([clean_str(v) for v in vals], dtype=object)
    return out


def save_health_token_npz(
    *,
    out_path: Path,
    dataset: str,
    x_health: np.ndarray,
    token_mask: np.ndarray,
    c_mask: np.ndarray,
    sample_meta: Dict[str, np.ndarray],
    fs: np.ndarray,
    rotation_hz: np.ndarray,
    load_value: np.ndarray,
    window_points: np.ndarray,
    window_stride: np.ndarray,
    window_seconds: np.ndarray,
    window_revolutions: np.ndarray,
    token_start_points: np.ndarray,
    feature_median: np.ndarray,
    feature_iqr: np.ndarray,
    extra_meta: Dict[str, object],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": np.asarray(dataset),
        "x_health": x_health.astype(np.float32, copy=False),
        "token_mask": token_mask.astype(bool, copy=False),
        "c_mask": c_mask.astype(bool, copy=False),
        "fs": fs.astype(np.float32, copy=False),
        "rotation_hz": rotation_hz.astype(np.float32, copy=False),
        "load_value": load_value.astype(np.float32, copy=False),
        "window_points": window_points.astype(np.int64, copy=False),
        "window_stride": window_stride.astype(np.int64, copy=False),
        "window_seconds": window_seconds.astype(np.float32, copy=False),
        "window_revolutions": window_revolutions.astype(np.float32, copy=False),
        "token_start_points": token_start_points.astype(np.int64, copy=False),
        "feature_names": np.asarray(FEATURE_NAMES, dtype=object),
        "feature_median": feature_median.astype(np.float32, copy=False),
        "feature_iqr": feature_iqr.astype(np.float32, copy=False),
        "meta_json": np.asarray(json.dumps(extra_meta, ensure_ascii=False, indent=2)),
    }
    payload.update(sample_meta)
    np.savez_compressed(out_path, **payload)


def config_to_jsonable(cfg: HealthTokenConfig) -> Dict[str, object]:
    return asdict(cfg)
