from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from preprocess_health_tokens.common.bearing_health_token_utils import (
    FEATURE_NAMES,
    HealthTokenConfig,
    append_sample_meta,
    choose_window_points,
    chunk_feature_sequence,
    config_to_jsonable,
    extract_health_features_1d,
    extract_local_feature_sequence,
    finalize_meta,
    infer_load_value,
    infer_rotation_hz,
    init_meta_accumulator,
    robust_fit_transform,
)
from preprocess_health_tokens.datasets.registry import DATASET_CONFIGS
from preprocess_health_tokens.main import OUT_ROOT, _read_segment_signal, progress_iter


DEFAULT_OUT_ROOT = OUT_ROOT / "bearing"


LOCAL_AGG_STATS: Tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "p05",
    "p95",
    "first",
    "last",
    "range",
    "slope",
)


def _local_agg_feature_names(base_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    return np.asarray([f"{name}_{stat}" for name in base_names for stat in LOCAL_AGG_STATS], dtype=object)


def _compute_slope(vals_tf: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals_tf, dtype=np.float64)
    t = np.arange(vals.shape[0], dtype=np.float64)
    if vals.shape[0] < 2:
        return np.zeros((vals.shape[1],), dtype=np.float32)
    t = t - float(np.mean(t))
    denom = float(np.sum(t ** 2))
    if denom <= 1e-12:
        return np.zeros((vals.shape[1],), dtype=np.float32)
    y = vals - np.mean(vals, axis=0, keepdims=True)
    return (np.sum(t[:, None] * y, axis=0) / denom).astype(np.float32)


def compute_local_agg_features(x_cmf: np.ndarray, token_mask_cm: np.ndarray) -> np.ndarray:
    x = np.asarray(x_cmf, dtype=np.float32)
    mask = np.asarray(token_mask_cm, dtype=bool)
    c, _, f = x.shape
    out = np.zeros((c, f * len(LOCAL_AGG_STATS)), dtype=np.float32)
    for ch in range(c):
        valid = mask[ch]
        if not np.any(valid):
            continue
        vals = x[ch, valid].astype(np.float64, copy=False)
        pieces = [
            np.mean(vals, axis=0),
            np.std(vals, axis=0),
            np.min(vals, axis=0),
            np.max(vals, axis=0),
            np.percentile(vals, 5, axis=0),
            np.percentile(vals, 95, axis=0),
            vals[0],
            vals[-1],
            np.max(vals, axis=0) - np.min(vals, axis=0),
            _compute_slope(vals),
        ]
        # Interleave by base feature: rms_mean, rms_std, ..., kurtosis_mean, ...
        stacked = np.stack(pieces, axis=1).reshape(-1)
        out[ch] = stacked.astype(np.float32)
    return out


def compute_global_raw_features(sig_cf: np.ndarray, fs: float) -> np.ndarray:
    sig = np.asarray(sig_cf, dtype=np.float32)
    if sig.ndim != 2:
        raise ValueError(f"Expected signal [C,N], got {sig.shape}")
    out = np.zeros((sig.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    for ch in range(sig.shape[0]):
        out[ch] = extract_health_features_1d(sig[ch], fs=float(fs))
    return out


def robust_fit_transform_channel_features(
    x_ncg: np.ndarray,
    c_mask_nc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_ncg, dtype=np.float32).copy()
    mask = np.asarray(c_mask_nc, dtype=bool)
    vals = x[mask]
    g = int(x.shape[-1])
    if vals.size == 0:
        return x, np.zeros((g,), dtype=np.float32), np.ones((g,), dtype=np.float32)
    median = np.median(vals, axis=0).astype(np.float32)
    q25 = np.percentile(vals, 25, axis=0).astype(np.float32)
    q75 = np.percentile(vals, 75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    iqr = np.where(np.abs(iqr) < 1e-6, 1.0, iqr).astype(np.float32)
    x[mask] = ((x[mask] - median[None, :]) / iqr[None, :]).astype(np.float32)
    x[~mask] = 0.0
    return x, median, iqr


def save_extended_npz(
    *,
    out_path: Path,
    cfg: HealthTokenConfig,
    x_health: np.ndarray,
    token_mask: np.ndarray,
    c_mask: np.ndarray,
    x_local_agg: np.ndarray,
    x_global_raw: np.ndarray,
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
    local_agg_median: np.ndarray,
    local_agg_iqr: np.ndarray,
    global_raw_median: np.ndarray,
    global_raw_iqr: np.ndarray,
    extra_meta: Dict[str, object],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": np.asarray(cfg.dataset),
        "x_health": x_health.astype(np.float32, copy=False),
        "token_mask": token_mask.astype(bool, copy=False),
        "c_mask": c_mask.astype(bool, copy=False),
        "x_local_agg": x_local_agg.astype(np.float32, copy=False),
        "x_global_raw": x_global_raw.astype(np.float32, copy=False),
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
        "local_agg_feature_names": _local_agg_feature_names(),
        "local_agg_median": local_agg_median.astype(np.float32, copy=False),
        "local_agg_iqr": local_agg_iqr.astype(np.float32, copy=False),
        "global_raw_feature_names": np.asarray(FEATURE_NAMES, dtype=object),
        "global_raw_median": global_raw_median.astype(np.float32, copy=False),
        "global_raw_iqr": global_raw_iqr.astype(np.float32, copy=False),
        "meta_json": np.asarray(json.dumps(extra_meta, ensure_ascii=False, indent=2)),
    }
    payload.update(sample_meta)
    np.savez_compressed(out_path, **payload)


def process_xjtu_snapshot_features(
    *,
    out_root: Path = DEFAULT_OUT_ROOT,
    limit_segments: Optional[int] = None,
) -> Dict[str, object]:
    cfg = DATASET_CONFIGS["xjtu"]
    module = importlib.import_module(cfg.source_module)
    segments = module.build_segments()
    if limit_segments is not None:
        segments = segments[: int(limit_segments)]

    print("=" * 100)
    print("[INFO] dataset      : xjtu")
    print("[INFO] variant      : local health tokens + local aggregation + global raw snapshot HI")
    print(f"[INFO] source       : {cfg.source_module}")
    print(f"[INFO] segments     : {len(segments)}")
    print(f"[INFO] out_root     : {out_root}")
    print(f"[INFO] output       : xjtu_health_tokens_snapshot_features.npz")
    print("=" * 100, flush=True)

    x_chunks: List[np.ndarray] = []
    token_masks: List[np.ndarray] = []
    c_masks: List[np.ndarray] = []
    local_agg_chunks: List[np.ndarray] = []
    global_raw_chunks: List[np.ndarray] = []
    token_starts: List[np.ndarray] = []

    fs_values: List[float] = []
    rotation_values: List[float] = []
    load_values: List[float] = []
    window_points_values: List[int] = []
    window_stride_values: List[int] = []
    window_seconds_values: List[float] = []
    window_revolutions_values: List[float] = []

    sample_meta_acc = init_meta_accumulator()
    cmax = 0
    num_segments_ok = 0
    num_segments_skipped = 0
    num_chunks_total = 0
    num_local_windows_total = 0
    rule_counts: Dict[str, int] = {}

    for seg in progress_iter(segments, "health_tokens:xjtu+snapshot_features"):
        try:
            meta = dict(getattr(seg, "metadata", {}) or {})
            sig, fs = _read_segment_signal(module, seg)
            cmax = max(cmax, int(sig.shape[0]))

            rotation_hz = infer_rotation_hz(cfg.dataset, meta, Path(seg.file_path))
            load_value = infer_load_value(meta, Path(seg.file_path))
            window_points, stride, rule, window_seconds, window_revolutions = choose_window_points(
                fs=fs,
                rotation_hz=rotation_hz,
                cfg=cfg,
            )
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

            feats, starts = extract_local_feature_sequence(
                sig,
                fs=fs,
                window_points=window_points,
                stride=stride,
            )
            global_raw = compute_global_raw_features(sig, fs=fs)
            num_local_windows_total += int(feats.shape[1])

            for chunk_id, (x, tok_mask, starts_chunk) in enumerate(
                chunk_feature_sequence(feats, starts, max_tokens=cfg.max_tokens)
            ):
                x_chunks.append(x)
                token_masks.append(tok_mask)
                cm = np.ones((x.shape[0],), dtype=bool)
                c_masks.append(cm)
                local_agg_chunks.append(compute_local_agg_features(x, tok_mask))
                global_raw_chunks.append(global_raw)
                token_starts.append(starts_chunk)

                fs_values.append(float(fs))
                rotation_values.append(float("nan") if rotation_hz is None else float(rotation_hz))
                load_values.append(float("nan") if load_value is None else float(load_value))
                window_points_values.append(int(window_points))
                window_stride_values.append(int(stride))
                window_seconds_values.append(float(window_seconds))
                window_revolutions_values.append(
                    float("nan") if window_revolutions is None else float(window_revolutions)
                )
                append_sample_meta(
                    sample_meta_acc,
                    dataset=cfg.dataset,
                    segment_id=seg.segment_id,
                    file_path=Path(seg.file_path),
                    meta=meta,
                    chunk_id=chunk_id,
                )
                num_chunks_total += 1

            num_segments_ok += 1
        except Exception as e:
            num_segments_skipped += 1
            print(f"[WARN] [xjtu] skip {getattr(seg, 'file_path', '<unknown>')}: {e}", flush=True)

    if not x_chunks:
        raise RuntimeError("No valid XJTU health-token chunks produced.")

    n = len(x_chunks)
    fdim = int(x_chunks[0].shape[-1])
    agg_dim = int(local_agg_chunks[0].shape[-1])
    global_dim = int(global_raw_chunks[0].shape[-1])

    x_all = np.zeros((n, cmax, cfg.max_tokens, fdim), dtype=np.float32)
    token_mask_all = np.zeros((n, cmax, cfg.max_tokens), dtype=bool)
    c_mask_all = np.zeros((n, cmax), dtype=bool)
    x_local_agg_all = np.zeros((n, cmax, agg_dim), dtype=np.float32)
    x_global_raw_all = np.zeros((n, cmax, global_dim), dtype=np.float32)

    for i, (x, tm, cm, agg, glob) in enumerate(
        zip(x_chunks, token_masks, c_masks, local_agg_chunks, global_raw_chunks)
    ):
        c = int(x.shape[0])
        x_all[i, :c] = x
        token_mask_all[i, :c] = tm
        c_mask_all[i, :c] = cm
        x_local_agg_all[i, :c] = agg
        x_global_raw_all[i, :c] = glob

    if cfg.apply_robust_scaling:
        x_all, feature_median, feature_iqr = robust_fit_transform(x_all, token_mask_all)
        x_local_agg_all, local_agg_median, local_agg_iqr = robust_fit_transform_channel_features(
            x_local_agg_all, c_mask_all
        )
        x_global_raw_all, global_raw_median, global_raw_iqr = robust_fit_transform_channel_features(
            x_global_raw_all, c_mask_all
        )
    else:
        feature_median = np.zeros((fdim,), dtype=np.float32)
        feature_iqr = np.ones((fdim,), dtype=np.float32)
        local_agg_median = np.zeros((agg_dim,), dtype=np.float32)
        local_agg_iqr = np.ones((agg_dim,), dtype=np.float32)
        global_raw_median = np.zeros((global_dim,), dtype=np.float32)
        global_raw_iqr = np.ones((global_dim,), dtype=np.float32)

    sample_meta = finalize_meta(sample_meta_acc)
    out_path = out_root / "xjtu_health_tokens_snapshot_features.npz"
    extra_meta = {
        "dataset": cfg.dataset,
        "source_module": cfg.source_module,
        "config": config_to_jsonable(cfg),
        "variant": "xjtu_snapshot_features",
        "num_segments_total": len(segments),
        "num_segments_ok": num_segments_ok,
        "num_segments_skipped": num_segments_skipped,
        "num_chunks_total": num_chunks_total,
        "num_local_windows_total": num_local_windows_total,
        "cmax": int(cmax),
        "max_tokens": int(cfg.max_tokens),
        "feature_dim": int(fdim),
        "local_agg_dim": int(agg_dim),
        "global_raw_dim": int(global_dim),
        "local_agg_stats": list(LOCAL_AGG_STATS),
        "rule_counts": rule_counts,
        "x_health_semantics": "[N, Cmax, Mmax, F] local health-token features",
        "x_local_agg_semantics": "[N, Cmax, A] aggregation over valid local health tokens in each chunk",
        "x_global_raw_semantics": "[N, Cmax, G] whole-snapshot raw-signal health indicators, repeated for chunks from the same snapshot",
        "scaling_note": "x_health, x_local_agg, and x_global_raw are robust-scaled separately.",
    }
    save_extended_npz(
        out_path=out_path,
        cfg=cfg,
        x_health=x_all,
        token_mask=token_mask_all,
        c_mask=c_mask_all,
        x_local_agg=x_local_agg_all,
        x_global_raw=x_global_raw_all,
        sample_meta=sample_meta,
        fs=np.asarray(fs_values, dtype=np.float32),
        rotation_hz=np.asarray(rotation_values, dtype=np.float32),
        load_value=np.asarray(load_values, dtype=np.float32),
        window_points=np.asarray(window_points_values, dtype=np.int64),
        window_stride=np.asarray(window_stride_values, dtype=np.int64),
        window_seconds=np.asarray(window_seconds_values, dtype=np.float32),
        window_revolutions=np.asarray(window_revolutions_values, dtype=np.float32),
        token_start_points=np.stack(token_starts, axis=0).astype(np.int64),
        feature_median=feature_median,
        feature_iqr=feature_iqr,
        local_agg_median=local_agg_median,
        local_agg_iqr=local_agg_iqr,
        global_raw_median=global_raw_median,
        global_raw_iqr=global_raw_iqr,
        extra_meta=extra_meta,
    )

    summary = {
        "dataset": cfg.dataset,
        "variant": "xjtu_snapshot_features",
        "out_path": str(out_path),
        "num_samples": int(n),
        "num_segments_total": int(len(segments)),
        "num_segments_ok": int(num_segments_ok),
        "num_segments_skipped": int(num_segments_skipped),
        "num_local_windows_total": int(num_local_windows_total),
        "cmax": int(cmax),
        "max_tokens": int(cfg.max_tokens),
        "feature_dim": int(fdim),
        "local_agg_dim": int(agg_dim),
        "global_raw_dim": int(global_dim),
        "rule_counts": rule_counts,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def run(out_root: Path = DEFAULT_OUT_ROOT, limit_segments: Optional[int] = None) -> Dict[str, object]:
    return process_xjtu_snapshot_features(out_root=out_root, limit_segments=limit_segments)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build XJTU health-token NPZ with local aggregation and global raw snapshot HI."
    )
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--limit-segments", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(out_root=args.out_root, limit_segments=args.limit_segments)
