from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from preprocess_health_tokens.common.bearing_health_token_utils import (
    HealthTokenConfig,
    append_sample_meta,
    choose_window_points,
    pack_observation_feature_sequence,
    clean_signal,
    config_to_jsonable,
    extract_local_feature_sequence,
    finalize_meta,
    infer_load_value,
    infer_rotation_hz,
    init_meta_accumulator,
    robust_fit_transform,
    save_health_token_npz,
)
from preprocess_health_tokens.datasets.registry import selected_configs
from preprocess_health_tokens.common.raw_token_contract import (
    RAW_OUTPUT_SUBDIR,
    RAW_PROTOCOL,
    as_raw_protocol_config,
    raw_contract_metadata,
    validate_raw_health_tokens,
)


def detect_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent.parent, *here.parents]:
        if (p / "data_phm" / "raw").exists():
            return p
    return here.parent.parent


PROJECT_ROOT = detect_project_root()
OUT_ROOT = PROJECT_ROOT / "data_phm" / "processed_health_tokens"


def progress_iter(items, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc, leave=False)
    except Exception:
        return items


def modality_out_root(out_root: Path, cfg: HealthTokenConfig) -> Path:
    modality = str(getattr(cfg, "modality", "bearing") or "bearing").lower()
    if modality not in {"bearing", "battery", "milling", "engine"}:
        raise ValueError(f"Unsupported health-token modality for {cfg.dataset}: {modality!r}")
    return Path(out_root) / modality


def _read_segment_signal(module, segment):
    dataset_name = str(getattr(module, "DATASET_NAME", "")).lower()
    if dataset_name == "xjtu":
        import pandas as pd
        arr = pd.read_csv(segment.file_path, dtype=np.float32).to_numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :] if arr.shape[0] <= 2 else arr[:, None]
        sig = arr[:, :2].T.astype(np.float32, copy=False)
        fs_from_file = None
    elif dataset_name == "ims":
        arr = np.loadtxt(segment.file_path, dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        sig = arr.T.astype(np.float32, copy=False)
        fs_from_file = None
    else:
        sig, fs_from_file = module.read_signal(segment)
    fs = float(fs_from_file if fs_from_file is not None else segment.sampling_rate)
    if dataset_name == 'kaist':
        return np.asarray(sig,dtype=np.float32),fs
    if not np.isfinite(sig).all():
        raise ValueError(f'Nonfinite raw sensor values: {segment.file_path}')
    return clean_signal(sig), fs


def process_dataset(cfg: HealthTokenConfig, *, out_root: Path) -> Dict[str, object]:
    module = importlib.import_module(cfg.source_module)
    out_root = modality_out_root(out_root, cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    custom_processor = getattr(module, "process_health_tokens", None)
    if callable(custom_processor):
        print("=" * 100)
        print(f"[INFO] dataset      : {cfg.dataset}")
        print(f"[INFO] source       : {cfg.source_module}")
        print(f"[INFO] processor    : custom dataset health-token builder")
        print(f"[INFO] out_root     : {out_root}")
        print(f"[INFO] max_tokens   : {cfg.max_tokens}")
        if str(getattr(cfg, "modality", "")).lower() == "battery":
            print(f"[INFO] capacity win : {cfg.battery_window_fraction:.3f}")
            print(f"[INFO] capacity step: {cfg.battery_stride_fraction:.3f}")
            print(f"[INFO] min valid tok: {cfg.battery_min_valid_tokens}")
        print("=" * 100, flush=True)
        summary = custom_processor(cfg, out_root=out_root, limit_segments=None)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return summary

    segments = module.build_segments()
    if cfg.modality == 'bearing':
        from preprocess_health_tokens.common.lifecycle import check_bearing_segments
        check_bearing_segments(cfg.dataset, segments)
    print("=" * 100)
    print(f"[INFO] dataset      : {cfg.dataset}")
    print(f"[INFO] source       : {cfg.source_module}")
    print(f"[INFO] segments     : {len(segments)}")
    print(f"[INFO] out_root     : {out_root}")
    print(f"[INFO] max_tokens   : {cfg.max_tokens}")
    print(f"[INFO] window rule  : fixed revolution if speed exists, else fixed duration")
    print("=" * 100, flush=True)

    x_samples: List[np.ndarray] = []
    token_masks: List[np.ndarray] = []
    c_masks: List[np.ndarray] = []
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
    num_samples_total = 0
    num_observations_uniformly_subsampled = 0
    num_local_windows_total = 0
    rule_counts: Dict[str, int] = {}

    for seg in progress_iter(segments, f"health_tokens:{cfg.dataset}"):
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

            observed_windows=None
            if cfg.dataset=='kaist':
                from preprocess_health_tokens.common.bearing_health_token_utils import extract_observed_feature_sequence
                feats,starts,observed_windows=extract_observed_feature_sequence(
                    sig,fs=fs,window_points=window_points,stride=stride)
            else:
                feats, starts = extract_local_feature_sequence(
                    sig,fs=fs,window_points=window_points,stride=stride)
            num_local_windows_total += int(feats.shape[1])

            x, tok_mask, packed_starts, selected_windows = pack_observation_feature_sequence(
                feats, starts, max_tokens=cfg.max_tokens
            )
            if observed_windows is not None:
                count = int(selected_windows.size)
                tok_mask[:, :count] &= observed_windows[:, selected_windows]
                x[~tok_mask] = 0.0
            if int(feats.shape[1]) > int(cfg.max_tokens):
                num_observations_uniformly_subsampled += 1

            x_samples.append(x)
            token_masks.append(tok_mask)
            c_masks.append(np.ones((x.shape[0],), dtype=bool))
            token_starts.append(packed_starts)

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
                chunk_id=0,
            )
            num_samples_total += 1

            num_segments_ok += 1
        except Exception as e:
            num_segments_skipped += 1
            if cfg.fail_on_segment_error:
                raise RuntimeError(
                    f"Strict preprocessing failed for dataset={cfg.dataset}, "
                    f"segment={getattr(seg, 'segment_id', '<unknown>')}, "
                    f"file={getattr(seg, 'file_path', '<unknown>')}"
                ) from e
            print(f"[WARN] [{cfg.dataset}] skip {getattr(seg, 'file_path', '<unknown>')}: {e}", flush=True)

    if not x_samples:
        raise RuntimeError(f"No valid health-token samples produced for dataset={cfg.dataset}")

    n = len(x_samples)
    fdim = int(x_samples[0].shape[-1])
    x_all = np.zeros((n, cmax, cfg.max_tokens, fdim), dtype=np.float32)
    token_mask_all = np.zeros((n, cmax, cfg.max_tokens), dtype=bool)
    c_mask_all = np.zeros((n, cmax), dtype=bool)

    for i, (x, tm, cm) in enumerate(zip(x_samples, token_masks, c_masks)):
        c = int(x.shape[0])
        x_all[i, :c] = x
        token_mask_all[i, :c] = tm
        c_mask_all[i, :c] = cm

    if cfg.apply_robust_scaling:
        x_all, feature_median, feature_iqr = robust_fit_transform(
            x_all,
            token_mask_all,
            clip_value=cfg.robust_clip_abs,
        )
    else:
        feature_median = np.zeros((fdim,), dtype=np.float32)
        feature_iqr = np.ones((fdim,), dtype=np.float32)

    raw_validation = None
    if bool(getattr(cfg, "scaling_deferred_to_experiment", False)):
        raw_validation = validate_raw_health_tokens(x_all, token_mask_all, c_mask_all)

    sample_meta = finalize_meta(sample_meta_acc)
    if cfg.modality == 'bearing':
        from preprocess_health_tokens.common.lifecycle import POLICY
        unit = sample_meta['sample_unit_id'].astype(str)
        order = sample_meta['sample_snapshot_index']
        endpoint = np.array([max(order[unit == u]) for u in unit], dtype=np.int64)
        sample_meta['endpoint_observed'] = np.full(len(unit), True)
        sample_meta['eol_order'] = endpoint
        sample_meta['y_rul'] = (endpoint-order).astype(np.float32)
        sample_meta['lifecycle_policy'] = np.asarray(POLICY)

    out_name = cfg.output_name or f"{cfg.dataset}_health_tokens.npz"
    out_path = out_root / out_name
    extra_meta = {
        "dataset": cfg.dataset,
        "debug_partial": False,
        "source_module": cfg.source_module,
        "config": config_to_jsonable(cfg),
        "num_segments_total": len(segments),
        "num_segments_ok": num_segments_ok,
        "num_segments_skipped": num_segments_skipped,
        "num_samples_total": num_samples_total,
        "num_chunks_total": num_samples_total,
        "sample_granularity": "one_source_observation_per_npz_row",
        "overfull_token_policy": "uniform_time_coverage_including_first_and_last",
        "num_observations_uniformly_subsampled": num_observations_uniformly_subsampled,
        "num_local_windows_total": num_local_windows_total,
        "cmax": int(cmax),
        "max_tokens": int(cfg.max_tokens),
        "feature_dim": int(fdim),
        "robust_scaling_applied": bool(cfg.apply_robust_scaling),
        "robust_clip_abs": (
            None if cfg.robust_clip_abs is None else float(cfg.robust_clip_abs)
        ),
        "strict_fail_on_segment_error": bool(cfg.fail_on_segment_error),
        "rule_counts": rule_counts,
        "x_shape_semantics": "[N, Cmax, Mmax, F]",
        "token_mask_semantics": "True means this local health token is valid.",
        "pretraining_note": (
            "Use x_health as local health-indicator tokens. Mask only valid tokens "
            "according to token_mask, and reconstruct F-dimensional health features."
        ),
    }
    if bool(getattr(cfg, "scaling_deferred_to_experiment", False)):
        extra_meta.update(raw_contract_metadata())
        extra_meta["raw_contract_validation"] = raw_validation
    save_health_token_npz(
        out_path=out_path,
        dataset=cfg.dataset,
        x_health=x_all,
        token_mask=token_mask_all,
        c_mask=c_mask_all,
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
        extra_meta=extra_meta,
    )

    summary = {
        "dataset": cfg.dataset,
        "out_path": str(out_path),
        "num_samples": int(n),
        "num_segments_total": int(len(segments)),
        "num_segments_ok": int(num_segments_ok),
        "num_segments_skipped": int(num_segments_skipped),
        "num_local_windows_total": int(num_local_windows_total),
        "cmax": int(cmax),
        "max_tokens": int(cfg.max_tokens),
        "feature_dim": int(fdim),
        "rule_counts": rule_counts,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild canonical non-fuelcell HealthToken NPZ files.")
    p.add_argument("--include", nargs="*", default=[], help="Dataset names to rebuild. Default: every canonical dataset.")
    p.add_argument("--exclude", nargs="*", default=[], help="Dataset names to exclude.")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return p.parse_args()



def main() -> None:
    args = parse_args()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    cfgs = selected_configs(args.include, args.exclude)
    args.out_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    for cfg in cfgs.values():
        summaries.append(process_dataset(cfg, out_root=args.out_root))

    summary_path = args.out_root / ('summary_' + '_'.join(cfgs) + '.json')
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] summary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
