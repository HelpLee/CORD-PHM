"""RealHyFC heavy-duty PEMFC durability data as operating snapshots.

The public raw release is a continuous 1 Hz stream split across ZIP members,
not reliably on physical cycle boundaries.  A snapshot is consequently a
non-overlapping ten-hour fixed-duration operating segment under the same
heavy-duty load protocol.  It contains about 36k real samples and no padding
or interpolation before local HealthToken extraction.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np

from preprocess_health_tokens.common.bearing_health_token_utils import (
    append_sample_meta,
    config_to_jsonable,
    extract_local_feature_sequence,
    finalize_meta,
    init_meta_accumulator,
    robust_fit_transform,
    save_health_token_npz,
)


DATASET_NAME = "realhyfc_stack_durability_upstream_fuelcell"
RAW_ARCHIVE = Path(__file__).resolve().parents[3] / "data_phm" / "raw" / "FuelCell" / "realhyfc_stack_durability" / "Loadcycle_data.zip"
SNAPSHOT_HOURS = 10.0
WINDOW_POINTS = 1800  # 30 minutes at the native 1 Hz sampling rate
WINDOW_STRIDE = 900   # 50% overlap
MIN_SNAPSHOT_POINTS = 8 * 3600


def _member_key(name: str) -> int:
    match = re.search(r"Loadcycle_(\d+)-\d+\.csv$", name)
    return int(match.group(1)) if match else 10**9


def _number(text: object) -> float:
    try:
        return float(str(text).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def _rows(*, strict: bool) -> Iterable[tuple[float, float, float]]:
    """Yield non-duplicated [age hours, average-cell voltage, current] rows."""
    last_age = -np.inf
    with zipfile.ZipFile(RAW_ARCHIVE) as zf:
        members = sorted(
            (x.filename for x in zf.infolist() if re.search(r"Loadcycle_\d+-\d+\.csv$", x.filename)),
            key=_member_key,
        )
        for member in members:
            with zf.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                reader = csv.DictReader(text, delimiter=";")
                for row_number, row in enumerate(reader, start=2):
                    age = _number(row.get("OpHrs [h]"))
                    voltage = _number(row.get("U.S.AveCell [V]"))
                    current = _number(row.get("I.S.act [A]"))
                    # ZIP members overlap at some file boundaries.  Retain
                    # one monotonically ordered copy of every raw timestamp.
                    if not (np.isfinite(age) and np.isfinite(voltage) and np.isfinite(current)):
                        if strict:
                            raise RuntimeError(
                                f"Non-finite age/voltage/current in {member}, row {row_number}"
                            )
                        continue
                    if age <= last_age:
                        continue
                    last_age = age
                    yield age, voltage, current


def process_health_tokens(cfg, *, out_root: Path, limit_segments=None):
    if not RAW_ARCHIVE.exists():
        raise FileNotFoundError(f"Missing RealHyFC raw archive: {RAW_ARCHIVE}")

    x_chunks: list[np.ndarray] = []
    token_masks: list[np.ndarray] = []
    starts_all: list[np.ndarray] = []
    ages: list[float] = []
    ends: list[float] = []
    meta_acc = init_meta_accumulator()

    active: list[tuple[float, float, float]] = []
    snapshot_index = 0
    snapshot_start: float | None = None

    def flush() -> None:
        nonlocal active, snapshot_index, snapshot_start
        if snapshot_start is None or len(active) < MIN_SNAPSHOT_POINTS:
            active = []
            return
        values = np.asarray(active, dtype=np.float32)
        voltage, current = values[:, 1], values[:, 2]
        signal = np.stack([voltage, current, voltage * current])
        feats, starts = extract_local_feature_sequence(
            signal, fs=1.0, window_points=WINDOW_POINTS, stride=WINDOW_STRIDE,
        )
        if feats.shape[1] > cfg.max_tokens:
            raise RuntimeError("RealHyFC snapshot unexpectedly exceeds the 64-token contract")
        x = np.zeros((3, cfg.max_tokens, feats.shape[-1]), dtype=np.float32)
        mask = np.zeros((3, cfg.max_tokens), dtype=bool)
        padded_starts = np.full((cfg.max_tokens,), -1, dtype=np.int64)
        count = feats.shape[1]
        x[:, :count] = feats
        mask[:, :count] = True
        padded_starts[:count] = starts
        x_chunks.append(x); token_masks.append(mask); starts_all.append(padded_starts)
        ages.append(float(snapshot_start)); ends.append(float(values[-1, 0]))
        append_sample_meta(meta_acc, dataset=cfg.dataset,
                           segment_id=f"realhyfc_stack__snapshot_{snapshot_index:04d}",
                           file_path=RAW_ARCHIVE,
                           meta={"group_id": "RealHyFC_stack", "unit_id": "RealHyFC_stack_01",
                                 "run_id": "heavy_duty_durability", "condition_id": "heavy_duty_load_cycle",
                                 "file_id": RAW_ARCHIVE.stem, "filename": RAW_ARCHIVE.name,
                                 "source_relpath": RAW_ARCHIVE.name, "source_split": "realhyfc_stack_durability",
                                 "file_index": 0, "snapshot_index": snapshot_index,
                                 "cycle_index": snapshot_index}, chunk_id=0)
        snapshot_index += 1
        active = []

    for age, voltage, current in _rows(strict=bool(cfg.fail_on_segment_error)):
        if snapshot_start is None:
            snapshot_start = age
        if age - snapshot_start >= SNAPSHOT_HOURS:
            flush()
            if limit_segments is not None and snapshot_index >= int(limit_segments):
                break
            snapshot_start = age
        active.append((age, voltage, current))
    if limit_segments is None or snapshot_index < int(limit_segments):
        flush()
    if not x_chunks:
        raise RuntimeError("No valid RealHyFC operating snapshots were produced")

    x_all = np.stack(x_chunks)
    token_mask = np.stack(token_masks)
    # Downstream scaling is deliberately deferred to each train fold.
    if cfg.apply_robust_scaling:
        x_all, median, iqr = robust_fit_transform(
            x_all,
            token_mask,
            clip_value=cfg.robust_clip_abs,
        )
    else:
        median = np.zeros((x_all.shape[-1],), dtype=np.float32)
        iqr = np.ones((x_all.shape[-1],), dtype=np.float32)
    sample_meta = finalize_meta(meta_acc)
    sample_meta["sample_age_hours"] = np.asarray(ages, dtype=np.float32)
    sample_meta["sample_snapshot_end_hours"] = np.asarray(ends, dtype=np.float32)
    out_path = Path(out_root) / (cfg.output_name or f"{cfg.dataset}_health_tokens.npz")
    n = x_all.shape[0]
    save_health_token_npz(
        out_path=out_path, dataset=cfg.dataset, x_health=x_all, token_mask=token_mask,
        c_mask=np.ones((n, 3), dtype=bool), sample_meta=sample_meta,
        fs=np.ones((n,), dtype=np.float32), rotation_hz=np.full((n,), np.nan, dtype=np.float32),
        load_value=np.full((n,), np.nan, dtype=np.float32),
        window_points=np.full((n,), WINDOW_POINTS, dtype=np.int64),
        window_stride=np.full((n,), WINDOW_STRIDE, dtype=np.int64),
        window_seconds=np.full((n,), WINDOW_POINTS, dtype=np.float32),
        window_revolutions=np.full((n,), np.nan, dtype=np.float32),
        token_start_points=np.stack(starts_all), feature_median=median, feature_iqr=iqr,
        extra_meta={"dataset": cfg.dataset, "source_module": __name__,
                    "config": config_to_jsonable(cfg),
                    "snapshot_definition": "non-overlapping 10-hour dense heavy-duty operating segment",
                    "local_window_definition": "30-minute window / 15-minute stride at 1 Hz",
                    "raw_archive": str(RAW_ARCHIVE), "num_snapshots": int(n),
                    "num_segments_total": int(n), "num_segments_ok": int(n),
                    "num_segments_skipped": 0,
                    "robust_scaling_applied": bool(cfg.apply_robust_scaling),
                    "robust_clip_abs": (
                        None if cfg.robust_clip_abs is None else float(cfg.robust_clip_abs)
                    ),
                    "strict_fail_on_segment_error": bool(cfg.fail_on_segment_error),
                    "scaling": (
                        "per-feature robust median/IQR over valid upstream tokens"
                        if cfg.apply_robust_scaling
                        else "none; downstream training fold must fit scaling on train only"
                    ),
                    "preprocessing_protocol": cfg.preprocessing_protocol,
                    "x_health_scaled": bool(cfg.apply_robust_scaling),
                    "scaling_deferred_to_experiment": bool(cfg.scaling_deferred_to_experiment),
                    "pre_scaling_clip_abs": None,
                    "padding_value": 0.0,
                    "causal": True,
                    "training_transform": "train-only valid-feature median/IQR, then one clip to [-20,20]",
                    },
    )
    return {"dataset": cfg.dataset, "output_path": str(out_path), "num_snapshots": int(n),
            "tokens_per_snapshot": int(token_mask[0, 0].sum())}
