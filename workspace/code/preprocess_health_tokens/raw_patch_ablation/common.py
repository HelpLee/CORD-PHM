from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


LOCAL_POINTS = 26
GLOBAL_POINTS = 26

# Handcrafted-only arrays must never enter the raw-input artifact.
HANDCRAFTED_KEYS = {
    "x_health",
    "x_global",
    "feature_mask",
    "global_feature_mask",
    "feature_names",
    "feature_median",
    "feature_iqr",
    "meta_json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resample_1d(values: np.ndarray, points: int) -> np.ndarray:
    """Linearly resample one finite physical trajectory without normalizing amplitude."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros((points,), dtype=np.float32)
    if not finite.all():
        idx = np.arange(x.size, dtype=np.float64)
        x = np.interp(idx, idx[finite], x[finite])
    if x.size == 1:
        return np.full((points,), x[0], dtype=np.float32)
    old = np.linspace(0.0, 1.0, x.size, dtype=np.float64)
    new = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    return np.interp(new, old, x).astype(np.float32)


def load_reference(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files if key not in HANDCRAFTED_KEYS}


def reference_sample_count(path: Path) -> int:
    with np.load(path, allow_pickle=True) as archive:
        return int(archive["token_mask"].shape[0])


def limit_payload(payload: Mapping[str, np.ndarray], count: int) -> Dict[str, np.ndarray]:
    """Limit only sample-aligned fields; scalar and dataset-level fields remain intact."""
    total = int(np.asarray(payload["token_mask"]).shape[0])
    result: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        result[key] = arr[:count] if arr.ndim > 0 and arr.shape[0] == total else arr
    return result


def save_raw_npz(
    output: Path,
    *,
    reference: Path,
    inherited: Mapping[str, np.ndarray],
    x_raw_local: np.ndarray,
    x_raw_global: np.ndarray,
    raw_signal_name: str,
    dataset: str,
    raw_roots: Iterable[Path],
    extra_metadata: Mapping[str, object],
) -> Dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    local = np.asarray(x_raw_local)
    global_ = np.asarray(x_raw_global)
    token_mask = np.asarray(inherited["token_mask"], dtype=bool)
    c_mask = np.asarray(inherited["c_mask"], dtype=bool)
    if local.ndim != 4 or local.shape[:3] != token_mask.shape:
        raise ValueError(f"x_raw_local/token_mask mismatch: {local.shape} vs {token_mask.shape}")
    if global_.ndim != 3 or global_.shape[:2] != c_mask.shape:
        raise ValueError(f"x_raw_global/c_mask mismatch: {global_.shape} vs {c_mask.shape}")
    if local.shape[-1] != LOCAL_POINTS or global_.shape[-1] != GLOBAL_POINTS:
        raise ValueError("Unexpected raw resampling length")
    if not np.isfinite(local).all() or not np.isfinite(global_).all():
        raise ValueError("Raw-patch arrays contain non-finite values")
    if np.count_nonzero(local[~token_mask]) != 0:
        raise ValueError("Invalid local-token slots must remain exactly zero")

    reference_hash = sha256(reference)
    metadata = {
        "dataset": dataset,
        "representation": "parameter_free_raw_waveform_26",
        "paired_handcrafted_npz": str(reference.resolve()),
        "paired_handcrafted_sha256": reference_hash,
        "sample_alignment": "exact row order, labels, token_mask and c_mask inherited from paired handcrafted NPZ",
        "x_raw_local_semantics": "[N,C,64,26]; each inherited raw window is linearly resampled to 26 values",
        "x_raw_global_semantics": "[N,C,26]; complete raw observation linearly resampled to 26 values",
        "raw_signal_name": raw_signal_name,
        "normalization": "none in preprocessing; downstream applies source-only channel/position median-IQR, asinh and clipping exactly like Handcrafted-26",
        "raw_roots": [str(Path(root).resolve()) for root in raw_roots],
        **dict(extra_metadata),
    }
    payload = dict(inherited)
    payload.update(
        x_raw_local=local.astype(np.float32, copy=False),
        x_raw_global=global_.astype(np.float32, copy=False),
        raw_signal_name=np.asarray(raw_signal_name),
        paired_handcrafted_sha256=np.asarray(reference_hash),
        raw_patch_meta_json=np.asarray(json.dumps(metadata, ensure_ascii=False, indent=2)),
    )
    np.savez_compressed(output, **payload)
    result = {
        "dataset": dataset,
        "output": str(output.resolve()),
        "samples": int(local.shape[0]),
        "x_raw_local_shape": list(local.shape),
        "x_raw_global_shape": list(global_.shape),
        "paired_handcrafted_sha256": reference_hash,
        "output_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(result | metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
