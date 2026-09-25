"""Build one immutable all-channel Milling runtime shared by both retrain arms."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
A_SUITE = HERE.parent
CODE = HERE / "three_domain" / "code"
sys.path.insert(0, str(CODE))

import data  # noqa: E402
sys.path.insert(1, str(data.ROOT / "code"))
import feature_observations as fo  # noqa: E402
import global_local_data as gl  # noqa: E402


DATA = data.ROOT / "code/data_phm/processed_health_tokens/milling"
RUNTIME = HERE / "runtime/milling"
DATASETS = (
    "luh_milling", "matwi_milling", "nonastreda_milling",
    "qit_cemc_milling", "piecuch_milling", "hmotp_milling", "nasa_milling",
)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def build_cache(name: str) -> Path:
    source = (DATA / f"{name}_health_tokens.npz").resolve()
    destination = RUNTIME / "cache" / name
    signature = dict(
        path=str(source), bytes=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns,
        sha256=digest(source), upstream_cap_per_unit=None, downstream=False,
        version=2, channel_policy="all recorded canonical channels",
    )
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            raise RuntimeError(f"Refusing stale cache for {name}")
        return destination

    with np.load(source, allow_pickle=False) as archive:
        x = np.asarray(archive["x_health"], dtype=np.float32)
        tm = np.asarray(archive["token_mask"], dtype=bool)
        cm = np.asarray(archive["c_mask"], dtype=bool)
        units = archive["sample_unit_id"].astype(str)
        order = np.asarray(archive["sample_snapshot_index"], dtype=float)
        features = archive["feature_names"].astype(str).tolist()
        metadata = json.loads(str(archive["meta_json"].item()))
    if x.ndim != 4 or x.shape[2:] != (64, 26) or tm.shape != x.shape[:3] or cm.shape != x.shape[:2]:
        raise ValueError(f"Invalid canonical contract for {name}: {x.shape}/{tm.shape}/{cm.shape}")
    if np.any(tm & ~cm[..., None]):
        raise ValueError(f"Token/channel masks disagree for {name}")
    valid = tm & cm[..., None]
    if not np.isfinite(x[valid]).all():
        raise ValueError(f"Non-finite observed value in {name}")
    x[~valid] = 0
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination / "x.npy", x)
    rows = np.arange(len(x), dtype=np.int64)
    np.savez(destination / "arrays.npz", tm=tm, cm=cm, y=np.zeros(len(x), np.float32),
             units=units, order=order, source_rows=rows)
    write(manifest_path, dict(signature=signature, domain="milling", downstream=False,
                              shape=list(x.shape), counts={u: int(np.sum(units == u)) for u in sorted(set(units))},
                              excluded={}, features=features, metadata=metadata))
    print("CACHE_DONE", name, tuple(x.shape), flush=True)
    return destination


def temporal_windows(store, units):
    blocks = [np.lib.stride_tricks.sliding_window_view(store.groups[u], 7)
              for u in units if len(store.groups[u]) >= 7]
    return np.concatenate(blocks) if blocks else np.empty((0, 7), dtype=int)


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    gl.OUT = fo.OUT = RUNTIME
    audit = []
    for name in DATASETS:
        local = fo.observed_store(data.Store(build_cache(name)))
        store = gl.GlobalLocalStore(local, calibrate=False, channels=local.x.shape[1])
        train, validation = store.split()
        norm = store.normalize(train)
        train_windows = temporal_windows(store, train)
        validation_windows = temporal_windows(store, validation)
        if not len(train_windows):
            raise ValueError(f"No seven-snapshot training windows for {name}")
        audit.append(dict(
            dataset=name, train=train, validation=validation,
            train_windows=int(len(train_windows)), validation_windows=int(len(validation_windows)),
            checkpoint_monitor=("unit-disjoint validation" if validation else
                                "train-only; excluded from checkpoint selection"),
            eligible_train=int(sum(len(v) for v in store.eligible(train).values())),
            eligible_validation=int(sum(len(v) for v in store.eligible(validation).values())),
            local_source=store.info["signature"], center_scale=[v.tolist() for v in norm],
            channels=int(local.x.shape[1]), channel_policy="all recorded canonical channels",
        ))
        print("RUNTIME_READY", name, train, validation, flush=True)
    write(RUNTIME / "upstream_splits_scalers.json", audit)
    write(RUNTIME / "status.json", dict(state="complete", datasets=list(DATASETS)))


if __name__ == "__main__":
    main()
