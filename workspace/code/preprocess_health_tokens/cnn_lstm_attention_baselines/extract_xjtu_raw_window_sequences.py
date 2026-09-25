#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw CSV -> CNN-LSTM-Attention baseline NPZ for XJTU-SY bearing RUL.

This is separated from the health-token / foundation-model preprocessing.
It reads original XJTU CSV files and saves fixed-length raw vibration windows.

Default input:
    data_phm/raw/Bearing/XJTU/{35Hz12kN,37.5Hz11kN,40Hz10kN}/Bearing*/1.csv ...

Default output:
    data_phm/processed_cnn_lstm_attention_baselines/bearing/raw_xjtu_window_sequences/
        cnn_lstm_raw_xjtu_snapshot_windows_len2048_tail.npz

Env:
    RAW_XJTU_ROOT
    CNN_LSTM_CACHE_DIR
    WINDOW_LEN=2048
    WINDOW_MODE=tail|center|head|uniform
    FORCE_REBUILD=1
"""
from __future__ import annotations
import json, os, re, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

CONDITION_TO_BEARINGS: Dict[str, List[str]] = {
    "35Hz12kN": ["Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
    "37.5Hz11kN": ["Bearing2_1", "Bearing2_2", "Bearing2_3", "Bearing2_4", "Bearing2_5"],
    "40Hz10kN": ["Bearing3_1", "Bearing3_2", "Bearing3_3", "Bearing3_4", "Bearing3_5"],
}

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def env_bool(name: str, default=False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return default if not v else v in {"1", "true", "yes", "on"}

def numeric_key(p: Path) -> Tuple[int, str]:
    m = re.search(r"(\d+)", p.stem)
    return (int(m.group(1)) if m else 10**12, p.name)

def read_csv_2ch(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        arr = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 2:
        raise ValueError(f"{path} has fewer than 2 columns: {arr.shape}")
    arr = arr[:, :2]
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] == 0:
        raise ValueError(f"{path} has no finite rows.")
    return arr.T.astype(np.float32)  # [2, N]

def select_window(x: np.ndarray, L: int, mode: str) -> np.ndarray:
    c, n = x.shape
    if mode == "uniform":
        if n == L:
            return x.astype(np.float32)
        pos = np.linspace(0, n - 1, L)
        lo = np.floor(pos).astype(np.int64)
        hi = np.ceil(pos).astype(np.int64)
        w = (pos - lo).astype(np.float32)
        return ((1 - w)[None, :] * x[:, lo] + w[None, :] * x[:, hi]).astype(np.float32)
    if n >= L:
        if mode == "head":
            return x[:, :L].astype(np.float32)
        if mode == "center":
            s = max(0, (n - L) // 2)
            return x[:, s:s+L].astype(np.float32)
        return x[:, -L:].astype(np.float32)  # tail
    out = np.zeros((c, L), dtype=np.float32)
    out[:, :n] = x
    if n > 0:
        out[:, n:] = x[:, -1:]
    return out

def main():
    pr = project_root()
    raw_root = Path(os.environ.get("RAW_XJTU_ROOT", pr / "data_phm/raw/Bearing/XJTU")).expanduser().resolve()
    cache_dir = Path(os.environ.get("CNN_LSTM_CACHE_DIR", pr / "data_phm/processed_cnn_lstm_attention_baselines/bearing/raw_xjtu_window_sequences")).expanduser().resolve()
    L = int(os.environ.get("WINDOW_LEN", "2048"))
    mode = os.environ.get("WINDOW_MODE", "tail").strip().lower()
    force = env_bool("FORCE_REBUILD", False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"cnn_lstm_raw_xjtu_snapshot_windows_len{L}_{mode}.npz"

    print("======================================")
    print("Extract raw windows for CNN-LSTM-Attention")
    print(f"raw_root : {raw_root}")
    print(f"out_path : {out_path}")
    print(f"L/mode   : {L}/{mode}")
    print("======================================")
    if out_path.exists() and not force:
        print(f"[SKIP] Existing file: {out_path}")
        print("       Set FORCE_REBUILD=1 to rebuild.")
        return
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    xs, conditions, bearings, file_indices, file_paths, npoints = [], [], [], [], [], []
    t0 = time.time()
    for cond, bs in CONDITION_TO_BEARINGS.items():
        for b in bs:
            d = raw_root / cond / b
            if not d.exists():
                print(f"[WARN] missing {d}")
                continue
            files = sorted(d.glob("*.csv"), key=numeric_key)
            print(f"[INFO] {cond}/{b}: {len(files)} csv")
            for p in files:
                try:
                    raw = read_csv_2ch(p)
                    win = select_window(raw, L, mode)
                    xs.append(win)
                    conditions.append(cond)
                    bearings.append(b)
                    file_indices.append(numeric_key(p)[0])
                    file_paths.append(str(p))
                    npoints.append(raw.shape[1])
                except Exception as e:
                    print(f"[WARN] failed {p}: {e!r}")
    if not xs:
        raise RuntimeError("No CSV files processed.")

    x_raw = np.stack(xs).astype(np.float32)   # [N,2,L]
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_root": str(raw_root),
        "window_len": L,
        "window_mode": mode,
        "shape": list(x_raw.shape),
        "description": "Raw fixed-length vibration windows for CNN-LSTM-Attention baseline, not health-token NPZ.",
    }
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        x_raw=x_raw,
        condition=np.asarray(conditions, dtype=object),
        bearing=np.asarray(bearings, dtype=object),
        file_index=np.asarray(file_indices, dtype=np.int64),
        file_path=np.asarray(file_paths, dtype=object),
        num_points=np.asarray(npoints, dtype=np.int64),
        channel_names=np.asarray(["horizontal", "vertical"], dtype=object),
        fs=np.asarray([25600.0], dtype=np.float32),
        window_len=np.asarray([L], dtype=np.int64),
        window_mode=np.asarray([mode], dtype=object),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    tmp.replace(out_path)
    print("======================================")
    print("[DONE]")
    print(f"saved: {out_path}")
    print(f"x_raw: {x_raw.shape}")
    print(f"time : {time.time() - t0:.1f}s")
    print("======================================")

if __name__ == "__main__":
    main()
