#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALCE CS2 battery -> CNN-LSTM-Attention baseline NPZ.

Default input:
    data_phm/raw/Battery/CALCE_CS2/{CS2_35,CS2_36,CS2_37,CS2_38}/

Default output:
    data_phm/processed_cnn_lstm_attention_baselines/battery/raw_calce_cs2_window_sequences/
        cnn_lstm_raw_calce_cs2_cycle_windows_len64_stride16.npz

NPZ keys:
    x_raw       float32 [N, 26, 64]  # 26 cycle features, 64-cycle sequence
    token_mask  bool    [N, 64]      # True for real cycles, False for left padding
    y_rul       float32 [N]          # remaining cycles
    y_rul_norm  float32 [N]
    cell, cycle_index, feature_names, meta_json, ...

Environment:
    RAW_CALCE_CS2_ROOT
    CNN_LSTM_BATTERY_CACHE_DIR
    MAX_TOKENS=64
    SEQ_STRIDE=16
    CALCE_CS2_TARGET_CELLS=CS2_35,CS2_36,CS2_37,CS2_38
    FORCE_REBUILD=1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess_health_tokens.common.battery_health_token_utils import (
    TARGET_CALCE_CS2_CELLS,
    default_meta,
    load_calce_cs2_trajectories,
    make_calce_cs2_sequence_cache,
    parse_cell_list,
    resolve_calce_cs2_raw_root,
    save_npz_with_meta,
    str_to_bool,
)
from preprocess_health_tokens.common.battery_health_token_utils import detect_project_root
from preprocess_health_tokens.datasets.battery.calce_cs2_battery import SEQ_STRIDE


def main() -> None:
    root = detect_project_root()
    raw_root = resolve_calce_cs2_raw_root(root)
    target_cells = parse_cell_list(os.environ.get("CALCE_CS2_TARGET_CELLS", ",".join(TARGET_CALCE_CS2_CELLS)))
    max_tokens = int(os.environ.get("MAX_TOKENS", "64"))
    stride = int(os.environ.get("SEQ_STRIDE", str(SEQ_STRIDE)))
    force = str_to_bool(os.environ.get("FORCE_REBUILD", ""), False)

    cache_dir = Path(os.environ.get("CNN_LSTM_BATTERY_CACHE_DIR", "")).expanduser()
    if not str(cache_dir).strip() or str(cache_dir) == ".":
        cache_dir = root / "data_phm" / "processed_cnn_lstm_attention_baselines" / "battery" / "raw_calce_cs2_window_sequences"
    out_path = cache_dir.resolve() / f"cnn_lstm_raw_calce_cs2_cycle_windows_len{max_tokens}_stride{stride}.npz"

    if out_path.exists() and not force:
        print(f"[SKIP] Existing file: {out_path}")
        print("Set FORCE_REBUILD=1 to rebuild.")
        return

    print("=" * 80)
    print("CALCE CS2 battery sequence preprocessing for CNN-LSTM-Attention")
    print(f"raw_root    : {raw_root}")
    print(f"out_path    : {out_path}")
    print(f"target_cells: {target_cells}")
    print(f"max/stride  : {max_tokens}/{stride}")
    print("=" * 80)

    trajectories = load_calce_cs2_trajectories(raw_root=raw_root, target_cells=target_cells)
    payload = make_calce_cs2_sequence_cache(
        trajectories=trajectories,
        max_tokens=max_tokens,
        stride=stride,
    )
    meta = default_meta(
        script=str(Path(__file__).resolve()),
        raw_root=raw_root,
        out_path=out_path,
        target_cells=target_cells,
        payload=payload,
        note="Battery 64-cycle sequence cache for CNN-LSTM-Attention baseline, independent of FM health-token NPZ.",
    )
    meta.update({"max_tokens": int(max_tokens), "sequence_stride": int(stride), "x_raw_semantics": "[N, 26, 64]"})
    save_npz_with_meta(out_path, payload, meta)

    print("[DONE]")
    print(f"saved: {out_path}")
    print(f"x_raw: {payload['x_raw'].shape}")


if __name__ == "__main__":
    main()
