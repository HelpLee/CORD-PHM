from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from preprocess_health_tokens.raw_patch_ablation.common import (
    GLOBAL_POINTS,
    LOCAL_POINTS,
    limit_payload,
    load_reference,
    resample_1d,
    save_raw_npz,
)


CHANNELS = ("horizontal_vibration", "vertical_vibration")


def _read_xjtu(path: Path) -> np.ndarray:
    import pandas as pd

    values = pd.read_csv(path, dtype=np.float32).to_numpy()
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Invalid XJTU CSV: {path} shape={values.shape}")
    signal = values[:, :2].T.astype(np.float32, copy=False)
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite XJTU signal: {path}")
    return signal


def build(
    *,
    raw_root: Path,
    reference: Path,
    output: Path,
    limit_samples: Optional[int] = None,
) -> Dict[str, object]:
    inherited = load_reference(reference)
    total = int(inherited["token_mask"].shape[0])
    count = total if limit_samples is None else min(total, int(limit_samples))
    inherited = limit_payload(inherited, count)
    token_mask = np.asarray(inherited["token_mask"], dtype=bool)
    starts = np.asarray(inherited["token_start_points"], dtype=np.int64)
    widths = np.asarray(inherited["window_points"], dtype=np.int64)
    relpaths = np.asarray(inherited["sample_source_relpath"]).astype(str)
    if token_mask.shape != (count, 2, 64) or starts.shape != (count, 64):
        raise ValueError(f"Unexpected XJTU reference contract: {token_mask.shape}, {starts.shape}")

    local = np.zeros((count, 2, 64, LOCAL_POINTS), dtype=np.float32)
    global_ = np.zeros((count, 2, GLOBAL_POINTS), dtype=np.float32)
    for row in range(count):
        signal = _read_xjtu(raw_root / Path(relpaths[row]))
        width = int(widths[row])
        for channel in range(2):
            global_[row, channel] = resample_1d(signal[channel], GLOBAL_POINTS)
            for token in np.flatnonzero(token_mask[row, channel]):
                start = int(starts[row, token])
                stop = min(start + width, signal.shape[1])
                if start < 0 or stop <= start:
                    raise ValueError(f"Invalid XJTU patch row={row}, token={token}, [{start}:{stop}]")
                local[row, channel, token] = resample_1d(signal[channel, start:stop], LOCAL_POINTS)
        if (row + 1) % 250 == 0:
            print(f"[xjtu] {row + 1}/{count}", flush=True)

    return save_raw_npz(
        output,
        reference=reference,
        inherited=inherited,
        x_raw_local=local,
        x_raw_global=global_,
        raw_signal_name="vibration_amplitude",
        dataset="xjtu_downstream_bearing",
        raw_roots=(raw_root,),
        extra_metadata={
            "physical_channels": list(CHANNELS),
            "window_alignment": "reference token_start_points and window_points; no re-derived split",
        },
    )
