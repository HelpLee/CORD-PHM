from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from preprocess_health_tokens.prepare_phm2010_downstream import CHANNELS, read_signal
from preprocess_health_tokens.raw_patch_ablation.common import (
    GLOBAL_POINTS,
    LOCAL_POINTS,
    limit_payload,
    load_reference,
    resample_1d,
    save_raw_npz,
)


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
    relpaths = np.asarray(inherited["sample_source_relpath"]).astype(str)
    if token_mask.shape != (count, 7, 64):
        raise ValueError(f"Unexpected PHM2010 reference contract: {token_mask.shape}")

    local = np.zeros((count, 7, 64, LOCAL_POINTS), dtype=np.float32)
    global_ = np.zeros((count, 7, GLOBAL_POINTS), dtype=np.float32)
    for row in range(count):
        signal = read_signal(raw_root / Path(relpaths[row]))
        edges = np.linspace(0, signal.shape[1], 33, dtype=np.int64)
        expected = np.zeros((7, 64), dtype=bool)
        expected[:, :32] = True
        if not np.array_equal(token_mask[row], expected):
            raise ValueError(f"PHM2010 row {row} does not follow the strict 32-valid/64-slot contract")
        for channel in range(7):
            global_[row, channel] = resample_1d(signal[channel], GLOBAL_POINTS)
            for token, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
                local[row, channel, token] = resample_1d(signal[channel, start:stop], LOCAL_POINTS)
        if (row + 1) % 100 == 0:
            print(f"[phm2010] {row + 1}/{count}", flush=True)

    return save_raw_npz(
        output,
        reference=reference,
        inherited=inherited,
        x_raw_local=local,
        x_raw_global=global_,
        raw_signal_name="native_sensor_amplitude",
        dataset="phm2010_milling_downstream",
        raw_roots=(raw_root,),
        extra_metadata={
            "physical_channels": list(CHANNELS),
            "window_alignment": "32 contiguous equal-index intervals from the complete cut; slots 32..63 stay masked zero",
            "wear_is_model_input": False,
        },
    )
