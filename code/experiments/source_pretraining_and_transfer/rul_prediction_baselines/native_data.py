"""Source-only normalized raw-waveform views aligned to the canonical RUL splits.

The raw files are independent of the handcrafted 26-feature HealthToken values:
each sensor window contains 64 x 26 parameter-free resampled amplitudes.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from models import InputShape
from run import Dataset


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
RAW = CODE / "data_phm" / "processed_raw_patches"
BEARING_ROWS = (CODE / "experiments" / "source_pretraining_and_transfer" /
                "shared_domain_components" / "bearing_runtime" /
                "global_local_cache" / "xjtu_condition2" / "arrays.npz")


class NativeDataset(Dataset):
    def __init__(self, domain: str, fraction: float, view: str = "source_scaled"):
        super().__init__(domain, fraction)
        path = next((RAW / domain).glob("*.npz"))
        with np.load(path, allow_pickle=False) as raw:
            if domain == "bearing":
                with np.load(BEARING_ROWS) as subset:
                    rows = subset["source_rows"]
                local = raw["x_raw_local"][rows]
                global_x = raw["x_raw_global"][rows]
                channel_mask = raw["c_mask"][rows]
                token_mask = raw["token_mask"][rows]
            else:
                local = raw["x_raw_local"]
                global_x = raw["x_raw_global"]
                channel_mask = raw["c_mask"]
                token_mask = raw["token_mask"]
                if "y_rul_norm" in raw:
                    if not np.allclose(raw["y_rul_norm"], self.y, atol=1e-6):
                        raise ValueError(f"{domain}: raw labels do not align with canonical split")
            if domain == "battery":
                # Capacity and cycle duration are observed cycle measurements,
                # not the RUL target. Do not include y_rul or wear labels.
                observed = np.stack((raw["cycle_capacity_ah"],
                                     raw["cycle_duration_s"]), axis=-1).astype(np.float32)
            else:
                observed = None
        if len(local) != len(self.y):
            raise ValueError(f"{domain}: raw sample count does not match canonical split")
        local = np.asarray(local, np.float32)
        global_x = np.asarray(global_x, np.float32)
        self.channel_mask = np.asarray(channel_mask, bool)
        original_mask = np.asarray(token_mask, bool)
        # Pack the actually observed raw windows; the upstream 64-slot layout
        # may contain only 7--9 valid bearing windows or 32 milling windows.
        # Interpolation is parameter-free and uses only this snapshot.
        packed = np.zeros_like(local)
        destination_axis = np.linspace(0, 1, local.shape[2] * local.shape[3])
        for row in range(len(local)):
            for channel in range(local.shape[1]):
                if not self.channel_mask[row, channel]:
                    continue
                observed_wave = local[row, channel, original_mask[row, channel]].reshape(-1)
                if not len(observed_wave):
                    continue
                source_axis = np.linspace(0, 1, len(observed_wave))
                packed[row, channel] = np.interp(
                    destination_axis, source_axis, observed_wave).reshape(local.shape[2:])
        local = packed
        self.token_mask = np.broadcast_to(
            self.channel_mask[..., None], original_mask.shape).copy()
        source_rows = np.unique(self.train[0].ravel())
        center = np.zeros((1, local.shape[1], 1, 1), np.float32)
        scale = np.ones_like(center)
        for channel in range(local.shape[1]):
            valid_rows = source_rows[self.channel_mask[source_rows, channel]]
            values = local[valid_rows, channel][self.token_mask[valid_rows, channel]]
            if len(values):
                center[0, channel, 0, 0] = np.median(values)
                q25, q75 = np.percentile(values, [25, 75])
                scale[0, channel, 0, 0] = max(float(q75 - q25), 1e-6)
        if view == "moment_raw":
            self.local = local.copy()
        elif view == "source_scaled":
            self.local = np.clip(np.arcsinh((local - center) / scale), -8, 8).astype(np.float32)
        else:
            raise ValueError(view)
        self.local *= (self.channel_mask[:, :, None, None] &
                       self.token_mask[:, :, :, None])
        self.view = view
        # Raw whole-snapshot waveform view, never the handcrafted global features.
        if observed is not None:
            global_x = np.concatenate((global_x, observed[:, None, :]), axis=-1)
        global_center = np.median(global_x[source_rows], axis=0, keepdims=True)
        global_q25, global_q75 = np.percentile(
            global_x[source_rows], [25, 75], axis=0, keepdims=True)
        global_scale = np.maximum(global_q75 - global_q25, 1e-6)
        self.global_x = np.clip(np.arcsinh(
            (global_x - global_center) / global_scale), -8, 8).astype(np.float32)
        self.shape = InputShape(self.train[0].shape[1], local.shape[1],
                                local.shape[2], local.shape[3])
        self._device_cache = {}
        self.raw_path = str(path)
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.raw_sha256 = sidecar["output_sha256"]
        self.raw_signal_name = sidecar["raw_signal_name"]
        self.normalization = {"source_rows": int(len(source_rows)),
                              "center": center.reshape(-1).tolist(),
                              "iqr": scale.reshape(-1).tolist()}

    def flatten(self, sample_set):
        """Handcrafted but model-generic waveform statistics for tree baselines."""
        sequence, history_mask = sample_set
        x = self.local[sequence].reshape(*sequence.shape, self.shape.channels, -1)
        valid = (self.channel_mask[sequence] & history_mask[:, :, None])
        features = np.stack([
            x.mean(-1), x.std(-1), np.sqrt(np.mean(x * x, -1)),
            x.min(-1), x.max(-1), np.percentile(x, 25, axis=-1),
            np.percentile(x, 75, axis=-1),
            np.mean(x * x, axis=-1),
        ], axis=-1)
        features *= valid[..., None]
        weight = valid[..., None].astype(np.float32)
        average = features.sum(1) / weight.sum(1).clip(min=1)
        first = features[:, 0]
        last = features[:, -1]
        trend = last - first
        waveform_stats = np.concatenate([last, average, trend], axis=-1).reshape(len(x), -1)
        global_values = self.global_x[sequence].reshape(
            len(sequence), sequence.shape[1], -1)
        global_values *= history_mask[..., None]
        global_summary = np.concatenate((
            global_values[:, -1],
            global_values.sum(1) / history_mask.sum(1)[:, None].clip(min=1),
            global_values[:, -1] - global_values[:, 0]), axis=-1)
        return np.concatenate((waveform_stats, global_summary), axis=-1)
