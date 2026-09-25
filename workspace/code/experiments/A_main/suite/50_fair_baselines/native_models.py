"""Native waveform baselines; no handcrafted HealthToken features enter these models."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models import InputShape, MomentRUL


def raw_series(local, channel_mask, history_mask, points_per_snapshot=32):
    b, h, c, t, f = local.shape
    wave = local.permute(0, 2, 1, 3, 4).reshape(b * c * h, 1, t * f)
    wave = F.interpolate(wave, size=points_per_snapshot, mode="linear",
                         align_corners=False)
    wave = wave.reshape(b, c, h, points_per_snapshot)
    wave = wave * history_mask[:, None, :, None]
    present = channel_mask.any(dim=1)
    return wave.reshape(b, c, h * points_per_snapshot), present


def instance_normalize(series):
    mean = series.mean(dim=-1, keepdim=True)
    scale = series.var(dim=-1, unbiased=False, keepdim=True).add(1e-5).sqrt()
    return (series - mean) / scale


class NativeCNNLSTM(nn.Module):
    def __init__(self, shape: InputShape, width=64, hidden=96, dropout=.05):
        super().__init__()
        self.channels = shape.channels
        self.extractor = nn.Sequential(
            nn.Conv1d(shape.channels, width, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=4, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.lifecycle = nn.LSTM(width, hidden, batch_first=True)
        self.global_projection = nn.LazyLinear(width)
        self.attention = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def snapshot(self, local):
        b, h, c, t, f = local.shape
        return self.extractor(local.reshape(b * h, c, t * f)).squeeze(-1).reshape(b, h, -1)

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        snapshot = self.snapshot(local) + self.global_projection(global_x.mean(dim=2))
        state, _ = self.lifecycle(snapshot)
        score = self.attention(state).squeeze(-1).masked_fill(~history_mask, -1e4)
        pooled = (state * torch.softmax(score, dim=1)[..., None]).sum(1)
        return torch.sigmoid(self.head(pooled).squeeze(-1))


class NativeTCN(nn.Module):
    def __init__(self, shape: InputShape, width=64, hidden=96, dropout=.05):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(shape.channels, width, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=4, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.project = nn.Conv1d(width, hidden, 1)
        self.global_projection = nn.LazyLinear(width)
        self.conv1 = nn.Conv1d(hidden, hidden, 3)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, dilation=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        b, h, c, t, f = local.shape
        snapshot = self.extractor(local.reshape(b * h, c, t * f))
        snapshot = snapshot.squeeze(-1).reshape(b, h, -1)
        state = (snapshot +
                 self.global_projection(global_x.mean(dim=2))) * history_mask[..., None]
        state = self.project(state.transpose(1, 2))
        state = F.gelu(self.conv1(F.pad(state, (2, 0))))
        state = F.gelu(self.conv2(F.pad(state, (4, 0))))
        return torch.sigmoid(self.head(state[:, :, -1]).squeeze(-1))


class NativePatchTST(nn.Module):
    def __init__(self, shape: InputShape, width=96, heads=4, layers=3,
                 patch_len=32, stride=16, dropout=.05):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.patch = nn.Linear(patch_len, width)
        length = shape.history * 32
        self.position = nn.Parameter(torch.zeros(1, 1 + (length - patch_len) // stride, width))
        block = nn.TransformerEncoderLayer(width, heads, width * 2, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(width))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.global_projection = nn.LazyLinear(width)

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        series, present = raw_series(local, channel_mask, history_mask)
        series = instance_normalize(series)
        b, c, _ = series.shape
        patches = self.patch(series.unfold(-1, self.patch_len, self.stride))
        n = patches.shape[2]
        encoded = self.encoder(patches.reshape(b * c, n, -1) + self.position[:, :n])
        state = encoded.mean(1).reshape(b, c, -1)
        weight = present[..., None].to(state.dtype)
        pooled = (state * weight).sum(1) / weight.sum(1).clamp_min(1)
        pooled = pooled + self.global_projection(global_x[:, -1].mean(dim=1))
        return torch.sigmoid(self.head(pooled).squeeze(-1))


class NativeITransformer(nn.Module):
    def __init__(self, shape: InputShape, width=96, heads=4, layers=3, dropout=.05):
        super().__init__()
        self.embedding = nn.Linear(shape.history * 32, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 2, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(width))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.global_projection = nn.LazyLinear(width)

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        series, present = raw_series(local, channel_mask, history_mask)
        tokens = self.embedding(instance_normalize(series))
        encoded = self.encoder(tokens, src_key_padding_mask=~present)
        weight = present[..., None].to(encoded.dtype)
        pooled = (encoded * weight).sum(1) / weight.sum(1).clamp_min(1)
        pooled = pooled + self.global_projection(global_x[:, -1].mean(dim=1))
        return torch.sigmoid(self.head(pooled).squeeze(-1))


class NativeMoment(nn.Module):
    """Official MOMENT embedding on a 512-point raw waveform history."""
    def __init__(self, shape: InputShape, config: dict, freeze: bool):
        super().__init__()
        wrapper = MomentRUL(shape, config["model_id"], config["revision"],
                            config["interface_channels"], 512, freeze)
        self.foundation = wrapper.foundation
        self.head = nn.Linear(self.foundation.config.d_model, 1)
        self.global_projection = nn.LazyLinear(self.foundation.config.d_model)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.foundation.parameters()):
            self.foundation.eval()
        return self

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        series, present = raw_series(local, channel_mask, history_mask,
                                     points_per_snapshot=max(1, 512 // history_mask.shape[1]))
        series = F.interpolate(series, size=512, mode="linear", align_corners=False)
        # MOMENT performs its own RevIN normalization internally.
        embedded = self.foundation(x_enc=series, input_mask=torch.ones(
            series.shape[0], 512, dtype=torch.long, device=series.device)).embeddings
        embedded = torch.nan_to_num(embedded)
        embedded = embedded + self.global_projection(global_x[:, -1].mean(dim=1))
        return torch.sigmoid(self.head(embedded).squeeze(-1))


def build_native(name: str, shape: InputShape, config: dict):
    if name == "native_cnn_lstm_attention":
        return NativeCNNLSTM(shape, config["width"], config["hidden"], config["dropout"])
    if name == "native_tcn":
        return NativeTCN(shape, config["width"], config["hidden"], config["dropout"])
    if name == "native_patchtst":
        return NativePatchTST(shape, config["width"], config["heads"], config["layers"],
                              config["patch_len"], config["stride"], config["dropout"])
    if name == "native_itransformer":
        return NativeITransformer(shape, config["width"], config["heads"],
                                  config["layers"], config["dropout"])
    if name in ("native_moment_frozen", "native_moment_full"):
        return NativeMoment(shape, config, freeze=name.endswith("frozen"))
    raise KeyError(name)
