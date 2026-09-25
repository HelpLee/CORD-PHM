"""Model-native adapters that all consume the complete HealthToken input."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch
from torch import nn


@dataclass
class InputShape:
    history: int
    channels: int
    tokens: int = 64
    features: int = 26

    @property
    def snapshot_flat(self) -> int:
        return self.channels * (
            self.tokens * self.features + self.features + self.tokens + 1)


def flatten_snapshot(local, global_x, channel_mask, token_mask):
    batch, history = local.shape[:2]
    return torch.cat([
        local.reshape(batch, history, -1),
        global_x.reshape(batch, history, -1),
        channel_mask.float().reshape(batch, history, -1),
        token_mask.float().reshape(batch, history, -1),
    ], dim=-1)


class CNNLSTMAttention(nn.Module):
    """CNN over local position, LSTM over lifecycle, masked time attention."""
    def __init__(self, shape: InputShape, width=64, hidden=96, dropout=.05):
        super().__init__()
        self.shape = shape
        self.local = nn.Sequential(
            nn.Conv1d(shape.features, width, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1), nn.GELU())
        self.global_projection = nn.Sequential(nn.LayerNorm(shape.features), nn.Linear(shape.features, width), nn.GELU())
        self.snapshot = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, hidden), nn.GELU())
        self.lifecycle = nn.LSTM(hidden, hidden, batch_first=True)
        self.attention = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        b, h, c, t, f = local.shape
        values = local.reshape(b * h * c, t, f).transpose(1, 2)
        encoded = self.local(values).transpose(1, 2).reshape(b, h, c, t, -1)
        valid = (token_mask & channel_mask[..., None]).unsqueeze(-1)
        pooled = (encoded * valid).sum(3) / valid.sum(3).clamp_min(1)
        global_state = self.global_projection(global_x)
        state = self.snapshot(torch.cat([pooled, global_state], dim=-1))
        state = (state * channel_mask[..., None]).sum(2) / channel_mask.sum(2, keepdim=True).clamp_min(1)
        state, _ = self.lifecycle(state)
        score = self.attention(state).squeeze(-1).masked_fill(~history_mask, float("-inf"))
        state = (state * torch.softmax(score, dim=1)[..., None]).sum(1)
        return torch.sigmoid(self.head(state).squeeze(-1))


class ControlledMLP(nn.Module):
    """Small non-pretrained regressor on the complete HealthToken interface."""
    def __init__(self, shape: InputShape, hidden=128, dropout=.05):
        super().__init__()
        self.snapshot = nn.Sequential(
            nn.LayerNorm(shape.snapshot_flat), nn.Linear(shape.snapshot_flat, hidden),
            nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        state = self.snapshot(flatten_snapshot(local, global_x, channel_mask, token_mask))
        weight = history_mask[..., None].to(state.dtype)
        state = (state * weight).sum(1) / weight.sum(1).clamp_min(1)
        return torch.sigmoid(self.head(state).squeeze(-1))


class ControlledTCN(nn.Module):
    """HealthToken snapshot projection followed by causal lifecycle convolutions."""
    def __init__(self, shape: InputShape, hidden=96, dropout=.05):
        super().__init__()
        self.snapshot = nn.Sequential(nn.LayerNorm(shape.snapshot_flat),
                                      nn.Linear(shape.snapshot_flat, hidden), nn.GELU())
        self.conv1 = nn.Conv1d(hidden, hidden, 3, dilation=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, dilation=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        state = self.snapshot(flatten_snapshot(local, global_x, channel_mask, token_mask))
        state = state * history_mask[..., None]
        temporal = torch.nn.functional.gelu(
            self.conv1(torch.nn.functional.pad(state.transpose(1, 2), (2, 0))))
        temporal = torch.nn.functional.gelu(
            self.conv2(torch.nn.functional.pad(temporal, (4, 0)))).transpose(1, 2)
        chosen = temporal[:, -1]
        return torch.sigmoid(self.head(chosen).squeeze(-1))


class FullSnapshotInterface(nn.Module):
    """Learned interface retaining every local/global value and validity mask."""
    def __init__(self, shape: InputShape, output: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(shape.snapshot_flat), nn.Linear(shape.snapshot_flat, output), nn.GELU())

    def forward(self, local, global_x, channel_mask, token_mask):
        return self.projection(flatten_snapshot(local, global_x, channel_mask, token_mask))


class PatchTSTRUL(nn.Module):
    """Channel-independent PatchTST adapted to complete snapshot inputs."""
    def __init__(self, shape: InputShape, variables=16, d_model=96, heads=4, layers=3, d_ff=192,
                 patch_len=4, stride=2, dropout=.05):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.variables = variables
        self.interface = FullSnapshotInterface(shape, variables)
        # The same patch embedding and Transformer are shared by all variables,
        # matching PatchTST's channel-independent inductive bias.
        self.patch_projection = nn.Linear(patch_len, d_model)
        maximum = 1 + max(0, (shape.history - patch_len) // stride)
        self.position = nn.Parameter(torch.zeros(1, maximum, d_model))
        block = nn.TransformerEncoderLayer(
            d_model, heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(d_model))
        self.head = nn.Sequential(nn.LayerNorm(variables * d_model), nn.Dropout(dropout),
                                  nn.Linear(variables * d_model, 1))

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        state = self.interface(local, global_x, channel_mask, token_mask)
        state = state * history_mask[..., None]
        if state.shape[1] < self.patch_len:
            padding = self.patch_len - state.shape[1]
            state = torch.cat([state.new_zeros(state.shape[0], padding, state.shape[2]), state], 1)
            history_mask = torch.cat([history_mask.new_zeros(history_mask.shape[0], padding), history_mask], 1)
        patches = state.transpose(1, 2).unfold(2, self.patch_len, self.stride)
        mask = history_mask.unfold(1, self.patch_len, self.stride).any(-1)
        patches = self.patch_projection(patches)
        b, variables, count, width = patches.shape
        patches = patches.reshape(b * variables, count, width)
        patches = patches + self.position[:, :count]
        expanded_mask = mask[:, None].expand(b, variables, count).reshape(b * variables, count)
        patches = self.encoder(patches, src_key_padding_mask=~expanded_mask)
        index = mask.sum(1).clamp_min(1) - 1
        index = index[:, None].expand(b, variables).reshape(-1)
        selected = patches[torch.arange(len(patches), device=patches.device), index]
        selected = selected.reshape(b, variables * width)
        return torch.sigmoid(self.head(selected).squeeze(-1))


class ITransformerRUL(nn.Module):
    """iTransformer variate-token attention with a scalar RUL readout.

    The snapshot interface is identical to PatchTST's. Each learned variate
    carries its complete lifecycle history, and the history mask is an extra
    variate token so padded positions remain observable to the model.
    """

    def __init__(self, shape: InputShape, variables=16, d_model=96, heads=4,
                 layers=3, d_ff=192, dropout=.05):
        super().__init__()
        self.interface = FullSnapshotInterface(shape, variables)
        self.embedding = nn.Linear(shape.history, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, heads, d_ff, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(d_model))
        self.head = nn.Sequential(
            nn.LayerNorm((variables + 1) * d_model), nn.Dropout(dropout),
            nn.Linear((variables + 1) * d_model, 1))

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        state = self.interface(local, global_x, channel_mask, token_mask)
        state = state * history_mask[..., None]
        variates = torch.cat(
            (state.transpose(1, 2), history_mask[:, None].to(state.dtype)), dim=1)
        encoded = self.encoder(self.embedding(variates))
        return torch.sigmoid(self.head(encoded.flatten(1)).squeeze(-1))


class MomentRUL(nn.Module):
    """MOMENT-1-base with a full-input learned interface and scalar RUL head."""
    def __init__(self, shape: InputShape, model_id: str, revision: str,
                 interface_channels=8, sequence_length=512, freeze=True):
        super().__init__()
        try:
            from momentfm import MOMENTPipeline
        except ImportError as error:
            raise RuntimeError("Install requirements-baselines.txt before running MOMENT") from error
        self.sequence_length = sequence_length
        model_id = os.environ.get("HEALTHTOKEN_MOMENT_MODEL", model_id)
        candidate = Path(model_id)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parent / candidate
        if candidate.is_dir():
            model_id = str(candidate)
        else:
            legacy_cache = (Path(__file__).resolve().parents[3] / "experiments" /
                            "source_pretraining_and_transfer" / "rul_prediction_baselines" /
                            "models" / "MOMENT-1-base")
            if legacy_cache.is_dir():
                model_id = str(legacy_cache)
        self.interface = FullSnapshotInterface(shape, interface_channels)
        cache_dir = (os.environ.get("HF_HUB_CACHE") or
                     os.environ.get("TRANSFORMERS_CACHE") or
                     str(Path(__file__).resolve().parent / ".hf_cache"))
        # The local transformers/huggingface_hub combination can fail to
        # retrieve config.json before constructing MOMENTPipeline (it then
        # raises ``__init__() missing config``).  This is the official
        # MOMENT-1-base config fallback; weights are still loaded from the Hub
        # or the staged cache and are never replaced by random initialization.
        moment_config = {
            "task_name": "reconstruction", "model_name": "MOMENT",
            "transformer_type": "encoder_only", "d_model": None,
            "seq_len": 512, "patch_len": 8, "patch_stride_len": 8,
            "device": "cpu", "transformer_backbone": "google/flan-t5-base",
            "enable_gradient_checkpointing": False,
            "model_kwargs": {}, "t5_config": {
                "architectures": ["T5ForConditionalGeneration"],
                "d_ff": 2048, "d_kv": 64, "d_model": 768,
                "decoder_start_token_id": 0, "dropout_rate": 0.1,
                "eos_token_id": 1, "feed_forward_proj": "gated-gelu",
                "initializer_factor": 1.0, "is_encoder_decoder": True,
                "layer_norm_epsilon": 1e-6, "model_type": "t5",
                "n_positions": 512, "num_decoder_layers": 12,
                "num_heads": 12, "num_layers": 12, "output_past": True,
                "pad_token_id": 0, "relative_attention_max_distance": 128,
                "relative_attention_num_buckets": 32, "tie_word_embeddings": False,
                "use_cache": True, "vocab_size": 32128,
            },
        }
        # momentfm >= 0.1 requires the Hub config to be forwarded explicitly;
        # older releases accepted only model_kwargs.  Supplying both is
        # compatible with the current API and avoids the missing-config error
        # when a cached checkpoint is present.
        self.foundation = MOMENTPipeline.from_pretrained(
            model_id, revision=revision,
            # The MOMENT checkpoint is already staged in the experiment cache.
            # Avoid a network lookup for every queued seed (the runner may be
            # used on an offline machine) and keep all seeds on the same files.
            cache_dir=cache_dir,
            local_files_only=Path(model_id).is_dir(),
            config=moment_config, model_kwargs={"task_name": "embedding"})
        self.foundation.init()
        for parameter in self.foundation.parameters():
            parameter.requires_grad_(not freeze)
        self.head = nn.LazyLinear(1)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.foundation.parameters()):
            self.foundation.eval()
        return self

    def forward(self, local, global_x, channel_mask, token_mask, history_mask):
        series = self.interface(local, global_x, channel_mask, token_mask).transpose(1, 2)
        # MOMENT marks a patch valid only when all eight positions are valid.
        # A six-observation history otherwise produces zero valid patches.
        # Causally fill missing initial observations with the first observation
        # and extend to the next whole patch using that same observation.
        history_mask = history_mask.bool()
        first_index = history_mask.long().argmax(dim=-1)
        first = series.gather(2, first_index[:, None, None].expand(-1, series.shape[1], 1))
        series = torch.where(history_mask[:, None], series, first)
        if series.shape[-1] > self.sequence_length:
            series = series[..., -self.sequence_length:]
        patch_len = self.foundation.patch_len
        patch_padding = (-series.shape[-1]) % patch_len
        if patch_padding:
            series = torch.cat((first.expand(-1, -1, patch_padding), series), dim=-1)
        valid_len = series.shape[-1]
        if valid_len > self.sequence_length:
            raise ValueError("MOMENT history exceeds sequence length after patch alignment")
        series = torch.nn.functional.pad(series, (self.sequence_length - valid_len, 0))
        valid = torch.nn.functional.pad(
            torch.ones((series.shape[0], valid_len), dtype=torch.long, device=series.device),
            (self.sequence_length - valid_len, 0))
        embedding = self.compact_embedding(series, valid)
        # Some battery snapshots can produce non-finite activations in the
        # third-party MOMENT reconstruction path.  Replace those activations
        # before the scalar RUL head so one bad foundation token cannot abort
        # the entire multi-seed baseline sweep.
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.sigmoid(self.head(embedding).squeeze(-1))

    def compact_embedding(self, series, valid):
        """MOMENT embedding with masked prefix patches omitted before T5.

        Patch embedding is computed at the original 512 positions, retaining
        exactly the checkpoint's positional embeddings. The removed tokens are
        attention-masked and contribute nothing to MOMENT's masked mean.
        """
        from momentfm.utils.masking import Masking

        foundation = self.foundation
        batch, channels, _ = series.shape
        normalized = foundation.normalizer(x=series, mask=valid, mode="norm")
        normalized = torch.nan_to_num(normalized, nan=0, posinf=0, neginf=0)
        patches = foundation.tokenizer(x=normalized)
        embedded = foundation.patch_embedding(patches, mask=valid)
        patch_mask = Masking.convert_seq_to_patch_view(valid, foundation.patch_len)
        keep = int(patch_mask.any(dim=0).sum().item())
        if keep < 1:
            raise RuntimeError("MOMENT received no valid patches")
        embedded = embedded[:, :, -keep:, :].reshape(batch * channels, keep, -1)
        patch_mask = patch_mask[:, -keep:]
        attention = patch_mask.repeat_interleave(channels, dim=0)
        encoded = foundation.encoder(inputs_embeds=embedded, attention_mask=attention)
        encoded = encoded.last_hidden_state.reshape(batch, channels, keep, -1).mean(dim=1)
        weights = patch_mask.unsqueeze(-1).to(encoded.dtype)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1)


def build_model(name: str, shape: InputShape, config: dict) -> nn.Module:
    if name == "controlled_mlp":
        return ControlledMLP(shape, config["hidden"], config["dropout"])
    if name == "controlled_tcn":
        return ControlledTCN(shape, config["hidden"], config["dropout"])
    if name in ("cnn_lstm_attention", "controlled_cnn_lstm_attention"):
        return CNNLSTMAttention(shape, config["cnn_width"], config["lstm_hidden"], config["dropout"])
    if name in ("patchtst", "controlled_patchtst"):
        patch = config["patch_len_by_history"][str(shape.history)]
        stride = config["stride_by_history"][str(shape.history)]
        return PatchTSTRUL(shape, config["interface_variables"], config["d_model"],
                           config["n_heads"], config["n_layers"], config["d_ff"],
                           patch, stride, config["dropout"])
    if name in ("itransformer", "controlled_itransformer"):
        return ITransformerRUL(
            shape, config["interface_variables"], config["d_model"],
            config["n_heads"], config["n_layers"], config["d_ff"],
            config["dropout"])
    if name in ("moment_frozen", "moment_full",
                "controlled_moment_frozen", "controlled_moment_full"):
        return MomentRUL(shape, config["model_id"], config["revision"],
                         config["interface_channels"], config["sequence_length"],
                         freeze=name.endswith("frozen"))
    raise ValueError(name)
