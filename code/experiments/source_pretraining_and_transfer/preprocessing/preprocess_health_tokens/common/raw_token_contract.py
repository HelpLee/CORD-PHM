from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import numpy as np


RAW_PROTOCOL = "three_domain_strict_raw_v2"
RAW_OUTPUT_SUBDIR = "three_domain_strict_raw"
FEATURE_DIM = 26
MAX_TOKENS = 64


def as_raw_protocol_config(cfg: Any) -> Any:
    """Return a config that defers every learned affine transform to training.

    Existing registered configs are immutable inputs to this conversion.  Their
    historical preprocessing behavior therefore remains the default whenever
    the raw-v2 command-line switch is not used.
    """

    return replace(
        cfg,
        apply_robust_scaling=False,
        robust_clip_abs=None,
        preprocessing_protocol=RAW_PROTOCOL,
        scaling_deferred_to_experiment=True,
    )


def raw_contract_metadata() -> Dict[str, object]:
    return {
        "preprocessing_protocol": RAW_PROTOCOL,
        "x_health_scaled": False,
        "scaling_deferred_to_experiment": True,
        "pre_scaling_clip_abs": None,
        "feature_dim": FEATURE_DIM,
        "max_tokens": MAX_TOKENS,
        "padding_value": 0.0,
        "causal": True,
        "training_transform": "train-only valid-feature median/IQR, then one clip to [-20,20]",
    }


def valid_token_mask(token_mask: np.ndarray, c_mask: np.ndarray) -> np.ndarray:
    token = np.asarray(token_mask, dtype=bool)
    channel = np.asarray(c_mask, dtype=bool)
    if token.ndim != 3 or channel.ndim != 2:
        raise ValueError(f"Expected token_mask [N,C,T] and c_mask [N,C], got {token.shape}, {channel.shape}")
    if token.shape[:2] != channel.shape:
        raise ValueError(f"token_mask/c_mask mismatch: {token.shape} vs {channel.shape}")
    return token & channel[..., None]


def validate_raw_health_tokens(
    x_health: np.ndarray,
    token_mask: np.ndarray,
    c_mask: np.ndarray,
    feature_mask: Optional[np.ndarray] = None,
    *,
    require_zero_padding: bool = True,
) -> Dict[str, object]:
    """Validate an unscaled [N,C,64,26] artifact without modifying values."""

    x = np.asarray(x_health)
    if x.ndim != 4 or x.shape[2:] != (MAX_TOKENS, FEATURE_DIM):
        raise ValueError(f"Raw HealthToken tensor must be [N,C,64,26], got {x.shape}")
    valid_token = valid_token_mask(token_mask, c_mask)
    if valid_token.shape != x.shape[:3]:
        raise ValueError(f"Mask/data mismatch: {valid_token.shape} vs {x.shape}")

    if feature_mask is None:
        valid_feature = np.broadcast_to(valid_token[..., None], x.shape)
    else:
        feature = np.asarray(feature_mask, dtype=bool)
        if feature.shape != x.shape:
            raise ValueError(f"feature_mask must match x_health, got {feature.shape} vs {x.shape}")
        valid_feature = feature & valid_token[..., None]

    if not np.isfinite(x[valid_feature]).all():
        raise ValueError("Raw HealthToken contains NaN/Inf in an observed feature")
    invalid_feature = ~valid_feature
    nonzero_padding = int(np.count_nonzero(x[invalid_feature]))
    if require_zero_padding and nonzero_padding:
        raise ValueError(f"Raw HealthToken has {nonzero_padding} non-zero values in padding/missing positions")

    observed = x[valid_feature]
    return {
        "shape": list(x.shape),
        "num_valid_tokens": int(valid_token.sum()),
        "num_observed_features": int(valid_feature.sum()),
        "num_nonzero_padding": nonzero_padding,
        "observed_min": float(observed.min()) if observed.size else 0.0,
        "observed_max": float(observed.max()) if observed.size else 0.0,
        "contains_values_outside_20": bool(np.any(np.abs(observed) > 20.0)) if observed.size else False,
    }


def sanitize_nonfinite_only(
    x_health: np.ndarray,
    token_mask: np.ndarray,
    c_mask: np.ndarray,
    feature_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero invalid positions and reject observed NaN/Inf; never magnitude-clip."""

    x = np.asarray(x_health, dtype=np.float32).copy()
    valid_token = valid_token_mask(token_mask, c_mask)
    valid_feature = np.broadcast_to(valid_token[..., None], x.shape).copy()
    if feature_mask is not None:
        feature = np.asarray(feature_mask, dtype=bool)
        if feature.shape != x.shape:
            raise ValueError(f"feature_mask must match x_health, got {feature.shape} vs {x.shape}")
        valid_feature &= feature
    if not np.isfinite(x[valid_feature]).all():
        raise ValueError("Observed raw feature contains NaN/Inf; fix the domain extractor instead of clipping")
    x[~valid_feature] = 0.0
    return x, valid_feature
