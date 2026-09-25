"""Canonical, complete registry for non-fuelcell processed HealthToken files.

The registry inventories the files under ``data_phm/processed_health_tokens``.
It is not a training admission list: R2F suitability and split selection are
experiment decisions, not preprocessing filters.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable

from preprocess_health_tokens.common.bearing_health_token_utils import HealthTokenConfig
from preprocess_health_tokens.common.raw_token_contract import as_raw_protocol_config

BEARING_DATASETS = (
    "cwru", "femto", "ferrara", "hust", "ims", "kaist", "mfpt",
    "paderborn", "seu", "unsw", "xjtu",
)
BATTERY_DATASETS = (
    "calce_cs2_downstream_battery", "hust_battery", "isu_ilcc_battery",
    "mich_exp_battery", "nasa_battery", "oxford_battery", "rwth_battery",
    "sdu_battery", "xjtu_battery",
)
MILLING_DATASETS = (
    "luh_milling", "matwi_milling", "nasa_milling",
    "nonastreda_milling", "phm2010_milling_downstream",
    "piecuch_milling", "qit_cemc_milling", "hmotp_milling",
)
ENGINE_DATASETS = ("ncmapss_ds02_downstream",)
ALL_DATASETS = BEARING_DATASETS + BATTERY_DATASETS + MILLING_DATASETS + ENGINE_DATASETS


def _raw(cfg: HealthTokenConfig) -> HealthTokenConfig:
    return replace(as_raw_protocol_config(cfg), fail_on_segment_error=True)


_bearing = {
    "xjtu": dict(window_revolutions=10.0, window_seconds=0.1, max_window_points=8192),
    "ims": dict(window_seconds=0.1, max_window_points=8192),
    "femto": dict(window_seconds=0.05, min_window_points=512, max_window_points=8192, use_fixed_revolution_when_available=False),
    "kaist": dict(window_seconds=0.1, max_window_points=8192),
    "ferrara": dict(window_revolutions=10.0, window_seconds=0.1, max_window_points=8192),
    "unsw": dict(window_revolutions=10.0, window_seconds=0.1, max_window_points=8192),
    "paderborn": dict(window_seconds=0.1, max_window_points=8192, use_fixed_revolution_when_available=False),
    "cwru": dict(window_seconds=0.1, min_window_points=512, max_window_points=8192, use_fixed_revolution_when_available=False),
    "hust": dict(window_revolutions=10.0, window_seconds=0.1, max_window_points=8192),
    "mfpt": dict(window_seconds=0.1, min_window_points=512, max_window_points=8192, use_fixed_revolution_when_available=False),
    "seu": dict(window_seconds=0.1, min_window_points=512, max_window_points=8192, use_fixed_revolution_when_available=False),
}

DATASET_CONFIGS: Dict[str, HealthTokenConfig] = {
    name: _raw(HealthTokenConfig(
        dataset=name,
        source_module=f"preprocess_health_tokens.datasets.bearing.{name}",
        modality="bearing",
        output_name=("xjtu_downstream_bearing_health_tokens.npz" if name == "xjtu" else f"{name}_bearing_health_tokens.npz"),
        **_bearing[name],
    ))
    for name in BEARING_DATASETS
}

for _name in BATTERY_DATASETS:
    DATASET_CONFIGS[_name] = _raw(HealthTokenConfig(
        dataset=_name,
        source_module=(
            "preprocess_health_tokens.datasets.battery.calce_cs2_downstream_4cells"
            if _name == "calce_cs2_downstream_battery"
            else f"preprocess_health_tokens.datasets.battery.{_name}"
        ),
        modality="battery",
        output_name=f"{_name}_health_tokens.npz",
        max_tokens=64,
        apply_robust_scaling=False,
    ))

for _name in MILLING_DATASETS:
    DATASET_CONFIGS[_name] = _raw(HealthTokenConfig(
        dataset=_name,
        source_module=f"preprocess_health_tokens.datasets.milling.{_name}",
        modality="milling",
        output_name=f"{_name}_health_tokens.npz",
        max_tokens=64,
        use_fixed_revolution_when_available=False,
        apply_robust_scaling=False,
    ))

DATASET_CONFIGS["ncmapss_ds02_downstream"] = _raw(HealthTokenConfig(
    dataset="ncmapss_ds02_downstream",
    source_module="preprocess_health_tokens.datasets.engine.ncmapss_ds02_downstream",
    modality="engine",
    output_name="ncmapss_ds02_downstream_health_tokens.npz",
    max_tokens=64,
    use_fixed_revolution_when_available=False,
    apply_robust_scaling=False,
))


def selected_configs(include: Iterable[str] = (), exclude: Iterable[str] = ()) -> Dict[str, HealthTokenConfig]:
    """Select canonical non-fuelcell datasets, preserving inventory order."""
    requested = ALL_DATASETS if not include else tuple(str(x).lower() for x in include)
    excluded = {str(x).lower() for x in exclude}
    unknown = sorted(set(requested) - set(DATASET_CONFIGS))
    if unknown:
        raise KeyError(f"Unknown dataset(s): {unknown}. Available: {list(ALL_DATASETS)}")
    return {name: DATASET_CONFIGS[name] for name in requested if name not in excluded}
