"""Migrate canonical Milling NPZ names and roles to the upstream convention."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "data_phm" / "processed_health_tokens" / "milling"
ARCHIVE = ROOT / "_archive" / "legacy_names"
RENAMES = {
    "hmotp_milling_downstream_wear": "hmotp_milling",
    "nasa_milling_downstream": "nasa_milling",
    "nonastreda_milling_snapshot": "nonastreda_milling",
    "qit_cemc_milling_snapshot": "qit_cemc_milling",
}
UPSTREAM_R2F = {"luh_milling", "matwi_milling", "piecuch_milling", "nasa_milling"}
UPSTREAM_SNAPSHOT = {"nonastreda_milling", "qit_cemc_milling", "hmotp_milling"}


def updated_meta(meta: dict, old: str, new: str) -> dict:
    meta = dict(meta)
    meta["dataset"] = new
    meta["canonical_dataset"] = new
    config = meta.get("config")
    if isinstance(config, dict):
        config = dict(config)
        config["dataset"] = new
        config["source_module"] = f"preprocess_health_tokens.datasets.milling.{new}"
        config["output_name"] = f"{new}_health_tokens.npz"
        meta["config"] = config
    if new == "hmotp_milling":
        meta["dataset_role"] = "upstream_self_supervised_snapshot_sequence"
        meta["upstream_eligible"] = True
        meta.pop("split_protocol", None)
        meta.pop("wear_target", None)
        meta["wear_label_note"] = "wear_um retained for audit only; never used by upstream SSL"
    elif new == "nasa_milling":
        meta["dataset_role"] = "upstream_self_supervised_r2f_sequence"
        meta["upstream_eligible"] = True
        meta.pop("excluded_from_upstream", None)
        meta["label_usage"] = "lifecycle metadata retained for audit; upstream SSL does not consume labels"
    else:
        meta["dataset_role"] = "upstream_self_supervised_snapshot_sequence"
        meta["upstream_eligible"] = True
    meta["renamed_from"] = old
    return meta


def migrate(old: str, new: str) -> None:
    old_npz = ROOT / f"{old}_health_tokens.npz"
    old_json = ROOT / f"{old}_health_tokens.lifecycle.json"
    new_npz = ROOT / f"{new}_health_tokens.npz"
    new_json = ROOT / f"{new}_health_tokens.lifecycle.json"
    if not old_npz.exists() or not old_json.exists():
        raise FileNotFoundError(f"Missing legacy artifact pair for {old}")
    if new_npz.exists() or new_json.exists():
        raise FileExistsError(f"Refusing to overwrite canonical artifact pair for {new}")
    with np.load(old_npz, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    meta = json.loads(str(payload["meta_json"].item()))
    payload["meta_json"] = np.asarray(json.dumps(updated_meta(meta, old, new), ensure_ascii=False))
    temporary = new_npz.with_name(new_npz.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    with np.load(temporary, allow_pickle=False) as check:
        if json.loads(str(check["meta_json"].item())).get("dataset") != new:
            raise ValueError(f"Metadata migration failed for {new}")
        if check["x_health"].shape != payload["x_health"].shape:
            raise ValueError(f"Array shape changed while migrating {new}")
    os.replace(temporary, new_npz)
    shutil.copy2(old_json, new_json)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_npz), str(ARCHIVE / old_npz.name))
    shutil.move(str(old_json), str(ARCHIVE / old_json.name))
    print(f"MIGRATED {old} -> {new}")


def normalize_upstream_artifact(dataset: str) -> None:
    """Make the role explicit without changing any feature or sample array."""
    path = ROOT / f"{dataset}_health_tokens.npz"
    lifecycle_path = ROOT / f"{dataset}_health_tokens.lifecycle.json"
    with np.load(path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    meta = json.loads(str(payload["meta_json"].item()))
    meta["dataset"] = dataset
    meta["canonical_dataset"] = dataset
    meta["upstream_eligible"] = True
    meta["dataset_role"] = (
        "upstream_self_supervised_r2f_sequence"
        if dataset in UPSTREAM_R2F
        else "upstream_self_supervised_snapshot_sequence"
    )
    meta["label_usage"] = "labels retained for audit only; upstream SSL does not consume RUL or wear labels"
    meta.pop("excluded_from_upstream", None)
    meta.pop("split_protocol", None)
    meta.pop("wear_target", None)
    payload["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)

    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    for row in lifecycle:
        status = str(row.get("status", ""))
        if status != "excluded":
            row["status"] = "accepted_upstream_ssl"
        if dataset == "hmotp_milling" and "endpoint_evidence" in row:
            row["endpoint_evidence"] = "not supplied; measured wear is audit metadata only"
    lifecycle_path.write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
    print(f"NORMALIZED upstream role: {dataset}")


def main() -> None:
    for old, new in RENAMES.items():
        old_path = ROOT / f"{old}_health_tokens.npz"
        new_path = ROOT / f"{new}_health_tokens.npz"
        if old_path.exists():
            migrate(old, new)
        elif not new_path.exists():
            raise FileNotFoundError(f"Neither legacy nor canonical artifact exists for {new}")
    for dataset in sorted(UPSTREAM_R2F | UPSTREAM_SNAPSHOT):
        normalize_upstream_artifact(dataset)


if __name__ == "__main__":
    main()
