"""Audit that the canonical non-fuelcell tree and registry have one-to-one coverage."""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from preprocess_health_tokens.datasets.registry import ALL_DATASETS, DATASET_CONFIGS


def meta_from_npz(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        if "meta_json.npy" not in archive.namelist():
            return {}
        value = np.load(io.BytesIO(archive.read("meta_json.npy")), allow_pickle=True).item()
    return json.loads(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data_phm" / "processed_health_tokens")
    args = parser.parse_args()
    expected = {args.root / cfg.modality / cfg.output_name: name for name, cfg in DATASET_CONFIGS.items()}
    actual = {
        p for p in args.root.rglob("*.npz")
        if "fuelcell" not in p.parts and "_archive" not in p.parts
    }
    missing, extra = sorted(set(expected)-actual), sorted(actual-set(expected))
    if missing or extra:
        raise SystemExit(f"Inventory mismatch: missing={missing}; extra={extra}")
    rows = []
    for path, name in expected.items():
        meta = meta_from_npz(path)
        declared = meta.get("dataset")
        if declared and declared != name:
            raise SystemExit(f"{path}: metadata dataset {declared!r} != registry {name!r}")
        rows.append(dict(dataset=name, path=str(path.relative_to(args.root)), modality=DATASET_CONFIGS[name].modality,
                         metadata_dataset=declared or name))
    print(json.dumps(dict(passed=True, datasets=len(rows), rows=rows), indent=2))


if __name__ == "__main__":
    main()
