"""Strictly compare a rebuilt non-fuelcell tree with canonical NPZ contents.

The comparison hashes every uncompressed NPY payload.  Metadata is compared as
JSON after removing only wall-clock/output-hash fields that are necessarily
created anew by the engine builder.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from preprocess_health_tokens.datasets.registry import DATASET_CONFIGS, selected_configs

VOLATILE_META = {"elapsed_seconds", "output_sha256"}


def stream_hash(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_meta(archive: zipfile.ZipFile) -> dict:
    value = np.load(io.BytesIO(archive.read("meta_json.npy")), allow_pickle=True).item()
    meta = json.loads(str(value))
    for key in VOLATILE_META:
        meta.pop(key, None)
    return meta


def path_for(root: Path, name: str) -> Path:
    cfg = DATASET_CONFIGS[name]
    return root / cfg.modality / cfg.output_name


def compare(reference: Path, candidate: Path) -> list[str]:
    if not candidate.exists():
        return ["candidate file missing"]
    errors: list[str] = []
    with zipfile.ZipFile(reference) as a, zipfile.ZipFile(candidate) as b:
        names_a, names_b = set(a.namelist()), set(b.namelist())
        if names_a != names_b:
            errors.append(f"NPY members differ: reference-only={sorted(names_a-names_b)}, candidate-only={sorted(names_b-names_a)}")
            return errors
        for name in sorted(names_a - {"meta_json.npy"}):
            if stream_hash(a, name) != stream_hash(b, name):
                errors.append(f"{name} content differs")
        if "meta_json.npy" in names_a and normalized_meta(a) != normalized_meta(b):
            errors.append("meta_json differs after removing only elapsed_seconds/output_sha256")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--include", nargs="*", default=[])
    args = parser.parse_args()
    failures = {}
    for name in selected_configs(args.include):
        result = compare(path_for(args.reference_root, name), path_for(args.candidate_root, name))
        print(f"{'FAIL' if result else 'PASS'} {name}" + (f": {'; '.join(result)}" if result else ""), flush=True)
        if result:
            failures[name] = result
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
