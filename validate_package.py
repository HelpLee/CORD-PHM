"""Verify bundled files and, optionally, the external NPZ dataset directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE


def filesystem_path(path: Path) -> str:
    value = str(path.resolve())
    return "\\\\?\\" + value if os.name == "nt" else value


def is_file(path: Path) -> bool:
    return os.path.isfile(filesystem_path(path))


def file_size(path: Path) -> int:
    return os.stat(filesystem_path(path)).st_size


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(filesystem_path(path), "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Directory corresponding to data_phm (contains processed_health_tokens).",
    )
    args = parser.parse_args()

    bundled_npz = list(HERE.rglob("*.npz"))
    if bundled_npz:
        raise RuntimeError(f"NPZ files must not be bundled: {bundled_npz[:3]}")

    manifest = json.loads((HERE / "file_checksums.json").read_text())
    errors = []
    for record in manifest["files"]:
        path = PACKAGE / record["path"]
        if not is_file(path):
            errors.append(f"missing bundled file: {record['path']}")
        elif file_size(path) != record["bytes"] or sha256(path) != record["sha256"]:
            errors.append(f"bundled hash mismatch: {record['path']}")

    if args.data_root:
        data = json.loads((HERE / "external_data_manifest.json").read_text())
        for record in data["files"]:
            if not record["required"]:
                continue
            path = args.data_root / Path(record["relative_path"])
            if not is_file(path):
                errors.append(f"missing external data: {record['relative_path']}")
            elif file_size(path) != record["bytes"] or sha256(path) != record["sha256"]:
                errors.append(f"external data hash mismatch: {record['relative_path']}")

    if errors:
        raise SystemExit("PACKAGE_INVALID\n" + "\n".join(errors))
    print(
        f"PACKAGE_VALID bundled={len(manifest['files'])} "
        f"external_data_checked={bool(args.data_root)}"
    )


if __name__ == "__main__":
    main()
