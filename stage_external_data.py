"""Stage external NPZ files into the isolated workspace using symbolic links."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = HERE / "workspace" / "code" / "data_phm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, help="Directory corresponding to data_phm")
    parser.add_argument("--include-archive", action="store_true")
    args = parser.parse_args()
    source_root = args.data_root.resolve()
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    linked = 0
    for record in manifest["files"]:
        if not record["required"] and not args.include_archive:
            continue
        source = source_root / Path(record["relative_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        target = TARGET / Path(record["relative_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise RuntimeError(f"Refusing to replace existing data path: {target}")
            continue
        os.symlink(source, target)
        linked += 1
    print(f"DATA_STAGED links_created={linked} target={TARGET}")


if __name__ == "__main__":
    main()
