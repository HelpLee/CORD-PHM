#!/usr/bin/env python3
"""Build strictly paired raw-patch NPZs for the three downstream ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from preprocess_health_tokens.raw_patch_ablation import battery_calce_cs2, bearing_xjtu, milling_phm2010


DATASETS = ("bearing", "battery", "milling")


def paths(code_root: Path):
    data = code_root / "data_phm"
    health = data / "processed_health_tokens"
    raw = data / "raw"
    output = data / "processed_raw_patches"
    return {
        "bearing": (
            bearing_xjtu,
            raw / "Bearing" / "XJTU",
            health / "bearing" / "xjtu_downstream_bearing_health_tokens.npz",
            output / "bearing" / "xjtu_downstream_bearing_raw_patches.npz",
        ),
        "battery": (
            battery_calce_cs2,
            raw / "Battery" / "CALCE_CS2",
            health / "battery" / "calce_cs2_downstream_battery_health_tokens.npz",
            output / "battery" / "calce_cs2_downstream_battery_raw_patches.npz",
        ),
        "milling": (
            milling_phm2010,
            raw / "Milling" / "phm2010",
            health / "milling" / "phm2010_milling_downstream_health_tokens.npz",
            output / "milling" / "phm2010_milling_downstream_raw_patches.npz",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--limit-samples", type=int, help="Diagnostic prefix only; omit for publishable artifacts")
    parser.add_argument("--output-root", type=Path, help="Override processed_raw_patches root")
    args = parser.parse_args()
    selected = paths(CODE_ROOT)
    summaries = []
    for name in args.include:
        module, raw_root, reference, output = selected[name]
        if args.output_root is not None:
            output = args.output_root.resolve() / name / output.name
        if not raw_root.exists():
            raise FileNotFoundError(raw_root)
        if not reference.exists():
            raise FileNotFoundError(reference)
        summaries.append(
            module.build(
                raw_root=raw_root,
                reference=reference,
                output=output,
                limit_samples=args.limit_samples,
            )
        )
    summary_path = (args.output_root.resolve() if args.output_root else CODE_ROOT / "data_phm" / "processed_raw_patches") / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
