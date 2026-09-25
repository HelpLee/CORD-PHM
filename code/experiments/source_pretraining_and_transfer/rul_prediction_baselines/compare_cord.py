"""Combine predictor baselines with CORD Scratch/Single/Multi-domain rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
METRICS = ("rmse", "mae", "r2")


def summarize(values):
    return {
        key: {
            "mean": float(np.mean([row[key] for row in values])),
            "std": float(np.std([row[key] for row in values], ddof=1))
            if len(values) > 1
            else None,
        }
        for key in METRICS
    }


def main() -> None:
    rows = []
    controls = SUITE / "adaptation_protocol_controls" / "reports" / "baseline.json"
    if controls.exists():
        source = json.loads(controls.read_text(encoding="utf-8"))
        for item in source["summary"]:
            rows.append(
                {
                    "family": "CORD",
                    "method": item["model"],
                    "adaptation": item["mode"],
                    "domain": item["domain"],
                    "fraction": item["fraction"],
                    "metrics": item["metrics"],
                    "source": str(controls.relative_to(SUITE)),
                }
            )

    transfer = SUITE / "included_type_adaptation_matrix" / "downstream"
    for initialization in ("single_domain", "multi_domain"):
        for adaptation in ("frozen", "partial"):
            for domain in ("bearing", "battery", "milling"):
                for percent in (10, 20, 100):
                    path = (
                        transfer
                        / initialization
                        / adaptation
                        / domain
                        / f"p{percent}"
                        / "results.json"
                    )
                    if not path.exists():
                        continue
                    source = json.loads(path.read_text(encoding="utf-8"))
                    complete = [
                        item["metrics"]
                        for item in source.get("rows", [])
                        if item.get("seed") in range(42, 47)
                    ]
                    if len(complete) != 5:
                        continue
                    rows.append(
                        {
                            "family": "CORD",
                            "method": initialization,
                            "adaptation": adaptation,
                            "domain": domain,
                            "fraction": percent / 100,
                            "metrics": summarize(complete),
                            "source": str(path.relative_to(SUITE)),
                        }
                    )

    baselines = HERE / "results_summary.json"
    if baselines.exists():
        for item in json.loads(baselines.read_text(encoding="utf-8"))["rows"]:
            rows.append(
                {
                    "family": "predictor_baseline",
                    "method": item["model"],
                    "adaptation": "model_native",
                    "domain": item["domain"],
                    "fraction": item["fraction"],
                    "metrics": item["metrics"],
                    "n": item["n"],
                    "source": str(baselines.relative_to(SUITE)),
                }
            )

    output = HERE / "comparison_with_cord.json"
    output.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"Collected {len(rows)} comparison rows")


if __name__ == "__main__":
    main()
