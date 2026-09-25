"""Aggregate completed five-seed baseline results as mean plus sample std."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_all import DOMAINS, FRACTIONS, MODELS, SEEDS, HERE


def main():
    rows = []
    missing = []
    for model in MODELS:
        for domain in DOMAINS:
            for fraction in FRACTIONS:
                group = []
                for seed in SEEDS:
                    path = (HERE / "results" / model / domain /
                            f"fraction{int(fraction * 100)}" / f"seed{seed}" / "metrics.json")
                    if not path.exists():
                        missing.append(str(path.relative_to(HERE)))
                        continue
                    group.append(json.loads(path.read_text(encoding="utf-8")))
                if group:
                    rows.append({
                        "model": model, "domain": domain, "fraction": fraction,
                        "n": len(group),
                        "metrics": {
                            metric: {
                                "mean": float(np.mean([item["metrics"][metric] for item in group])),
                                "std": (float(np.std([item["metrics"][metric] for item in group], ddof=1))
                                        if len(group) > 1 else None),
                            }
                            for metric in ("rmse", "mae", "r2", "bias")
                        },
                    })
    (HERE / "results_summary.json").write_text(
        json.dumps({"complete": not missing, "rows": rows, "missing": missing}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
