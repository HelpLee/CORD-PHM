"""Combine new baseline summaries with TRACE Scratch/Single/Three-domain rows.

No historical baseline score is imported. Partial/frozen TRACE rows are
included when their corresponding C-group result files exist.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
METRICS = ("rmse", "mae", "r2")


def summarize(values):
    return {key: {"mean": float(np.mean([row[key] for row in values])),
                  "std": float(np.std([row[key] for row in values], ddof=1))
                  if len(values) > 1 else None}
            for key in METRICS}


def main():
    rows = []
    trace = STUDY / "A_main" / "05_reports" / "e38_baseline.json"
    if trace.exists():
        source = json.loads(trace.read_text(encoding="utf-8"))
        for item in source["summary"]:
            rows.append({"family": "TRACE", "method": item["model"],
                         "adaptation": item["mode"], "domain": item["domain"],
                         "fraction": item["fraction"], "metrics": item["metrics"],
                         "source": str(trace.relative_to(STUDY))})
    transfer = STUDY / "C_transfer_label_efficiency" / "01_frozen_partial_full" / "results"
    for initialization in ("single_domain", "three_domain"):
        for adaptation in ("frozen", "partial"):
            for domain in ("bearing", "battery", "milling"):
                for percent in (10, 20, 100):
                    path = (transfer / initialization / adaptation / domain /
                            f"p{percent}" / "results.json")
                    if not path.exists():
                        continue
                    source = json.loads(path.read_text(encoding="utf-8"))
                    complete = [item["metrics"] for item in source.get("rows", [])
                                if item.get("seed") in range(42, 47)]
                    if len(complete) != 5:
                        continue
                    rows.append({"family": "TRACE", "method": initialization,
                                 "adaptation": adaptation, "domain": domain,
                                 "fraction": percent / 100, "metrics": summarize(complete),
                                 "source": str(path.relative_to(STUDY))})
    baselines = HERE / "results_summary.json"
    if baselines.exists():
        for item in json.loads(baselines.read_text(encoding="utf-8"))["rows"]:
            rows.append({"family": "native" if item["model"].startswith("native_")
                         else "controlled", "method": item["model"],
                         "adaptation": "model_native", "domain": item["domain"],
                         "fraction": item["fraction"], "metrics": item["metrics"],
                         "n": item["n"], "source": str(baselines.relative_to(STUDY))})
    (HERE / "comparison_with_trace.json").write_text(
        json.dumps({"rows": rows,
                    "warning": "Native and controlled groups answer different questions; "
                    "do not attribute native-group differences solely to pretraining."},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Collected {len(rows)} comparison rows")


if __name__ == "__main__":
    main()
