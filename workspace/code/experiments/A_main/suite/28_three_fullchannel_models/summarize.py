"""Aggregate the 27 downstream cells after all dependencies succeed."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rows = []
summary = []
for model in ("three4", "three5", "single5"):
    for domain in ("bearing", "battery", "milling"):
        for fraction in (.1, .2, 1.0):
            path = HERE / "downstream" / model / domain / f"p{round(fraction * 100)}/results.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("complete"):
                raise RuntimeError(f"Incomplete result: {path}")
            group = payload["rows"]
            if len(group) != 5:
                raise RuntimeError(f"Expected five seeds: {path} ({len(group)})")
            rows.extend(group)
            metrics = {}
            for metric in ("rmse", "mae", "r2"):
                values = [row["metrics"][metric] for row in group]
                metrics[metric] = {"mean": float(np.mean(values)),
                                   "std": float(np.std(values, ddof=1))}
            summary.append({"model": model, "domain": domain, "fraction": fraction,
                            "n": len(group), "metrics": metrics})
(HERE / "comparison.json").write_text(
    json.dumps({"complete": True, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
