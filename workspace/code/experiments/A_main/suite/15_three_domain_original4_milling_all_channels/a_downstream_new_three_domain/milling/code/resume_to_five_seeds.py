"""Resume the all-seven-channel scratch ablation with seeds 44--46 only."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

import run_all_channels_scratch as experiment
import downstream_interval_val200 as down


MISSING_SEEDS = (44, 45, 46)
FRACTIONS = (0.1, 0.2)


def aggregate(rows):
    return {metric: {"mean": float(np.mean([row["metrics"][metric] for row in rows])),
                     "std": float(np.std([row["metrics"][metric] for row in rows], ddof=1))}
            for metric in ("rmse", "mae", "r2", "bias")}


def main():
    previous = json.loads((experiment.PACKAGE / "results.json").read_text(encoding="utf-8"))
    rows = list(previous["rows"])
    store = experiment.load_store()
    down.prepare_store = lambda _path: store
    down.OUT = experiment.RUNTIME
    down.ARMS = ("scratch",)
    down.MAX_UPDATES = 0
    down.MAX_EPOCHS = 200
    down.PATIENCE = 15
    down.VAL_INTERVAL = 2
    down.TRAIN_VALIDATION_FRACTION = .2
    down.TEST_VALIDATION_FRACTION = 0
    down.NESTED_LABELS = False
    down.source_split = experiment.interval_split

    for fraction in FRACTIONS:
        for seed in MISSING_SEEDS:
            if any(row["seed"] == seed and row["fraction"] == fraction for row in rows):
                continue
            down.SEED, down.LABEL_FRACTION = seed, fraction
            down.NAME = f"all7_scratch_seed{seed}_{int(fraction * 100)}pct"
            experiment.write(experiment.PACKAGE / "status.json", {
                "state": "running", "seed": seed, "fraction": fraction,
                "completed_runs": len(rows),
            })
            down.main()
            payload = json.loads((experiment.RUNTIME / down.NAME / "development_summary.json").read_text(encoding="utf-8"))
            row = payload["rows"][0]
            row["fraction"] = fraction
            rows.append(row)
            experiment.write(experiment.PACKAGE / "results.json", {"complete": False, "rows": rows})

    summary = []
    for fraction in FRACTIONS:
        group = [row for row in rows if row["fraction"] == fraction]
        assert len(group) == 5, (fraction, len(group))
        summary.append({"fraction": fraction, "n": 5, "metrics": aggregate(group)})
    experiment.write(experiment.PACKAGE / "results.json", {"complete": True, "rows": rows, "summary": summary})
    subprocess.run([sys.executable, str(experiment.PACKAGE / "summarize_comparison.py")], check=True)
    experiment.write(experiment.PACKAGE / "status.json", {"state": "complete", "runs": len(rows)})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        experiment.write(experiment.PACKAGE / "status.json", {"state": "failed", "error": repr(error)})
        raise
