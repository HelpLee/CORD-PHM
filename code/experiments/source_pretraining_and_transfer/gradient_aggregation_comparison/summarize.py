"""Transparent all-arm reporting: no winner selected using test performance."""
import json
from pathlib import Path
import numpy as np
from submit import ARMS, DOMAINS

HERE = Path(__file__).resolve().parent
rows, summary, missing = [], [], []
for arm in (*ARMS, "joint_selection"):
    targets = DOMAINS if arm.startswith("joint_") else (arm.split("_")[1],)
    for domain in targets:
        for fraction in (.1, .2, 1.):
            path = HERE / "downstream" / arm / domain / f"p{round(fraction*100)}/results.json"
            if not path.exists():
                missing.append(str(path.relative_to(HERE))); continue
            result = json.loads(path.read_text())
            group = result["rows"]
            if not result.get("complete") or len(group) != 5 or {r["seed"] for r in group} != set(range(42, 47)):
                missing.append(str(path.relative_to(HERE))); continue
            rows.extend(dict(r, model=arm, domain=domain, fraction=fraction) for r in group)
            summary.append(dict(model=arm, domain=domain, fraction=fraction, n=5,
                metrics={m: dict(mean=float(np.mean([r["metrics"][m] for r in group])),
                                  std=float(np.std([r["metrics"][m] for r in group], ddof=1)))
                         for m in ("rmse", "mae", "r2")}))
paired = []
for arm in ("joint_raw", "joint_selection", "joint_norm", "joint_pcgrad", "joint_cagrad"):
    for domain in DOMAINS:
        control = f"single_{domain}_" + ("raw" if arm in ("joint_raw", "joint_selection") else "norm")
        for fraction in (.1, .2, 1.):
            a = {r["seed"]: r for r in rows if r["model"] == arm and r["domain"] == domain and r["fraction"] == fraction}
            b = {r["seed"]: r for r in rows if r["model"] == control and r["domain"] == domain and r["fraction"] == fraction}
            if set(a) != set(range(42, 47)) or set(b) != set(a):
                continue
            delta = np.array([a[s]["metrics"]["rmse"] - b[s]["metrics"]["rmse"] for s in sorted(a)])
            paired.append(dict(model=arm, control=control, domain=domain, fraction=fraction,
                delta_rmse_mean=float(delta.mean()), delta_rmse_std=float(delta.std(ddof=1)),
                wins=int((delta < 0).sum()), paired_deltas=delta.tolist(),
                interpretation="negative delta favors joint; downstream seeds, one upstream seed only"))
payload = dict(complete=not missing, missing=missing, summary=summary, paired=paired, rows=rows)
(HERE / "comparison.json").write_text(json.dumps(payload, indent=2))
lines = ["# Experiment 29 (all prespecified arms)", "", "Incomplete cells: " + str(len(missing)), "",
         "| Model | Domain | Labels | RMSE | MAE | R2 |", "|---|---|---|---|---|---|"]
for item in summary:
    metrics = [f'{item["metrics"][m]["mean"]:.4f} ± {item["metrics"][m]["std"]:.4f}' for m in ("rmse", "mae", "r2")]
    lines.append(f'| {item["model"]} | {item["domain"]} | {item["fraction"]:.0%} | ' + " | ".join(metrics) + " |")
lines += ["", "All arms reported; no method selection by test results. Normalized joint arms compare primarily with normalized own-domain singles.",
          "Five downstream seeds are conditional on one upstream seed, not five independent pretraining runs."]
(HERE / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(dict(complete=not missing, cells=len(summary), missing=len(missing)), indent=2))
