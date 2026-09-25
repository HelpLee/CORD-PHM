import json
from pathlib import Path
import numpy as np
from config import DOMAINS, FRACTIONS, HERE, MODELS, MODES, SEEDS, result_path


def main():
    rows=[]; missing=[]
    for model in MODELS:
        for mode in MODES:
            for domain in DOMAINS:
                for fraction in FRACTIONS:
                    path=result_path(model,mode,domain,fraction)
                    if not path.is_file(): missing.append(str(path)); continue
                    result=json.loads(path.read_text())
                    values=result.get("rows",[])
                    seeds={r.get("seed") for r in values}
                    if not result.get("complete") or seeds != set(SEEDS):
                        missing.append(str(path)); continue
                    metrics={k:{"mean":float(np.mean([r["metrics"][k] for r in values])),
                                "std":float(np.std([r["metrics"][k] for r in values]))}
                             for k in ("rmse","mae","r2")}
                    rows.append(dict(model=model,mode=mode,domain=domain,fraction=fraction,
                                     metrics=metrics,source=str(path)))
    report=dict(complete=not missing,missing=missing,summary=rows,
                protocol="matched inputs/splits/seeds; scratch full is in E38 baseline")
    out=HERE/"reports";out.mkdir(exist_ok=True)
    (out/"adaptation_matrix.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=["| Model | Mode | Domain | Labels | RMSE | MAE | R2 |","|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        m=r["metrics"]
        lines.append(f'| {r["model"]} | {r["mode"]} | {r["domain"]} | {round(100*r["fraction"])}% | '
                     f'{m["rmse"]["mean"]:.5f} +/- {m["rmse"]["std"]:.5f} | '
                     f'{m["mae"]["mean"]:.5f} | {m["r2"]["mean"]:.5f} |')
    (out/"adaptation_matrix.md").write_text("\n".join(lines)+"\n")
    print("SUMMARY",len(rows),"/",len(MODELS)*len(MODES)*len(DOMAINS)*len(FRACTIONS),"missing",len(missing))


if __name__ == "__main__": main()

