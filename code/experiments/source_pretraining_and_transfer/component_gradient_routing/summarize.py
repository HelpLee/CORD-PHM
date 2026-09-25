"""Complete-cell summaries and paired joint-minus-single comparisons."""
import json
import numpy as np
from config import ARMS, DOMAINS, FRACTIONS, HERE, JOINT_ARMS, SEEDS, control_for, settings


def main():
    rows=[]; summary=[]; missing=[]
    for arm in ARMS:
        for domain in settings(arm)["domains"]:
            for fraction in FRACTIONS:
                path=HERE/"downstream"/arm/domain/f"p{round(100*fraction)}"/"results.json"
                if not path.exists(): missing.append(str(path.relative_to(HERE))); continue
                data=json.loads(path.read_text()); group=data.get("rows", [])
                if not data.get("complete") or {r["seed"] for r in group} != set(SEEDS):
                    missing.append(str(path.relative_to(HERE))); continue
                rows += [dict(r, model=arm, domain=domain, fraction=fraction) for r in group]
                summary.append(dict(model=arm, domain=domain, fraction=fraction,
                    metrics={m: dict(mean=float(np.mean([r["metrics"][m] for r in group])),
                                     std=float(np.std([r["metrics"][m] for r in group], ddof=1)))
                             for m in ("rmse","mae","r2")}))
    paired=[]
    for arm in JOINT_ARMS:
        for domain in DOMAINS:
            control=control_for(arm,domain)
            for fraction in FRACTIONS:
                a={r["seed"]:r for r in rows if r["model"]==arm and r["domain"]==domain and r["fraction"]==fraction}
                b={r["seed"]:r for r in rows if r["model"]==control and r["domain"]==domain and r["fraction"]==fraction}
                if set(a)!=set(SEEDS) or set(b)!=set(SEEDS): continue
                delta=np.array([a[s]["metrics"]["rmse"]-b[s]["metrics"]["rmse"] for s in SEEDS])
                paired.append(dict(model=arm,control=control,domain=domain,fraction=fraction,
                    mean_delta_rmse=float(delta.mean()),sd_delta_rmse=float(delta.std(ddof=1)),
                    wins=int((delta<0).sum()),deltas=delta.tolist()))
    payload=dict(complete=not missing,missing=missing,summary=summary,paired=paired,rows=rows)
    (HERE/"comparison.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(dict(complete=not missing,cells=len(summary),missing=len(missing))))


if __name__=="__main__": main()
