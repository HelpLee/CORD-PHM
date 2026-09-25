"""Report all arms, missing cells, paired gains and training diagnostics."""
import json
import numpy as np
from config import HERE, ARMS, JOINT_ARMS, DOMAINS, FRACTIONS, SEEDS, settings, control_for


def summarize():
    rows, summary, missing, upstream, paired = [], [], [], {}, []
    for arm in ARMS:
        folder = HERE / 'models' / arm / 'training'
        if (folder / 'status.json').exists():
            upstream[arm] = json.loads((folder / 'status.json').read_text())
        for domain in settings(arm)['domains']:
            for fraction in FRACTIONS:
                path = HERE / 'downstream' / arm / domain / f'p{round(fraction*100)}/results.json'
                if not path.exists():
                    missing.append(str(path.relative_to(HERE)))
                    continue
                data = json.loads(path.read_text())
                group = data['rows']
                if not data.get('complete') or len(group) != 5 or {r['seed'] for r in group} != set(SEEDS):
                    missing.append(str(path.relative_to(HERE)))
                # Incomplete groups stay visible, not silently removed.
                rows.extend(dict(r, model=arm, domain=domain, fraction=fraction) for r in group)
                if group:
                    summary.append(dict(model=arm, domain=domain, fraction=fraction, n=len(group),
                        metrics={m: dict(mean=float(np.mean([r['metrics'][m] for r in group])),
                            std=float(np.std([r['metrics'][m] for r in group], ddof=1)) if len(group)>1 else None)
                                 for m in ('rmse', 'mae', 'r2')}))
    for arm in JOINT_ARMS:
        for domain in DOMAINS:
            control = control_for(arm, domain)
            for fraction in FRACTIONS:
                a = {r['seed']: r for r in rows if r['model']==arm and r['domain']==domain and r['fraction']==fraction}
                b = {r['seed']: r for r in rows if r['model']==control and r['domain']==domain and r['fraction']==fraction}
                if set(a) != set(SEEDS) or set(b) != set(SEEDS):
                    continue
                delta = np.array([a[s]['metrics']['rmse'] - b[s]['metrics']['rmse'] for s in SEEDS])
                paired.append(dict(model=arm, control=control, domain=domain, fraction=fraction,
                    mean_delta_rmse=float(delta.mean()), sd_delta_rmse=float(delta.std(ddof=1)),
                    wins=int((delta < 0).sum()), deltas=delta.tolist()))
    payload = dict(complete=not missing, missing=missing, upstream=upstream, summary=summary, paired=paired, rows=rows)
    (HERE / 'comparison.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    lines = ['# E33: all prespecified results', '', f'Missing/incomplete cells: {len(missing)}', '',
             '| Model | Domain | Labels | Seeds | RMSE | MAE | R2 |', '|---|---|---|---|---|---|---|']
    for item in summary:
        metrics = []
        for metric in ('rmse', 'mae', 'r2'):
            value = item['metrics'][metric]
            peers = [s['metrics'][metric]['mean'] for s in summary if s['domain']==item['domain'] and s['fraction']==item['fraction'] and s['n']==5]
            text = f"{value['mean']:.4f}" + (f" ± {value['std']:.4f}" if value['std'] is not None else '')
            if item['n']==5 and peers and value['mean']==(max(peers) if metric=='r2' else min(peers)):
                text = '<u><strong>' + text + '</strong></u>'
            metrics.append(text)
        lines.append(f"| {item['model']} | {item['domain']} | {item['fraction']:.0%} | {item['n']} | " + ' | '.join(metrics) + ' |')
    lines += ['', 'Negative paired RMSE differences favor joint training. Each modified joint model is compared with the corresponding own-domain control.',
              'One upstream seed per arm: the five downstream seeds do not establish robustness to pretraining initialization.',
              'All E33 models use FP32 on gpu; old BF16/A100 results are context, not the primary control.']
    (HERE / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(complete=not missing, cells=len(summary), missing=len(missing))))


if __name__ == '__main__':
    summarize()
