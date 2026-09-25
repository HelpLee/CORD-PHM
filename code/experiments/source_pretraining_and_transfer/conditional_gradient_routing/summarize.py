"""Paired comparisons against matched controls AND fixed COMPONENT_ROUTING component controls."""
import json
import numpy as np
from config import ARMS, DOMAINS, COMPONENT_ROUTING, FRACTIONS, HERE, JOINT_ARMS, SEEDS, control_for, settings


def group(root, arm, domain, fraction):
    path = root / 'downstream' / arm / domain / f'p{round(fraction*100)}/results.json'
    if not path.exists(): return None
    data = json.loads(path.read_text())
    rows = data.get('rows', [])
    if not data.get('complete') or len(rows) != 5 or {r['seed'] for r in rows} != set(SEEDS): return None
    return {r['seed']: r for r in rows}


def main():
    summary = []; missing = []; paired = []
    for arm in ARMS:
        for domain in settings(arm)['domains']:
            for f in FRACTIONS:
                a = group(HERE, arm, domain, f)
                if a is None: missing.append([arm, domain, f]); continue
                summary.append(dict(model=arm, domain=domain, fraction=f,
                    metrics={m: dict(mean=float(np.mean([r['metrics'][m] for r in a.values()])),
                                     std=float(np.std([r['metrics'][m] for r in a.values()], ddof=1)))
                             for m in ('rmse', 'mae', 'r2')}))
                if arm not in JOINT_ARMS: continue
                root, control = control_for(arm, domain)
                for kind, cr, ca in [('matched', root, control), ('fixed_component', COMPONENT_ROUTING, f'single_{domain}_component')]:
                    b = group(cr, ca, domain, f)
                    if b is None:
                        missing.append(['control', str(cr), ca, domain, f]); continue
                    delta = {m: [a[s]['metrics'][m] - b[s]['metrics'][m] for s in SEEDS] for m in ('rmse','mae','r2')}
                    baseline = float(np.mean([b[s]['metrics']['rmse'] for s in SEEDS]))
                    paired.append(dict(model=arm, control=ca, control_experiment=cr.name, comparison=kind,
                        domain=domain, fraction=f, deltas=delta, wins=sum(v<0 for v in delta['rmse']),
                        mean_delta_rmse=float(np.mean(delta['rmse'])),
                        relative_rmse_change=float(np.mean(delta['rmse']))/baseline))
    aggregate = []
    for arm in JOINT_ARMS:
        for kind in ('matched', 'fixed_component'):
            cells = [x for x in paired if x['model']==arm and x['comparison']==kind]
            if len(cells)==6:
                aggregate.append(dict(model=arm, comparison=kind,
                    macro_relative_rmse_change=float(np.mean([x['relative_rmse_change'] for x in cells])),
                    improved_cells=sum(x['mean_delta_rmse']<0 for x in cells),
                    seed_wins=sum(x['wins'] for x in cells),
                    worst_relative_change=max(x['relative_rmse_change'] for x in cells)))
    payload=dict(complete=not missing,missing=missing,summary=summary,paired=paired,aggregate=aggregate,
                 caveat='Exploratory follow-up informed by previous test results; confirm on held-out devices and upstream seeds.')
    (HERE/'comparison.json').write_text(json.dumps(payload,indent=2))
    lines=['# CONDITIONAL_ROUTING low-label comparison','', '| Model | Domain | Labels | RMSE | MAE | R2 |', '|---|---|---|---|---|---|']
    for x in summary:
        ms=['%.4f ± %.4f'%(x['metrics'][m]['mean'],x['metrics'][m]['std']) for m in ('rmse','mae','r2')]
        lines.append('| '+ ' | '.join([x['model'],x['domain'],f"{x['fraction']:.0%}"]+ms)+' |')
    lines += ['', 'Paired comparisons and fixed-control comparisons: see comparison.json.', payload['caveat']]
    (HERE/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(complete=not missing,cells=len(summary),missing=len(missing),aggregate=aggregate)))


if __name__ == '__main__': main()
