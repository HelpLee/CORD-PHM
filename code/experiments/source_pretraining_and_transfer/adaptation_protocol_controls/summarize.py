"""Summarize every seed/cell; label incomplete comparisons, no test-driven selection."""
import argparse
import json
import math
import statistics
from config import HERE,DOMAINS,SEEDS,METHODS,existing_result

def cells(phase):
    if phase=='baseline':
        return [(m,'full',d,f) for m in ('scratch','single_domain','multi_domain') for d in DOMAINS for f in (.1,.2,1.)]
    if phase=='adaptive':
        return [(m,'full',d,f) for method in METHODS for m,domains in
            [('joint_'+method,DOMAINS)]+[('single_'+d+'_'+method,(d,)) for d in DOMAINS]
            for d in domains for f in (.1,.2)]
    return [(m,mode,'bearing',.1) for m in ('multi_domain','single_domain') for mode in ('partial','l2sp1','l2sp2')]

def path_for(phase,model,mode,domain,fraction):
    if phase=='baseline' and model!='scratch' and fraction!=1.:
        return existing_result(model,domain,fraction)
    return HERE/'downstream'/phase/model/mode/domain/f'p{round(100*fraction)}/results.json'

def group(path):
    if not path or not path.exists():return None
    data=json.loads(path.read_text());rows=data.get('rows',[])
    if not data.get('complete') or len(rows)!=5 or {r['seed'] for r in rows}!=set(SEEDS):return None
    if not all(math.isfinite(r['metrics'][k]) for r in rows for k in ('rmse','mae','r2')):raise RuntimeError('Nonfinite result: '+str(path))
    return {r['seed']:r['metrics'] for r in rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);args=p.parse_args()
    summary=[];missing=[];paired=[]
    for model,mode,domain,fraction in cells(args.phase):
        path=path_for(args.phase,model,mode,domain,fraction);a=group(path)
        if a is None:missing.append(str(path));continue
        metrics={k:dict(mean=statistics.mean(r[k] for r in a.values()),std=statistics.stdev(r[k] for r in a.values())) for k in ('rmse','mae','r2')}
        summary.append(dict(model=model,mode=mode,domain=domain,fraction=fraction,metrics=metrics,source=str(path)))
        refs=[]
        if args.phase=='adaptive':
            refs=[('fixed_SELECTED_MODEL_joint',existing_result('multi_domain',domain,fraction))]
            if model.startswith('joint_'):
                refs.append(('matched_single',path_for('adaptive','single_'+domain+'_'+model.split('_')[-1],'full',domain,fraction)))
        elif args.phase=='bearing10':
            refs=[('same_checkpoint_full',existing_result(model,domain,fraction))]
            if model=='multi_domain':refs.append(('same_adaptation_single',path_for('bearing10','single_domain',mode,domain,fraction)))
        elif model!='scratch':refs=[('scratch',path_for('baseline','scratch','full',domain,fraction))]
        for label,bp in refs:
            b=group(bp)
            if b is None:missing.append(str(bp));continue
            delta={k:[a[s][k]-b[s][k] for s in SEEDS] for k in ('rmse','mae','r2')}
            paired.append(dict(model=model,mode=mode,domain=domain,fraction=fraction,reference=label,
                deltas=delta,rmse_wins=sum(v<0 for v in delta['rmse']),
                relative_rmse_change=statistics.mean(delta['rmse'])/statistics.mean(r['rmse'] for r in b.values())))
    result=dict(complete=not missing,missing=sorted(set(missing)),summary=summary,paired=paired,
        caveat='Exploratory methods informed by previous test results. Five downstream seeds, one upstream seed. All cells reported; no automatic best-method selection.')
    out=HERE/'reports';out.mkdir(exist_ok=True)
    (out/(args.phase+'.json')).write_text(json.dumps(result,indent=2))
    lines=['| Model | Mode | Domain | Labels | RMSE | MAE | R2 |','|---|---|---|---|---|---|---|']
    for x in summary:
        vals=['%.5f +/- %.5f'%(x['metrics'][k]['mean'],x['metrics'][k]['std']) for k in ('rmse','mae','r2')]
        lines.append('| '+' | '.join([x['model'],x['mode'],x['domain'],str(x['fraction'])]+vals)+' |')
    (out/(args.phase+'.md')).write_text('\n'.join(lines))
    print(args.phase,'complete',not missing,'cells',len(summary),'missing',len(set(missing)))

if __name__=='__main__':main()
