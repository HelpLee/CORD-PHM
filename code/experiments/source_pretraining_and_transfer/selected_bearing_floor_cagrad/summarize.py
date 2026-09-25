"""Report new candidates against matched, fixed and parent joint references."""
import importlib.util
import json
import numpy as np
from config import HERE,CONDITIONAL_ROUTING,NEW_JOINT_ARMS,PARENT_ARMS,DOMAINS,FRACTIONS,SEEDS,parent_for


def main():
    spec=importlib.util.spec_from_file_location('cord_selected_summary_base',CONDITIONAL_ROUTING/'summarize.py')
    base=importlib.util.module_from_spec(spec)
    source=(CONDITIONAL_ROUTING/'summarize.py').read_text()
    old="('fixed_component', COMPONENT_ROUTING, f'single_{domain}_component')"
    assert source.count(old)==1
    source=source.replace(old,"('fixed_component', HERE if domain=='battery' else COMPONENT_ROUTING, f'single_{domain}_component')")
    exec(compile(source,str(CONDITIONAL_ROUTING/'summarize.py'),'exec'),base.__dict__);base.main()
    path=HERE/'comparison.json'; data=json.loads(path.read_text()); parent=[]
    for arm in NEW_JOINT_ARMS:
        for domain in DOMAINS:
            for f in FRACTIONS:
                a=base.group(HERE,arm,domain,f);b=base.group(HERE,parent_for(arm),domain,f)
                if a is None or b is None: continue
                metrics={}
                for metric in ('rmse','mae','r2'):
                    delta=np.array([a[s]['metrics'][metric]-b[s]['metrics'][metric] for s in SEEDS])
                    metrics[metric]=dict(mean_delta=float(delta.mean()),std_delta=float(delta.std(ddof=1)),
                        wins=int((delta>0 if metric=='r2' else delta<0).sum()),deltas=delta.tolist())
                parent.append(dict(model=arm,parent=parent_for(arm),domain=domain,fraction=f,metrics=metrics))
    data['parent_joint_comparisons']=parent
    extension=[]
    for arm in PARENT_ARMS+('single_battery_conditional','single_battery_component'):
        from config import continuation_source
        root=continuation_source(arm).parent.parent
        for domain in (DOMAINS if arm.startswith('joint_') else ('battery',)):
            for f in FRACTIONS:
                a=base.group(HERE,arm,domain,f);b=base.group(root,arm,domain,f)
                if a is None or b is None: continue
                extension.append(dict(model=arm,domain=domain,fraction=f,
                    mean_delta={m:float(np.mean([a[s]['metrics'][m]-b[s]['metrics'][m] for s in SEEDS]))
                                for m in ('rmse','mae','r2')}))
    data['ceiling_extension_only_comparisons']=extension
    data['budget']='All active arms max2000, patience30. Reused bearing/milling singles already early-stopped. No 2000-vs-capped500 control in primary comparisons.'
    data['caveat']='Exploratory CONDITIONAL_ROUTING-informed candidates, one upstream seed and five downstream seeds; training-gradient constraints do not guarantee transfer improvements. Compare both matched and fixed singles.'
    path.write_text(json.dumps(data,indent=2))
    md=HERE/'comparison.md'; md.write_text(md.read_text().replace('# CONDITIONAL_ROUTING low-label comparison','# SELECTED_MODEL low-label comparison'),encoding='utf-8')
    print('PARENT_COMPARISONS',len(parent))


if __name__=='__main__':main()
