"""Create unified evidence figures for unknown-device degradation transfer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
E = HERE.parent
ROOT = next(p for p in HERE.parents if (p/'code/data_phm').is_dir())
SUITE = ROOT/'code/experiments/A_main/suite'
AUDIT = E/'mechanism_audit'
OUT = HERE/'outputs'
DOMAINS = ('bearing','battery','milling')
TITLES = {'bearing':'Bearing','battery':'Battery','milling':'Milling'}
VARIANTS = ('single_domain','three_domain_adapter')
LAYERS = ('input_tokens','block1','block2','hidden','projector')
LAYER_LABELS = ('Pooled\ninput','Block 1','Block 2','FinalNorm\nhidden','Snapshot\nprojector')
PACKAGES = {
    'bearing': ('42_downstream_bearing_only_adapter','41_downstream_bearing_three_domain_adapter'),
    'battery': ('32_downstream_battery_only_adapter','31_downstream_battery_three_domain_adapter'),
    'milling': ('21_downstream_milling_only_adapter','20_downstream_milling_three_domain_adapter'),
}


def cosine_distance(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return float(1-np.dot(a,b)/max(na*nb,1e-12))


def extract_domain(domain):
    sys.path.insert(0,str(AUDIT))
    import analyze as audit
    import torch
    arrays,y,units,order,train,test=audit.dataset(domain)
    source=np.isin(units,train)
    bins=np.digitize(np.clip(y,0,1),[.2,.4,.6,.8],right=False)
    report={'domain':domain,'source_units':list(train),'test_unit':str(test),
            'bins':['[0,.2)','[.2,.4)','[.4,.6)','[.6,.8)','[.8,1]'],'variants':{}}
    for variant in VARIANTS:
        model,checkpoint=audit.ex.load_encoder(domain,variant,torch.device('cpu'))
        features=audit.extract(model,domain,arrays)
        result={}
        for layer in LAYERS:
            scaler=StandardScaler().fit(features[layer][source])
            z=scaler.transform(features[layer])
            centroids={}
            counts={}
            for unit in (*train,test):
                for stage in range(5):
                    ids=np.flatnonzero((units==unit)&(bins==stage))
                    if len(ids):
                        centroids[(str(unit),stage)]=z[ids].mean(0)
                        counts[f'{unit}:{stage}']=int(len(ids))
            matched=[]
            all_units=[str(u) for u in (*train,test)]
            for stage in range(5):
                available=[u for u in all_units if (u,stage) in centroids]
                for i,u in enumerate(available):
                    for v in available[i+1:]:
                        matched.append(cosine_distance(centroids[(u,stage)],centroids[(v,stage)]))
            health=[]
            for unit in all_units:
                stages=[s for s in range(5) if (unit,s) in centroids]
                for i,s in enumerate(stages):
                    for t in stages[i+1:]:
                        health.append(cosine_distance(centroids[(unit,s)],centroids[(unit,t)]))
            result[layer]={'matched_health_device_distance':float(np.mean(matched)),
                           'health_stage_separation':float(np.mean(health)),
                           'ratio_device_over_health':float(np.mean(matched)/np.mean(health)),
                           'matched_pairs':len(matched),'health_pairs':len(health),'counts':counts}
        report['variants'][variant]={'checkpoint':str(checkpoint),'layers':result}
    (OUT/f'{domain}_conditioned_geometry.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report),flush=True)


def plot_transfer_gain():
    rows=json.loads((AUDIT/'outputs/paired_results.json').read_text(encoding='utf8'))
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True,layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        group=[r for r in rows if r['domain']==domain and r['arm']=='frozen_probe']
        group.sort(key=lambda r:r['fraction'])
        for i,row in enumerate(group):
            gain=np.asarray(row['paired_gain'])
            ax.scatter(i+np.linspace(-.13,.13,5),gain,s=32,color='#3978a8',alpha=.85)
            ax.plot([i-.19,i+.19],[gain.mean()]*2,color='#c8503c',lw=3)
            ax.text(i,gain.mean()+.0025,f"{gain.mean():+.3f}",ha='center',fontsize=8)
        ax.axhline(0,color='black',lw=.9)
        ax.set_xticks(range(3),['10%','20%','100%'])
        ax.set_title(TITLES[domain]); ax.set_xlabel('Labeled training fraction')
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Paired RMSE gain: Single − Joint\npositive favors joint pretraining')
    fig.suptitle('Unknown-device transfer with frozen encoders (5 paired downstream seeds)')
    fig.savefig(OUT/'F1_frozen_transfer_gain.png',dpi=240)
    fig.savefig(OUT/'F1_frozen_transfer_gain.pdf'); plt.close(fig)


def plot_conditioned_geometry():
    reports={d:json.loads((OUT/f'{d}_conditioned_geometry.json').read_text(encoding='utf8')) for d in DOMAINS}
    fig,axes=plt.subplots(2,3,figsize=(14.5,7.2),layout='constrained')
    for col,domain in enumerate(DOMAINS):
        for variant,color,label in zip(VARIANTS,('#777777','#2878b5'),('Single-domain','Joint three-domain')):
            layers=reports[domain]['variants'][variant]['layers']
            axes[0,col].plot(range(5),[layers[x]['matched_health_device_distance'] for x in LAYERS],
                             'o-',lw=2,color=color,label=label)
            axes[1,col].plot(range(5),[layers[x]['health_stage_separation'] for x in LAYERS],
                             'o-',lw=2,color=color,label=label)
        axes[0,col].set_title(TITLES[domain]); axes[0,col].grid(axis='y',alpha=.2)
        axes[1,col].grid(axis='y',alpha=.2)
        for row in range(2): axes[row,col].set_xticks(range(5),LAYER_LABELS,fontsize=8)
    axes[0,0].set_ylabel('Cross-device distance at matched health ↓')
    axes[1,0].set_ylabel('Within-device health-stage separation ↑')
    axes[1,1].set_xlabel('Frozen representation depth')
    axes[0,2].legend(fontsize=8)
    fig.suptitle('Does joint pretraining suppress device variation while preserving health structure?\nFive normalized-RUL bins; source-fitted scaling; cosine distance between device-stage centroids')
    fig.savefig(OUT/'F2_health_conditioned_geometry.png',dpi=240)
    fig.savefig(OUT/'F2_health_conditioned_geometry.pdf'); plt.close(fig)


def plot_interface_summary():
    # Use the representation actually consumed by each archived downstream:
    # Bearing/Battery use snapshot e_t; Milling rebuilds its state from hidden.
    readout={'bearing':'projector','battery':'projector','milling':'hidden'}
    device=[]; health=[]; ratio=[]
    for domain in DOMAINS:
        report=json.loads((OUT/f'{domain}_conditioned_geometry.json').read_text(encoding='utf8'))
        layer=readout[domain]
        single=report['variants']['single_domain']['layers'][layer]
        joint=report['variants']['three_domain_adapter']['layers'][layer]
        device.append(joint['matched_health_device_distance']/single['matched_health_device_distance'])
        health.append(joint['health_stage_separation']/single['health_stage_separation'])
        ratio.append(joint['ratio_device_over_health']/single['ratio_device_over_health'])
    x=np.arange(3); width=.24
    fig,ax=plt.subplots(figsize=(9.2,4.7),layout='constrained')
    ax.bar(x-width,device,width,label='Matched-health device distance',color='#d07a4b')
    ax.bar(x,health,width,label='Health-stage separation',color='#4c9a73')
    ax.bar(x+width,ratio,width,label='Device / health ratio',color='#3978a8')
    ax.axhline(1,color='black',lw=.9)
    ax.set_xticks(x,[TITLES[d] for d in DOMAINS])
    ax.set_ylabel('Joint / Single-domain (1 = unchanged)')
    ax.set_ylim(.9,1.04); ax.grid(axis='y',alpha=.2)
    ax.legend(ncol=3,fontsize=8,loc='lower center')
    ax.set_title('Health-conditioned geometry at the representation used by each downstream')
    fig.savefig(OUT/'F4_downstream_interface_geometry.png',dpi=240)
    fig.savefig(OUT/'F4_downstream_interface_geometry.pdf'); plt.close(fig)


def prediction_file(package,seed):
    base=SUITE/package
    if 'milling' in package:
        matches=list((base/'runtime').glob(f'*seed{seed}_10pct/frozen_probe/predictions.npz'))
    else:
        matches=[base/'runs'/f'seed{seed}'/'fraction10'/'frozen_probe'/'predictions.npz']
    assert len(matches)==1 and matches[0].exists(),(package,seed,matches)
    return matches[0]


def plot_predictions():
    fig,axes=plt.subplots(3,1,figsize=(12.5,9),layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        series={}
        reference=None
        for package,key in zip(PACKAGES[domain],('single','joint')):
            predictions=[]
            for seed in range(42,47):
                z=np.load(prediction_file(package,seed))
                idx=np.argsort(z['rows'],kind='stable')
                y=z['y'][idx]; pred=z['pred'][idx]
                if reference is None: reference=y
                else: np.testing.assert_allclose(reference,y,rtol=0,atol=1e-7)
                predictions.append(pred)
            series[key]=np.stack(predictions)
        x=np.arange(len(reference))
        ax.plot(x,reference,color='black',lw=1.8,label='True normalized RUL')
        for key,color,label in [('single','#777777','Single-domain'),('joint','#2878b5','Joint three-domain')]:
            mean=series[key].mean(0); std=series[key].std(0,ddof=1)
            ax.plot(x,mean,color=color,lw=1.5,label=label)
            ax.fill_between(x,mean-std,mean+std,color=color,alpha=.16,lw=0)
        rmses={key:np.sqrt(np.mean((value-reference[None])**2,axis=1)).mean() for key,value in series.items()}
        ax.set_title(f"{TITLES[domain]} held-out device — RMSE {rmses['single']:.3f} → {rmses['joint']:.3f}")
        ax.set_ylabel('Normalized RUL'); ax.set_ylim(-.08,1.08); ax.grid(alpha=.18)
    axes[-1].set_xlabel('Chronological test snapshot / endpoint')
    axes[0].legend(ncol=3,fontsize=8)
    fig.suptitle('10% labels, frozen encoder: prediction on an unseen device (mean ± SD, 5 seeds)')
    fig.savefig(OUT/'F3_unseen_device_predictions.png',dpi=240)
    fig.savefig(OUT/'F3_unseen_device_predictions.pdf'); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--domain',choices=DOMAINS)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.domain:
        extract_domain(args.domain); return
    for domain in DOMAINS:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--domain',domain],check=True)
    plot_transfer_gain(); plot_conditioned_geometry(); plot_interface_summary(); plot_predictions()
    print('WROTE',OUT)


if __name__=='__main__': main()
