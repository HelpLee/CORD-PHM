"""Explain why self-supervised pretraining helps unseen-device RUL prediction."""
import argparse, importlib, json, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
E=HERE.parent
ROOT=next(p for p in HERE.parents if (p/'code/data_phm').is_dir())
SUITE=ROOT/'code/experiments/source_pretraining_and_transfer'
OUT=HERE/'outputs'
AUDIT=E/'mechanism_audit'
DOMAINS=('bearing','battery','milling')
TITLES={'bearing':'Bearing','battery':'Battery','milling':'Milling'}
PACKAGES={'bearing':'41_downstream_bearing_three_domain_adapter',
          'battery':'31_downstream_battery_three_domain_adapter',
          'milling':'20_downstream_milling_three_domain_adapter'}
SCRATCH_PACKAGES={'bearing':'40_downstream_bearing_scratch',
                  'battery':'30_downstream_battery_scratch',
                  'milling':'22_downstream_milling_scratch'}
READOUT={'bearing':'projector','battery':'projector','milling':'hidden'}


def cosine_distance(a,b):
    return float(1-np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))


def representation_stats(domain):
    sys.path.insert(0,str(AUDIT)); import analyze as audit; import torch
    arrays,y,units,order,train,test=audit.dataset(domain)
    pretrained,_=audit.ex.load_encoder(domain,'three_domain_adapter',torch.device('cpu'))
    joint=importlib.import_module('joint_model')
    torch.manual_seed(42)
    random=joint.JointModel().encoder.eval()
    models={'Random initialization':random,'Self-supervised pretrained':pretrained}
    layer=READOUT[domain]; source=np.isin(units,train)
    bins=np.digitize(np.clip(y,0,1),[.2,.4,.6,.8],right=False)
    report={'domain':domain,'readout':layer,'models':{}}
    for name,model in models.items():
        z=audit.extract(model,domain,arrays)[layer]
        scaler=StandardScaler().fit(z[source]); zs=scaler.transform(z)
        health_scores=[]; knn_errors=[]
        for unit in (*train,test):
            ids=np.flatnonzero(units==unit); ids=ids[np.argsort(order[ids],kind='stable')]
            if len(ids)>250: ids=ids[np.unique(np.rint(np.linspace(0,len(ids)-1,250)).astype(int))]
            dz=pdist(zs[ids],metric='euclidean'); dy=pdist(y[ids,None],metric='cityblock')
            health_scores.append(float(spearmanr(dz,dy).statistic))
            matrix=squareform(dz); np.fill_diagonal(matrix,np.inf)
            neighbors=np.argpartition(matrix,5,axis=1)[:,:5]
            knn_errors.append(float(np.mean(np.abs(y[ids,None]-y[ids][neighbors]))))
        centroids={}
        for unit in (*train,test):
            for stage in range(5):
                ids=np.flatnonzero((units==unit)&(bins==stage))
                if len(ids): centroids[(str(unit),stage)]=zs[ids].mean(0)
        dev=[]; health=[]; all_units=[str(u) for u in (*train,test)]
        for stage in range(5):
            available=[u for u in all_units if (u,stage) in centroids]
            for i,u in enumerate(available):
                for v in available[i+1:]: dev.append(cosine_distance(centroids[(u,stage)],centroids[(v,stage)]))
        for unit in all_units:
            stages=[s for s in range(5) if (unit,s) in centroids]
            for i,s in enumerate(stages):
                for t in stages[i+1:]: health.append(cosine_distance(centroids[(unit,s)],centroids[(unit,t)]))
        report['models'][name]={'health_distance_spearman':float(np.mean(health_scores)),
          'knn_lifecycle_mae':float(np.mean(knn_errors)),
          'matched_health_device_distance':float(np.mean(dev)),
          'health_stage_separation':float(np.mean(health)),
          'device_over_health_ratio':float(np.mean(dev)/np.mean(health)),
          'per_device_spearman':health_scores,'per_device_knn_mae':knn_errors}
    (OUT/f'{domain}_representation_stats.json').write_text(json.dumps(report,indent=2),encoding='utf8')


def rows():
    output=[]
    for domain in DOMAINS:
        pretrained=json.loads((SUITE/PACKAGES[domain]/'results.json').read_text(encoding='utf8'))['rows']
        scratch=json.loads((SUITE/SCRATCH_PACKAGES[domain]/'results.json').read_text(encoding='utf8'))['rows']
        for fraction in (.1,.2,1.):
            for seed in range(42,47):
                selected=[x for x in pretrained if x['fraction']==fraction and x['seed']==seed and x['arm']=='frozen_probe']
                selected += [x for x in scratch if x['fraction']==fraction and x['seed']==seed and x['arm']=='scratch']
                pair={x['arm']:x for x in selected}
                assert set(pair)=={'scratch','frozen_probe'}
                output.append({'domain':domain,'fraction':fraction,'seed':seed,
                  'scratch_rmse':pair['scratch']['metrics']['rmse'],'pretrained_rmse':pair['frozen_probe']['metrics']['rmse'],
                  'scratch_bias':pair['scratch']['metrics']['bias'],'pretrained_bias':pair['frozen_probe']['metrics']['bias']})
    return output


def pred_file(domain,seed,arm):
    base=SUITE/(SCRATCH_PACKAGES[domain] if arm=='scratch' else PACKAGES[domain])
    if domain=='milling': matches=list((base/'runtime').glob(f'*seed{seed}_10pct/{arm}/predictions.npz'))
    else: matches=[base/'runs'/f'seed{seed}'/'fraction10'/arm/'predictions.npz']
    assert len(matches)==1 and matches[0].exists(),matches
    return matches[0]


def figures(records):
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True,layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        for i,fraction in enumerate((.1,.2,1.)):
            r=[x for x in records if x['domain']==domain and x['fraction']==fraction]
            gain=np.array([x['scratch_rmse']-x['pretrained_rmse'] for x in r])
            ax.scatter(i+np.linspace(-.13,.13,5),gain,s=32,color='#2b78b5')
            ax.plot([i-.19,i+.19],[gain.mean()]*2,color='#c84e3a',lw=3)
            ax.text(i,gain.mean()+.003,f'{gain.mean():+.3f}',ha='center',fontsize=8)
        ax.axhline(0,color='black',lw=.9); ax.set_xticks(range(3),['10%','20%','100%'])
        ax.set_title(TITLES[domain]); ax.set_xlabel('Labeled training fraction'); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('RMSE gain: Scratch − Pretrained Frozen\npositive favors pretraining')
    fig.suptitle('Pretraining transfer gain is strongest in the low-label regime (5 paired seeds)')
    fig.savefig(OUT/'P1_pretraining_transfer_gain.png',dpi=240); fig.savefig(OUT/'P1_pretraining_transfer_gain.pdf'); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2),sharey=False,layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        x=np.arange(3); bias=[]; centered=[]
        for f in (.1,.2,1.):
            r=[q for q in records if q['domain']==domain and q['fraction']==f]
            bias.append(np.mean([q['scratch_bias']**2-q['pretrained_bias']**2 for q in r]))
            centered.append(np.mean([(q['scratch_rmse']**2-q['scratch_bias']**2)-(q['pretrained_rmse']**2-q['pretrained_bias']**2) for q in r]))
        ax.bar(x-.18,bias,.36,label='Bias² reduction',color='#3978a8')
        ax.bar(x+.18,centered,.36,label='Centered-error reduction',color='#df8a4d')
        ax.axhline(0,color='black',lw=.8); ax.set_xticks(x,['10%','20%','100%']); ax.set_title(TITLES[domain]); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Scratch − Pretrained error component\npositive favors pretraining'); axes[-1].legend(fontsize=8)
    fig.suptitle('How pretraining reduces downstream MSE')
    fig.savefig(OUT/'P2_pretraining_error_decomposition.png',dpi=240); fig.savefig(OUT/'P2_pretraining_error_decomposition.pdf'); plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(12.5,9),layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        series={}; reference=None
        for arm,key in [('scratch','Scratch'),('frozen_probe','Pretrained Frozen')]:
            values=[]
            for seed in range(42,47):
                z=np.load(pred_file(domain,seed,arm)); idx=np.argsort(z['rows'],kind='stable'); y=z['y'][idx]
                if reference is None: reference=y
                else: np.testing.assert_allclose(reference,y,rtol=0,atol=1e-7)
                values.append(z['pred'][idx])
            series[key]=np.stack(values)
        xx=np.arange(len(reference)); ax.plot(xx,reference,color='black',lw=1.8,label='True RUL')
        for key,color in [('Scratch','#888888'),('Pretrained Frozen','#2878b5')]:
            mean=series[key].mean(0); sd=series[key].std(0,ddof=1); ax.plot(xx,mean,color=color,lw=1.4,label=key); ax.fill_between(xx,mean-sd,mean+sd,color=color,alpha=.16,lw=0)
        rm={k:np.sqrt(np.mean((v-reference[None])**2,axis=1)).mean() for k,v in series.items()}
        ax.set_title(f"{TITLES[domain]} unseen device — RMSE {rm['Scratch']:.3f} → {rm['Pretrained Frozen']:.3f}")
        ax.set_ylim(-.08,1.08); ax.set_ylabel('Normalized RUL'); ax.grid(alpha=.18)
    axes[0].legend(ncol=3,fontsize=8); axes[-1].set_xlabel('Chronological test snapshot / endpoint')
    fig.suptitle('10% labels: Scratch versus a frozen self-supervised encoder (mean ± SD, 5 seeds)')
    fig.savefig(OUT/'P3_scratch_vs_pretrained_predictions.png',dpi=240); fig.savefig(OUT/'P3_scratch_vs_pretrained_predictions.pdf'); plt.close(fig)

    stats={d:json.loads((OUT/f'{d}_representation_stats.json').read_text(encoding='utf8')) for d in DOMAINS}
    metrics=[('health_distance_spearman','Health-distance correlation ↑'),('knn_lifecycle_mae','5-NN lifecycle error ↓'),('device_over_health_ratio','Device / health ratio ↓')]
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.3),layout='constrained')
    for ax,(metric,title) in zip(axes,metrics):
        x=np.arange(3); random=[stats[d]['models']['Random initialization'][metric] for d in DOMAINS]; pre=[stats[d]['models']['Self-supervised pretrained'][metric] for d in DOMAINS]
        ax.bar(x-.18,random,.36,color='#aaaaaa',label='Random encoder'); ax.bar(x+.18,pre,.36,color='#2878b5',label='Pretrained encoder')
        ax.set_xticks(x,[TITLES[d] for d in DOMAINS]); ax.set_title(title); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Frozen representation metric'); axes[-1].legend(fontsize=8)
    fig.suptitle('What self-supervised pretraining changes before any RUL-head training')
    fig.savefig(OUT/'P4_pretraining_representation_mechanism.png',dpi=240); fig.savefig(OUT/'P4_pretraining_representation_mechanism.pdf'); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--domain',choices=DOMAINS); a=p.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if a.domain: representation_stats(a.domain); return
    for d in DOMAINS: subprocess.run([sys.executable,str(Path(__file__).resolve()),'--domain',d],check=True)
    r=rows(); (OUT/'paired_scratch_pretrained.json').write_text(json.dumps(r,indent=2),encoding='utf8'); figures(r)
    print('WROTE',OUT)
if __name__=='__main__': main()
