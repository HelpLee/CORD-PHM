"""Frozen layer-wise cross-tool linear health probe for PHM2010.

The probe is selected only with C1/C4. C6 is evaluated once and never selects
layer, regularization, sign, normalization, labels, or stopping.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
E = HERE.parent
AUDIT = E / 'mechanism_audit'
OUT = HERE / 'outputs'
sys.path.insert(0, str(AUDIT))
import analyze as audit

LAYERS = audit.LAYERS
LABELS = ('Pooled input', 'Block 1', 'Block 2', 'Final hidden', 'Projector')
VARIANTS = audit.VARIANTS
ALPHAS = (.1, 1., 10., 100., 1000.)


def metric(y, pred):
    error = pred-y
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))), bias=float(error.mean()),
                centered_rmse=float(np.sqrt(np.mean((error-error.mean())**2))),
                spearman=float(spearmanr(pred, y).statistic))


def eligible_by_unit(units, order):
    keep = np.zeros(len(units), dtype=bool)
    for unit in ('C1', 'C4', 'C6'):
        ids = np.flatnonzero(units == unit)
        ids = ids[np.argsort(order[ids], kind='stable')]
        keep[ids[19:]] = True
    return keep


def selected_labels(units, order, eligible, fraction):
    keep = np.zeros(len(units), dtype=bool)
    for unit in ('C1', 'C4'):
        ids = np.flatnonzero(eligible & (units == unit))
        ids = ids[np.argsort(order[ids], kind='stable')]
        n = max(1, int(np.ceil(len(ids)*fraction)))
        chosen = np.unique(np.rint(np.linspace(0, len(ids)-1, n)).astype(int))
        keep[ids[chosen]] = True
    return keep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, y, units, order, train_units, test_unit = audit.dataset('milling')
    eligible = eligible_by_unit(units, order)
    target = eligible & (units == test_unit)
    records = []
    extracted = {}
    for variant in VARIANTS:
        model, checkpoint = audit.ex.load_encoder('milling', variant, audit.torch.device('cpu'))
        extracted[variant] = audit.extract(model, 'milling', arrays)
        del model
        for fraction in (.1, .2, 1.):
            chosen = selected_labels(units, order, eligible, fraction)
            for layer in LAYERS:
                z = extracted[variant][layer]
                source_scores = []
                for alpha in ALPHAS:
                    fold = []
                    for held in train_units:
                        tr = chosen & (units != held) & np.isin(units, train_units)
                        va = chosen & (units == held)
                        probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                        probe.fit(z[tr], y[tr])
                        fold.append(metric(y[va], probe.predict(z[va]))['rmse'])
                    source_scores.append(float(np.mean(fold)))
                best = int(np.argmin(source_scores))
                probe = make_pipeline(StandardScaler(), Ridge(alpha=ALPHAS[best]))
                probe.fit(z[chosen], y[chosen])
                pred = probe.predict(z[target])
                row = dict(variant=variant, fraction=fraction, layer=layer,
                           alpha=ALPHAS[best], source_loco_rmse=source_scores[best],
                           source_loco_curve=source_scores, n_train=int(chosen.sum()),
                           n_test=int(target.sum()), checkpoint=str(checkpoint),
                           **metric(y[target], pred))
                records.append(row)
                np.savez_compressed(OUT/f'{variant}_{layer}_{int(fraction*100)}.npz',
                                    pred=pred, y=y[target], rows=np.flatnonzero(target))
    (OUT/'results.json').write_text(json.dumps(records, indent=2), encoding='utf8')

    # Do C1 and C4 induce the same linear health direction in a common feature
    # coordinate system?  Fit a common unlabeled source scaler, choose alpha by
    # symmetric C1<->C4 transfer, then compare the two independently fitted
    # coefficient vectors. C6 is not used anywhere in this calculation.
    directions = []
    source = eligible & np.isin(units, train_units)
    for variant in VARIANTS:
        for layer in LAYERS:
            z = extracted[variant][layer]
            scaler = StandardScaler().fit(z[source])
            zs = scaler.transform(z)
            folds = []
            for alpha in ALPHAS:
                errors = []
                for fit_unit, eval_unit in (('C1','C4'),('C4','C1')):
                    fit = eligible & (units == fit_unit)
                    eva = eligible & (units == eval_unit)
                    probe = Ridge(alpha=alpha).fit(zs[fit], y[fit])
                    errors.append(metric(y[eva], probe.predict(zs[eva]))['rmse'])
                folds.append(float(np.mean(errors)))
            best = int(np.argmin(folds))
            coefs = []
            transfer = []
            target_predictions = []
            for fit_unit, eval_unit in (('C1','C4'),('C4','C1')):
                fit = eligible & (units == fit_unit)
                eva = eligible & (units == eval_unit)
                probe = Ridge(alpha=ALPHAS[best]).fit(zs[fit], y[fit])
                coefs.append(probe.coef_)
                transfer.append(metric(y[eva], probe.predict(zs[eva]))['rmse'])
                target_predictions.append(probe.predict(zs[target]))
            cosine = float(np.dot(coefs[0],coefs[1]) /
                           (np.linalg.norm(coefs[0])*np.linalg.norm(coefs[1])+1e-12))
            directions.append(dict(variant=variant, layer=layer, alpha=ALPHAS[best],
                                   direction_cosine=cosine,
                                   source_cross_tool_rmse=float(np.mean(transfer)),
                                   c6_ensemble=metric(y[target], np.mean(target_predictions,axis=0))))
    (OUT/'source_direction_consistency.json').write_text(
        json.dumps(directions, indent=2), encoding='utf8')

    # Unsupervised post-hoc device separability.  Lower between/within scatter
    # and silhouette imply less cutter-identity structure. This uses C6 device
    # identity only for analysis and never tunes or trains a downstream model.
    invariance = []
    analysis_ids = np.flatnonzero(eligible)
    for variant in VARIANTS:
        for layer in LAYERS:
            z = StandardScaler().fit_transform(extracted[variant][layer][analysis_ids])
            labels = units[analysis_ids]
            mean = z.mean(0)
            between = 0.; within = 0.
            for unit in ('C1','C4','C6'):
                current = z[labels == unit]
                center = current.mean(0)
                between += len(current)*float(np.sum((center-mean)**2))
                within += float(np.sum((current-center)**2))
            invariance.append(dict(variant=variant, layer=layer,
                                   between_within_ratio=between/max(within,1e-12),
                                   device_silhouette=float(silhouette_score(z,labels,metric='euclidean'))))
    (OUT/'device_invariance.json').write_text(json.dumps(invariance,indent=2),encoding='utf8')

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    for ax, fraction in zip(axes, (.1, .2, 1.)):
        for variant, color, label in zip(VARIANTS, ('#777777','#2468a0'),
                                         ('Milling-only', 'Three-domain')):
            rows = [r for r in records if r['fraction']==fraction and r['variant']==variant]
            rows.sort(key=lambda r: LAYERS.index(r['layer']))
            ax.plot(range(5), [r['rmse'] for r in rows], 'o-', color=color, label=label)
        ax.set_xticks(range(5), LABELS, rotation=20, ha='right')
        ax.set_title(f'{int(fraction*100)}% source labels')
        ax.set_ylabel('C6 linear-probe RMSE (lower better)')
        ax.grid(axis='y', alpha=.2)
    axes[-1].legend()
    fig.suptitle('Frozen layer-wise cross-tool health decoding: C1+C4 train -> C6 test')
    fig.savefig(OUT/'layerwise_transfer_probe.png', dpi=220)
    fig.savefig(OUT/'layerwise_transfer_probe.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
    for variant, color, label in zip(VARIANTS, ('#777777','#2468a0'),
                                     ('Milling-only','Three-domain')):
        rows=[r for r in directions if r['variant']==variant]
        rows.sort(key=lambda r: LAYERS.index(r['layer']))
        ax.plot(range(5),[r['direction_cosine'] for r in rows],'o-',color=color,label=label)
    ax.axhline(0,color='black',lw=.8); ax.set_ylim(-1,1)
    ax.set_xticks(range(5),LABELS,rotation=20,ha='right')
    ax.set_ylabel('Cosine(C1 health direction, C4 health direction)')
    ax.set_title('Source-only health-direction consistency (higher is better)')
    ax.grid(axis='y',alpha=.2); ax.legend()
    fig.savefig(OUT/'source_direction_consistency.png',dpi=220)
    fig.savefig(OUT/'source_direction_consistency.pdf')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    for variant,color,label in zip(VARIANTS,('#777777','#2468a0'),
                                   ('Milling-only','Three-domain')):
        rows=[r for r in invariance if r['variant']==variant]
        rows.sort(key=lambda r: LAYERS.index(r['layer']))
        ax.plot(range(5),[r['between_within_ratio'] for r in rows],'o-',color=color,label=label)
    ax.set_xticks(range(5),LABELS,rotation=20,ha='right')
    ax.set_ylabel('Cutter between/within scatter (lower = invariant)')
    ax.set_title('Frozen representation cutter-identity structure')
    ax.grid(axis='y',alpha=.2); ax.legend()
    fig.savefig(OUT/'device_invariance.png',dpi=220)
    fig.savefig(OUT/'device_invariance.pdf'); plt.close(fig)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
