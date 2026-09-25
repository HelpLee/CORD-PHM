"""Exploratory, fixed-protocol audit: source-device retrieval and paired RUL gains.

No encoder/head training. All source snapshots form a diagnostic reference bank;
this is not a new 10/20/100% supervised benchmark. Never select k using targets.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
E = HERE.parent
OUT = HERE / 'outputs'
sys.path.insert(0, str(E))
import extract_representations as ex

DOMAINS = ('bearing', 'battery', 'milling')
VARIANTS = ('single_domain', 'three_domain_adapter')
LAYERS = ('input_tokens', 'block1', 'block2', 'hidden', 'projector')
LABELS = ('Pooled\ninput', 'Block 1\n+ Adapter', 'Block 2\n+ Adapter', 'FinalNorm\nhidden', 'Snapshot\nprojector')
PACKAGES = {
    'bearing': ('42_downstream_bearing_only_adapter', '41_downstream_bearing_three_domain_adapter'),
    'battery': ('32_downstream_battery_only_adapter', '31_downstream_battery_three_domain_adapter'),
    'milling': ('21_downstream_milling_only_adapter', '20_downstream_milling_three_domain_adapter'),
}


def dataset(domain):
    if domain == 'battery':
        sys.path.insert(0, str(ex.SUITE / 'shared_domain_components/battery_code'))
        m = importlib.import_module('run_battery_state_trend')
        s = m.Downstream()
        train, test = m.TRAIN, m.TEST
        ids = np.concatenate([s.groups[u] for u in (*train, test)])
        arrays = tuple(a[ids] for a in (s.xn, s.gn, s.cm, s.tm))
        return arrays, s.y[ids], s.units[ids], s.order[ids], train, test
    if domain == 'bearing':
        sys.path.insert(0, str(ex.SUITE / 'shared_domain_components/bearing_code'))
        gl = importlib.import_module('global_local_data')
        gl.OUT = ex.SUITE / 'shared_domain_components/bearing_runtime'
        s = gl.GlobalLocalStore(gl.build_condition2(), channels=2)
        train = ('Bearing2_2', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5')
        test = 'Bearing2_1'
    else:
        sys.path.insert(0, str(ex.SUITE / '20_downstream_milling_three_domain_adapter/code'))
        d = importlib.import_module('data')
        d.OUT = ex.SUITE / 'shared_domain_components/milling_runtime'
        m = importlib.import_module('run_milling_global_local_multiscale12')
        s = m.prepare_store(d.OUT / 'raw_cache' / d.DOWN['milling'])
        train, test = ('C1', 'C4'), 'C6'
    ids = np.concatenate([s.groups[u] for u in (*train, test)])
    return s.transform(ids, s.normalize(train)), s.y[ids], s.units[ids], s.order[ids], train, test


def extract(model, domain, arrays):
    captured = {}
    def before(module, args):
        captured['input_tokens'] = args[0]
    def after(name):
        def hook(module, args, output):
            captured[name] = output
        return hook
    handles = [model.transformer.layers[0].register_forward_pre_hook(before)]
    handles += [layer.register_forward_hook(after(f'block{i+1}')) for i, layer in enumerate(model.transformer.layers)]
    handles.append(model.final_norm.register_forward_hook(after('hidden')))
    values = {key: [] for key in LAYERS}
    model.requires_grad_(False).eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(arrays[0]), 64):
                batch = tuple(torch.as_tensor(a[start:start+64]) for a in arrays)
                captured.clear()
                embedding, _ = model(domain, *batch)
                valid = (batch[3].bool() & batch[2].bool()[..., None]).any(1)
                for key in LAYERS[:-1]:
                    h = captured[key]
                    local = (h[:, 1:] * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
                    values[key].append(torch.cat((h[:, 0], local), -1).numpy().copy())
                values['projector'].append(embedding.numpy().copy())
    finally:
        for handle in handles:
            handle.remove()
    return {key: np.concatenate(parts).astype('float64') for key, parts in values.items()}


def distance(a, b):
    return np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T, 0)


def analyze_domain(domain):
    # Each domain runs in a separate process to isolate identically named modules.
    arrays, y, units, order, train, test = dataset(domain)
    source_ids = np.flatnonzero(np.isin(units, train))
    target_ids = np.flatnonzero(units == test)
    assert not set(source_ids) & set(target_ids)
    source_y, target_y = y[source_ids], y[target_ids]
    report = {'domain': domain, 'train_units': list(train), 'test_unit': test,
              'source_n': len(source_ids), 'target_n': len(target_ids),
              'reference_bank': 'all source snapshots; diagnostic only',
              'metric': 'mean absolute true normalized RUL difference of query and source neighbors',
              'primary_k': 5, 'sensitivity_k': [1, 10], 'variants': {}}
    for variant in VARIANTS:
        model, checkpoint = ex.load_encoder(domain, variant, torch.device('cpu'))
        z = extract(model, domain, arrays)
        cached = np.load(E/'outputs'/domain/f'{variant}.npz', allow_pickle=False)
        idx = target_ids[np.argsort(order[target_ids], kind='stable')]
        cache_idx = np.argsort(cached['order'], kind='stable')
        np.testing.assert_allclose(y[idx], cached['y'][cache_idx], rtol=0, atol=1e-6)
        np.testing.assert_allclose(z['projector'][idx], cached['embedding'][cache_idx], rtol=1e-4, atol=1e-5)
        max_difference = float(np.max(np.abs(z['projector'][idx]-cached['embedding'][cache_idx])))
        result = {}
        for layer in LAYERS:
            ranking = np.argsort(distance(z[layer][target_ids], z[layer][source_ids]), axis=1, kind='stable')[:, :10]
            errors = np.abs(target_y[:, None] - source_y[ranking])
            result[layer] = {str(k): float(errors[:, :k].mean()) for k in (1, 5, 10)}
            np.savez_compressed(OUT / f'{domain}_{variant}_{layer}_cross_device.npz',
                                target_y=target_y, source_y=source_y,
                                source_neighbor_ids=source_ids[ranking], target_ids=target_ids,
                                query_error_k5=errors[:, :5].mean(1))
        # Intermediate 192D pooled h reproduces the tensor entering projector.
        # Downstream Milling consumes this h via a NEW learned token_fusion.
        report['variants'][variant] = {'scores': result, 'checkpoint': str(checkpoint),
                                      'sha256': ex.sha256(checkpoint),
                                      'E1_embedding_max_absolute_difference': max_difference}
    (OUT / f'{domain}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(domain, json.dumps({v: report['variants'][v]['scores'] for v in VARIANTS}), flush=True)


def paired_results():
    results = []
    for domain in DOMAINS:
        loaded = [json.loads((ex.SUITE / package / 'results.json').read_text(encoding='utf8'))['rows']
                  for package in PACKAGES[domain]]
        maps = [{(r['fraction'], r['arm'], r['seed']): r for r in rows} for rows in loaded]
        for fraction in (.1, .2, 1.):
            for arm in ('frozen_probe', 'partial_finetune', 'full_finetune'):
                keys = [(fraction, arm, seed) for seed in range(42, 47)]
                assert all(k in m for k in keys for m in maps)
                a, b = [np.array([m[k]['metrics']['rmse'] for k in keys]) for m in maps]
                bias_a, bias_b = [np.array([m[k]['metrics']['bias'] for k in keys]) for m in maps]
                delta = a-b
                results.append(dict(domain=domain, fraction=fraction, arm=arm,
                    single_mean=float(a.mean()), three_mean=float(b.mean()),
                    single_std=float(a.std(ddof=1)), three_std=float(b.std(ddof=1)),
                    paired_gain=delta.tolist(), mean_gain=float(delta.mean()),
                    three_wins=int((delta>0).sum()),
                    single_mse=float(np.mean(a*a)), three_mse=float(np.mean(b*b)),
                    bias_squared_gain=float(np.mean(bias_a*bias_a-bias_b*bias_b)),
                    centered_mse_gain=float(np.mean((a*a-bias_a*bias_a)-(b*b-bias_b*bias_b)))))
    (OUT/'paired_results.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), constrained_layout=True)
    for ax, domain in zip(axes, DOMAINS):
        group = [r for r in results if r['domain'] == domain and r['arm']=='frozen_probe']
        for i, r in enumerate(group):
            gains = np.asarray(r['paired_gain'])
            ax.scatter(np.full(5, i)+np.linspace(-.12,.12,5), gains, color='#2468a0', s=25)
            ax.plot([i-.18,i+.18], [gains.mean()]*2, color='#c6543c', lw=3)
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(range(3), ['10%', '20%', '100%'])
        ax.set_xlabel('Downstream label budget')
        ax.set_title(domain.title())
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('RMSE(single) - RMSE(three)\nPositive favors three-domain')
    fig.suptitle('Frozen encoder transfer: five paired downstream seeds\nDots = paired seeds; red line = mean (one upstream checkpoint per setting)')
    fig.savefig(OUT/'frozen_paired_gain.png', dpi=220)
    fig.savefig(OUT/'frozen_paired_gain.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for ax, domain in zip(axes, DOMAINS):
        group = [r for r in results if r['domain']==domain and r['arm']=='frozen_probe']
        x = np.arange(3)
        ax.bar(x-.18, [r['bias_squared_gain'] for r in group], width=.35,
               label='Bias-squared reduction', color='#2468a0')
        ax.bar(x+.18, [r['centered_mse_gain'] for r in group], width=.35,
               label='Centered-error reduction', color='#e19245')
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(x, ['10%', '20%', '100%'])
        ax.set_title(domain.title())
        ax.set_xlabel('Downstream label budget')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Single minus three-domain error component\nPositive favors three-domain')
    axes[2].legend(fontsize=8)
    fig.suptitle('Frozen downstream: where does the average MSE improvement come from?\nMSE = (mean prediction error)^2 + mean centered squared error; averaged over 5 seeds')
    fig.savefig(OUT/'error_decomposition.png', dpi=220)
    fig.savefig(OUT/'error_decomposition.pdf')
    plt.close(fig)
    return results


def plot_retrieval():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    reports = {}
    for ax, domain in zip(axes, DOMAINS):
        d = json.loads((OUT/f'{domain}.json').read_text(encoding='utf8'))
        reports[domain] = d
        for variant, color, label in zip(VARIANTS, ['#777777', '#2468a0'], ['Single-domain + Adapter', 'Three-domain + Adapter']):
            ax.plot(range(5), [d['variants'][variant]['scores'][layer]['5'] for layer in LAYERS],
                    'o-', label=label, color=color, lw=2)
        ax.set_xticks(range(5), LABELS, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{domain.title()}: source devices -> {d['test_unit']}")
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Cross-device 5-NN RUL discrepancy (lower is better)')
    axes[2].legend(fontsize=8)
    fig.suptitle('Frozen representation audit: query the held-out device, retrieve ONLY from source devices\nAll source snapshots; no target neighbors, no probe fitting, no target-selected k')
    fig.savefig(OUT/'cross_device_retrieval.png', dpi=220)
    fig.savefig(OUT/'cross_device_retrieval.pdf')
    plt.close(fig)
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=DOMAINS)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.domain:
        analyze_domain(args.domain)
        return
    for domain in DOMAINS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--domain', domain], check=True)
    retrieval = plot_retrieval()
    paired = paired_results()
    lines = ['# Exploratory mechanism audit', '',
             'All comparisons are descriptive; no model or training protocol is changed.',
             'Existing E1--E4 are preserved. k=5 primary; k=1,10 sensitivity, all reported.',
             'Pairs within a device are dependent. Five downstream seeds are not five independent pretrained encoders.', '',
             '| Domain | Readout | Single | Three | Single minus three |', '|---|---|---:|---:|---:|']
    for domain, d in retrieval.items():
        for layer in LAYERS:
            a,b = [d['variants'][v]['scores'][layer]['5'] for v in VARIANTS]
            lines.append(f'| {domain} | {layer} | {a:.5f} | {b:.5f} | {a-b:+.5f} |')
    lines += ['', '| Domain | Label | Arm | Paired RMSE gain | Three wins / 5 |', '|---|---:|---|---:|---:|']
    for r in paired:
        lines.append(f"| {r['domain']} | {r['fraction']:.0%} | {r['arm']} | {r['mean_gain']:+.5f} | {r['three_wins']} |")
    lines += ['', '| Domain | Labels | Frozen bias-squared reduction | Frozen centered-MSE reduction |',
              '|---|---:|---:|---:|']
    for r in paired:
        if r['arm']=='frozen_probe':
            lines.append(f"| {r['domain']} | {r['fraction']:.0%} | {r['bias_squared_gain']:+.6f} | {r['centered_mse_gain']:+.6f} |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
