"""Read-only diagnostics of E28/E29; safe to run on the cluster login node."""
import json
import argparse
from pathlib import Path
import numpy as np


def inspect(suite):
    report = {}
    for experiment in ('full_channel_source_models', 'gradient_aggregation_comparison'):
        arms = {}
        for folder in sorted((suite / experiment / 'models').glob('*')):
            status_path = folder / 'training/status.json'
            history_path = folder / 'training/history.json'
            if not status_path.exists() or not history_path.exists():
                continue
            status = json.loads(status_path.read_text())
            history = json.loads(history_path.read_text())
            if not history:
                continue
            chosen = next((h for h in history if h['epoch'] == status.get('best_epoch')), history[-1])
            item = {'status': status, 'domain_minima': {}}
            for domain, detail in chosen['validation'].items():
                available = [h for h in history if domain in h['validation']]
                best = min(available, key=lambda h: h['validation'][domain]['total'])
                item['domain_minima'][domain] = {
                    'epoch': best['epoch'], 'minimum': best['validation'][domain]['total'],
                    'at_selected_checkpoint': detail['total'],
                    'relative_regret': detail['total'] / best['validation'][domain]['total'] - 1,
                    'datasets': {name: {'mask': v['mask'], 'temporal': v['temporal'],
                        'temporal_share': .2 * v['temporal'] / max(v['total'], 1e-12)}
                        for name, v in detail['datasets'].items()}}
            gs = [g for h in history[-50:] for g in h.get('gradients', [])]
            if gs and len(gs[0]['norms']) == 3:
                cos = np.asarray([g['cosine'] for g in gs])
                item['gradients_last50'] = {
                    'mean_norms': np.mean([g['norms'] for g in gs], axis=0).tolist(),
                    'mean_cosine': cos.mean(0).tolist(),
                    'conflict_frequency': (cos < 0).mean(0).tolist(),
                    'applied_direction_mean_cosine': np.mean([g['direction_cosine'] for g in gs], axis=0).tolist()}
            arms[folder.name] = item
        report[experiment] = arms
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    report = inspect(Path(__file__).resolve().parent.parent)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.compact:
        for experiment, arms in report.items():
            for arm, record in arms.items():
                if arm == 'smoke':
                    continue
                for domain in record['domain_minima'].values():
                    domain.pop('datasets', None)
                print(json.dumps(dict(experiment=experiment, arm=arm, **record)))
    else:
        print(json.dumps(report, indent=2))
