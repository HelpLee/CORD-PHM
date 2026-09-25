"""Resume the Domain Adapter milling evaluation without repeating completed groups."""
import json
import os
from pathlib import Path

import numpy as np

import run_experiment as experiment
import downstream_interval_val200 as down


def configure(store):
    down.prepare_store = lambda _path: store
    down.OUT = experiment.RUNTIME
    down.SOURCE = experiment.CONVERTED
    down.MAX_UPDATES = 0
    down.MAX_EPOCHS = 200
    down.PATIENCE = 15
    down.VAL_INTERVAL = 2
    down.TRAIN_VALIDATION_FRACTION = .2
    down.TEST_VALIDATION_FRACTION = 0
    down.NESTED_LABELS = False
    down.ARMS = ('scratch', 'frozen_probe', 'partial_finetune', 'full_finetune')
    down.source_split = experiment.interval_split


def completed_rows():
    rows = []
    for seed in (42, 43, 44, 45, 46):
        for fraction in (.1, .2, 1.0):
            path = experiment.RUNTIME / f'adapter_transfer_seed{seed}_{int(fraction * 100)}pct' / 'development_summary.json'
            if not path.exists():
                continue
            content = json.loads(path.read_text(encoding='utf-8'))
            assert content['complete'] and len(content['rows']) == 4
            for row in content['rows']:
                row['fraction'] = fraction
                rows.append(row)
    return rows


def aggregate(rows):
    summary = []
    for fraction in (.1, .2, 1.0):
        for arm in down.ARMS:
            group = [r for r in rows if r['fraction'] == fraction and r['arm'] == arm]
            assert len(group) == 5
            summary.append(dict(fraction=fraction, arm=arm, n=5, metrics={
                metric: dict(mean=float(np.mean([r['metrics'][metric] for r in group])),
                             std=float(np.std([r['metrics'][metric] for r in group], ddof=1)))
                for metric in ('rmse', 'mae', 'r2', 'bias')}))
    return summary


def main():
    upstream = json.loads((experiment.PACKAGE.parent / 'training/status.json').read_text())
    assert upstream['state'] == 'complete'
    experiment.convert_checkpoint()
    store = experiment.load_store()
    configure(store)
    rows = completed_rows()
    experiment.write(experiment.PACKAGE / 'results.json', {'complete': False, 'rows': rows})

    for seed in (42, 43, 44, 45, 46):
        for fraction in (.1, .2, 1.0):
            output = experiment.RUNTIME / f'adapter_transfer_seed{seed}_{int(fraction * 100)}pct'
            if (output / 'development_summary.json').exists():
                continue
            assert not output.exists(), f'Incomplete output must be archived before resume: {output}'
            down.SEED = seed
            down.LABEL_FRACTION = fraction
            down.NAME = output.name
            experiment.write(experiment.PACKAGE / 'status.json', {
                'state': 'running', 'seed': seed, 'fraction': fraction,
                'pid': os.getpid(), 'completed_rows': len(rows),
            })
            down.main()
            content = json.loads((output / 'development_summary.json').read_text(encoding='utf-8'))
            for row in content['rows']:
                row['fraction'] = fraction
                rows.append(row)
            experiment.write(experiment.PACKAGE / 'results.json', {'complete': False, 'rows': rows})

    experiment.write(experiment.PACKAGE / 'results.json', {
        'complete': True, 'rows': rows, 'summary': aggregate(rows)})
    experiment.write(experiment.PACKAGE / 'status.json', {'state': 'completed', 'runs': len(rows)})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        experiment.write(experiment.PACKAGE / 'status.json', {'state': 'failed', 'error': repr(error)})
        raise
