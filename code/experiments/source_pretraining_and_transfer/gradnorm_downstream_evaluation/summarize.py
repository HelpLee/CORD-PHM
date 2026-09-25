"""Create paired single/three-domain RMSE, MAE and R2 tables after all six jobs."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    summary = []
    for domain in ('bearing', 'battery', 'milling'):
        for initialization in ('single', 'three'):
            package = HERE / 'runs' / domain / initialization
            payload = json.loads((package / 'results.json').read_text())
            assert payload['complete'] and len(payload['rows']) == 15
            source = json.loads((package / 'source.json').read_text())
            for fraction in (.1, .2, 1.):
                rows = [r for r in payload['rows'] if r['fraction'] == fraction and r['arm'] == 'frozen_probe']
                assert sorted(r['seed'] for r in rows) == [42, 43, 44, 45, 46]
                summary.append(dict(domain=domain, initialization=initialization, fraction=fraction, n=5,
                    source=source, metrics={m: dict(mean=float(np.mean([r['metrics'][m] for r in rows])),
                    std=float(np.std([r['metrics'][m] for r in rows], ddof=1))) for m in ('rmse', 'mae', 'r2')}))
    (HERE / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# GradNorm upstream: unseen-device frozen-probe comparison', '',
             'BF16 on A100; five downstream seeds; each upstream selects its validation-best checkpoint.', '',
             '| Domain | Labels | Source | RMSE | MAE | R2 |', '|---|---|---|---:|---:|---:|']
    for row in summary:
        peers = [r for r in summary if (r['domain'], r['fraction']) == (row['domain'], row['fraction'])]
        values = []
        for metric in ('rmse', 'mae', 'r2'):
            value = row['metrics'][metric]
            text = '%.4f ± %.4f' % (value['mean'], value['std'])
            best = (max if metric == 'r2' else min)(r['metrics'][metric]['mean'] for r in peers)
            values.append('<u>**' + text + '**</u>' if value['mean'] == best else text)
        lines.append('| %s | %d%% | %s | %s |' % (row['domain'], row['fraction'] * 100, row['initialization'], ' | '.join(values)))
    (HERE / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
