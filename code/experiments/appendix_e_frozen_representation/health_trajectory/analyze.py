"""Joint-PCA visualization of frozen health trajectories."""
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VARIANTS = ('single_domain', 'three_domain_no_adapter', 'three_domain_adapter')
LABELS = ('Single-domain', 'Three-domain without Adapter', 'Three-domain + Adapter')
DOMAINS = ('bearing', 'battery', 'milling')


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit('matplotlib is required for the trajectory figures') from error
    out = HERE / 'outputs'; out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for domain in DOMAINS:
        payloads = [np.load(ROOT / 'outputs' / domain / f'{variant}.npz') for variant in VARIANTS]
        y, order = payloads[0]['y'], payloads[0]['order']
        assert all(np.array_equal(y, p['y']) and np.array_equal(order, p['order']) for p in payloads[1:])
        combined = np.concatenate([p['embedding'] for p in payloads])
        centered = combined - combined.mean(0, keepdims=True)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        coordinates = centered @ vt[:2].T
        explained = (singular[:2] ** 2 / np.square(singular).sum()).tolist()
        pieces = np.split(coordinates, len(VARIANTS))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        # Use the continuous normalized RUL itself as the trajectory colour.
        # The reversed map makes the healthy (RUL=1) end visually distinct
        # from the failed (RUL=0) end while preserving a single shared scale.
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.get_cmap('viridis_r')
        csv_path = out / f'{domain}_pca_coordinates.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle); writer.writerow(['variant','sample','order','rul','stage','pc1','pc2'])
            for ax, label, coords in zip(axes, LABELS, pieces):
                ax.plot(coords[:, 0], coords[:, 1], color='#777777', alpha=.28, linewidth=.8)
                stages = np.where(y > .67, 'Early', np.where(y > .33, 'Middle', 'Late'))
                ax.scatter(coords[:, 0], coords[:, 1], s=20, alpha=.86,
                           c=y, cmap=cmap, norm=norm)
                ax.set_title(label); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(alpha=.15)
                for i, (xy, rul, st) in enumerate(zip(coords, y, stages)):
                    writer.writerow([label, i, float(order[i]), float(rul), st,
                                     float(xy[0]), float(xy[1])])
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        fig.colorbar(scalar, ax=axes, label='RUL (continuous; 1.0 healthy → 0.0 failed)')
        fig.suptitle(f'{domain.capitalize()} frozen health trajectory')
        fig.savefig(out / f'{domain}_pca_health_trajectory.png', dpi=180)
        plt.close(fig)
        summary[domain] = dict(samples=int(len(y)), joint_pca=True,
                               explained_variance_ratio=explained,
                               test_unit=str(payloads[0]['unit'].item()))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
