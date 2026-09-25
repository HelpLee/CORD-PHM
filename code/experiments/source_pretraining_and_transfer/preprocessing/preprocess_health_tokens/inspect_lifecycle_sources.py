"""Read-only structural audit of local lifecycle sources."""
from pathlib import Path
import zipfile
import json
import csv
import io
import numpy as np
from scipy.io import loadmat
import h5py

ROOT = Path(__file__).resolve().parents[1] / 'data_phm' / 'raw'

def main():
    for name in ('Ferrara', 'UNSW', 'KAIST', 'FEMTO', 'XJTU'):
        root = ROOT / 'Bearing' / name
        files = sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.mat', '.csv'))
        counts = {}
        for p in files:
            counts[str(p.parent.relative_to(root))] = counts.get(str(p.parent.relative_to(root)), 0) + 1
        print(name, json.dumps(counts), flush=True)
        for p in files[:1]:
            print('FIRST', p, flush=True)
            if p.suffix == '.mat':
                d = loadmat(p, simplify_cells=True)
                print({k: (np.shape(v), str(v)[:180] if np.size(v) < 8 else '') for k,v in d.items() if not k.startswith('__')}, flush=True)
            else:
                with p.open() as stream: print(stream.readline()[:250], flush=True)
    for name in ('luh_tool_wear', 'piecuch_2025'):
        root = ROOT / 'Milling' / name
        for p in root.glob('*.csv'):
            with p.open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            print(name, p.name, len(rows), rows[:1], flush=True)
        for p in root.glob('*.zip'):
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                print(p.name, len(names), names[:8], flush=True)
                if name == 'luh_tool_wear':
                    member = next(n for n in names if n.endswith(('.h5', '.hdf5')))
                    with h5py.File(io.BytesIO(z.read(member)), 'r') as f:
                        f.visititems(lambda n,v: print(n, getattr(v,'shape',None), dict(v.attrs), flush=True))
                else:
                    member = next(n for n in names if n.endswith('.csv'))
                    with z.open(member) as f:
                        print('HEADER', f.readline()[:1200], flush=True)
                        print('ROW', f.readline()[:400], flush=True)

if __name__ == '__main__':
    main()
