"""Deterministic thin upstream cache and full downstream trajectories from raw NPZ."""
import hashlib
import json
import os
import zipfile
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'code/data_phm/processed_health_tokens').is_dir())
OUT = Path(__file__).resolve().parents[1] / 'runtime'
if __import__('os').name == 'nt':
    OUT = Path('\\\\?\\' + str(OUT))
DATA = ROOT / 'code/data_phm/processed_health_tokens'
UP = {'bearing': ['cwru','femto','ferrara','hust','ims','kaist','mfpt','paderborn','seu','unsw'],
      'battery': ['hust','isu_ilcc','mich_exp','nasa','oxford','rwth','sdu','xjtu'],
      'milling': ['luh','matwi','piecuch']}
DOWN = {'bearing': 'xjtu_downstream_bearing', 'battery': 'calce_cs2_downstream_battery',
        'milling': 'phm2010_milling_downstream'}
RESERVED = {'bearing': 'Bearing3_4', 'battery': 'CS2_38', 'milling': 'C6'}
FPT = {'Bearing3_1':2344, 'Bearing3_3':340, 'Bearing3_4':1418, 'Bearing3_5':9}

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    os.replace(temp, path)

def key(x):
    return hashlib.sha256(str(x).encode()).hexdigest()

def read_selected(path, rows):
    """Stream a compressed NPY once, keeping only chosen rows; never allocate full ISU."""
    with zipfile.ZipFile(path) as z, z.open('x_health.npy') as src:
        version = np.lib.format.read_magic(src)
        shape, fortran, dtype = np.lib.format._read_array_header(src, version)
        assert not fortran and dtype == np.dtype('float32')
        result = np.empty((len(rows), *shape[1:]), dtype=dtype)
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, int(rows[-1]) + 1, 512):
            n = min(512, shape[0] - start)
            block = src.read(n * row_bytes)
            assert len(block) == n * row_bytes
            lo, hi = np.searchsorted(rows, [start, start+n])
            if hi > lo:
                a = np.frombuffer(block, dtype).reshape(n, *shape[1:])
                result[lo:hi] = a[rows[lo:hi]-start]
    return result

def build_one(domain, stem, downstream=False, source_path=None):
    """Build a thin cache from the default corpus or an explicitly supplied NPZ.

    ``source_path`` is for auxiliary corpora deliberately kept outside the
    default strict lifecycle directory.  The cache manifest records its exact
    origin, so it cannot silently be reused for a different NPZ.
    """
    path = (Path(source_path) if source_path is not None else DATA / domain / f'{stem}_health_tokens.npz').resolve()
    signature = dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                     upstream_cap_per_unit=24, downstream=downstream, version=1)
    cache = OUT / 'raw_cache' / stem
    if (cache / 'manifest.json').exists():
        assert json.loads((cache / 'manifest.json').read_text())['signature'] == signature
        return cache
    with np.load(path, allow_pickle=True) as z:
        units = z['sample_unit_id'].astype(str)
        conditions = z['sample_condition_id'].astype(str)
        order_key = 'sample_cycle_index' if domain == 'battery' else 'sample_snapshot_index'
        order = np.asarray(z[order_key], float)
        groups = {u: np.flatnonzero(units==u) for u in sorted(set(units))}
        excluded = {}
        if downstream and domain == 'bearing':
            for u in list(groups):
                if u not in FPT:
                    excluded[u] = 'outside 40Hz or no historical FPT'
                    del groups[u]
                else:
                    ids = groups[u][np.argsort(order[groups[u]], kind='stable')]
                    groups[u] = ids[min(FPT[u]-1, max(0,len(ids)-8)):]
        selected = []
        for u, ids in groups.items():
            ids = ids[np.argsort(order[ids], kind='stable')]
            if not downstream:
                ids = ids[np.unique(np.linspace(0, len(ids)-1, min(24,len(ids)), dtype=int))]
            selected.extend(ids.tolist())
        selected = np.array(sorted(selected), np.int64)
        tm = np.asarray(z['token_mask'][selected], bool)
        cm = np.asarray(z['c_mask'][selected], bool)
        y = z['y_rul_norm'][selected].astype('float32') if downstream and domain != 'bearing' else np.zeros(len(selected), 'float32')
        sub_units, sub_order = units[selected], order[selected]
        if downstream and domain == 'bearing':
            for u in groups:
                ids = np.flatnonzero(sub_units == u)
                ids = ids[np.argsort(sub_order[ids], kind='stable')]
                y[ids] = np.linspace(1,0,len(ids),dtype='float32')
        metadata = json.loads(str(z['meta_json'].item()))
        features = z['feature_names'].astype(str).tolist()
        median, iqr = z['feature_median'], z['feature_iqr']
        assert np.all(median == 0) and np.all(iqr == 1), 'Expected unscaled raw-v2 features'
    print('CACHE_START', stem, len(selected), flush=True)
    x = read_selected(path, selected)
    valid = tm & cm[...,None]
    assert np.isfinite(x[valid]).all()
    x[~valid] = 0
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / 'x.npy', x)
    np.savez(cache / 'arrays.npz', tm=tm, cm=cm, y=y, units=sub_units, order=sub_order, source_rows=selected)
    report = dict(signature=signature, domain=domain, downstream=downstream, shape=list(x.shape),
                  counts={u:int(np.sum(sub_units==u)) for u in sorted(set(sub_units))},
                  excluded=excluded, features=features, metadata=metadata)
    write(cache / 'manifest.json', report)
    print('CACHE_DONE', stem, list(x.shape), flush=True)
    return cache

class Store:
    def __init__(self, cache):
        self.path = Path(cache)
        self.info = json.loads((self.path/'manifest.json').read_text(encoding='utf8'))
        self.domain = self.info['domain']; self.down = self.info['downstream']; self.name = self.path.name
        self.x = np.load(self.path/'x.npy', mmap_mode='r')
        with np.load(self.path/'arrays.npz') as z:
            self.tm, self.cm, self.y, self.units, self.order = [z[k] for k in ('tm','cm','y','units','order')]
        self.groups = {u: np.flatnonzero(self.units==u)[np.argsort(self.order[self.units==u],kind='stable')]
                       for u in sorted(set(self.units))}

    def rows(self, units):
        return np.concatenate([self.groups[u] for u in units])

    def split(self):
        units = sorted(self.groups, key=lambda u:key((42,self.name,u)))
        nv = min(len(units)-1, max(1, round(.2*len(units)))) if len(units)>1 else 0
        return units[nv:], units[:nv]

    def normalize(self, train, kind):
        assert not self.down or RESERVED[self.domain] not in train
        ids = self.rows(train)
        x = np.array(self.x[ids]); valid = self.tm[ids] & self.cm[ids,:,None]
        centers, scales = [], []
        channelwise = kind in ('channel_asinh','causal_relative')
        for c in range(x.shape[1] if channelwise else 1):
            vals = x[:,c][valid[:,c]] if channelwise else x[valid]
            if kind == 'pooled_zscore':
                center = vals.mean(axis=0, dtype=np.float64); scale = vals.std(axis=0, dtype=np.float64)
            else:
                q25, center, q75 = np.percentile(vals, [25,50,75], axis=0); scale=q75-q25
            centers.append(center); scales.append(np.where(scale>1e-6,scale,1.0))
        center = np.asarray(centers,'float32')[:,None,:]
        scale = np.asarray(scales,'float32')[:,None,:]
        return center, scale

    def transform(self, rows, norm, kind):
        x = np.array(self.x[rows])
        if kind == 'causal_relative':
            # First RETAINED snapshot, not a whole-life or label-dependent fit.
            # For bearing it is post-FPT, so do not call it a healthy reference.
            baseline = np.empty_like(x[..., :1, :])
            row_units = self.units[rows]
            for u in np.unique(row_units):
                first = self.groups[str(u)][0]
                valid = self.tm[first] & self.cm[first,:,None]
                base = (np.asarray(self.x[first])*valid[...,None]).sum(-2) / valid.sum(-1).clip(1)[:,None]
                baseline[row_units==u] = base[:,None,:]
            x = (x-baseline)/norm[1]
        else:
            x = (x-norm[0])/norm[1]
        if kind in ('channel_asinh','causal_relative'): x = np.arcsinh(x)
        x = np.clip(x,-20,20)
        cm, tm = self.cm[rows], self.tm[rows]
        valid=tm & cm[...,None]; x[~valid]=0
        assert np.isfinite(x).all()
        return x,cm,tm

    def sequences(self, units, length=10):
        return np.concatenate([ids[np.maximum(np.arange(len(ids))[:,None]-np.arange(length-1,-1,-1),0)]
                               for ids in [self.groups[u] for u in units]])

    def sequences_with_mask(self, units, length=10):
        sequences, masks = [], []
        offsets = np.arange(length - 1, -1, -1)
        for unit in units:
            ids = self.groups[unit]
            positions = np.arange(len(ids))[:, None] - offsets
            masks.append(positions >= 0)
            sequences.append(ids[np.maximum(positions, 0)])
        return np.concatenate(sequences), np.concatenate(masks)

def main():
    for domain, stem in DOWN.items(): build_one(domain, stem, True)
    for domain, names in UP.items():
        for name in names: build_one(domain, f'{name}_{domain}')
    print('PREPARED_ALL', flush=True)

if __name__=='__main__': main()
