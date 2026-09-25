"""Read verified downstream tensors without raw-data paths or cache mutation."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = next(p for p in HERE.parents if (p / 'code/data_phm').is_dir())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_store(domain):
    import data
    import global_local_data as gl
    folder = HERE / 'inputs' / domain
    audit = json.loads((folder / 'bundle.json').read_text())
    for name, digest in audit['files'].items():
        if sha(folder / name) != digest:
            raise RuntimeError('Portable input checksum mismatch: ' + name)
    source = WORKSPACE / audit['source_relative']
    if sha(source) != audit['source_sha256']:
        raise RuntimeError('Canonical NPZ differs from the exported downstream data')
    local = data.Store(folder / 'cache')
    local.info['signature']['source'] = str(source)
    local.info['signature']['path'] = str(source)
    local.feature_mask = np.load(folder / 'feature_mask.npy', mmap_mode='r')
    global_x = np.load(folder / 'global.npy', mmap_mode='r')
    original_raw, original_aggregate = gl.raw_globals, gl.battery_globals
    try:
        gl.raw_globals = gl.battery_globals = lambda store: global_x
        store = gl.GlobalLocalStore(local, calibrate=domain == 'bearing', channels=local.x.shape[1])
    finally:
        gl.raw_globals, gl.battery_globals = original_raw, original_aggregate
    with np.load(folder / 'reference.npz') as reference:
        norm = store.normalize(audit['train_units'])
        for expected, actual in zip(reference['norm'], norm):
            np.testing.assert_array_equal(expected, actual)
        for key, actual in zip(('x', 'g', 'cm', 'tm'), store.transform(reference['rows'], norm)):
            if key in ('cm', 'tm'):
                np.testing.assert_array_equal(reference[key], actual)
            else:
                # Windows/Linux NumPy arcsinh may differ by a few float32 ULPs.
                # Raw files and train-only scalers remain checksum/exact checked.
                np.testing.assert_allclose(reference[key], actual, rtol=1e-6, atol=1e-7)
    return store
