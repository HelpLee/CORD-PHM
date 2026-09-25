"""Export exact historical bearing/milling input bundles; no new feature recipe."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from portable_inputs import sha, HERE, WORKSPACE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=('bearing', 'milling'))
    args = parser.parse_args()
    domain = args.domain
    suite = HERE.parent
    sys.path.insert(0, str(WORKSPACE / 'code'))
    if domain == 'bearing':
        sys.path.insert(0, str(suite / '01_shared_dependencies/bearing_code'))
        import bearing_adapter_study as original
        import global_local_data as gl
        store, norm, *_ = original.prepare()
        train = original.TRAIN
        source = WORKSPACE / 'code/data_phm/processed_health_tokens/bearing/xjtu_downstream_bearing_health_tokens.npz'
        global_x = gl.raw_globals(store.local)
    else:
        sys.path.insert(0, str(suite / '16_update_matched_all7_fullchannels/downstream/code'))
        import run_all_channels_scratch as original
        import global_local_data as gl
        # The old data cache is read; new provenance is written only here.
        original.PACKAGE = HERE / 'inputs/milling'
        original.PACKAGE.mkdir(parents=True, exist_ok=True)
        store = original.load_store()
        train = ('C1', 'C4')
        norm = store.normalize(train)
        source = WORKSPACE / 'code/data_phm/processed_health_tokens/milling/phm2010_milling_downstream_health_tokens.npz'
        global_x = gl.battery_globals(store.local)
    folder = HERE / 'inputs' / domain
    cache = folder / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    local = store.local
    np.save(cache / 'x.npy', np.asarray(local.x))
    with np.load(local.path / 'arrays.npz') as source_arrays:
        arrays = {k: source_arrays[k] for k in source_arrays.files}
    np.savez(cache / 'arrays.npz', **arrays)
    (cache / 'manifest.json').write_text(json.dumps(local.info, indent=2), encoding='utf-8')
    np.save(folder / 'global.npy', np.asarray(global_x))
    np.save(folder / 'feature_mask.npy', np.asarray(store.feature_mask))
    rows = np.linspace(0, len(store.x) - 1, 16, dtype=int)
    x, g, cm, tm = store.transform(rows, norm)
    np.savez(folder / 'reference.npz', rows=rows, norm=np.asarray(norm), x=x, g=g, cm=cm, tm=tm)
    names = ['cache/x.npy', 'cache/arrays.npz', 'cache/manifest.json', 'global.npy', 'feature_mask.npy', 'reference.npz']
    audit = dict(domain=domain, train_units=list(train), source_relative=source.relative_to(WORKSPACE).as_posix(),
                 source_sha256=sha(source), files={name: sha(folder / name) for name in names})
    (folder / 'bundle.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('EXPORTED', domain, store.x.shape, flush=True)


if __name__ == '__main__':
    main()
