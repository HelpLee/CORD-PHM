"""Seed42 milling transfer test using the three-domain joint upstream model."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import data as data_module
import downstream_interval_val200 as down
from adapter_encoder import MillingAdapterEncoder as Attention65Encoder
import global_local_data as gl
import feature_observations as fo
from run_milling_global_local_multiscale12 import prepare_store


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = next(p for p in Path(__file__).resolve().parents
                 if (p / 'outputs/reproduction_final').is_dir())
UPSTREAM_PACKAGE = PACKAGE.parent / '10_upstream_three_domain_adapter'
JOINT = UPSTREAM_PACKAGE / 'training/encoder.pt'
OLD_RUNTIME = PACKAGE.parent / '01_shared_dependencies/milling_runtime'
RUNTIME = PACKAGE / 'runtime'
CONVERTED = PACKAGE / 'milling_encoder_from_joint.pt'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def convert_checkpoint():
    source = torch.load(JOINT, map_location='cpu', weights_only=True)
    converted = {key: value for key, value in source.items()
                 if not key.startswith(('stems.', 'adapters.'))}
    for key, value in source.items():
        if key.startswith('adapters.milling.'):
            converted[key.replace('adapters.milling.', 'adapters.', 1)] = value
    mapping = {
        'local_norm.weight': 'stems.milling.local.0.weight',
        'local_norm.bias': 'stems.milling.local.0.bias',
        'local_proj.weight': 'stems.milling.local.1.weight',
        'local_proj.bias': 'stems.milling.local.1.bias',
        'global_norm.weight': 'stems.milling.global.0.weight',
        'global_norm.bias': 'stems.milling.global.0.bias',
        'global_proj.weight': 'stems.milling.global.1.weight',
        'global_proj.bias': 'stems.milling.global.1.bias',
    }
    for target, origin in mapping.items():
        converted[target] = source[origin]
    model = Attention65Encoder(channels=3, dropout=.05)
    model.load_state_dict(converted, strict=True)
    torch.save(converted, CONVERTED)
    write(PACKAGE / 'checkpoint_provenance.json', {
        'joint_checkpoint': str(JOINT),
        'joint_sha256': file_hash(JOINT),
        'converted_checkpoint': str(CONVERTED),
        'converted_sha256': file_hash(CONVERTED),
        'mapping': mapping,
        'strict_load': True,
        'tensor_count': len(converted),
        'adapter_mapping': 'adapters.milling.* -> adapters.*; discard bearing/battery adapters',
    })
    return source, converted


def load_store():
    cache = OLD_RUNTIME / 'raw_cache' / data_module.DOWN['milling']
    manifest = json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    source = Path(manifest['signature']['path'])
    assert source.resolve() == (data_module.DATA/'milling/phm2010_milling_downstream_health_tokens.npz').resolve()
    assert source.stat().st_size == manifest['signature']['bytes']
    assert source.stat().st_mtime_ns == manifest['signature']['mtime_ns']
    gl.OUT = fo.OUT = OLD_RUNTIME
    store = prepare_store(cache)
    write(PACKAGE/'data_provenance.json', dict(npz=str(source),npz_sha256=file_hash(source),
        raw_cache=str(cache),feature_and_global_cache=str(OLD_RUNTIME),
        channels='force XYZ',train=['C1','C4'],test='C6'))
    return store


def interval_split(store):
    trains, validations, counts = [], [], {}
    for unit in down.TRAIN:
        sequences = down.full_sequences(store, (unit,))
        count = max(1, int(np.ceil(len(sequences) * down.LABEL_FRACTION)))
        selected = np.unique(np.rint(np.linspace(0, len(sequences) - 1, count)).astype(np.int64))
        chosen = sequences[selected]
        validation_count = max(1, int(np.ceil(len(chosen) * .2)))
        validation_index = np.unique(np.rint(
            np.linspace(0, len(chosen) - 1, validation_count)).astype(np.int64))
        train_index = np.setdiff1d(np.arange(len(chosen)), validation_index)
        assert len(train_index) > 0 and len(validation_index) > 0
        trains.append(chosen[train_index])
        validations.append(chosen[validation_index])
        counts[unit] = {
            'total': len(sequences), 'selected': len(chosen),
            'train': len(train_index), 'validation': len(validation_index),
            'train_endpoints': chosen[train_index, -1].tolist(),
            'validation_endpoints': chosen[validation_index, -1].tolist(),
        }
    train = np.concatenate(trains)
    validation = np.concatenate(validations)
    assert not set(train[:, -1]) & set(validation[:, -1])
    return train, validation, counts


def main():
    status=json.loads((UPSTREAM_PACKAGE/'training/status.json').read_text())
    assert status['state']=='complete', 'Upstream must finish before downstream training.'
    assert JOINT.exists()
    convert_checkpoint()
    # Reuse only the immutable raw-data cache from the preserved successful run.
    store = load_store()
    down.prepare_store = lambda _path: store
    down.OUT = RUNTIME
    down.SOURCE = CONVERTED
    down.SEED = 42
    down.MAX_UPDATES = 0
    down.FROZEN_EPOCHS = 0
    down.FROZEN_EPOCHS = 0
    down.MAX_EPOCHS = 200
    down.PATIENCE = 15
    down.VAL_INTERVAL = 2
    down.TRAIN_VALIDATION_FRACTION = .2
    down.TEST_VALIDATION_FRACTION = 0
    down.NESTED_LABELS = False
    if os.environ.get('HEALTHTOKEN_B2_ARMS', '0') != '1':
        down.ARMS = ('scratch', 'frozen_probe', 'partial_finetune', 'full_finetune')
    down.source_split = interval_split

    write(PACKAGE / 'protocol.json', {
        'seeds': [42,43,44,45,46], 'fractions': [.1, .2, 1.0],
        'arms': list(down.ARMS), 'train': list(down.TRAIN), 'test': down.HELD,
        'max_epochs': 200, 'validation_interval': 2, 'patience': 15,
        'validation': '20% interval sample within independently selected labels',
        'nested_labels': False, 'dropout': .05, 'batch': 32, 'microbatch': 8,
        'loss': 'SmoothL1 beta=0.05', 'weight_decay': 1e-4,
        'scratch_lr': 1e-3, 'pretrained_lr': 1e-4, 'new_layers_lr': 1e-3,
        'frozen_probe': 'freeze complete pretrained encoder; train adapter/TCN/GRU/RUL head',
        'partial_finetune': 'from epoch 1 train Transformer block 2, final norm, and Adapter 2; other encoder parameters frozen',
        'full_finetune': 'train entire encoder from epoch 1',
    })

    rows = []
    if os.environ.get('HEALTHTOKEN_RESUME_RESULTS', '0') == '1':
        previous = PACKAGE / 'results.json'
        if previous.exists():
            payload = json.loads(previous.read_text(encoding='utf-8'))
            rows = list(payload.get('rows', []))
    for seed, fraction in ((s,f) for s in (42,43,44,45,46) for f in (.1,.2,1.0)):
        down.SEED = seed
        down.LABEL_FRACTION = fraction
        down.NAME = f'adapter_transfer_seed{seed}_{int(fraction * 100)}pct'
        write(PACKAGE / 'status.json', {
            'state': 'running', 'seed': seed, 'fraction': fraction,
            'pid': os.getpid(), 'completed_rows': len(rows),
        })
        existing = [row for row in rows if row.get('seed') == seed and
                    row.get('fraction') == fraction and
                    row.get('arm') in down.ARMS]
        if len(existing) == len(down.ARMS):
            # Preserve completed summaries after a killed process; the
            # generated runtime directory may safely be absent.
            continue
        rows = [row for row in rows if not (row.get('seed') == seed and
                                            row.get('fraction') == fraction)]
        down.main()
        result = json.loads((RUNTIME / down.NAME / 'development_summary.json').read_text())
        for row in result['rows']:
            row['fraction'] = fraction
            rows.append(row)
        write(PACKAGE / 'results.json', {'complete': False, 'rows': rows})

    summary=[]
    for fraction in (.1,.2,1.0):
        for arm in down.ARMS:
            group=[r for r in rows if r['fraction']==fraction and r['arm']==arm]
            assert len(group)==5
            summary.append(dict(fraction=fraction,arm=arm,n=5,metrics={
                m:dict(mean=float(np.mean([r['metrics'][m] for r in group])),
                       std=float(np.std([r['metrics'][m] for r in group],ddof=1)))
                for m in ('rmse','mae','r2','bias')}))
    write(PACKAGE / 'results.json', {'complete': True, 'rows': rows,'summary':summary})
    write(PACKAGE / 'status.json', {'state': 'completed', 'runs': len(rows)})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(PACKAGE / 'status.json', {'state': 'failed', 'error': repr(error)})
        raise
