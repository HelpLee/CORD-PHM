"""One GPU per (target domain, pretraining source), 15 paired frozen probes."""
import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from portable_inputs import HERE, WORKSPACE, sha, load_store

SUITE = HERE.parent
UPSTREAM = SUITE / 'gradnorm_multidomain_pretraining'
SEEDS = (42, 43, 44, 45, 46)
FRACTIONS = (.1, .2, 1.)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def checkpoint(domain, initialization, package):
    source = UPSTREAM / ('three_domain' if initialization == 'three' else 'single_domain/' + domain) / 'training/encoder.pt'
    status = json.loads((source.parent / 'status.json').read_text())
    if status['state'] != 'complete':
        raise RuntimeError('Upstream has not completed successfully: ' + str(source))
    snapshot = package / 'source_snapshot/training/encoder.pt'
    digest = sha(source)
    if snapshot.is_file():
        if sha(snapshot) != digest:
            raise RuntimeError('Refusing to mix results from different upstream checkpoints')
    else:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
    write(snapshot.parent / 'status.json', status)
    write(package / 'source.json', dict(path=str(source), sha256=digest, selected_epoch=status['best_epoch'],
                                      final_epoch=status['epoch'], domain=domain, initialization=initialization))
    return snapshot


def milling(package, source, verify_only=False):
    import numpy as np
    import torch
    import downstream_interval_val200 as down
    from adapter_encoder import MillingAdapterEncoder
    state = torch.load(source, map_location='cpu', weights_only=True)
    converted = {k: v for k, v in state.items() if not k.startswith(('stems.', 'adapters.'))}
    for key, value in state.items():
        if key.startswith('adapters.milling.'):
            converted[key.replace('adapters.milling.', 'adapters.', 1)] = value
    for target, origin in (('local_norm', 'local.0'), ('local_proj', 'local.1'), ('global_norm', 'global.0'), ('global_proj', 'global.1')):
        for field in ('weight', 'bias'):
            converted[target + '.' + field] = state['stems.milling.' + origin + '.' + field]
    MillingAdapterEncoder(channels=7, dropout=.05).load_state_dict(converted, strict=True)
    converted_path = package / 'converted_encoder.pt'
    torch.save(converted, converted_path)
    store = load_store('milling')
    assert store.x.shape[1:] == (7, 64, 26)

    def interval_split(store):
        trains, vals, counts = [], [], {}
        for unit in down.TRAIN:
            sequences = down.full_sequences(store, (unit,))
            count = max(1, int(np.ceil(len(sequences) * down.LABEL_FRACTION)))
            selected = np.unique(np.rint(np.linspace(0, len(sequences) - 1, count)).astype(np.int64))
            chosen = sequences[selected]
            nv = max(1, int(np.ceil(len(chosen) * .2)))
            vi = np.unique(np.rint(np.linspace(0, len(chosen) - 1, nv)).astype(np.int64))
            ti = np.setdiff1d(np.arange(len(chosen)), vi)
            trains.append(chosen[ti]); vals.append(chosen[vi])
            counts[unit] = dict(total=len(sequences), selected=len(chosen), train=len(ti), validation=len(vi))
        return np.concatenate(trains), np.concatenate(vals), counts

    down.prepare_store = lambda _: store
    down.source_split = interval_split
    down.SOURCE = converted_path
    down.OUT = package / 'runtime'
    down.MAX_EPOCHS = 200
    down.MAX_UPDATES = 0
    down.FROZEN_EPOCHS = 0
    down.PATIENCE = 15
    down.VAL_INTERVAL = 2
    down.TRAIN_VALIDATION_FRACTION = .2
    down.TEST_VALIDATION_FRACTION = 0
    down.NESTED_LABELS = False
    down.ARMS = ('frozen_probe',)
    write(package / 'protocol.json', dict(seeds=SEEDS, fractions=FRACTIONS, arm='frozen_probe', max_epochs=200,
          patience=15, validation_interval=2, train=['C1', 'C4'], test='C6', channels=7, precision='BF16',
          source_sha256=sha(source), validation='20% interval within selected source labels'))
    if verify_only:
        for fraction in FRACTIONS:
            down.LABEL_FRACTION = fraction
            tr, va, counts = interval_split(store)
            assert not set(tr[:, -1]) & set(va[:, -1])
        write(package / 'verification.json', dict(passed=True, strict_load=True))
        return
    rows = []
    for seed in SEEDS:
        for fraction in FRACTIONS:
            down.SEED, down.LABEL_FRACTION = seed, fraction
            down.NAME = 'seed%d_p%d' % (seed, round(fraction * 100))
            summary = down.OUT / down.NAME / 'development_summary.json'
            write(package / 'status.json', dict(state='running', seed=seed, fraction=fraction, completed=len(rows), pid=os.getpid()))
            if not summary.is_file():
                down.main()
            row = dict(json.loads(summary.read_text())['rows'][0], seed=seed, fraction=fraction)
            assert row['initial_encoder_hash'] == row['final_encoder_hash']
            rows.append(row)
            write(package / 'results.json', dict(complete=False, rows=rows))
    write(package / 'results.json', dict(complete=True, rows=rows))
    write(package / 'status.json', dict(state='complete', runs=len(rows)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=('bearing', 'battery', 'milling'))
    parser.add_argument('--initialization', required=True, choices=('single', 'three'))
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verify-checkpoint', type=Path, help='Verification only; never used for a training run')
    args = parser.parse_args()
    if args.verify_checkpoint and not args.verify_only:
        parser.error('--verify-checkpoint requires --verify-only')
    package = HERE / ('_verification' if args.verify_only else 'runs') / args.domain / args.initialization
    package.mkdir(parents=True, exist_ok=True)
    try:
        source = args.verify_checkpoint.resolve() if args.verify_checkpoint else checkpoint(args.domain, args.initialization, package)
        shared = SUITE / ('multidomain_original_cutting_tool_sources/downstream_milling_10pct/code' if args.domain == 'milling'
                         else 'shared_domain_components/' + args.domain + '_code')
        sys.path.insert(0, str(shared))
        sys.path.insert(0, str(HERE / (args.domain + '_code')))
        import torch
        torch.set_num_threads(4)
        if not args.verify_only:
            assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'A100/BF16 GPU required'
            write(package / 'device.json', dict(name=torch.cuda.get_device_name(0), job_id=os.environ.get('SLURM_JOB_ID'), precision='BF16'))
        if args.domain == 'milling':
            milling(package, source, args.verify_only)
        else:
            module = __import__('bearing_adapter_study' if args.domain == 'bearing' else 'adapter_downstream')
            write(package / 'config.json', dict(checkpoint=str(source), upstream=None, arms=['frozen_probe']))
            sys.argv = [sys.argv[0]] + (['--verify-only'] if args.verify_only else [])
            module.main(package)
    except BaseException as error:
        write(package / 'status.json', dict(state='failed', error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
