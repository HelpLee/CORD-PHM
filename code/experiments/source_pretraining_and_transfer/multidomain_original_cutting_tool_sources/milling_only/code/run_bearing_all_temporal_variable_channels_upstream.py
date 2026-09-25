"""Variable-channel bearing pretraining over every usable canonical source.

The optimization protocol intentionally matches run_bearing_mask_dynamics_upstream:
datasets are visited round-robin, units and rows are sampled uniformly, and the
objective is masked reconstruction plus 0.2 * six-to-one temporal prediction.
Only the source inventory and channel interface differ.  A temporal window is
never allowed to cross a physical unit/condition/recording boundary.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import DATA, OUT, Store, read_selected
from feature_observations import observed_store
from global_local_data import GlobalLocalStore, file_sha
from compression_model import Attention65MaskedModel as GlobalLocalMaskedModel
from io_utils import resilient_write as write
from helpers import hh, st


NAME = os.environ.get('BEARING_DYNAMICS_NAME', 'attention65_upstream_seed42')
SEED = 42
MAX_EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 1e-4
UPDATES = 20
BATCH = 32
MICRO = 8
HISTORY = 6
DYN_WEIGHT = 0.2
MAX_CHANNELS = 8
INDEPENDENT_CHANNELS = os.environ.get('BEARING_INDEPENDENT_CHANNELS', '0') == '1'

# HUST and MFPT do not contain a genuine seven-observation sequence.  Every
# listed source below has at least one auditable, ordered 7-snapshot trajectory.
SOURCES = ('cwru', 'femto', 'ferrara', 'ims', 'kaist', 'seu', 'unsw')


class TemporalPredictor(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True)
        self.projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, history):
        _, hidden = self.gru(history)
        return self.projection(hidden[-1])


def natural_key(value):
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r'(\d+)', str(value)))


def source_layout(stem, z):
    """Return selected source rows, physical units, sequence ids and order."""
    n = len(z['x_health'])
    source_rows = np.arange(n, dtype=np.int64)
    units = z['sample_unit_id'].astype(str)
    conditions = z['sample_condition_id'].astype(str)
    segments = z['sample_segment_id'].astype(str)
    files = z['sample_filename'].astype(str)
    chunks = np.asarray(z['sample_chunk_id'], dtype=np.int64)
    snapshots = np.asarray(z['sample_snapshot_index'], dtype=np.int64)

    if stem == 'unsw':
        source_rows = source_rows[conditions == '6Hz']
    units = units[source_rows]
    conditions = conditions[source_rows]
    segments = segments[source_rows]
    files = files[source_rows]
    chunks = chunks[source_rows]
    snapshots = snapshots[source_rows]

    # Static/fault corpora may only form time sequences inside one uninterrupted
    # acquisition.  Lifecycle corpora may continue across their ordered files.
    if stem in ('cwru', 'seu'):
        physical = np.asarray([f'{u}::{c}' for u,c in zip(units,conditions)], dtype=str)
        sequence = segments.copy()
        keys = [(int(chunk),) for chunk in chunks]
    else:
        physical = np.asarray([f'{u}::{c}' for u, c in zip(units, conditions)], dtype=str)
        sequence = physical.copy()
        if stem == 'femto':
            keys = [(int(s), int(c)) for s, c in zip(snapshots, chunks)]
        else:
            keys = [(natural_key(f), int(c)) for f, c in zip(files, chunks)]
        if stem == 'kaist':
            sequence = segments.copy()

    keep = []
    order = np.zeros(len(source_rows), dtype=np.float64)
    for seq in sorted(set(sequence)):
        ids = np.flatnonzero(sequence == seq)
        ids = np.asarray(sorted(ids, key=lambda i: keys[i]), dtype=np.int64)
        # Reject duplicate/non-increasing acquisition keys rather than silently
        # inventing a temporal link.
        unique = [ids[0]] if len(ids) else []
        for idx in ids[1:]:
            assert keys[idx] > keys[unique[-1]], (stem, seq, 'duplicate chronology')
            unique.append(idx)
        if len(unique) >= HISTORY + 1:
            for rank, idx in enumerate(unique):
                order[idx] = rank
            keep.extend(unique)
    keep = np.asarray(sorted(keep), dtype=np.int64)
    return source_rows[keep], physical[keep], sequence[keep], order[keep]


def build_temporal_cache(stem):
    source = (DATA / 'bearing' / f'{stem}_bearing_health_tokens.npz').resolve()
    cache = OUT / 'all_bearing_temporal_cache_v4' / f'{stem}_bearing'
    signature = dict(path=str(source), bytes=source.stat().st_size,
                     mtime_ns=source.stat().st_mtime_ns, version=4, row_scoped_globals=True,
                     selection='all rows belonging to a genuine ordered sequence of length >=7',
                     no_cross_boundary=True)
    if (cache / 'manifest.json').exists():
        assert json.loads((cache / 'manifest.json').read_text())['signature'] == signature
        return cache
    with np.load(source, allow_pickle=True) as z:
        rows, units, sequences, order = source_layout(stem, z)
        tm = np.asarray(z['token_mask'][rows], bool)
        cm = np.asarray(z['c_mask'][rows], bool)
        metadata = json.loads(str(z['meta_json'].item()))
        assert metadata['preprocessing_protocol'] == 'three_domain_strict_raw_v2'
        assert metadata['x_health_scaled'] is False
        features = z['feature_names'].astype(str).tolist()
    assert len(rows) and cm.shape[1] <= MAX_CHANNELS
    x = read_selected(source, rows)
    valid = tm & cm[..., None]
    assert np.isfinite(x[valid]).all()
    x[~valid] = 0
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / 'x.npy', x)
    np.savez(cache / 'arrays.npz', tm=tm, cm=cm,
             y=np.zeros(len(rows), np.float32), units=units, order=order,
             source_rows=rows, sequence_key=sequences)
    write(cache / 'manifest.json', dict(
        signature=signature, domain='bearing', downstream=False,
        shape=list(x.shape), counts={u: int(np.sum(units == u)) for u in sorted(set(units))},
        temporal_sequences=int(len(set(sequences))), features=features, metadata=metadata))
    return cache


def temporal_windows(store, units):
    with np.load(store.path / 'arrays.npz') as z:
        sequence_key = z['sequence_key'].astype(str)
    allowed = np.isin(store.units.astype(str), np.asarray(units, dtype=str))
    result = []
    for key in sorted(set(sequence_key[allowed])):
        ids = np.flatnonzero(allowed & (sequence_key == key))
        ids = ids[np.argsort(store.order[ids], kind='stable')]
        result.extend(ids[i:i + HISTORY + 1] for i in range(len(ids) - HISTORY))
    return np.asarray(result, dtype=np.int64)


def bt(store, ids, norm):
    return tuple(torch.as_tensor(v, device='cuda') for v in store.transform(ids, norm))


def mask_loss(model, store, ids, norm, fixed=None):
    x, g, cm, tm = bt(store, ids, norm)
    pred, mask, _ = model(x, g, cm, tm, fixed)
    observed = torch.as_tensor(np.asarray(store.feature_mask[ids]), device='cuda')
    valid = mask[..., None] & observed
    if INDEPENDENT_CHANNELS:
        counts=valid.sum((2,3))
        losses=((pred.float()-x.float()).square()*valid).sum((2,3))/counts.clamp_min(1)
        eligible=counts>0
        per_snapshot=(losses*eligible).sum(1)/eligible.sum(1).clamp_min(1)
        return per_snapshot.mean()
    return ((pred.float() - x.float()).square() * valid).sum() / valid.sum().clamp_min(1)


def dynamics_loss(model, predictor, store, sequence, norm):
    b, width = sequence.shape
    assert width == HISTORY + 1
    rows = sequence.reshape(-1)
    x, g, cm, tm = bt(store, rows, norm)
    state = model.encoder(x, g, cm, tm)['snapshot'].reshape(b, width, -1)
    predicted = predictor(state[:, :-1])
    return (predicted.float() - state[:, -1].detach().float()).square().mean()


@torch.no_grad()
def validate(model, predictor, pool):
    model.eval(); predictor.eval(); by_dataset = {}
    for name, item in pool.items():
        store, _, val_units, norm, _, val_windows = item
        groups = store.eligible(val_units)
        if not groups or not len(val_windows):
            continue
        ids = np.concatenate(list(groups.values()))
        ids = ids[np.unique(np.linspace(0, len(ids)-1, min(64, len(ids)), dtype=int))]
        chosen = val_windows[np.unique(np.linspace(0, len(val_windows)-1, min(64, len(val_windows)), dtype=int))]
        mv = [(float(mask_loss(model, store, ids[s:s+MICRO], norm, 9000+s)), len(ids[s:s+MICRO]))
              for s in range(0, len(ids), MICRO)]
        dv = [(float(dynamics_loss(model, predictor, store, chosen[s:s+MICRO], norm)), len(chosen[s:s+MICRO]))
              for s in range(0, len(chosen), MICRO)]
        mask = sum(v*n for v, n in mv) / sum(n for _, n in mv)
        dyn = sum(v*n for v, n in dv) / sum(n for _, n in dv)
        by_dataset[name] = dict(mask=mask, dynamics=dyn, total=mask + DYN_WEIGHT*dyn)
    return by_dataset


def main():
    torch.set_num_threads(4)
    output = OUT / NAME
    assert not output.exists(), 'Use a fresh upstream output directory'
    (output / 'upstream').mkdir(parents=True)
    write(output/'submission.json', dict(
        state='preparing', source_root=str(DATA/'bearing'), sources=list(SOURCES),
        temporal_rule='within continuous recording or real ordered lifecycle measurements',
        channel_rule='native C per dataset batch; attention-pool channel locals and globals before 65-token Transformer',
        independent_channels=INDEPENDENT_CHANNELS,
        downstream=dict(test='Bearing2_1',fractions=[0.1,0.2],seeds=[42,43,44,45,46],updates=500,nested=True,dropout=.05,deterministic=True),
        code_sha256={p.name:file_sha(p) for p in (
            Path(__file__),Path(__file__).with_name('global_local_model.py'),
            Path(__file__).with_name('global_local_data.py'),
            Path(__file__).with_name('compression_model.py'),
            Path(__file__).with_name('run_bearing_delta6_no_b24_fewshot.py'))}))
    pool = {}; audit = []
    for stem in SOURCES:
        local = observed_store(Store(build_temporal_cache(stem)))
        store = GlobalLocalStore(local, channels=local.x.shape[1])
        train_units, val_units = store.split()
        norm = store.normalize(train_units)
        train_windows = temporal_windows(store, train_units)
        val_windows = temporal_windows(store, val_units)
        assert len(train_windows)
        pool[stem] = (store, train_units, val_units, norm, train_windows, val_windows)
        audit.append(dict(dataset=stem, native_channels=int(local.x.shape[1]),
                          batch_channels=int(local.x.shape[1]),
                          train=list(train_units), val=list(val_units),
                          train_windows=int(len(train_windows)), val_windows=int(len(val_windows)),
                          local_source=store.info['signature'],
                          global_cache=file_sha(OUT/'global_feature_cache'/'row_scoped_v4'/store.name/'manifest.json'),
                          calibration=store.calibration,
                          center_scale=[v.tolist() for v in norm]))
    write(output/'upstream_splits_scalers.json', audit)

    torch.manual_seed(SEED)
    if INDEPENDENT_CHANNELS:
        from channel_independent_model import ChannelIndependentMaskedModel
        model = ChannelIndependentMaskedModel().cuda()
    else:
        model = GlobalLocalMaskedModel(channels=MAX_CHANNELS, variable_channels=True).cuda()
    predictor = TemporalPredictor().cuda()
    initial = hh(st(model.encoder))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(predictor.parameters()),
                                  lr=1e-4, weight_decay=1e-4)
    names = list(pool); best = float('inf'); history = []; stale = 0; started = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); predictor.train(); epoch_rows = []
        for update in range(UPDATES):
            rng = np.random.default_rng(SEED + epoch*10000 + update*10)
            name = names[((epoch-1)*UPDATES + update) % len(names)]
            store, train_units, _, norm, train_windows, _ = pool[name]
            groups = store.eligible(train_units)
            ids = np.asarray([rng.choice(groups[str(rng.choice(list(groups)))]) for _ in range(BATCH)])
            chosen = train_windows[rng.integers(0, len(train_windows), size=BATCH)]
            torch.manual_seed(SEED + epoch*10000 + update*10)
            optimizer.zero_grad(set_to_none=True); total_mask = total_dyn = 0.0
            for start in range(0, BATCH, MICRO):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    lm = mask_loss(model, store, ids[start:start+MICRO], norm)
                    ld = dynamics_loss(model, predictor, store, chosen[start:start+MICRO], norm)
                    loss = lm + DYN_WEIGHT*ld
                assert torch.isfinite(loss)
                (loss/(BATCH/MICRO)).backward()
                total_mask += float(lm)/(BATCH/MICRO); total_dyn += float(ld)/(BATCH/MICRO)
            gradient = torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(predictor.parameters()), 5)
            assert torch.isfinite(gradient); optimizer.step()
            epoch_rows.append((total_mask, total_dyn))
        scores = validate(model, predictor, pool)
        monitor = float(np.mean([v['total'] for v in scores.values()]))
        train_mask = float(np.mean([v[0] for v in epoch_rows]))
        train_dyn = float(np.mean([v[1] for v in epoch_rows]))
        history.append(dict(epoch=epoch, train_mask=train_mask, train_dynamics=train_dyn,
                            val_total=monitor, by_dataset=scores, seconds=time.time()-started))
        if monitor < best-MIN_DELTA:
            best=monitor; best_epoch=epoch; best_encoder=st(model.encoder); best_predictor=st(predictor); stale=0
            torch.save(best_encoder, output/'upstream/encoder.pt')
            torch.save(dict(model=st(model),predictor=best_predictor,epoch=epoch),output/'upstream/best_pretraining.pt')
        else:
            stale += 1
        write(output/'upstream/history.json', history)
        if epoch % 5 == 0:
            print('BEARING_ALL_VARIABLE', epoch, f'mask={train_mask:.5f}', f'dyn={train_dyn:.5f}',
                  f'val={monitor:.5f}', f'best={best:.5f}@{best_epoch}', f'stale={stale}/{PATIENCE}', flush=True)
        if stale >= PATIENCE:
            stop_reason='early_stopping'; break
    else:
        stop_reason='max_epochs'
    torch.save(best_encoder, output/'upstream/encoder.pt')
    torch.save(best_predictor, output/'upstream/temporal_predictor.pt')
    write(output/'upstream/summary.json', dict(
        best_epoch=best_epoch, val_total=best, encoder_hash=hh(best_encoder),
        initial_encoder_hash=initial, stopped_epoch=epoch, stop_reason=stop_reason,
        updates=epoch*UPDATES, global_batch=BATCH, history_snapshots=HISTORY,
        dynamics_weight=DYN_WEIGHT, max_channels=MAX_CHANNELS, variable_channels=True,
        independent_channels=INDEPENDENT_CHANNELS,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, seconds=time.time()-started))
    write(output/'protocol.json', dict(
        name=NAME, seed=SEED, source_root=str(DATA/'bearing'), datasets=list(SOURCES),
        excluded={'hust':'no genuine contiguous 7-snapshot sequence',
                  'paderborn':'two chunks per recording; separate acquisitions cannot be concatenated',
                  'mfpt':'no genuine contiguous 7-snapshot sequence',
                  'xjtu_downstream':'strict downstream-only target'},
        sampling='original dataset round-robin; uniform unit/row and uniform temporal-window sampling; no R2F weighting',
        temporal_boundary='never cross unit, condition, or uninterrupted-recording sequence',
        channel_interface='native C retained in input; attention compressed to 65 tokens before Transformer; channel ID only in reconstruction decoder',
        global_token=('one Global per channel' if INDEPENDENT_CHANNELS else
                      'shared per-channel LN(26)->Linear(26,96), attention pooling over valid channels'),
        encoder=('shared single-channel 65-token Encoder + zero-initialized channel attention pooling'
                 if INDEPENDENT_CHANNELS else 'attention-compressed 64 Local + 1 Global; D96, 2-layer, 4-head Transformer'),
        mask_reduction=('feature mean per channel then channel mean per snapshot'
                        if INDEPENDENT_CHANNELS else 'all observed masked features mean'),
        objective='L_mask + 0.2 * L_inter_snapshot_temporal_dynamics',
        selection='mean source validation mask+dynamics objective; patience20, max200'))
    print('BEARING_ALL_VARIABLE_COMPLETE', json.dumps(dict(best_epoch=best_epoch,val_total=best)), flush=True)


if __name__ == '__main__':
    main()
