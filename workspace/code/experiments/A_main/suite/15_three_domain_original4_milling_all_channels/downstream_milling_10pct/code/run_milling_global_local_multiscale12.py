"""Milling Global+Local masked pretraining and multi-scale PHM2010 LOCO RUL."""
import hashlib
import json
import shutil
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import DOWN, OUT, Store
from feature_observations import observed_store
from global_local_data import GlobalLocalStore, file_sha
from global_local_model import GlobalLocalEncoder, GlobalLocalMaskedModel
from io_utils import resilient_write as write
from run_battery_global_local_delta6 import state, state_hash
from sensor_selection import select_sensors


NAME = os.environ.get('MILLING_EXPERIMENT_NAME', 'milling_global_local_multiscale12_seed42_v2')
SEED = 42
LENGTH = 12
EPOCHS = 100
# Channel-ablation condition: retain every PHM2010 sensor stream rather than
# the canonical force-only Fx/Fy/Fz subset.
CHANNELS = 7
# Stems refer to explicit lifecycle caches, rather than reconstructing names
# from a two-item historical default.  This keeps the lightweight model
# unchanged while allowing auditable auxiliary upstream sources.
UPSTREAM = tuple(item.strip() for item in os.environ.get(
    'MILLING_UPSTREAM_STEMS',
    'luh_milling,matwi_milling,nonastreda_milling_snapshot,qit_cemc_milling_snapshot'
).split(',') if item.strip())
RUN_DOWNSTREAM = os.environ.get('RUN_MILLING_DOWNSTREAM', '0') == '1'
# Auxiliary corpora have no observed end-of-life and therefore do not belong in
# lifecycle_cache.  raw_cache retains a deterministic 24-snapshot/unit sample
# without using their wear metadata.
UPSTREAM_CACHE = os.environ.get('MILLING_UPSTREAM_CACHE', 'raw_cache')
MAX_UPSTREAM_EPOCHS = int(os.environ.get('MILLING_MAX_UPSTREAM_EPOCHS', '200'))
EARLY_STOP_PATIENCE = int(os.environ.get('MILLING_EARLY_STOP_PATIENCE', '20'))
EARLY_STOP_MIN_DELTA = float(os.environ.get('MILLING_EARLY_STOP_MIN_DELTA', '1e-4'))
UNITS = ('C1', 'C4', 'C6')


class MillingMultiScaleRULModel(nn.Module):
    """Current cut state plus one-cut and four-cut learned degradation changes."""
    def __init__(self):
        super().__init__()
        self.encoder = GlobalLocalEncoder(channels=CHANNELS)
        self.input_projection = nn.Sequential(nn.LayerNorm(3 * 96), nn.Linear(3 * 96, 96), nn.GELU())
        self.gru = nn.GRU(96, 96, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, x, global_x, channel_mask, token_mask, history_mask):
        batch, steps, channels, tokens, features = x.shape
        states = self.encoder(x.reshape(batch * steps, channels, tokens, features),
                              global_x.reshape(batch * steps, channels, features),
                              channel_mask.reshape(batch * steps, channels),
                              token_mask.reshape(batch * steps, channels, tokens))['snapshot'].reshape(batch, steps, -1)
        lengths = history_mask.sum(1).long()
        compact = states.new_zeros(states.shape)
        for index in range(batch):
            compact[index, :lengths[index]] = states[index, history_mask[index]]
        delta_one = compact.new_zeros(compact.shape)
        delta_one[:, 1:] = compact[:, 1:] - compact[:, :-1]
        delta_four = compact.new_zeros(compact.shape)
        delta_four[:, 4:] = compact[:, 4:] - compact[:, :-4]
        trajectory = self.input_projection(torch.cat((compact, delta_one, delta_four), -1))
        valid = torch.arange(steps, device=trajectory.device)[None, :] < lengths[:, None]
        trajectory = trajectory * valid[..., None]
        packed = nn.utils.rnn.pack_padded_sequence(trajectory, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(-1)


def prepare_store(path):
    return GlobalLocalStore(select_sensors(observed_store(Store(path)), 'all'), calibrate=False, channels=CHANNELS)


def batch(store, rows, norm):
    return tuple(torch.as_tensor(value, device='cuda') for value in store.transform(rows, norm))


def masked_loss(model, store, rows, norm, fixed=None):
    x, global_x, channel_mask, token_mask = batch(store, rows, norm)
    prediction, masked, _ = model(x, global_x, channel_mask, token_mask, fixed)
    observed = torch.as_tensor(store.feature_mask[rows], device='cuda')
    valid = masked[..., None] & observed
    return ((prediction.float() - x.float()).square() * valid).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def validate_masked(model, pools):
    model.eval()
    scores = {}
    for name, (store, _, validation_units, norm) in pools.items():
        # QIT-CEMC is a single chronologically ordered tool sequence.  It may
        # contribute unlabeled SSL updates, but cannot supply a unit-disjoint
        # validation set and therefore must not influence checkpoint choice.
        if not validation_units:
            continue
        groups = store.eligible(validation_units)
        rows = np.concatenate(list(groups.values()))
        rows = rows[np.unique(np.linspace(0, len(rows) - 1, min(64, len(rows)), dtype=int))]
        values = []
        for start in range(0, len(rows), 8):
            current = rows[start:start + 8]
            values.append((float(masked_loss(model, store, current, norm, 9000 + start)), len(current)))
        scores[name] = sum(value * n for value, n in values) / sum(n for _, n in values)
    return scores


def run_upstream(output):
    pools = {}
    audit = []
    for stem in UPSTREAM:
        store = prepare_store(OUT / UPSTREAM_CACHE / stem)
        train_units, validation_units = store.split()
        norm = store.normalize(train_units)
        pools[store.name] = (store, train_units, validation_units, norm)
        cache_name = store.name + '__force_' + '_'.join(map(str, store.local.sensor_selection['indices']))
        audit.append(dict(dataset=store.name, sensors=store.local.sensor_selection, train=train_units, validation=validation_units,
                          checkpoint_monitor='unit-disjoint validation' if validation_units else 'train-only; one source unit',
                          eligible_train=sum(map(len, store.eligible(train_units).values())),
                          eligible_validation=sum(map(len, store.eligible(validation_units).values())),
                          global_cache_sha256=file_sha(OUT / 'global_feature_cache' / cache_name / 'manifest.json'),
                          center_scale=[value.tolist() for value in norm]))
    write(output / 'upstream_splits_scalers.json', audit)
    torch.manual_seed(SEED)
    model = GlobalLocalMaskedModel(channels=CHANNELS).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    names = list(pools)
    history = []
    best_loss = float('inf')
    stale_epochs = 0
    started = time.time()
    for epoch in range(1, MAX_UPSTREAM_EPOCHS + 1):
        model.train()
        losses = []
        for update in range(20):
            rng = np.random.default_rng(SEED + epoch * 10000 + update * 10)
            name = names[((epoch - 1) * 20 + update) % len(names)]
            store, train_units, _, norm = pools[name]
            groups = store.eligible(train_units)
            rows = np.asarray([rng.choice(groups[str(rng.choice(list(groups)))]) for _ in range(32)])
            torch.manual_seed(SEED + epoch * 10000 + update * 10)
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for start in range(0, 32, 8):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = masked_loss(model, store, rows[start:start + 8], norm)
                assert torch.isfinite(loss)
                (loss / 4).backward()
                total += float(loss) / 4
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            assert torch.isfinite(gradient)
            optimizer.step()
            losses.append(total)
        by_dataset = validate_masked(model, pools)
        monitor = float(np.mean(list(by_dataset.values())))
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_loss=monitor,
                            by_dataset=by_dataset, seconds=time.time() - started))
        if monitor < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = monitor
            best_epoch = epoch
            best_encoder = state(model.encoder)
            stale_epochs = 0
        else:
            stale_epochs += 1
        write(output / 'upstream/history.json', history)
        if epoch % 8 == 0:
            print('MILLING_GLOBAL_LOCAL_UP', epoch, round(monitor, 5),
                  f'best={best_loss:.5f}', f'stale={stale_epochs}/{EARLY_STOP_PATIENCE}', flush=True)
        if stale_epochs >= EARLY_STOP_PATIENCE:
            stop_reason = 'early_stopping'
            break
    else:
        stop_reason = 'max_epochs'
    (output / 'upstream').mkdir(exist_ok=True)
    torch.save(best_encoder, output / 'upstream/encoder.pt')
    write(output / 'upstream/summary.json', dict(best_epoch=best_epoch, validation_loss=best_loss,
          encoder_hash=state_hash(best_encoder), updates=epoch * 20, global_batch=32,
          max_epochs=MAX_UPSTREAM_EPOCHS, patience=EARLY_STOP_PATIENCE,
          min_delta=EARLY_STOP_MIN_DELTA, stopped_epoch=epoch, stop_reason=stop_reason,
          seconds=time.time() - started))
    del model, optimizer
    torch.cuda.empty_cache()
    return best_encoder


@torch.no_grad()
def score(model, gpu, sequences, history_mask, target_by_row):
    model.eval()
    predictions = []
    for start in range(0, len(sequences), 64):
        rows = torch.as_tensor(sequences[start:start + 64], device='cuda')
        mask = torch.as_tensor(history_mask[start:start + 64], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(*(value[rows] for value in gpu), mask)
        predictions.extend(prediction.float().cpu().tolist())
    predictions = np.asarray(predictions)
    targets = target_by_row[sequences[:, -1]].astype(float)
    error = predictions - targets
    denominator = np.sum((targets - targets.mean()) ** 2)
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))),
                r2=float(1 - np.sum(error ** 2) / denominator), n=len(error)), predictions, targets


def run_downstream(output, pretrained):
    store = prepare_store(OUT / 'raw_cache' / DOWN['milling'])
    units = list(UNITS)
    assert sorted(store.groups) == sorted(units)
    rows = []
    for held in units:
        train_units = [unit for unit in units if unit != held]
        norm = store.normalize(train_units)
        train_sequences, train_mask = store.sequences_with_mask(train_units, LENGTH)
        test_sequences, test_mask = store.sequences_with_mask([held], LENGTH)
        allowed = store.rows(units)
        gpu = []
        for value in store.transform(allowed, norm):
            tensor = torch.zeros((len(store.x), *value.shape[1:]), dtype=torch.from_numpy(value).dtype, device='cuda')
            tensor[torch.as_tensor(allowed, device='cuda')] = torch.as_tensor(value, device='cuda')
            gpu.append(tensor)
        expected_head = None
        expected_order = None
        for arm in ('scratch', 'only'):
            torch.manual_seed(SEED)
            model = MillingMultiScaleRULModel().cuda()
            head_hash = state_hash({key: value for key, value in state(model).items() if not key.startswith('encoder.')})
            expected_head = head_hash if expected_head is None else expected_head
            assert head_hash == expected_head
            if arm == 'only':
                model.encoder.load_state_dict(pretrained, strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            order_hash = hashlib.sha256()
            started = time.time()
            for epoch in range(1, EPOCHS + 1):
                torch.manual_seed(SEED + epoch)
                model.train()
                order = np.random.default_rng(SEED + epoch).permutation(len(train_sequences))[:128]
                order_hash.update(order.tobytes())
                for start in range(0, len(order), 32):
                    selected = order[start:start + 32]
                    optimizer.zero_grad(set_to_none=True)
                    for micro_start in range(0, len(selected), 8):
                        take = selected[micro_start:micro_start + 8]
                        sequence = torch.as_tensor(train_sequences[take], device='cuda')
                        mask = torch.as_tensor(train_mask[take], device='cuda')
                        targets = torch.as_tensor(store.y[train_sequences[take, -1]], device='cuda')
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            loss = F.smooth_l1_loss(model(*(value[sequence] for value in gpu), mask), targets, beta=.05)
                        assert torch.isfinite(loss)
                        (loss * len(take) / len(selected)).backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    assert torch.isfinite(gradient)
                    optimizer.step()
            metrics, predictions, targets = score(model, gpu, test_sequences, test_mask, store.y)
            expected_order = order_hash.hexdigest() if expected_order is None else expected_order
            assert order_hash.hexdigest() == expected_order
            destination = output / 'downstream' / f'{held}_{arm}_seed42'
            destination.mkdir(parents=True)
            np.savez(destination / 'predictions.npz', pred=predictions, y=targets, rows=test_sequences[:, -1])
            row = dict(held=held, train=train_units, arm=arm, metrics=metrics, seconds=time.time() - started,
                       initial_encoder_hash=state_hash(pretrained) if arm == 'only' else None,
                       final_encoder_hash=state_hash(state(model.encoder)), head_hash=head_hash,
                       sample_order_hash=order_hash.hexdigest())
            write(destination / 'summary.json', row)
            rows.append(row)
            del model, optimizer
            torch.cuda.empty_cache()
        del gpu
        fold = {row['arm']: row['metrics'] for row in rows if row['held'] == held}
        print('MILLING_MULTISCALE_FOLD', json.dumps(dict(held=held, **fold)), flush=True)
    macro = {arm: {metric: float(np.mean([row['metrics'][metric] for row in rows if row['arm'] == arm]))
                   for metric in ('rmse', 'mae', 'r2')} for arm in ('scratch', 'only')}
    pooled = {}
    for arm in ('scratch', 'only'):
        predictions, targets = [], []
        for unit in units:
            with np.load(output / 'downstream' / f'{unit}_{arm}_seed42' / 'predictions.npz') as values:
                predictions.append(values['pred'])
                targets.append(values['y'])
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        error = prediction - target
        pooled[arm] = dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))),
                           r2=float(1 - np.sum(error ** 2) / np.sum((target - target.mean()) ** 2)), n=len(target))
    result = dict(complete=True, macro=macro, pooled=pooled, rows=rows)
    write(output / 'development_summary.json', result)
    print('MILLING_MULTISCALE_RESULT', json.dumps(result), flush=True)


def main():
    torch.set_num_threads(4)
    output = OUT / NAME
    assert not output.exists(), 'Fresh experiment name required'
    output.mkdir(parents=True)
    sources = [Path(__file__), Path(__file__).with_name('global_local_data.py'),
               Path(__file__).with_name('global_local_model.py')]
    write(output / 'protocol.json', dict(
        name=NAME, seed=SEED,
        upstream=dict(cache=UPSTREAM_CACHE, stems=UPSTREAM,
                      labels='no wear/RUL labels; unit-disjoint validation within each corpus'),
        encoder='bearing-matched Global+Local D96 two-layer four-head Transformer; three physical force channels',
        pretraining='30% valid local-token masked reconstruction; observed-window whole-cut force summary as global token; best validation checkpoint',
        downstream='PHM2010 C1/C4/C6 three-fold leave-one-cutter-out; other two cutters train; no validation',
        input='12 causal cuts; fused state plus one-cut and four-cut embedding changes',
        temporal='projection to D96 -> GRU96 -> MLP64 normalized RUL',
        features='26 raw_v2 statistics/frequency descriptors including RMS, energy, kurtosis, spectral centroid, dominant frequency and band energies',
        optimization='upstream 48x20 batch32; downstream fixed100x4 batch32 micro8; AdamW1e-4; SmoothL1 beta0.05',
        source_sha256={path.name: file_sha(path) for path in sources}))
    (output / 'source').mkdir()
    for path in sources:
        shutil.copy2(path, output / 'source' / path.name)
    pretrained = run_upstream(output)
    if RUN_DOWNSTREAM:
        run_downstream(output, pretrained)


if __name__ == '__main__':
    main()
