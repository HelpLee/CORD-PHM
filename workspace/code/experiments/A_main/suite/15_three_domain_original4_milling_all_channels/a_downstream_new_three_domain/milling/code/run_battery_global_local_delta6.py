"""Battery-only Global+Local masked pretraining and CS2 delta-6 LOCO transfer."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import DOWN, OUT, Store
from feature_observations import observed_store
from global_local_data import GlobalLocalStore, file_sha
from global_local_model import DeltaGlobalLocalRULModel, GlobalLocalMaskedModel
from io_utils import resilient_write as write


NAME = 'battery_global_local_delta6_seed42_v1'
SEED = 42
LENGTH = 6
EPOCHS = 100
UPSTREAM = ('hust', 'isu_ilcc', 'mich_exp', 'xjtu')
UNITS = ('CS2_35', 'CS2_36', 'CS2_37', 'CS2_38')


def state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_hash(sd):
    return hashlib.sha256(b''.join(value.numpy().tobytes() for key, value in sorted(sd.items()))).hexdigest()


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
        groups = store.eligible(validation_units)
        if not groups:
            continue
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
    for source_name in UPSTREAM:
        store = GlobalLocalStore(observed_store(Store(OUT / 'lifecycle_cache' / f'{source_name}_battery')), calibrate=False)
        train_units, validation_units = store.split()
        norm = store.normalize(train_units)
        pools[store.name] = (store, train_units, validation_units, norm)
        audit.append(dict(dataset=store.name, train=train_units, validation=validation_units,
                          eligible_train=sum(map(len, store.eligible(train_units).values())),
                          eligible_validation=sum(map(len, store.eligible(validation_units).values())),
                          local_source=store.info['signature'],
                          global_cache=file_sha(OUT / 'global_feature_cache' / store.name / 'manifest.json'),
                          center_scale=[value.tolist() for value in norm]))
    write(output / 'upstream_splits_scalers.json', audit)
    torch.manual_seed(SEED)
    model = GlobalLocalMaskedModel().cuda()
    initial_hash = state_hash(state(model.encoder))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    names = list(pools)
    best_loss = float('inf')
    history = []
    started = time.time()
    for epoch in range(1, 49):
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
        if monitor < best_loss:
            best_loss = monitor
            best_epoch = epoch
            best_encoder = state(model.encoder)
        write(output / 'upstream/history.json', history)
        if epoch % 8 == 0:
            print('BATTERY_GLOBAL_LOCAL_UP', epoch, round(monitor, 5), flush=True)
    (output / 'upstream').mkdir(exist_ok=True)
    torch.save(best_encoder, output / 'upstream/encoder.pt')
    write(output / 'upstream/summary.json', dict(best_epoch=best_epoch, validation_loss=best_loss,
          initial_hash=initial_hash, encoder_hash=state_hash(best_encoder), updates=960,
          global_batch=32, seconds=time.time() - started))
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
    denominator = ((targets - targets.mean()) ** 2).sum()
    metrics = dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))),
                   r2=float(1 - (error ** 2).sum() / denominator) if denominator else float('nan'), n=len(error))
    return metrics, predictions, targets


def run_downstream(output, pretrained):
    store = GlobalLocalStore(observed_store(Store(OUT / 'raw_cache' / DOWN['battery'])), calibrate=False)
    units = list(UNITS)
    assert sorted(store.groups) == sorted(units)
    rows = []
    write(output / 'downstream/data.json', dict(units=units, counts={unit: len(store.groups[unit]) for unit in units},
          source=store.info['signature'], global_summary='observed local mean plus exact cycle duration/capacity'))
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
            model = DeltaGlobalLocalRULModel().cuda()
            head_hash = state_hash({key: value for key, value in state(model).items() if not key.startswith('encoder.')})
            expected_head = head_hash if expected_head is None else expected_head
            assert head_hash == expected_head
            if arm == 'only':
                model.encoder.load_state_dict(pretrained, strict=True)
            initial_encoder_hash = state_hash(state(model.encoder))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            draw_hash = hashlib.sha256()
            started = time.time()
            for epoch in range(1, EPOCHS + 1):
                torch.manual_seed(SEED + epoch)
                model.train()
                order = np.random.default_rng(SEED + epoch).permutation(len(train_sequences))[:128]
                draw_hash.update(order.tobytes())
                for start in range(0, len(order), 32):
                    selected = order[start:start + 32]
                    optimizer.zero_grad(set_to_none=True)
                    for micro_start in range(0, len(selected), 8):
                        take = selected[micro_start:micro_start + 8]
                        seq = torch.as_tensor(train_sequences[take], device='cuda')
                        mask = torch.as_tensor(train_mask[take], device='cuda')
                        targets = torch.as_tensor(store.y[train_sequences[take, -1]], device='cuda')
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            loss = F.smooth_l1_loss(model(*(value[seq] for value in gpu), mask), targets, beta=.05)
                        assert torch.isfinite(loss)
                        (loss * len(take) / len(selected)).backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    assert torch.isfinite(gradient)
                    optimizer.step()
            metrics, predictions, targets = score(model, gpu, test_sequences, test_mask, store.y)
            expected_order = draw_hash.hexdigest() if expected_order is None else expected_order
            assert draw_hash.hexdigest() == expected_order
            destination = output / 'downstream' / f'{held}_{arm}_seed42'
            destination.mkdir(parents=True)
            np.savez(destination / 'predictions.npz', pred=predictions, y=targets, rows=test_sequences[:, -1])
            row = dict(held=held, train=train_units, arm=arm, seed=SEED, epoch=EPOCHS, metrics=metrics,
                       initial_encoder_hash=initial_encoder_hash, final_encoder_hash=state_hash(state(model.encoder)),
                       head_hash=head_hash, sample_order_hash=draw_hash.hexdigest(), seconds=time.time() - started)
            write(destination / 'summary.json', row)
            rows.append(row)
            del model, optimizer
            torch.cuda.empty_cache()
        del gpu
        fold = {row['arm']: row['metrics'] for row in rows if row['held'] == held}
        improvement = 100 * (fold['scratch']['rmse'] - fold['only']['rmse']) / fold['scratch']['rmse']
        print('BATTERY_DELTA6_FOLD', json.dumps(dict(held=held, scratch=fold['scratch'], only=fold['only'],
              rmse_improvement_percent=improvement)), flush=True)
    macro = {arm: {metric: float(np.mean([row['metrics'][metric] for row in rows if row['arm'] == arm]))
                   for metric in ('rmse', 'mae', 'r2')} for arm in ('scratch', 'only')}
    pooled = {}
    for arm in ('scratch', 'only'):
        prediction_parts, target_parts = [], []
        for unit in units:
            with np.load(output / 'downstream' / f'{unit}_{arm}_seed42' / 'predictions.npz') as values:
                prediction_parts.append(values['pred']); target_parts.append(values['y'])
        prediction = np.concatenate(prediction_parts); target = np.concatenate(target_parts); error = prediction - target
        pooled[arm] = dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))),
                           r2=float(1 - np.sum(error ** 2) / np.sum((target - target.mean()) ** 2)), n=len(target))
    result = dict(complete=True, macro=macro, pooled=pooled, rows=rows)
    write(output / 'development_summary.json', result)
    print('BATTERY_DELTA6_RESULT', json.dumps(dict(macro=macro, pooled=pooled)), flush=True)


def main():
    torch.set_num_threads(4)
    output = OUT / NAME
    assert not output.exists(), 'Fresh experiment name required'
    output.mkdir(parents=True)
    sources = [Path(__file__), Path(__file__).with_name('global_local_data.py'), Path(__file__).with_name('global_local_model.py')]
    protocol = dict(name=NAME, seed=SEED,
      upstream='HUST/ISU-ILCC/MICH-EXP/XJTU battery-only lifecycle rows; no RUL labels',
      upstream_split='deterministic unit-disjoint 80/20 train/validation inside every source',
      downstream='CALCE CS2_35/36/37/38 four-fold leave-one-cell-out; other three cells train',
      snapshot='one 26D whole-discharge global summary + 64 local battery HealthTokens',
      global_summary='observed local-token mean, with exact whole-cycle duration and capacity when present',
      encoder='D96, two-layer, four-head Transformer; fusion(global hidden, pooled local hidden)',
      pretraining='random 30% valid local-token masked MSE; global token always visible; best validation checkpoint',
      downstream_model='six causal [e_t, e_t-e_(t-1)] states -> projection -> GRU96 -> MLP64 -> normalized RUL',
      downstream_selection='fixed epoch100, no validation set',
      budgets='upstream 48x20 updates batch32 micro8x4; downstream 100x4 updates batch32 micro8x4',
      arms={'scratch': 'random encoder, full training', 'only': 'battery-only pretrained encoder, full finetuning'},
      source_sha256={path.name: file_sha(path) for path in sources})
    write(output / 'protocol.json', protocol)
    (output / 'source').mkdir()
    for path in sources:
        shutil.copy2(path, output / 'source' / path.name)
    pretrained = run_upstream(output)
    run_downstream(output, pretrained)


if __name__ == '__main__':
    main()
