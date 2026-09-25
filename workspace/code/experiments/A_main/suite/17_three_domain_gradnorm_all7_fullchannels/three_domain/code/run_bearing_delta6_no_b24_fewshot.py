"""Label-efficient four-bearing LOBO: scratch versus full finetune.

Only the selected endpoint labels supervise the loss.  All source snapshots
remain available as causal input history, and the held bearing never enters
selection, normalization, or training.
"""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
if os.environ.get('BEARING_FEWSHOT_DETERMINISTIC', '0') == '1':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

from data import OUT
from global_local_data import GlobalLocalStore, build_condition2
from compression_model import Attention65RUL as DeltaGlobalLocalRULModel
from io_utils import resilient_write as write
from helpers import score, hh, st


SEED = int(os.environ.get('BEARING_FEWSHOT_SEED', '42'))
NAME = os.environ.get('BEARING_FEWSHOT_NAME',
                      f'bearing_delta6_no_b24_fewshot_seed{SEED}_v1')
FRACTIONS = tuple(float(item) for item in os.environ.get(
    'BEARING_FEWSHOT_FRACTIONS', '0.05,0.10,0.20').split(','))
EPOCHS = int(os.environ.get('BEARING_FEWSHOT_EPOCHS', '100'))
MAX_UPDATES = int(os.environ.get('BEARING_FEWSHOT_MAX_UPDATES', '0'))
ENCODER_LR = float(os.environ.get('BEARING_FEWSHOT_ENCODER_LR', '1e-4'))
HEAD_LR = float(os.environ.get('BEARING_FEWSHOT_HEAD_LR', '1e-4'))
MODEL_DROPOUT = float(os.environ.get('BEARING_FEWSHOT_DROPOUT', '0.1'))
MODEL_CHANNELS = int(os.environ.get('BEARING_FEWSHOT_CHANNELS', '2'))
VARIABLE_CHANNELS = os.environ.get('BEARING_FEWSHOT_VARIABLE_CHANNELS', '0') == '1'
INDEPENDENT_CHANNELS = os.environ.get('BEARING_INDEPENDENT_CHANNELS', '0') == '1'
SWA_EPOCHS = int(os.environ.get('BEARING_FEWSHOT_SWA_EPOCHS', '0'))
NESTED_LABELS = os.environ.get('BEARING_FEWSHOT_NESTED_LABELS', '0') == '1'
DETERMINISTIC = os.environ.get('BEARING_FEWSHOT_DETERMINISTIC', '0') == '1'
ARMS = tuple(os.environ.get(
    'BEARING_FEWSHOT_ARMS', 'scratch,finetune').split(','))
LENGTH = 6
UNITS = tuple(os.environ.get(
    'BEARING_FEWSHOT_UNITS', 'Bearing2_1,Bearing2_2,Bearing2_3,Bearing2_5').split(','))
HELD_UNITS = tuple(os.environ.get('BEARING_FEWSHOT_HELD', ','.join(UNITS)).split(','))
_excluded = os.environ.get('BEARING_FEWSHOT_EXCLUDED', 'Bearing2_4')
EXCLUDED = None if _excluded.strip().lower() in ('', 'none', 'null') else _excluded
SOURCE = Path(os.environ.get(
    'BEARING_UPSTREAM_CHECKPOINT',
    str(OUT/'bearing_four_lobo_finetune5_up200_v1'/'upstream'/'upstream'/'encoder.pt'),
))


def fraction_key(fraction):
    return f'{fraction:.2f}'.replace('.', 'p')


def nested_uniform_order(size):
    """Deterministic, nested coverage order over a one-dimensional lifetime."""
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    selected = [size // 2]
    if size > 1:
        selected.extend(position for position in (0, size - 1) if position not in selected)
    while len(selected) < size:
        remaining = np.setdiff1d(np.arange(size), np.asarray(selected), assume_unique=True)
        distances = np.min(np.abs(remaining[:, None] - np.asarray(selected)[None, :]), axis=1)
        selected.append(int(remaining[np.argmax(distances)]))
    return np.asarray(selected, dtype=np.int64)


def evenly_spaced_labeled_sequences(store, sequences, masks, train_units, fraction):
    """Choose labels uniformly across each source-bearing life, not by loss."""
    selected = []
    counts = {}
    endpoint_rows = sequences[:, -1]
    for unit in train_units:
        candidates = np.flatnonzero(store.units[endpoint_rows].astype(str) == unit)
        assert len(candidates) > 0
        count = max(1, int(np.ceil(fraction * len(candidates))))
        if NESTED_LABELS:
            positions = nested_uniform_order(len(candidates))[:count]
        else:
            positions = np.unique(np.rint(np.linspace(0, len(candidates) - 1, count)).astype(np.int64))
        picked = candidates[positions]
        selected.extend(picked.tolist())
        counts[unit] = dict(total=int(len(candidates)), labeled=int(len(picked)))
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return sequences[selected], masks[selected], counts


def train_and_score(model, store, gpu, train_sequences, train_masks, test_sequences,
                    test_masks, seed, arm):
    if arm == 'finetune' and ENCODER_LR != HEAD_LR:
        encoder_parameters = list(model.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        head_parameters = [parameter for parameter in model.parameters()
                           if id(parameter) not in encoder_ids]
        optimizer = torch.optim.AdamW([
            {'params': encoder_parameters, 'lr': ENCODER_LR},
            {'params': head_parameters, 'lr': HEAD_LR},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=1e-4)
    order_hash = hashlib.sha256()
    swa = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
    swa_count = 0
    update_count = 0
    started = time.time()
    epoch = 0
    while (update_count < MAX_UPDATES if MAX_UPDATES > 0 else epoch < EPOCHS):
        epoch += 1
        torch.manual_seed(seed + epoch)
        model.train()
        # Every selected label contributes once per epoch; no unselected label
        # is sampled or used for loss.
        order = np.random.default_rng(seed + epoch).permutation(len(train_sequences))
        order_hash.update(order.tobytes())
        for start in range(0, len(order), 32):
            selected = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0, len(selected), 8):
                take = selected[micro:micro + 8]
                sequence = torch.as_tensor(train_sequences[take], device='cuda')
                history_mask = torch.as_tensor(train_masks[take], device='cuda')
                target = torch.as_tensor(store.y[train_sequences[take, -1]], device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.smooth_l1_loss(
                        model(*(value[sequence] for value in gpu), history_mask),
                        target, beta=.05,
                    )
                assert torch.isfinite(loss)
                (loss * len(take) / len(selected)).backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            assert torch.isfinite(gradient)
            optimizer.step()
            update_count += 1
            if update_count % 100 == 0:
                print('DOWNSTREAM_UPDATE',seed,arm,len(train_sequences),update_count,flush=True)
            if MAX_UPDATES > 0 and update_count >= MAX_UPDATES:
                break
        if SWA_EPOCHS > 0 and epoch > EPOCHS - SWA_EPOCHS:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    swa[name].add_(parameter)
            swa_count += 1
    if swa_count:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(swa[name] / swa_count)
    metrics, prediction, target = score(model, gpu, test_sequences, test_masks, store.y)
    return metrics, prediction, target, order_hash.hexdigest(), time.time() - started, epoch, update_count


def configure_arm(model, pretrained, arm):
    """Keep the trajectory head trainable; choose encoder adaptation depth."""
    if arm == 'scratch':
        return 'random encoder, fully trainable'
    model.encoder.load_state_dict(pretrained, strict=True)
    if arm == 'finetune':
        return 'pretrained encoder, fully trainable'
    if arm == 'linear_probe':
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        return 'pretrained encoder frozen; train downstream trajectory head only'
    if arm == 'partial_finetune':
        # The encoder has two Transformer blocks.  Preserve tokenisation and
        # the first block; adapt the second block plus snapshot readout.
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.encoder.named_parameters():
            if (name.startswith('transformer.layers.1.') or
                    name.startswith('final_norm.') or name.startswith('fusion.')):
                parameter.requires_grad_(True)
        return 'pretrained last Transformer block + snapshot readout trainable'
    raise ValueError(f'Unknown arm: {arm}')


def main():
    torch.set_num_threads(4)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    assert SOURCE.exists(), f'Missing upstream checkpoint: {SOURCE}'
    output = OUT / NAME
    assert not output.exists(), 'Use a fresh output directory'
    output.mkdir(parents=True)
    pretrained = torch.load(SOURCE, map_location='cpu', weights_only=True)
    store = GlobalLocalStore(build_condition2(), channels=MODEL_CHANNELS)
    assert set(UNITS).issubset(store.groups) and set(HELD_UNITS).issubset(UNITS)
    if EXCLUDED is not None:
        assert EXCLUDED in store.groups and EXCLUDED not in UNITS
    results = []
    for fraction in FRACTIONS:
        assert 0 < fraction <= 1
        for held in HELD_UNITS:
            train_units = tuple(unit for unit in UNITS if unit != held)
            norm = store.normalize(train_units)
            source_sequences, source_masks = store.sequences_with_mask(train_units, LENGTH)
            train_sequences, train_masks, label_counts = evenly_spaced_labeled_sequences(
                store, source_sequences, source_masks, train_units, fraction,
            )
            write(output/f'labels_{fraction_key(fraction)}.json',dict(rows=train_sequences[:,-1].tolist(),counts=label_counts))
            write(output/'scalers.json',dict(train=list(train_units),center_scale=[v.tolist() for v in norm]))
            test_sequences, test_masks = store.sequences_with_mask((held,), LENGTH)
            allowed = store.rows(UNITS)
            gpu = []
            for value in store.transform(allowed, norm):
                tensor = torch.zeros((len(store.x), *value.shape[1:]),
                                     dtype=torch.from_numpy(value).dtype, device='cuda')
                tensor[torch.as_tensor(allowed, device='cuda')] = torch.as_tensor(value, device='cuda')
                gpu.append(tensor)
            expected_head = expected_order = None
            for arm in ARMS:
                torch.manual_seed(SEED)
                model = DeltaGlobalLocalRULModel(
                    channels=MODEL_CHANNELS, variable_channels=VARIABLE_CHANNELS,
                    independent_channels=INDEPENDENT_CHANNELS,
                    dropout=MODEL_DROPOUT, head_dropout=MODEL_DROPOUT).cuda()
                head_hash = hh({key: value for key, value in st(model).items()
                                if not key.startswith('encoder.')})
                expected_head = head_hash if expected_head is None else expected_head
                assert head_hash == expected_head
                arm_description = configure_arm(model, pretrained, arm)
                initial_encoder_hash = hh(st(model.encoder))
                metrics, prediction, target, order_hash, seconds, epochs_run, updates = train_and_score(
                    model, store, gpu, train_sequences, train_masks,
                    test_sequences, test_masks, SEED, arm,
                )
                expected_order = order_hash if expected_order is None else expected_order
                assert order_hash == expected_order
                destination = output / f'fraction_{fraction_key(fraction)}' / f'{held}_{arm}_seed{SEED}'
                destination.mkdir(parents=True)
                torch.save(st(model),destination/'model.pt')
                np.savez(destination/'predictions.npz', pred=prediction, y=target,
                         rows=test_sequences[:, -1])
                row = dict(seed=SEED, fraction=fraction, held=held,
                           train=list(train_units), label_counts=label_counts,
                           labeled_total=int(len(train_sequences)), arm=arm,
                           arm_description=arm_description,
                           epoch=epochs_run, optimizer_updates=updates, metrics=metrics,
                           encoder_lr=ENCODER_LR if arm == 'finetune' else HEAD_LR,
                           head_lr=HEAD_LR, swa_epochs=SWA_EPOCHS,
                           dropout=MODEL_DROPOUT,
                           nested_labels=NESTED_LABELS, deterministic=DETERMINISTIC,
                           initial_encoder_hash=initial_encoder_hash,
                           final_encoder_hash=hh(st(model.encoder)), head_hash=head_hash,
                           sample_order_hash=order_hash, seconds=seconds)
                write(destination/'summary.json', row)
                results.append(row)
                print('BEARING_FEWSHOT_RESULT', json.dumps(row), flush=True)
                del model
                torch.cuda.empty_cache()
            del gpu
    aggregate = {}
    for fraction in FRACTIONS:
        aggregate[str(fraction)] = {
            arm: {metric: float(np.mean([row['metrics'][metric] for row in results
                                         if row['fraction'] == fraction and row['arm'] == arm]))
                  for metric in ('rmse', 'mae', 'r2')}
            for arm in ARMS
        }
    write(output/'development_summary.json', dict(complete=True, aggregate=aggregate, rows=results))
    write(output/'protocol.json', dict(
        name=NAME, seed=SEED, fractions=list(FRACTIONS), units=list(UNITS), held_units=list(HELD_UNITS), excluded=EXCLUDED,
        outer='held-bearing folds; every nonheld listed bearing is source only',
        labels='per source bearing, uniformly spaced endpoint labels only; all selected labels used once per epoch',
        validation=None, selection=(f'fixed optimizer updates={MAX_UPDATES}' if MAX_UPDATES > 0
                                    else f'fixed epoch{EPOCHS}'),
        stability=dict(deterministic=DETERMINISTIC, nested_labels=NESTED_LABELS,
                       encoder_lr=ENCODER_LR, head_lr=HEAD_LR,
                       swa_epochs=SWA_EPOCHS, max_updates=MAX_UPDATES,
                       dropout=MODEL_DROPOUT, staged_unfreezing=False),
        channel_interface=dict(max_channels=MODEL_CHANNELS,
                               variable_channels=VARIABLE_CHANNELS,
                               independent_channels=INDEPENDENT_CHANNELS),
        input='six causal [e_t,e_t-e_(t-1)] states; unlabeled source snapshots are causal input only',
        arms={arm: 'configured by configure_arm' for arm in ARMS},
        upstream_checkpoint=str(SOURCE),
    ))
    print('BEARING_FEWSHOT_COMPLETE', json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
