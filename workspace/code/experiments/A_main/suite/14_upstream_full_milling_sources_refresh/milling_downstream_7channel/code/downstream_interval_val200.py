"""PHM2010 C1+C4 -> C6: HealthTokens, residual tool adapter, causal TCN and GRU."""
import hashlib
import json
import os
import random
import time

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

from data import DOWN, OUT
from adapter_encoder import MillingAdapterEncoder as GlobalLocalEncoder
from io_utils import resilient_write as write
from run_battery_global_local_delta6 import state, state_hash
from run_milling_global_local_multiscale12 import CHANNELS, prepare_store


SEED = int(os.environ.get('MILLING_ADAPTER_SEED', '42'))
NAME = os.environ.get('MILLING_ADAPTER_NAME', f'milling_tool_adapter_tcn20_c6_seed{SEED}_v1')
SOURCE = OUT / 'milling_mask_dynamics_upstream_seed42_v1' / 'upstream' / 'encoder.pt'
TRAIN = ('C1', 'C4')
HELD = 'C6'
LENGTH = 20
MAX_EPOCHS = 100
FROZEN_EPOCHS = 0
PATIENCE = 15
VAL_INTERVAL = 2
BATCH = 32
MICRO = 8
DROPOUT = .05
HEAD_LR = 1e-3
LAST_BLOCK_LR = 1e-4
LABEL_FRACTION = float(os.environ.get('MILLING_ADAPTER_FRACTION', '1.0'))
MAX_UPDATES = int(os.environ.get('MILLING_ADAPTER_MAX_UPDATES', '0'))
TRAIN_VALIDATION_FRACTION = float(os.environ.get('MILLING_ADAPTER_TRAIN_VALIDATION_FRACTION', '0.2'))
TEST_VALIDATION_FRACTION = float(os.environ.get('MILLING_ADAPTER_TEST_VALIDATION_FRACTION', '0'))
PRELOAD_INDICES = os.environ.get('HEALTHTOKEN_PRELOAD_INDICES', '0') == '1'
NESTED_LABELS = os.environ.get('MILLING_ADAPTER_NESTED_LABELS', '0') == '1'
ARMS = tuple(value.strip() for value in os.environ.get(
    'MILLING_ADAPTER_ARMS', 'scratch,partial_finetune').split(',') if value.strip())


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(96, 96, kernel_size=3, dilation=dilation)
        self.norm = nn.LayerNorm(96)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, value):
        residual = value
        hidden = F.pad(value.transpose(1, 2), (self.padding, 0))
        hidden = self.conv(hidden).transpose(1, 2)
        return residual + self.dropout(F.gelu(self.norm(hidden)))


class ToolAdapterTCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GlobalLocalEncoder(channels=CHANNELS, dropout=DROPOUT)
        self.token_fusion = nn.Linear(192, 96)
        self.adapter_down = nn.Linear(96, 24)
        self.adapter_up = nn.Linear(24, 96)
        self.adapter_norm = nn.LayerNorm(96)
        self.tcn = nn.ModuleList([CausalBlock(dilation) for dilation in (1, 2, 4)])
        self.gru = nn.GRU(96, 96, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 64), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(64, 1))

    def cut_state(self, x, g, cm, tm):
        encoded = self.encoder(x, g, cm, tm)
        valid = encoded['valid'][..., None]
        local = (encoded['local_hidden'] * valid).sum(1) / valid.sum(1).clamp_min(1)
        state = self.token_fusion(torch.cat((encoded['global_hidden'], local), dim=-1))
        adapted = state + self.adapter_up(F.gelu(self.adapter_down(state)))
        return self.adapter_norm(adapted)

    def forward(self, x, g, cm, tm):
        batch, steps, channels, tokens, features = x.shape
        sequence = self.cut_state(
            x.reshape(batch * steps, channels, tokens, features),
            g.reshape(batch * steps, channels, features),
            cm.reshape(batch * steps, channels),
            tm.reshape(batch * steps, channels, tokens),
        ).reshape(batch, steps, -1)
        for block in self.tcn:
            sequence = block(sequence)
        _, hidden = self.gru(sequence)
        return torch.sigmoid(self.head(hidden[-1]).squeeze(-1))


def full_sequences(store, units):
    sequence, mask = store.sequences_with_mask(units, LENGTH)
    keep = mask.sum(1) == LENGTH
    return sequence[keep]


def nested_uniform_order(size):
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


def source_split(store):
    train_parts, val_parts, counts = [], [], {}
    for unit in TRAIN:
        sequence = full_sequences(store, (unit,))
        candidates = sequence
        count = max(1, int(np.ceil(len(candidates) * LABEL_FRACTION)))
        if NESTED_LABELS:
            selected = nested_uniform_order(len(candidates))[:count]
        else:
            selected = np.unique(np.rint(np.linspace(0, len(candidates) - 1, count)).astype(np.int64))
        chosen = candidates[selected]
        n_val = max(1, int(np.ceil(len(chosen) * TRAIN_VALIDATION_FRACTION)))
        val_idx = np.arange(len(chosen))[-n_val:]
        train_idx = np.arange(len(chosen))[:-n_val]
        if len(train_idx) == 0:
            train_idx, val_idx = val_idx[:1], val_idx[1:]
        train_parts.append(chosen[train_idx])
        val_parts.append(chosen[val_idx])
        counts[unit] = dict(total=len(sequence), selected=int(len(chosen)), train=int(len(train_idx)),
                            validation=int(len(val_idx)))
    return np.concatenate(train_parts), np.concatenate(val_parts), counts


def configure_partial(model, stage):
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    if stage == 2:
        for name, parameter in model.encoder.named_parameters():
            if name.startswith(('transformer.layers.1.', 'final_norm.', 'adapters.1.')):
                parameter.requires_grad_(True)


def optimizer_for(model, arm, stage):
    if arm == 'scratch':
        return torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=1e-4)
    configure_partial(model, stage)
    last_block, downstream = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('encoder.'):
            last_block.append(parameter)
        else:
            downstream.append(parameter)
    groups = [{'params': downstream, 'lr': HEAD_LR}]
    if last_block:
        groups.append({'params': last_block, 'lr': LAST_BLOCK_LR})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def optimizer_for_full(model):
    """Full transfer: all pretrained encoder weights train from epoch 1."""
    encoder, downstream = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith('encoder.') else downstream).append(parameter)
    return torch.optim.AdamW(
        [{'params': downstream, 'lr': HEAD_LR},
         {'params': encoder, 'lr': LAST_BLOCK_LR}], weight_decay=1e-4)


@torch.no_grad()
def score(model, gpu, sequences, target_by_row):
    model.eval()
    predictions = []
    sequence_tensor = (torch.as_tensor(sequences, device='cuda')
                       if PRELOAD_INDICES else None)
    for start in range(0, len(sequences), 64):
        rows = (sequence_tensor[start:start + 64] if sequence_tensor is not None
                else torch.as_tensor(sequences[start:start + 64], device='cuda'))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(*(value[rows] for value in gpu))
        predictions.extend(prediction.float().cpu().tolist())
    prediction = np.asarray(predictions)
    target = target_by_row[sequences[:, -1]].astype(float)
    error = prediction - target
    metrics = dict(
        rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))),
        r2=float(1 - np.sum(error ** 2) / np.sum((target - target.mean()) ** 2)),
        bias=float(error.mean()), n=len(error))
    return metrics, prediction, target


def train(model, arm, store, gpu, train_sequences, val_sequences):
    stage = 2 if arm == 'full_finetune' else 1
    optimizer = optimizer_for_full(model) if arm == 'full_finetune' else optimizer_for(model, arm, stage)
    order_hash = hashlib.sha256()
    best = None
    stale = 0
    updates = 0
    history = []
    started = time.time()
    sequence_tensor = (torch.as_tensor(train_sequences, device='cuda')
                       if PRELOAD_INDICES else None)
    target_tensor = (torch.as_tensor(store.y[train_sequences[:, -1]], device='cuda')
                     if PRELOAD_INDICES else None)
    epoch_limit = MAX_UPDATES if MAX_UPDATES > 0 else MAX_EPOCHS
    for epoch in range(1, epoch_limit + 1):
        next_stage = 2 if (arm == 'full_finetune' or epoch > FROZEN_EPOCHS) else 1
        if arm == 'partial_finetune' and next_stage != stage:
            stage = next_stage
            optimizer = optimizer_for(model, arm, stage)
            stale = 0
        torch.manual_seed(SEED + epoch)
        model.train()
        if arm == 'frozen_probe':
            model.encoder.eval()
        elif arm == 'partial_finetune':
            model.encoder.eval()
            if stage == 2:
                model.encoder.transformer.layers[1].train()
                model.encoder.final_norm.train()
                model.encoder.adapters[1].train()
        elif arm == 'full_finetune':
            model.encoder.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(train_sequences))
        order_hash.update(order.tobytes())
        losses = []
        for start in range(0, len(order), BATCH):
            selected = order[start:start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(selected), MICRO):
                take = selected[micro_start:micro_start + MICRO]
                if sequence_tensor is None:
                    rows = torch.as_tensor(train_sequences[take], device='cuda')
                    target = torch.as_tensor(store.y[train_sequences[take, -1]], device='cuda')
                else:
                    take_tensor = torch.as_tensor(take, device='cuda')
                    rows = sequence_tensor[take_tensor]
                    target = target_tensor[take_tensor]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(*(value[rows] for value in gpu))
                    loss = F.smooth_l1_loss(prediction, target, beta=.05)
                assert torch.isfinite(loss)
                (loss * len(take) / len(selected)).backward()
                losses.append(float(loss.detach()))
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5)
            assert torch.isfinite(gradient)
            optimizer.step()
            updates += 1
            if MAX_UPDATES > 0 and updates >= MAX_UPDATES:
                break
        row = dict(epoch=epoch, stage=stage, loss=float(np.mean(losses)), updates=updates)
        if MAX_UPDATES <= 0 and epoch % VAL_INTERVAL == 0:
            validation, _, _ = score(model, gpu, val_sequences, store.y)
            row['validation'] = validation
            # The partial arm must enter stage 2 before it may be selected.
            eligible = arm in ('scratch', 'frozen_probe', 'full_finetune') or stage == 2
            if eligible and (best is None or validation['rmse'] < best['metrics']['rmse']):
                best = dict(epoch=epoch, stage=stage, metrics=validation, model=state(model))
                stale = 0
            elif eligible:
                stale += VAL_INTERVAL
        history.append(row)
        if MAX_UPDATES > 0 and updates >= MAX_UPDATES:
            break
        if MAX_UPDATES <= 0 and best is not None and stale >= PATIENCE:
            break
    if MAX_UPDATES > 0:
        final_metrics, _, _ = score(model, gpu, val_sequences, store.y)
        best = dict(epoch=epoch, stage=stage, metrics=final_metrics, model=state(model))
    assert best is not None
    model.load_state_dict(best.pop('model'), strict=True)
    return best, history, updates, order_hash.hexdigest(), time.time() - started


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    output = OUT / NAME
    # This package runs Scratch only: no upstream checkpoint is accessed.
    requires_checkpoint = any(arm in ('frozen_probe', 'partial_finetune', 'full_finetune')
                              for arm in ARMS)
    if requires_checkpoint:
        assert SOURCE.exists(), f'Missing checkpoint: {SOURCE}'
    assert not output.exists(), 'Use a fresh output directory'
    output.mkdir(parents=True)
    pretrained = (torch.load(SOURCE, map_location='cpu', weights_only=True)
                  if requires_checkpoint else None)
    store = prepare_store(OUT / 'raw_cache' / DOWN['milling'])
    train_sequences, val_sequences, counts = source_split(store)
    test_sequences = full_sequences(store, (HELD,))
    held_validation = None
    if TEST_VALIDATION_FRACTION > 0:
        n_val = max(1, int(np.ceil(len(test_sequences) * TEST_VALIDATION_FRACTION)))
        held_validation, test_sequences = test_sequences[:n_val], test_sequences[n_val:]
        assert len(test_sequences) > 0
    norm = store.normalize(TRAIN)
    allowed = store.rows(('C1', 'C4', 'C6'))
    gpu = []
    for value in store.transform(allowed, norm):
        tensor = torch.zeros((len(store.x), *value.shape[1:]),
                             dtype=torch.from_numpy(value).dtype, device='cuda')
        tensor[torch.as_tensor(allowed, device='cuda')] = torch.as_tensor(value, device='cuda')
        gpu.append(tensor)
    write(output / 'protocol.json', dict(
        name=NAME, seed=SEED, train=list(TRAIN), held=HELD, length=LENGTH,
        checkpoint=str(SOURCE), source_split=counts,
        tokens='pretrained global token concatenated with valid-local-token mean -> Linear192x96',
        adapter='residual 96->24->96 tool adapter',
        temporal='causal TCN96 dilation1,2,4 -> one-layer GRU96 -> sigmoid normalized RUL',
        arms=list(ARMS), label_fraction=LABEL_FRACTION,
        finetune='fully trainable from update1; encoder 1e-4; adapter/TCN/GRU/head 1e-3',
        validation=('20% of selected C1/C4 training labels; early stopping'
                     if TRAIN_VALIDATION_FRACTION > 0 else 'none'),
        max_updates=MAX_UPDATES, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        test_validation_fraction=TEST_VALIDATION_FRACTION,
        train_validation_fraction=TRAIN_VALIDATION_FRACTION,
        nested_labels=NESTED_LABELS, dropout=DROPOUT,
        leakage='C6 labels/features never used for training, validation, normalization, stopping or model selection'))
    rows = []
    expected_head = None
    for arm in ARMS:
        torch.manual_seed(SEED)
        model = ToolAdapterTCNModel().cuda()
        head_hash = state_hash({key: value for key, value in state(model).items()
                                if not key.startswith('encoder.')})
        expected_head = head_hash if expected_head is None else expected_head
        assert head_hash == expected_head
        if arm in ('frozen_probe', 'partial_finetune', 'full_finetune'):
            assert pretrained is not None
            model.encoder.load_state_dict(pretrained, strict=True)
        initial_encoder_hash = state_hash(state(model.encoder))
        effective_val = held_validation if held_validation is not None else val_sequences
        best, history, updates, order_hash, seconds = train(
            model, arm, store, gpu, train_sequences, effective_val)
        metrics, prediction, target = score(model, gpu, test_sequences, store.y)
        destination = output / arm
        destination.mkdir()
        torch.save(state(model),destination/'model.pt')
        np.savez(destination / 'predictions.npz', pred=prediction, y=target,
                 rows=test_sequences[:, -1])
        row = dict(seed=SEED, arm=arm, train=list(TRAIN), held=HELD,
                   best=best, optimizer_updates=updates, metrics=metrics,
                   initial_encoder_hash=initial_encoder_hash,
                   final_encoder_hash=state_hash(state(model.encoder)),
                   head_hash=head_hash, sample_order_hash=order_hash, seconds=seconds)
        write(destination / 'summary.json', row)
        write(destination / 'history.json', history)
        rows.append(row)
        print('MILLING_ADAPTER_RESULT', json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    write(output / 'development_summary.json', dict(complete=True, rows=rows))
    print('MILLING_ADAPTER_COMPLETE', json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
