"""Seven-source battery pretraining: all rows, C=1, Global+Local and 6->1.

CALCE CS2 is reserved for downstream; ISU-ILCC is explicitly excluded.
Uses the historical GlobalLocalMaskedModel without changing other experiments.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from global_local_model import GlobalLocalMaskedModel

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'code/data_phm/processed_health_tokens').is_dir())
DATA = ROOT / 'code/data_phm/processed_health_tokens/battery'
BASE = ROOT / 'outputs/local_transfer_research/battery_fullrows_65token_no_isu_v1'
SOURCES = ('hust', 'mich_exp', 'nasa', 'oxford', 'rwth', 'sdu', 'xjtu')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    os.replace(tmp, path)


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(96, 96, batch_first=True)
        self.projection = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 96))

    def forward(self, x):
        return self.projection(self.gru(x)[1][-1])


class Corpus:
    def __init__(self, name):
        self.name = name
        source = DATA / f'{name}_battery_health_tokens.npz'
        dest = BASE / 'cache' / name
        dest.mkdir(parents=True, exist_ok=True)
        signature = dict(path=str(source), bytes=source.stat().st_size,
                         mtime_ns=source.stat().st_mtime_ns, version=1)
        ready = dest / 'manifest.json'
        if ready.exists():
            # A copied cache may contain the original Windows absolute path
            # and mtime. Bytes and cache version are the portable invariants;
            # the source above is the canonical NPZ on the current machine.
            saved = json.loads(ready.read_text())['signature']
            assert saved['bytes'] == signature['bytes'], source
            assert saved.get('version') == signature['version'], source
        else:
            print('EXTRACT', name, flush=True)
            # Stream full arrays to disk without allocating the whole corpus.
            with zipfile.ZipFile(source) as archive:
                for key in ('x_health', 'feature_mask', 'token_mask', 'c_mask'):
                    with archive.open(key + '.npy') as src, (dest / (key + '.npy')).open('wb') as dst:
                        shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
        self.x, self.fm, self.tm, self.cm = [np.load(dest / (k + '.npy'), mmap_mode='r')
                                          for k in ('x_health', 'feature_mask', 'token_mask', 'c_mask')]
        assert self.x.shape[1:] == (1, 64, 26), self.x.shape
        with np.load(source, allow_pickle=True) as z:
            self.units = z['sample_unit_id'].astype(str)
            self.order = z['sample_cycle_index'].astype(float)
            runs = z['sample_run_id'].astype(str)
            conditions = z['sample_condition_id'].astype(str)
            features = z['feature_names'].astype(str).tolist()
            capacity, duration = z['cycle_capacity_ah'], z['cycle_duration_s']
            assert np.all(z['feature_median'] == 0) and np.all(z['feature_iqr'] == 1)
        self.valid_rows = np.any(self.tm & self.cm[..., None], axis=(1, 2))
        self.groups = {u: np.flatnonzero(self.units == u) for u in sorted(set(self.units))}
        units = sorted(self.groups, key=lambda u: hashlib.sha256(str((42, name+'_battery', u)).encode()).hexdigest())
        nv = min(len(units)-1, max(1, round(.2 * len(units))))
        self.train, self.val = units[nv:], units[:nv]
        assert set(self.train).isdisjoint(self.val)
        self.windows = {}
        window_audit = []
        for part, selected_units in [('train', self.train), ('val', self.val)]:
            blocks = []
            for u in selected_units:
                rows = self.groups[u]
                keys = np.char.add(np.char.add(runs[rows], '\x1f'), conditions[rows])
                for key in np.unique(keys):
                    ids = rows[keys == key]
                    ids = ids[np.argsort(self.order[ids], kind='stable')]
                    assert np.all(np.diff(self.order[ids]) > 0), (name, u)
                    if len(ids) < 7:
                        continue
                    win = np.lib.stride_tricks.sliding_window_view(ids, 7)
                    win = win[np.all(self.valid_rows[win], axis=1)]
                    blocks.append(win)
                    window_audit.append(dict(part=part, unit=u, windows=len(win),
                                             min_cycle_gap=float(np.diff(self.order[ids]).min()),
                                             max_cycle_gap=float(np.diff(self.order[ids]).max())))
            self.windows[part] = np.concatenate(blocks) if blocks else np.empty((0, 7), dtype=np.int64)
        assert len(self.windows['train']), (name, 'no training sequences')
        if not ready.exists():
            global_x = np.lib.format.open_memmap(dest/'global.npy', mode='w+', dtype='float32', shape=(len(self.x), 1, 26))
            for start in range(0, len(self.x), 1024):
                sl = slice(start, start+1024)
                obs = self.fm[sl] & self.tm[sl, ..., None] & self.cm[sl, ..., None, None]
                values = np.asarray(self.x[sl])
                assert np.isfinite(values[obs]).all()
                global_x[sl] = np.where(obs, values, 0).sum(2) / obs.sum(2).clip(min=1)
            global_x[:, 0, features.index('local_duration')] = duration
            global_x[:, 0, features.index('local_capacity')] = capacity
            global_x.flush()
            del global_x
            write(ready, dict(signature=signature, rows=len(self.x), shape=list(self.x.shape),
                              features=features, snapshot_cap=None,
                              global_features='Observed-window feature means; duration/capacity replaced by saved full-cycle values'))
        self.g = np.load(dest/'global.npy', mmap_mode='r')
        self.gf = np.any(self.fm & self.tm[..., None] & self.cm[..., None, None], axis=2)
        self.gf &= np.isfinite(self.g)
        train_mask = np.isin(self.units, self.train) & self.valid_rows
        norm_path = dest/'scalers.json'
        if norm_path.exists():
            saved = json.loads(norm_path.read_text())
            assert saved['train_units'] == self.train
            self.norm = np.asarray(saved['norm'], dtype=np.float32)
        else:
            print('FIT_TRAIN_ONLY_SCALERS', name, flush=True)
            norms = [[], [], [], []]
            for f in range(26):
                obs = self.fm[:, 0, :, f] & self.tm[:, 0] & self.cm[:, 0, None] & train_mask[:, None]
                lv = self.x[:, 0, :, f][obs]
                gv = self.g[:, 0, f][self.gf[:, 0, f] & train_mask]
                for vals, pos in ((lv, 0), (gv, 2)):
                    if len(vals):
                        q25, med, q75 = np.percentile(vals, [25, 50, 75])
                        scale = q75-q25
                    else:
                        med, scale = 0., 1.
                    norms[pos].append(float(med))
                    norms[pos+1].append(float(scale) if scale > 1e-6 else 1.)
            self.norm = np.asarray(norms, dtype=np.float32)
            write(norm_path, dict(train_units=self.train, norm=self.norm.tolist()))
        self.train_groups = [ids[self.valid_rows[ids]] for u, ids in self.groups.items() if u in self.train]
        self.train_groups = [ids for ids in self.train_groups if len(ids)]
        self.val_rows = np.flatnonzero(np.isin(self.units, self.val) & self.valid_rows)
        self.audit = dict(source=signature, rows=len(self.x), valid_rows=int(self.valid_rows.sum()),
                          train_units=self.train, validation_units=self.val,
                          train_windows=len(self.windows['train']), validation_windows=len(self.windows['val']),
                          trajectories=window_audit, snapshot_cap=None)
        print('DATA_READY', name, len(self.x), len(self.windows['train']), len(self.windows['val']), flush=True)

    def batch(self, ids):
        lc, ls, gc, gs = self.norm
        tm, cm = np.asarray(self.tm[ids]), np.asarray(self.cm[ids])
        observed = self.fm[ids] & tm[..., None] & cm[..., None, None]
        x = np.where(observed, np.clip(np.arcsinh((self.x[ids]-lc)/ls), -20, 20), 0)
        g = np.where(self.gf[ids], np.clip(np.arcsinh((self.g[ids]-gc)/gs), -20, 20), 0)
        assert np.isfinite(x).all() and np.isfinite(g).all()
        return tuple(torch.as_tensor(np.array(v), device='cuda') for v in (x, g, cm, tm, observed))


def losses(model, predictor, corpus, ids, sequences, fixed=None):
    x, g, cm, tm, observed = corpus.batch(ids)
    pred, masked, _ = model(x, g, cm, tm, fixed)
    mask = masked[..., None] & observed
    reconstruction = ((pred.float()-x.float()).square()*mask).sum()/mask.sum().clamp_min(1)
    x, g, cm, tm, _ = corpus.batch(sequences.reshape(-1))
    e = model.encoder(x, g, cm, tm)['snapshot'].reshape(len(sequences), 7, 96)
    dynamics = (predictor(e[:, :6]).float()-e[:, 6].detach().float()).square().mean()
    return reconstruction, dynamics


@torch.no_grad()
def validate(model, predictor, corpora):
    model.eval(); predictor.eval()
    scores = {}
    for c in corpora:
        if not len(c.windows['val']):
            continue
        n = min(64, len(c.val_rows), len(c.windows['val']))
        ids = c.val_rows[np.linspace(0, len(c.val_rows)-1, n, dtype=int)]
        seq = c.windows['val'][np.linspace(0, len(c.windows['val'])-1, n, dtype=int)]
        sums = np.zeros(2)
        for start in range(0, n, 8):
            a, b = losses(model, predictor, c, ids[start:start+8], seq[start:start+8], 9000+start)
            sums += np.array([a.item(), b.item()])*len(ids[start:start+8])
        a, b = sums/n
        scores[c.name] = dict(mask=float(a), dynamics=float(b), total=float(a+.2*b))
    assert scores
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    if args.smoke:
        model = GlobalLocalMaskedModel(channels=1).cuda()
        predictor = Predictor().cuda()
        x = torch.randn(14, 1, 64, 26, device='cuda'); g = x.mean(2)
        cm = torch.ones(14, 1, dtype=torch.bool, device='cuda')
        tm = torch.ones(14, 1, 64, dtype=torch.bool, device='cuda')
        seen = []
        hook = model.encoder.transformer.register_forward_pre_hook(lambda _, inputs: seen.append(tuple(inputs[0].shape)))
        p, mask, out = model(x, g, cm, tm)
        e = out['snapshot'].reshape(2, 7, 96)
        loss = (p-x).square()[mask].mean()+.2*(predictor(e[:, :6])-e[:, 6].detach()).square().mean()
        loss.backward(); hook.remove()
        assert seen == [(14, 65, 96)] and torch.isfinite(loss)
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        print('SMOKE_OK', seen, float(loss), flush=True)
        return
    out = BASE/'run_seed42'
    assert not (out/'status.json').exists(), 'Existing run: inspect status before restarting'
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=42, data_root=str(DATA), sources=list(SOURCES),
                    excluded_upstream=['isu_ilcc', 'calce_cs2_downstream'],
                    downstream_npz=str(DATA/'calce_cs2_downstream_battery_health_tokens.npz'),
                    channels=1, tokens=65, d=96, layers=2, heads=4, ff=192, dropout=.1,
                    objectives='MSE masked valid features + 0.2 MSE stop-gradient next embedding', mask_ratio=.3,
                    dynamics='6->1 adjacent available measurements within same unit/run/condition; Oxford is sparsely sampled cycles',
                    global_features='Observed local-window mean in 26 coordinates; full-cycle duration/capacity replacement',
                    preprocessing='All NPZ rows cached; no snapshot cap; training-unit-only median/IQR then asinh and clip20',
                    sampling='Dataset round-robin; reconstruction uniform units then rows; dynamics uniform eligible windows',
                    epochs=200, updates_per_epoch=20, batch=32, microbatch=8, lr=1e-4, weight_decay=1e-4,
                    selection='20% unit-disjoint source validation; macro mean; patience20; min_delta1e-4',
                    budget_note='All valid rows available to sampling, not a guaranteed full traversal every epoch',
                    torch=torch.__version__)
    write(out/'protocol.json', protocol)
    src = out/'source'; src.mkdir(exist_ok=True)
    for filename in (Path(__file__), Path(__file__).with_name('global_local_model.py'), Path(__file__).with_name('masking.py')):
        shutil.copy2(filename, src/filename.name)
    started = time.time()
    write(out/'status.json', dict(state='preparing', pid=os.getpid()))
    try:
        corpora = [Corpus(name) for name in SOURCES]
        write(out/'data_audit.json', [c.audit for c in corpora])
        model = GlobalLocalMaskedModel(channels=1).cuda(); predictor = Predictor().cuda()
        params = list(model.parameters())+list(predictor.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        best, stale, history = float('inf'), 0, []
        for epoch in range(1, 201):
            model.train(); predictor.train()
            train = []
            for update in range(20):
                seed = 42+epoch*10000+update*10
                rng = np.random.default_rng(seed); torch.manual_seed(seed)
                c = corpora[((epoch-1)*20+update) % len(corpora)]
                ids = np.array([rng.choice(c.train_groups[rng.integers(len(c.train_groups))]) for _ in range(32)])
                sequences = c.windows['train'][rng.integers(len(c.windows['train']), size=32)]
                optimizer.zero_grad(set_to_none=True)
                current = np.zeros(2)
                for start in range(0, 32, 8):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        a, b = losses(model, predictor, c, ids[start:start+8], sequences[start:start+8])
                        loss = a+.2*b
                    assert torch.isfinite(loss)
                    (loss/4).backward()
                    current += np.array([a.item(), b.item()])/4
                gradient = torch.nn.utils.clip_grad_norm_(params, 5)
                assert torch.isfinite(gradient)
                optimizer.step(); train.append(current)
                if update == 0:
                    print('TRAIN_UPDATE', epoch, c.name, current.tolist(), flush=True)
                    write(out/'status.json', dict(state='training', epoch=epoch, update=1, pid=os.getpid()))
            scores = validate(model, predictor, corpora)
            monitor = float(np.mean([v['total'] for v in scores.values()]))
            if monitor < best-1e-4:
                best, stale, best_epoch = monitor, 0, epoch
                torch.save(state(model.encoder), out/'encoder.pt')
                torch.save(dict(model=state(model), predictor=state(predictor), optimizer=optimizer.state_dict(), epoch=epoch), out/'best_training.pt')
            else:
                stale += 1
            history.append(dict(epoch=epoch, train_mask=float(np.mean(train, axis=0)[0]),
                                train_dynamics=float(np.mean(train, axis=0)[1]), validation=monitor,
                                by_dataset=scores, best_epoch=best_epoch, seconds=time.time()-started))
            write(out/'history.json', history)
            write(out/'status.json', dict(state='training', epoch=epoch, best_epoch=best_epoch,
                                         best_validation=best, stale=stale, pid=os.getpid()))
            print('EPOCH', epoch, 'val', monitor, 'best', best_epoch, best, flush=True)
            if stale >= 20:
                break
        write(out/'status.json', dict(state='completed', epoch=epoch, best_epoch=best_epoch,
                                     best_validation=best, seconds=time.time()-started,
                                     checkpoint=str(out/'encoder.pt')))
        print('COMPLETE', out, flush=True)
    except BaseException as error:
        write(out/'status.json', dict(state='failed', error=repr(error), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
