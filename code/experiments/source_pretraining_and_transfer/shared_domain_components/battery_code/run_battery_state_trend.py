"""Prepare/verify by default. --run launches 12 fixed-budget downstream fits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import io
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from battery_state_trend import StateTrendRUL
from run_battery_fullrows_65token import ROOT, DATA, BASE, write, state

OUT = ROOT/'outputs/local_transfer_research/battery_state_trend_fixed500_v1'
UPSTREAM = BASE/'run_seed42'
TRAIN = ('CS2_35', 'CS2_36', 'CS2_37')
TEST = 'CS2_38'


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def nested_order(n):
    selected = list(dict.fromkeys([n//2, 0, n-1]))
    distances = np.min(np.abs(np.arange(n)[:, None]-np.array(selected)[None, :]), axis=1)
    while len(selected) < n:
        nxt = int(np.argmax(distances))
        selected.append(nxt)
        distances = np.minimum(distances, np.abs(np.arange(n)-nxt))
    return np.array(selected)


class Downstream:
    def __init__(self):
        self.path = DATA/'calce_cs2_downstream_battery_health_tokens.npz'
        with np.load(self.path, allow_pickle=True) as z:
            self.x = z['x_health'].astype('float32')
            self.fm, self.tm, self.cm = [z[k].astype(bool) for k in ('feature_mask', 'token_mask', 'c_mask')]
            self.units = z['sample_unit_id'].astype(str)
            self.order = z['sample_cycle_index'].astype(float)
            self.y = z['y_rul_norm'].astype('float32')
            features = z['feature_names'].astype(str).tolist()
            observed = self.fm & self.tm[..., None] & self.cm[..., None, None]
            self.g = np.where(observed, self.x, 0).sum(2)/observed.sum(2).clip(min=1)
            # Preserve the exact upstream snapshot input construction.
            # These are encoder inputs, never a separate downstream bypass branch.
            self.g[:, 0, features.index('local_duration')] = z['cycle_duration_s']
            self.g[:, 0, features.index('local_capacity')] = z['cycle_capacity_ah']
        self.gf = np.any(observed, axis=2) & np.isfinite(self.g)
        assert self.x.shape[1:] == (1, 64, 26)
        assert set(self.units) == set(TRAIN+(TEST,))
        assert np.isfinite(self.y).all() and np.all((self.y >= 0) & (self.y <= 1))
        self.groups = {u: np.flatnonzero(self.units == u) for u in TRAIN+(TEST,)}
        self.sequences = np.zeros((len(self.x), 20), dtype=np.int64)
        self.hm = np.zeros((len(self.x), 20), dtype=bool)
        self.labels = {fraction: [] for fraction in (.1, .2)}
        for unit, ids in self.groups.items():
            ids = ids[np.argsort(self.order[ids], kind='stable')]
            self.groups[unit] = ids
            assert np.all(np.diff(self.order[ids]) > 0)
            for j, row in enumerate(ids):
                history = ids[max(0, j-19):j+1]
                self.sequences[row, :len(history)] = history
                self.sequences[row, len(history):] = history[-1]
                self.hm[row, :len(history)] = True
            if unit in TRAIN:
                priority = nested_order(len(ids))
                for fraction in self.labels:
                    self.labels[fraction].extend(ids[priority[:int(np.ceil(fraction*len(ids)))]] .tolist())
        self.labels = {f: np.array(sorted(ids), dtype=np.int64) for f, ids in self.labels.items()}
        assert set(self.labels[.1]) < set(self.labels[.2])
        self.norm = self.fit_scaler()
        # Cache the deterministic snapshot-level normalization once.  The
        # training protocol is unchanged; this only removes repeated NumPy
        # arcsinh/clip work from every microbatch.
        lc, ls, gc, gs = self.norm
        observed_all = self.fm & self.tm[..., None] & self.cm[..., None, None]
        self.xn = np.where(observed_all, np.clip(np.arcsinh((self.x-lc)/ls), -20, 20), 0).astype('float32')
        self.gn = np.where(self.gf, np.clip(np.arcsinh((self.g-gc)/gs), -20, 20), 0).astype('float32')
        assert np.isfinite(self.xn).all() and np.isfinite(self.gn).all()
        self._cuda_cache = None
        self.cuda_y = None

    def fit_scaler(self):
        rows = np.isin(self.units, TRAIN)
        result = [[], [], [], []]
        for f in range(26):
            mask = self.fm[:, 0, :, f] & self.tm[:, 0] & self.cm[:, 0, None] & rows[:, None]
            for values, index in ((self.x[:, 0, :, f][mask], 0),
                                  (self.g[:, 0, f][self.gf[:, 0, f] & rows], 2)):
                q25, med, q75 = np.percentile(values, [25, 50, 75]) if len(values) else (0., 0., 1.)
                result[index].append(float(med))
                result[index+1].append(float(q75-q25) if q75-q25 > 1e-6 else 1.)
        return np.array(result, dtype=np.float32)

    def batch(self, rows, device):
        if device == 'cuda' and self._cuda_cache is not None:
            ids = torch.as_tensor(rows, device='cuda')
            xn, gn, cm_all, tm_all, sequences, hm = self._cuda_cache
            seq = sequences[ids]
            return xn[seq], gn[seq], cm_all[seq], tm_all[seq], hm[ids]
        seq = self.sequences[rows]
        cm, tm = self.cm[seq], self.tm[seq]
        x = self.xn[seq]
        g = self.gn[seq]
        return tuple(torch.as_tensor(v, device=device) for v in (x, g, cm, tm, self.hm[rows]))

    def enable_cuda_cache(self):
        """Move immutable normalized arrays once; disabled in canonical runs."""
        self._cuda_cache = tuple(torch.as_tensor(value, device='cuda') for value in
                                 (self.xn, self.gn, self.cm, self.tm, self.sequences, self.hm))
        self.cuda_y = torch.as_tensor(self.y, device='cuda')


def load_checkpoint(path):
    payload = path.read_bytes()
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True), hashlib.sha256(payload).hexdigest()


def make_model(seed, arm, checkpoint, device='cpu'):
    seed_all(seed)
    model = StateTrendRUL()
    if arm == 'finetune':
        model.encoder.load_state_dict(checkpoint, strict=True)
    model.to(device)
    assert all(p.requires_grad for p in model.parameters())
    encoder = list(model.encoder.parameters())
    other = [p for name, p in model.named_parameters() if not name.startswith('encoder.')]
    groups = [dict(params=encoder, lr=1e-4 if arm == 'finetune' else 1e-3), dict(params=other, lr=1e-3)]
    return model, groups


def verify(data, checkpoint):
    torch.set_num_threads(4)
    scratch, _ = make_model(42, 'scratch', checkpoint)
    ft, _ = make_model(42, 'finetune', checkpoint)
    assert all(torch.equal(v, scratch.state_dict()[k]) for k, v in ft.state_dict().items() if not k.startswith('encoder.'))
    for k, v in checkpoint.items():
        assert torch.equal(ft.encoder.state_dict()[k], v)
    e = torch.arange(20).float()[None, :, None].expand(1, 20, 96)
    assert torch.allclose(ft.trend(e, torch.tensor([20])), torch.full((1, 96), 5.))
    assert torch.count_nonzero(ft.trend(e, torch.tensor([5]))) == 0
    ft.eval()
    e = torch.randn(1, 20, 96)
    e2 = e.clone(); e2[:, 10:] = torch.randn_like(e2[:, 10:])*100
    with torch.no_grad():
        a = ft.temporal_states(e, torch.ones(1, 20, dtype=torch.bool))
        b = ft.temporal_states(e2, torch.ones(1, 20, dtype=torch.bool))
        assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    rows = data.groups[TRAIN[0]][[0, 19]]
    batch = data.batch(rows, 'cpu')
    before = state(ft.encoder)
    ft.train(); optimizer = torch.optim.AdamW(ft.parameters(), lr=1e-4)
    loss = F.smooth_l1_loss(ft(*batch), torch.tensor(data.y[rows]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in ft.parameters() if p.grad is not None)
    optimizer.step()
    assert any(not torch.equal(v, before[k]) for k, v in ft.encoder.state_dict().items())
    # Test-set changes cannot alter source scalers or few-shot label indices.
    norm = data.norm.copy(); test = data.groups[TEST]
    old = data.x[test].copy(); data.x[test] = 1e6
    assert np.array_equal(norm, data.fit_scaler())
    data.x[test] = old
    for fraction, ids in data.labels.items():
        assert set(data.units[ids]) == set(TRAIN)
    # Same seed, same initialized model and dropout sequence on CPU.
    outputs = []
    for _ in range(2):
        m, _ = make_model(43, 'finetune', checkpoint)
        outputs.append(m(*batch).detach())
    assert torch.equal(*outputs)
    return dict(strict_checkpoint_load=True, paired_head_initialization=True, full_encoder_update=True,
                finite_loss=float(loss.detach()), deterministic_forward=True, causal_mask=True,
                nested_labels=True, test_excluded_from_scalers=True, trend_formula=True)


def run(data, checkpoint, checkpoint_hash):
    status = json.loads((UPSTREAM/'status.json').read_text())
    assert status['state'] == 'completed', 'Wait for upstream completion before the 12 comparisons'
    assert torch.cuda.is_available()
    source = OUT/'source'; source.mkdir(exist_ok=True)
    for name in ('run_battery_state_trend.py', 'battery_state_trend.py', 'global_local_model.py', 'masking.py', 'run_battery_fullrows_65token.py'):
        shutil.copy2(Path(__file__).with_name(name), source/name)
    pinned = OUT/'upstream_encoder.pt'
    if pinned.exists():
        checkpoint, checkpoint_hash = load_checkpoint(pinned)
    else:
        torch.save(checkpoint, pinned)
        checkpoint, checkpoint_hash = load_checkpoint(pinned)
    for seed in (42, 43, 44):
        for fraction in (.1, .2):
            for arm in ('scratch', 'finetune'):
                dest = OUT/f'seed{seed}'/f'fraction{int(fraction*100)}'/arm
                if (dest/'metrics.json').exists():
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                model, groups = make_model(seed, arm, checkpoint, 'cuda')
                model.train()
                optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
                rng = np.random.default_rng(seed); labels = data.labels[fraction]
                updates, history = 0, []
                while updates < 500:
                    order = rng.permutation(labels)
                    for start in range(0, len(order), 32):
                        selected = order[start:start+32]
                        optimizer.zero_grad(set_to_none=True)
                        value = 0.
                        for j in range(0, len(selected), 4):
                            rows = selected[j:j+4]
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                pred = model(*data.batch(rows, 'cuda'))
                                loss = F.smooth_l1_loss(pred.float(), torch.as_tensor(data.y[rows], device='cuda'))
                            assert torch.isfinite(loss)
                            weight = len(rows)/len(selected)
                            (loss*weight).backward(); value += loss.item()*weight
                        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 5))
                        optimizer.step(); updates += 1; history.append(value)
                        if updates % 50 == 0:
                            print(seed, fraction, arm, updates, value, flush=True)
                        if updates == 500:
                            break
                model.eval(); test = data.groups[TEST]; predictions = []
                with torch.no_grad():
                    for i in range(0, len(test), 4):
                        predictions.extend(model(*data.batch(test[i:i+4], 'cuda')).cpu().tolist())
                pred = np.asarray(predictions); y = data.y[test]; error = pred-y
                metrics = dict(seed=seed, fraction=fraction, arm=arm, updates=updates,
                               rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))),
                               r2=float(1-np.sum(error**2)/np.sum((y-y.mean())**2)), bias=float(error.mean()),
                               upstream_sha256=checkpoint_hash)
                np.savez(dest/'predictions.npz', pred=pred, y=y, source_rows=test, cycles=data.order[test])
                torch.save(state(model), dest/'model.pt')
                write(dest/'train_loss.json', history); write(dest/'metrics.json', metrics)
                del model, optimizer, groups
                torch.cuda.empty_cache()
    summaries = []
    for fraction in (.1, .2):
        for arm in ('scratch', 'finetune'):
            rows = [json.loads((OUT/f'seed{s}'/f'fraction{int(fraction*100)}'/arm/'metrics.json').read_text()) for s in (42, 43, 44)]
            summaries.append(dict(fraction=fraction, arm=arm, **{k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows], ddof=1))) for k in ('rmse', 'mae', 'r2', 'bias')}))
    write(OUT/'summary.json', summaries)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(4)
    data = Downstream()
    checkpoint, sha = load_checkpoint(UPSTREAM/'encoder.pt')
    OUT.mkdir(parents=True, exist_ok=True)
    report = verify(data, checkpoint)
    write(OUT/'verification.json', dict(**report, inspected_upstream_sha256=sha))
    write(OUT/'protocol.json', dict(train=list(TRAIN), test=TEST, validation=None, seed=[42, 43, 44],
        fractions=[.1, .2], updates=500, batch=32, microbatch=4, dropout=.05, s=20,
        temporal=dict(layers=2, d=96, heads=4, ff=192, causal=True),
        head='MLP+sigmoid([h_t, mean(last5)-mean(previous5), e_t]); no additional physical bypass',
        short_history='right padding+mask; trend zero before ten real observations',
        lr=dict(scratch=1e-3, finetune_encoder=1e-4, finetune_head=1e-3), weight_decay=1e-4,
        finetune='all snapshot encoder parameters updated from step1; no freezing',
        labels='existing y_rul_norm; deterministic nested source endpoint subsets; all source histories available',
        scaling='all source input rows only; median/IQR+asinh+clip20, identical to upstream transform',
        snapshot_global='identical upstream construction, including existing cycle capacity/duration coordinates; these do not enter head separately',
        data_path=str(data.path), upstream=str(UPSTREAM/'encoder.pt'),
        schedule='12 runs, start only after upstream completion; one pinned checkpoint shared by all runs',
        std='sample standard deviation, ddof=1', deterministic=True))
    write(OUT/'labels.json', {str(f): dict(source_rows=ids.tolist(), counts={u:int(np.sum(data.units[ids] == u)) for u in TRAIN}) for f, ids in data.labels.items()})
    write(OUT/'scalers.json', dict(train_units=list(TRAIN), norm=data.norm.tolist()))
    print('VERIFIED', json.dumps(report), flush=True)
    if args.run:
        run(data, checkpoint, sha)
    else:
        print('Prepared and verified only; no downstream experiment launched.', flush=True)


if __name__ == '__main__':
    main()
