"""Extract frozen downstream snapshot embeddings for E1 and E2."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'code/data_phm').is_dir())
SUITE = ROOT / 'code/experiments/A_main/suite'
NO_ADAPTER = ROOT / 'outputs/reproduction_final/joint_upstream500_seed42'
DOMAINS = ('bearing', 'battery', 'milling')
VARIANTS = ('single_domain', 'three_domain_no_adapter', 'three_domain_adapter')
SINGLE = {
    'bearing': SUITE / '12_upstream_bearing_only_adapter/training/encoder.pt',
    'battery': SUITE / '13_upstream_battery_only_adapter/training/encoder.pt',
    'milling': SUITE / '11_upstream_milling_only_adapter/training/encoder.pt',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_encoder(domain, variant, device):
    if variant == 'three_domain_no_adapter':
        code = NO_ADAPTER / 'code'
        checkpoint = NO_ADAPTER / 'training/encoder.pt'
    else:
        code = SUITE / '10_upstream_three_domain_adapter/code'
        checkpoint = (SINGLE[domain] if variant == 'single_domain' else
                      SUITE / '10_upstream_three_domain_adapter/training/encoder.pt')
    sys.path.insert(0, str(code))
    joint = importlib.import_module('joint_model')
    encoder = joint.JointModel().encoder
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if variant == 'single_domain':
        for name in tuple(encoder.stems):
            if name != domain:
                del encoder.stems[name]
        for name in tuple(encoder.adapters):
            if name != domain:
                del encoder.adapters[name]
    encoder.load_state_dict(state, strict=True)
    encoder.eval().to(device)
    return encoder, checkpoint


def bearing_inputs():
    code = SUITE / '01_shared_dependencies/bearing_code'
    sys.path.insert(0, str(code))
    gl = importlib.import_module('global_local_data')
    gl.OUT = SUITE / '01_shared_dependencies/bearing_runtime'
    store = gl.GlobalLocalStore(gl.build_condition2(), channels=2)
    ids = store.groups['Bearing2_1']
    ids = ids[np.argsort(store.order[ids], kind='stable')]
    norm = store.normalize(('Bearing2_2', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5'))
    return store.transform(ids, norm), store.y[ids], store.order[ids], ids, 'Bearing2_1'


def battery_inputs():
    code = SUITE / '01_shared_dependencies/battery_code'
    sys.path.insert(0, str(code))
    module = importlib.import_module('run_battery_state_trend')
    data = module.Downstream()
    ids = data.groups[module.TEST]
    arrays = (data.xn[ids], data.gn[ids], data.cm[ids], data.tm[ids])
    return arrays, data.y[ids], data.order[ids], ids, module.TEST


def milling_inputs():
    code = SUITE / '20_downstream_milling_three_domain_adapter/code'
    sys.path.insert(0, str(code))
    data = importlib.import_module('data')
    # The reproducibility package centralizes immutable caches here; the copied
    # downstream module's historical local runtime directory is intentionally empty.
    data.OUT = SUITE / '01_shared_dependencies/milling_runtime'
    module = importlib.import_module('run_milling_global_local_multiscale12')
    store = module.prepare_store(data.OUT / 'raw_cache' / data.DOWN['milling'])
    ids = store.groups['C6']
    ids = ids[np.argsort(store.order[ids], kind='stable')]
    norm = store.normalize(('C1', 'C4'))
    return store.transform(ids, norm), store.y[ids], store.order[ids], ids, 'C6'


def inputs(domain):
    return dict(bearing=bearing_inputs, battery=battery_inputs,
                milling=milling_inputs)[domain]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=DOMAINS)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    device = torch.device(args.device)

    encoder, checkpoint = load_encoder(args.domain, args.variant, device)
    arrays, y, order, rows, unit = inputs(args.domain)
    if args.verify_only:
        arrays, y, order, rows = tuple(v[:2] for v in arrays), y[:2], order[:2], rows[:2]

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(y), args.batch_size):
            batch = tuple(torch.as_tensor(v[start:start + args.batch_size], device=device)
                          for v in arrays)
            embedding, _ = encoder(args.domain, *batch)
            embeddings.append(embedding.float().cpu().numpy())

    embedding = np.concatenate(embeddings)
    assert embedding.shape == (len(y), 96) and np.isfinite(embedding).all()
    if args.verify_only:
        print(json.dumps(dict(passed=True, domain=args.domain, variant=args.variant,
                              checkpoint=str(checkpoint), shape=list(embedding.shape))), flush=True)
        return

    destination = HERE / 'outputs' / args.domain
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / f'{args.variant}.npz', embedding=embedding,
             y=np.asarray(y, np.float32), order=np.asarray(order), rows=np.asarray(rows), unit=unit)
    metadata = dict(domain=args.domain, variant=args.variant, unit=unit, samples=len(y),
                    checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    embedding='frozen upstream snapshot projector output e_t', dimension=96,
                    labels_used_for_extraction=False)
    (destination / f'{args.variant}.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
