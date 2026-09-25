"""E28 fine-tuning protocol, isolated E33 checkpoints and consistent V100 FP32."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import contextlib
import hashlib
import importlib.util
import json
import sys
import traceback
import torch
from config import ARMS, DOMAINS, HERE, SUITE


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fp32_driver(name):
    """Specialize metadata in memory; source files and old runs are untouched."""
    spec = importlib.util.find_spec(name)
    text = open(spec.origin, encoding='utf-8').read()
    updated = text.replace("precision='BF16'", "precision='FP32'")
    updated = updated.replace('encoder_lr=.0001', 'encoder_lr=.0003')
    updated = updated.replace('pretrained_lr=.0001', 'pretrained_lr=.0003')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(updated, spec.origin, 'exec'), module.__dict__)
    return module, dict(source=spec.origin, original_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        effective_sha256=hashlib.sha256(updated.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--domain', required=True, choices=DOMAINS)
    parser.add_argument('--fraction', type=float, choices=(.1, .2, 1.), required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    arm = args.model.removeprefix('smoke_')
    if arm not in ARMS or (arm.startswith('single_') and arm.split('_')[1] != args.domain):
        raise ValueError('Invalid model/domain pairing')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Covers inherited scoring helpers as well as training, in this process only.
    torch.autocast = lambda *a, **kw: contextlib.nullcontext()
    d = load('down28', SUITE / 'full_channel_source_models/run_downstream.py')
    d.HERE = HERE
    package = HERE / ('smoke_downstream' if args.verify_only else 'downstream') / args.model / args.domain / f'p{round(100*args.fraction)}'
    package.mkdir(parents=True, exist_ok=True)
    checkpoint = d.upstream(args.model, args.domain, package)
    if (package / 'results.json').exists() and not args.verify_only:
        result = json.loads((package / 'results.json').read_text())
        if result.get('complete') and {r['seed'] for r in result['rows']} == set(range(42, 47)):
            print('ALREADY_COMPLETE', package, flush=True)
            return
    try:
        if args.domain == 'milling':
            d.SEEDS = () if args.verify_only else tuple(range(42, 47))
            d.run_milling(args.model, args.fraction, checkpoint, package)
            provenance = {'source': str(d.EXP15 / 'downstream_milling_10pct/code/downstream_interval_val200.py')}
        else:
            sys.path[:0] = [str(d.WORKSPACE), str(d.WORKSPACE / 'data_phm'),
                str(SUITE / 'shared_domain_components' / f'{args.domain}_code'),
                str(d.EXP19 / f'{args.domain}_code'), str(d.EXP19)]
            name = 'bearing_adapter_study' if args.domain == 'bearing' else 'adapter_downstream'
            driver, provenance = fp32_driver(name)
            if args.verify_only:
                original = driver.main
                def verify(dest):
                    sys.argv = [sys.argv[0], '--verify-only']
                    original(dest)
                driver.main = verify
            d.run_bearing_or_battery(args.domain, args.model, args.fraction, checkpoint, package)
        d.write(package / 'effective_protocol.json', dict(precision='FP32', tf32=False,
            encoder_lr=3e-4, head_lr=1e-3, seeds=list(range(42, 47)), fraction=args.fraction,
            source=provenance, inherited='experiment28 full fine-tuning; same devices/splits/early stopping',
            upstream_arm=args.model))
        if args.verify_only:
            d.write(package / 'status.json', dict(state='verified_only', not_a_result=True))
        else:
            result = json.loads((package / 'results.json').read_text())
            assert result.get('complete') and len(result['rows']) == 5
            assert {r['seed'] for r in result['rows']} == set(range(42, 47))
        print('DOWNSTREAM_OK', args.model, args.domain, args.fraction, flush=True)
    except BaseException as error:
        d.write(package / 'status.json', dict(state='failed', error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    main()
