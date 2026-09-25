"""CPU-only data/import checks before submitting any GPU computation."""
import json
import sys
import subprocess
from pathlib import Path
from config import HERE, DOMAINS, ARMS, settings


def main():
    subprocess.run([sys.executable, str(HERE / 'test_methods.py')], check=True, cwd=HERE)
    from train import load_trainer
    trainer = load_trainer('preflight')
    pools = trainer.prepare()
    audit = {d: [dict(name=p.name, train_windows=len(p.tw), validation_windows=len(p.vw)) for p in pools[d]] for d in DOMAINS}
    assert tuple(p.name for p in pools['milling']) == (
        'luh_milling', 'matwi_milling', 'nonastreda_milling', 'qit_cemc_milling', 'hmotp_milling')
    for domain in DOMAINS:
        assert pools[domain]
        assert all(len(p.tw) and p.groups for p in pools[domain])
    print('PREFLIGHT_DATA_OK', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
