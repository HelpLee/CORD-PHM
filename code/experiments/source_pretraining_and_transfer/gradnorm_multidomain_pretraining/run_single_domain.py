"""Run one matched single-domain control without touching the joint run."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def command_for(domain, resume=False):
    arm = HERE / 'single_domain' / domain
    command = [sys.executable, '-u', str(HERE / 'three_domain/code/train_joint.py'),
               '--domain', domain]
    if resume and (arm / 'training/last.pt').is_file():
        command.append('--resume')
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=('bearing', 'battery', 'milling'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    status = HERE / 'single_domain' / args.domain / 'training/status.json'
    if status.is_file() and json.loads(status.read_text())['state'] == 'complete':
        print('Already complete:', args.domain, flush=True)
        return
    # Milling uses exactly the runtime already prepared for experiment 17.
    # Never rebuild it concurrently or silently change its audited splits.
    if args.domain == 'milling':
        audit = HERE / 'runtime/milling/upstream_splits_scalers.json'
        if not audit.is_file():
            raise FileNotFoundError('Prepare experiment 17 Milling runtime first: ' + str(audit))
    subprocess.run(command_for(args.domain, args.resume), cwd=str(HERE), check=True)


if __name__ == '__main__':
    main()
