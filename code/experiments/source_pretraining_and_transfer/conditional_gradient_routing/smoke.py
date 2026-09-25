"""Exercise all joint strategies, both singles, and three downstream loaders."""
import json
import subprocess
import sys
from config import ARMS, DOMAINS, COMPONENT_ROUTING, HERE, SEEDS


def main():
    subprocess.run([sys.executable, 'test_routing.py'], cwd=HERE, check=True)
    # Reused controls must actually be complete; never treat absent results as zero.
    for domain in DOMAINS:
        for pct in (10, 20):
            path = COMPONENT_ROUTING / 'downstream' / f'single_{domain}_component' / domain / f'p{pct}/results.json'
            data = json.loads(path.read_text())
            assert data.get('complete') and len(data['rows']) == 5
            assert {r['seed'] for r in data['rows']} == set(SEEDS)
    for arm in ARMS:
        subprocess.run([sys.executable, '-u', 'train.py', '--arm', arm, '--smoke', '--resume'], cwd=HERE, check=True)
    for domain in DOMAINS:
        subprocess.run([sys.executable, '-u', 'downstream.py', '--model', 'smoke_joint_conditional_cagrad',
                        '--domain', domain, '--fraction', '.1', '--verify-only'], cwd=HERE, check=True)
    print('CONDITIONAL_ROUTING_SMOKE_OK', flush=True)


if __name__ == '__main__': main()
