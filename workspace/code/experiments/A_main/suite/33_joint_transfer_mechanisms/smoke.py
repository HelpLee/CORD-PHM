"""Dependency gate: real-data one-update paths and all downstream conversions."""
import subprocess
import sys
from config import HERE


def run(*arguments):
    subprocess.run([sys.executable, '-u', *arguments], cwd=HERE, check=True)


if __name__ == '__main__':
    run('test_methods.py')
    for arm in ('joint_control', 'joint_famo', 'joint_private_both'):
        run('train.py', '--arm', arm, '--smoke', '--resume')
    for domain in ('bearing', 'battery', 'milling'):
        run('downstream.py', '--model', 'smoke_joint_private_both', '--domain', domain,
            '--fraction', '.1', '--verify-only')
    print('SMOKE_ALL_PASSED', flush=True)
