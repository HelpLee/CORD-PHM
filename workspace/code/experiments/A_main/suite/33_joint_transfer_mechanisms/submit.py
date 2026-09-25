"""Idempotent bounded GPU lanes; successful smoke -> upstream -> own downstream."""
import argparse
import json
import subprocess
from config import HERE, ARMS, FRACTIONS, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    (HERE / 'logs').mkdir(exist_ok=True)
    path = HERE / 'submissions.json'
    ledger = json.loads(path.read_text()) if path.exists() else {}
    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(ledger, indent=2))
        temp.replace(path)
    counter = [0]
    def submit(key, command, walltime, dependencies=''):
        if key in ledger:
            return ledger[key]['job_id']
        options = ['sbatch', '--parsable', '--partition=gpu', '--job-name=e33_' + key,
                   '--time=' + walltime]
        if command and command[0] == 'train.py':
            options += ['--signal=USR1@300']
        if dependencies:
            options += ['--dependency=' + dependencies, '--kill-on-invalid-dep=yes']
        full = options + ['run_gpu.sh'] + list(command)
        if args.dry_run:
            counter[0] += 1
            job = str(9900000 + counter[0])
        else:
            job = subprocess.check_output(full, cwd=HERE, text=True).strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError(job)
        ledger[key] = dict(job_id=job, partition='gpu', walltime=walltime,
                           dependencies=dependencies, command=full)
        if not args.dry_run:
            save()
        print(key, job, walltime, dependencies, flush=True)
        return job
    smoke = submit('smoke', ['smoke.py'], '00:45:00')
    if args.smoke_only:
        return
    # Two complete pipelines concurrently, limiting GPU and filesystem pressure.
    # Prioritize structural hypotheses; FAMO gets an independent lane next.
    order = ('joint_private_dynamics', 'joint_dataset_heads', 'joint_private_both',
             'joint_famo', 'joint_control') + tuple(a for a in ARMS if a.startswith('single_'))
    lanes = [None, None]
    downstream = []
    for i, arm in enumerate(order):
        lane = i % 2
        dependency = 'afterok:' + smoke
        if lanes[lane]:
            dependency += ',afterany:' + lanes[lane]
        up = submit('up_' + arm, ['train.py', '--arm', arm, '--resume'],
                    '10:00:00' if arm.startswith('joint_') else '04:00:00', dependency)
        tail = None
        for domain in settings(arm)['domains']:
            for fraction in FRACTIONS:
                dependency = 'afterok:' + up
                if tail:
                    dependency += ',afterany:' + tail
                key = f'down_{arm}_{domain}_p{round(100*fraction)}'
                tail = submit(key, ['downstream.py', '--model', arm, '--domain', domain,
                                    '--fraction', str(fraction)], '02:00:00', dependency)
                downstream.append(tail)
        lanes[lane] = tail
    # A short summary still uses gpu as explicitly requested for this new batch.
    submit('summary', ['summarize.py'], '00:10:00', 'afterany:' + ':'.join(downstream))
    print('SUBMITTED', len(ledger), 'jobs; 11 upstream / 63 downstream cells / 315 downstream seed-runs')


if __name__ == '__main__':
    main()
