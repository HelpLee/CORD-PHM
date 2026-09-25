"""Idempotent gpua100 submission; smoke -> upstream -> low-label downstream."""
import argparse
import json
import subprocess
from config import ARMS, FRACTIONS, HERE, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    path = HERE / 'submissions.json'
    ledger = json.loads(path.read_text()) if path.exists() else {}
    if not args.dry_run: (HERE / 'logs').mkdir(exist_ok=True)

    def submit(key, arguments, walltime, dependency=''):
        if key in ledger: return ledger[key]['job_id']
        cmd = ['sbatch', '--parsable', '--partition=gpua100', '--exclude=cluster-gpu13',
               '--job-name=e35_' + key, '--time=' + walltime]
        if arguments[0] == 'train.py': cmd += ['--signal=USR1@300']
        if dependency: cmd += ['--dependency=' + dependency, '--kill-on-invalid-dep=yes']
        cmd += ['run_gpu.sh'] + arguments
        job = str(9900000 + len(ledger)) if args.dry_run else subprocess.check_output(cmd, cwd=HERE, text=True).strip().split(';')[0]
        if not job.isdigit(): raise RuntimeError(job)
        ledger[key] = dict(job_id=job, command=cmd, partition='gpua100', walltime=walltime, dependency=dependency)
        if not args.dry_run:
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(ledger, indent=2)); tmp.replace(path)
        print(key, job, flush=True)
        return job

    smoke = submit('smoke', ['smoke.py'], '00:45:00')
    lanes = [None, None]; jobs = []
    for i, arm in enumerate(ARMS):
        lane = i % 2
        dep = 'afterok:' + smoke
        if lanes[lane]: dep += ',afterany:' + lanes[lane]
        up = submit('up_' + arm, ['train.py', '--arm', arm, '--resume'],
                    '04:00:00' if arm.startswith('joint_') else '02:00:00', dep)
        tail = None
        for domain in settings(arm)['domains']:
            for fraction in FRACTIONS:
                dep = 'afterok:' + up
                if tail: dep += ',afterany:' + tail
                tail = submit(f'down_{arm}_{domain}_p{round(100*fraction)}',
                              ['downstream.py', '--model', arm, '--domain', domain, '--fraction', str(fraction)],
                              '01:00:00', dep)
                jobs.append(tail)
        lanes[lane] = tail
    submit('summary', ['summarize.py'], '00:10:00', 'afterany:' + ':'.join(jobs))
    print('SUBMITTED', len(ledger), 'jobs: 7 upstream, 34 downstream cells (170 seed runs), smoke and summary')


if __name__ == '__main__': main()
