"""Submit six A100 jobs with per-source afterok dependencies and one CPU summary."""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / 'gradnorm_multidomain_pretraining'
UPSTREAM_JOBS = {'three': '1900024', 'bearing': '1900025', 'battery': '1900026', 'milling': '1900027'}
LIMITS = {'bearing': '01:00:00', 'battery': '02:00:00', 'milling': '03:00:00'}
FILE = HERE / 'submissions.json'


def output(*args):
    return subprocess.check_output(args, text=True, cwd=str(HERE)).strip()


def write(records):
    temp = FILE.with_suffix('.tmp')
    temp.write_text(json.dumps(records, indent=2), encoding='utf-8')
    temp.replace(FILE)


def active(name):
    rows = output('squeue', '--noheader', '--user=anonymous_user', '--name=' + name, '--format=%i').splitlines()
    if len(rows) > 1:
        raise RuntimeError('Multiple active jobs match ' + name)
    return rows[0].strip() if rows else None


def dependency(key):
    package = UPSTREAM / ('three_domain' if key == 'three' else 'single_domain/' + key) / 'training'
    status = package / 'status.json'
    if status.is_file() and json.loads(status.read_text()).get('state') == 'complete' and (package / 'encoder.pt').is_file():
        return None
    job = UPSTREAM_JOBS[key]
    states = output('sacct', '-X', '-n', '-P', '-j', job, '--format=JobID,State').splitlines()
    rows = [row.split('|') for row in states if row.split('|')[0] == job]
    if not rows or rows[0][1].split()[0] not in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'REQUEUED'):
        raise RuntimeError('Upstream incomplete and not running: ' + key + ' ' + repr(states))
    return job


def main():
    (HERE / 'logs').mkdir(exist_ok=True)
    records = json.loads(FILE.read_text()) if FILE.is_file() else {}
    deps = {key: dependency(key) for key in UPSTREAM_JOBS}
    ids = []
    for domain in ('bearing', 'battery', 'milling'):
        for initialization in ('single', 'three'):
            key = domain + '_' + initialization
            name = 'gn19_' + key
            running = active(name)
            state = HERE / 'runs' / domain / initialization / 'status.json'
            if not running and state.is_file() and json.loads(state.read_text()).get('state') == 'complete':
                print('COMPLETE', key, flush=True)
                continue
            dep = deps['three' if initialization == 'three' else domain]
            if not running:
                command = ['sbatch', '--parsable', '--job-name=' + name, '--time=' + LIMITS[domain]]
                if dep:
                    command += ['--dependency=afterok:' + dep, '--kill-on-invalid-dep=yes']
                command += ['run_cluster.sh', domain, initialization]
                running = output(*command).split(';')[0]
            ids.append(running)
            records[key] = dict(job_id=running, upstream_dependency=dep, partition='gpua100', gpus=1,
                                walltime=LIMITS[domain], downstream_fits=15)
            write(records)
            print('SUBMITTED', key, running, 'afterok=' + str(dep), flush=True)
    summary = active('gn19_summary')
    if not summary and not (HERE / 'comparison.json').is_file():
        command = ['sbatch', '--parsable']
        if ids:
            command += ['--dependency=afterok:' + ':'.join(ids), '--kill-on-invalid-dep=yes']
        summary = output(*command, 'summarize_cluster.sh').split(';')[0]
    records['summary'] = dict(job_id=summary, dependency=ids, partition='cpu_short', gpus=0)
    write(records)
    print('SUMMARY', summary, flush=True)


if __name__ == '__main__':
    main()
