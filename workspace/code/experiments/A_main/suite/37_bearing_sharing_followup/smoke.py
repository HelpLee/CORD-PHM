"""gpu_test FP32 verification gates all BF16 production runs."""
import json
import subprocess
import sys
from config import HERE, E34, E35, DOMAINS, SEEDS, ARMS, settings, continuation_source


def run(*args):
    subprocess.run([sys.executable,'-u',*args],cwd=HERE,check=True)


if __name__ == '__main__':
    run('test_methods.py')
    # Check continuation eligibility before releasing the production dependency.
    from continuation import validate_protocol
    for arm in ARMS:
        source=continuation_source(arm)
        if source is None: continue
        old=json.loads((source/'protocol.json').read_text())
        new=dict(old,max_epochs=2000,settings=settings(arm))
        validate_protocol(old,new)
        status=json.loads((source/'training/status.json').read_text())
        assert status['reason']=='max_epochs' and status['epoch']==500
        assert (source/'training/last.pt').is_file() and (source/'training/encoder.pt').is_file()
    for root, arms in ((E34, [f'single_{d}_component' for d in DOMAINS]),
                       (E35, ['single_bearing_conditional','single_battery_conditional',
                              'joint_conditional_cagrad','joint_conditional_pcgrad'])):
        for arm in arms:
            domains=DOMAINS if arm.startswith('joint_') else (arm.split('_')[1],)
            for domain in domains:
                for pct in (10,20):
                    value=json.loads((root/'downstream'/arm/domain/f'p{pct}/results.json').read_text())
                    assert value.get('complete') and len(value['rows'])==5
                    assert {r['seed'] for r in value['rows']}==set(SEEDS)
    for arm in ('joint_lowbearing_floor_cagrad','single_bearing_lowbearing'):
        run('train.py','--arm',arm,'--smoke','--resume')
    for domain in DOMAINS:
        run('downstream.py','--model','smoke_joint_lowbearing_floor_cagrad','--domain',domain,
            '--fraction','.1','--verify-only')
    print('E37_REAL_DATA_SMOKE_PASSED',flush=True)
