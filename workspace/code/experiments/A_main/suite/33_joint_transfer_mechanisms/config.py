"""Prespecified E33 arms: one transferable encoder per upstream run."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
DOMAINS = ('bearing', 'battery', 'milling')
JOINT_ARMS = ('joint_control', 'joint_famo', 'joint_private_dynamics',
              'joint_dataset_heads', 'joint_private_both')
SINGLE_ARMS = tuple(f'single_{d}_{kind}' for d in DOMAINS for kind in ('control', 'dataset_heads'))
ARMS = JOINT_ARMS + SINGLE_ARMS
FRACTIONS = (.1, .2, 1.)
SEEDS = (42, 43, 44, 45, 46)


def settings(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return dict(domains=DOMAINS if arm.startswith('joint_') else (arm.split('_')[1],),
                famo=arm == 'joint_famo',
                private_dynamics=arm in ('joint_private_dynamics', 'joint_private_both'),
                dataset_heads=arm.endswith('dataset_heads') or arm == 'joint_private_both')


def control_for(arm, domain):
    return f'single_{domain}_' + ('dataset_heads' if settings(arm)['dataset_heads'] else 'control')
