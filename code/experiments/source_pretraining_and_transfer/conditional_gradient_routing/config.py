"""CONDITIONAL_ROUTING: exploratory low-label follow-up; one joint encoder per treatment."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
COMPONENT_ROUTING = SUITE / 'component_gradient_routing'
DOMAINS = ('bearing', 'battery', 'milling')
FRACTIONS = (.1, .2)
SEEDS = (42, 43, 44, 45, 46)
JOINT_ARMS = ('joint_component_pcgrad', 'joint_component_cagrad',
              'joint_conditional', 'joint_conditional_pcgrad', 'joint_conditional_cagrad')
SINGLE_ARMS = ('single_bearing_conditional', 'single_battery_conditional')
ARMS = JOINT_ARMS + SINGLE_ARMS


def settings(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    conditional = 'conditional' in arm
    return dict(domains=DOMAINS if arm.startswith('joint_') else (arm.split('_')[1],),
        shared_reconstruction_fraction=1.,
        reconstruction_routing={'bearing': .3 if conditional else 1.,
                                'battery': .3 if conditional else 1., 'milling': 1.},
        aggregation='pcgrad' if arm.endswith('pcgrad') else 'cagrad' if arm.endswith('cagrad') else 'mean',
        experiment='CONDITIONAL_ROUTING', downstream_fractions=FRACTIONS,
        inference_architecture='unchanged; a single jointly pretrained encoder checkpoint')


def control_for(arm, domain):
    if 'conditional' in arm and domain != 'milling':
        return HERE, f'single_{domain}_conditional'
    return COMPONENT_ROUTING, f'single_{domain}_component'
