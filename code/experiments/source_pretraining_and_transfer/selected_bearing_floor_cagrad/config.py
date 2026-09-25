"""SELECTED_MODEL: prespecified follow-up to CONDITIONAL_ROUTING, with matched and fixed references."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
COMPONENT_ROUTING = SUITE / 'component_gradient_routing'
CONDITIONAL_ROUTING = SUITE / 'conditional_gradient_routing'
DOMAINS = ('bearing', 'battery', 'milling')
FRACTIONS = (.1, .2)
SEEDS = (42, 43, 44, 45, 46)
NEW_JOINT_ARMS = ('joint_lowbearing_pcgrad', 'joint_lowbearing_cagrad',
              'joint_bearingfloor_cagrad', 'joint_lowbearing_floor_cagrad')
PARENT_ARMS = ('joint_conditional_pcgrad', 'joint_conditional_cagrad')
JOINT_ARMS = NEW_JOINT_ARMS + PARENT_ARMS
ARMS = JOINT_ARMS + ('single_bearing_lowbearing', 'single_battery_conditional', 'single_battery_component')
MAX_EPOCHS = 2000


def continuation_source(arm):
    if arm in PARENT_ARMS or arm == 'single_battery_conditional':
        return CONDITIONAL_ROUTING / 'models' / arm
    if arm == 'single_battery_component':
        return COMPONENT_ROUTING / 'models' / arm
    return None


def settings(arm):
    if arm not in ARMS: raise ValueError(arm)
    low = 'lowbearing' in arm
    return dict(domains=DOMAINS if arm.startswith('joint_') else (arm.split('_')[1],),
        shared_reconstruction_fraction=1., aggregation='pcgrad' if arm.endswith('pcgrad') else 'mean' if arm.startswith('single_') else 'cagrad',
        reconstruction_routing=dict(bearing=.1 if low else .3, battery=1. if arm.endswith('_component') else .3, milling=1.),
        bearing_floor='floor' in arm,
        floor_definition='dot(g_bearing,direction) >= squared_norm(g_bearing)/number_of_domains before clipping/AdamW',
        methods_sha256=hashlib.sha256((HERE/'methods.py').read_bytes()).hexdigest(),
        experiment='SELECTED_MODEL', architecture='unchanged CONDITIONAL_ROUTING', downstream_fractions=FRACTIONS)


def control_for(arm, domain):
    if domain == 'bearing' and 'lowbearing' in arm:
        return HERE, 'single_bearing_lowbearing'
    if domain == 'battery':
        return HERE, 'single_battery_conditional'
    if domain == 'bearing':
        return CONDITIONAL_ROUTING, 'single_bearing_conditional'
    return COMPONENT_ROUTING, 'single_milling_component'


def parent_for(arm):
    return 'joint_conditional_pcgrad' if arm.endswith('pcgrad') else 'joint_conditional_cagrad'
