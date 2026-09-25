from pathlib import Path

HERE=Path(__file__).resolve().parent
SUITE=HERE.parent
COMPONENT_ROUTING=SUITE/'component_gradient_routing'
CONDITIONAL_ROUTING=SUITE/'conditional_gradient_routing'
SELECTED_MODEL=SUITE/'selected_bearing_floor_cagrad'
DOMAINS=('bearing','battery','milling')
SEEDS=(42,43,44,45,46)
METHODS=('cosine','learned','layerwise')
ARMS=tuple('joint_'+m for m in METHODS)+tuple('single_'+d+'_'+m for m in METHODS for d in DOMAINS)
MAX_EPOCHS=2000

def settings(arm):
    if arm not in ARMS: raise ValueError(arm)
    return dict(domains=DOMAINS if arm.startswith('joint_') else (arm.split('_')[1],),
        aggregation='cagrad',shared_reconstruction_fraction=1.,method=arm.split('_')[-1],
        rho_init=.5,rho_bounds=(.05,1.),ema=.9,gate_lr=.05,entropy_weight=.01,
        reference_interval=5,reference_batch=8,experiment='ADAPTATION_CONTROLSB',bearing_floor=True,
        gate_objective='normalized first-order auxiliary usefulness on independent TRAIN dynamics gradients; no differentiation through CAGrad/AdamW',
        selection='same legacy macro validation and patience30',seed=42)

def source_for(model,domain):
    if model=='scratch': return None
    if model=='multi_domain': return SELECTED_MODEL/'models/joint_bearingfloor_cagrad/training/encoder.pt'
    if model=='single_domain':
        return {'bearing':CONDITIONAL_ROUTING/'models/single_bearing_conditional/training/encoder.pt',
                'battery':SELECTED_MODEL/'models/single_battery_conditional/training/encoder.pt',
                'milling':COMPONENT_ROUTING/'models/single_milling_component/training/encoder.pt'}[domain]
    if model.removeprefix('smoke_') not in ARMS: raise ValueError(model)
    return HERE/'models'/model/'training/encoder.pt'

def existing_result(model,domain,fraction):
    if fraction==1. or model=='scratch': return None
    source=source_for(model,domain)
    return source.parents[3]/'downstream'/source.parents[1].name/domain/f'p{round(fraction*100)}/results.json'
