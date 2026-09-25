from pathlib import Path

HERE=Path(__file__).resolve().parent
SUITE=HERE.parent
E34=SUITE/'34_shared_degradation_gradient_routing'
E35=SUITE/'35_lowlabel_conditional_sharing'
E37=SUITE/'37_bearing_sharing_followup'
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
        reference_interval=5,reference_batch=8,experiment='E38B',bearing_floor=True,
        gate_objective='normalized first-order auxiliary usefulness on independent TRAIN dynamics gradients; no differentiation through CAGrad/AdamW',
        selection='same legacy macro validation and patience30',seed=42)

def source_for(model,domain):
    if model=='scratch': return None
    if model=='e37_joint': return E37/'models/joint_bearingfloor_cagrad/training/encoder.pt'
    if model=='e37_single':
        return {'bearing':E35/'models/single_bearing_conditional/training/encoder.pt',
                'battery':E37/'models/single_battery_conditional/training/encoder.pt',
                'milling':E34/'models/single_milling_component/training/encoder.pt'}[domain]
    if model.removeprefix('smoke_') not in ARMS: raise ValueError(model)
    return HERE/'models'/model/'training/encoder.pt'

def existing_result(model,domain,fraction):
    if fraction==1. or model=='scratch': return None
    source=source_for(model,domain)
    return source.parents[3]/'downstream'/source.parents[1].name/domain/f'p{round(fraction*100)}/results.json'
