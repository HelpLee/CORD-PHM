"""Physical sensor selection; no labels or development errors used at runtime.

MATWI README.md MD5 fc91a416b3e19af8e7979b0a9b3a673a matches the
publisher's KU Leuven RDR record doi:10.48804/GK6LHH. Columns 2:5 (zero
based) are force XYZ; native.py preserves this order. LUH loads force XYZ
explicitly. PHM2010 meta_json.sensor_channels identifies force XYZ first.
Piecuch has accelerometers/current, so is outside the force-only subset.
"""
import numpy as np

FORCE_CHANNELS={'luh_milling':(0,1,2),'matwi_milling':(2,3,4),
                'phm2010_milling_downstream':(0,1,2),
                # Both auxiliary corpora were preprocessed explicitly from
                # Fx/Fy/Fz, in that order.
                'nonastreda_milling_snapshot':(0,1,2),
                'qit_cemc_milling_snapshot':(0,1,2)}

def select_sensors(store, mode):
    if mode=='all' or store.domain!='milling':return store
    assert mode=='force' and store.name in FORCE_CHANNELS, store.name
    indices=list(FORCE_CHANNELS[store.name])
    assert store.x.ndim==4 and store.x.shape[1]>max(indices)
    # Advanced indexing copies only selected sensors; source arrays remain intact.
    store.x=np.asarray(store.x[:,indices]).copy()
    store.cm=store.cm[:,indices].copy();store.tm=store.tm[:,indices].copy()
    if hasattr(store,'feature_mask'):
        store.feature_mask=np.asarray(store.feature_mask[:,indices]).copy()
        assert store.feature_mask.shape==store.x.shape
        store.all_features_observed=bool(store.feature_mask[store.tm&store.cm[...,None]].all())
    store.sensor_selection=dict(mode=mode,indices=indices,channels=['force_x','force_y','force_z'])
    return store
