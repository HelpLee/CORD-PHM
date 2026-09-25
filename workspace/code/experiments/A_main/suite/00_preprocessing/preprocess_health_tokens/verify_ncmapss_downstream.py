"""Audit every source flight/label and reproduce local features for every engine."""
import json
from pathlib import Path
import h5py
import numpy as np
from .prepare_ncmapss_downstream import CODE,CHANNELS,descriptors,sha256

def main():
    p=CODE/'data_phm/processed_health_tokens/engine/ncmapss_ds02_downstream_health_tokens.npz'
    raw=CODE/'data_phm/raw/Engine/N-CMAPSS_DS02-006.h5'
    with np.load(p) as z,h5py.File(raw) as f:
        x=z['x_health'];mask=z['feature_mask'];u=z['sample_engine_id'];cy=z['sample_flight_index'];sp=z['sample_official_split']
        assert x.shape==(648,14,64,26)
        assert np.isfinite(x).all() and np.all(x[~mask]==0)
        assert np.all(z['c_mask']) and np.all(z['token_mask'])
        assert set(u)=={2,5,10,16,18,20,11,14,15}
        for split in ['dev','test']:
            a=f['A_'+split][:,:2].astype(int)
            pairs=np.unique(a,axis=0)
            assert set(map(tuple,pairs))==set(zip(u[sp==split],cy[sp==split]))
            for i in np.flatnonzero(sp==split):
                lo,hi=z['sample_source_row_range'][i]
                assert np.all(a[lo:hi]==[u[i],cy[i]])
                y=f['Y_'+split][lo:hi]
                assert np.all(y==z['y_rul'][i])
        cols=[[v.decode() for v in f['X_s_var'][:]].index(c) for c in CHANNELS]
        checked=0
        for engine in np.unique(u):
            ids=np.flatnonzero(u==engine)
            assert np.all(np.diff(cy[ids])>0)
            assert np.allclose(z['y_rul_norm'][ids],z['y_rul'][ids]/max(z['y_rul'][ids].max(),1))
            for i in [ids[0],ids[-1]]:
                lo,hi=z['sample_source_row_range'][i];edges=np.linspace(0,hi-lo,65,dtype=int)
                for c in [0,13]:
                    for j in [0,32,63]:
                        y=f['X_s_'+sp[i]][lo+edges[j]:lo+edges[j+1],cols[c]]
                        values,valid=descriptors(y)
                        assert np.array_equal(values,x[i,c,j]) and np.array_equal(valid,mask[i,c,j])
                        assert np.isclose(values[-1],y[-1]-y[0],rtol=1e-6,atol=1e-6)
                        checked+=1
        report=dict(passed=True,flights=len(x),engines=len(set(u)),reproduced_channel_windows=checked,
                    all_source_flights_and_labels_verified=True,npz_sha256=sha256(p))
        p.with_suffix('.verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))

if __name__=='__main__':main()
