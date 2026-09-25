"""Respect supplied feature observability; retain legacy protocol as a control."""
import hashlib,json
from pathlib import Path
import numpy as np
from data import OUT,Store,RESERVED
from io_utils import resilient_write as write

class ObservedStore(Store):
    def normalize(self,train,kind):
        assert not getattr(self,'down',False) or RESERVED[self.domain] not in train
        if self.all_features_observed:return super().normalize(train,kind)
        assert kind in ('channel_asinh','pooled_zscore')
        ids=self.rows(train);x=np.asarray(self.x[ids]);valid=self.feature_mask[ids]&(self.tm[ids]&self.cm[ids,:,None])[...,None]
        centers=[];scales=[]
        for c in range(x.shape[1] if kind=='channel_asinh' else 1):
            values=x[:,c] if kind=='channel_asinh' else x
            observed=valid[:,c] if kind=='channel_asinh' else valid
            cc=[];ss=[]
            for f in range(x.shape[-1]):
                v=values[...,f][observed[...,f]]
                if not len(v):center,scale=0.,1.
                elif kind=='channel_asinh':
                    q25,center,q75=np.percentile(v,[25,50,75]);scale=q75-q25
                else:center,scale=v.mean(dtype=np.float64),v.std(dtype=np.float64)
                cc.append(center);ss.append(scale if scale>1e-6 else 1.)
            centers.append(cc);scales.append(ss)
        return np.asarray(centers,np.float32)[:,None,:],np.asarray(scales,np.float32)[:,None,:]

    def transform(self,rows,norm,kind):
        x,cm,tm=super().transform(rows,norm,kind)
        x[~self.feature_mask[rows]]=0
        return x,cm,tm

def observed_store(store):
    assert type(store) is Store,'Observability variant currently supports local tokens only'
    with np.load(store.path/'arrays.npz') as z:rows=z['source_rows']
    signature=dict(source=store.info['signature'],rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
    dest=OUT/'feature_mask_cache'/store.path.parent.name/store.name
    if (dest/'manifest.json').exists():
        assert json.loads((dest/'manifest.json').read_text())['signature']==signature
    else:
        with np.load(Path(store.info['signature']['path']),allow_pickle=True) as z:
            explicit='feature_mask' in z.files
            fm=z['feature_mask'][rows].astype(bool) if explicit else np.broadcast_to((store.tm&store.cm[...,None])[...,None],store.x.shape).copy()
        dest.mkdir(parents=True,exist_ok=True);np.save(dest/'feature_mask.npy',fm)
        write(dest/'manifest.json',dict(signature=signature,explicit_source_mask=explicit))
    store.feature_mask=np.load(dest/'feature_mask.npy',mmap_mode='r')
    assert store.feature_mask.shape==store.x.shape
    store.all_features_observed=bool(store.feature_mask[store.tm&store.cm[...,None]].all())
    store.__class__=ObservedStore
    return store
