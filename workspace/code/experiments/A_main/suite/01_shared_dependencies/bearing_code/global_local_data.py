"""Global raw-snapshot feature cache plus local HealthToken views."""
import hashlib,importlib,json,sys
from pathlib import Path

import numpy as np

from data import OUT,ROOT,Store,write,read_selected
sys.path.insert(1,str(Path(__file__).resolve().parents[2]/'00_preprocessing'))
from preprocess_health_tokens.main import _read_segment_signal
from preprocess_health_tokens.datasets.bearing.xjtu_snapshot_features import compute_global_raw_features
from preprocess_health_tokens.common.bearing_health_token_utils import clean_signal

FPT2={'Bearing2_1':455,'Bearing2_2':48,'Bearing2_3':327,'Bearing2_4':32,'Bearing2_5':141}
LINEAR=('mean','abs_mean','std','rms','peak','ptp','min','max','root_amplitude')
QUADRATIC=('var','energy')


def file_sha(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def build_condition2():
    source=ROOT/'code/data_phm/processed_health_tokens/bearing/xjtu_downstream_bearing_health_tokens.npz'
    dest=OUT/'global_local_cache/xjtu_condition2'
    signature=dict(source=str(source),source_sha256=file_sha(source),bytes=source.stat().st_size,
                   condition='37.5Hz11kN',fpt_ordinal_1based=FPT2,version=1)
    if (dest/'manifest.json').exists():
        manifest=json.loads((dest/'manifest.json').read_text())
        previous=manifest['signature']
        # Legacy caches recorded an absolute Windows path and mtime.  Those
        # fields are not portable even when the uploaded NPZ is byte-identical.
        # Validate the stable protocol fields and, when available, content
        # hash.  Keep the legacy manifest byte-for-byte unchanged because
        # downstream global-feature caches key their provenance by its hash.
        for field in ('bytes','condition','fpt_ordinal_1based','version'):
            assert previous[field]==signature[field],field
        if previous.get('source_sha256') is not None:
            assert previous['source_sha256']==signature['source_sha256']
        return Store(dest)
    with np.load(source,allow_pickle=True) as z:
        units=z['sample_unit_id'].astype(str);file_ids=z['sample_file_id'].astype(str)
        selected=[];counts={}
        for unit,fpt in FPT2.items():
            ids=np.flatnonzero(units==unit)
            ids=ids[np.argsort(np.asarray([int(file_ids[i]) for i in ids]),kind='stable')]
            keep=ids[fpt-1:];assert len(keep)>=8
            selected.extend(keep.tolist());counts[unit]=len(keep)
        selected=np.asarray(sorted(selected),dtype=np.int64)
        tm=np.asarray(z['token_mask'][selected],bool);cm=np.asarray(z['c_mask'][selected],bool)
        sub_units=units[selected];order=np.asarray([int(v) for v in file_ids[selected]],float)
        y=np.zeros(len(selected),np.float32)
        for unit in FPT2:
            ids=np.flatnonzero(sub_units==unit);ids=ids[np.argsort(order[ids])]
            y[ids]=np.linspace(1,0,len(ids),dtype=np.float32)
        features=z['feature_names'].astype(str).tolist();meta=json.loads(str(z['meta_json'].item()))
    x=read_selected(source,selected);valid=tm&cm[...,None];x[~valid]=0
    dest.mkdir(parents=True);np.save(dest/'x.npy',x)
    np.savez(dest/'arrays.npz',tm=tm,cm=cm,y=y,units=sub_units,order=order,source_rows=selected)
    write(dest/'manifest.json',dict(signature=signature,domain='bearing',downstream=True,shape=list(x.shape),
          counts=counts,features=features,metadata=meta,
          rul='Linear 1->0 over each post-FPT trajectory; FPT map is historical ordinal_1based'))
    return Store(dest)


def raw_globals(store):
    """Compute exact 26 features from complete raw signal for cached rows."""
    if store.info['signature'].get('row_scoped_globals'):
        return row_scoped_globals(store)
    selection=getattr(store,'sensor_selection',None)
    suffix='' if selection is None else '__'+selection['mode']+'_'+'_'.join(map(str,selection['indices']))
    dest=OUT/'global_feature_cache'/(store.name+suffix)
    with np.load(store.path/'arrays.npz') as z:source_rows=z['source_rows']
    source=Path(store.info['signature'].get('path',store.info['signature'].get('source')))
    signature=dict(store_manifest=file_sha(store.path/'manifest.json'),source=str(source),sensor_selection=selection,
                   source_rows_sha=hashlib.sha256(source_rows.tobytes()).hexdigest(),version=2,
                   missing_value_policy='common clean_signal before whole-snapshot features')
    if (dest/'manifest.json').exists():
        if json.loads((dest/'manifest.json').read_text())['signature']==signature:
            return np.load(dest/'global.npy',mmap_mode='r')
    with np.load(source,allow_pickle=True) as z:
        meta=json.loads(str(z['meta_json'].item()));segment_ids=z['sample_segment_id'][source_rows].astype(str)
    module=importlib.import_module(meta['source_module']);segments={s.segment_id:s for s in module.build_segments()}
    result=np.zeros((len(source_rows),store.x.shape[1],store.x.shape[-1]),np.float32);cache={}
    for i,sid in enumerate(segment_ids):
        if sid not in cache:
            signal,fs=_read_segment_signal(module,segments[sid]);cache[sid]=compute_global_raw_features(clean_signal(signal),fs)
        g=cache[sid]
        if selection is not None:g=g[selection['indices']]
        assert len(g)<=result.shape[1]
        result[i,:len(g)]=g
        if (i+1)%100==0:print('GLOBAL_FEATURES',store.name,i+1,len(source_rows),flush=True)
    assert np.isfinite(result[store.cm]).all()
    dest.mkdir(parents=True,exist_ok=True);np.save(dest/'global.npy',result)
    write(dest/'manifest.json',dict(signature=signature,shape=list(result.shape),
          extraction='Exact common 26 features over the complete raw snapshot signal; one vector per channel.'))
    return np.load(dest/'global.npy',mmap_mode='r')


def row_scoped_globals(store):
    """Global descriptors for precisely the current row's acquisition extent."""
    dest=OUT/'global_feature_cache'/'row_scoped_v4'/store.name
    source=Path(store.info['signature']['path'])
    signature=dict(store_manifest=file_sha(store.path/'manifest.json'),
                   source=str(source),version='row_scoped_v1')
    if (dest/'manifest.json').exists():
        if json.loads((dest/'manifest.json').read_text())['signature']==signature:
            return np.load(dest/'global.npy',mmap_mode='r')
    with np.load(store.path/'arrays.npz') as arrays: rows=arrays['source_rows']
    with np.load(source,allow_pickle=True) as z:
        meta=json.loads(str(z['meta_json'].item()))
        all_segments=z['sample_segment_id'].astype(str)
        ids=all_segments[rows]
        starts=z['token_start_points'][rows]
        widths=z['window_points'][rows]
        unique,counts=np.unique(all_segments,return_counts=True)
        multiplicity=dict(zip(unique,counts))
    module=importlib.import_module(meta['source_module'])
    segments={s.segment_id:s for s in module.build_segments()}
    result=np.zeros((len(rows),store.x.shape[1],26),np.float32)
    last_id=None
    for i,sid in enumerate(ids):
        if sid!=last_id:
            signal,fs=_read_segment_signal(module,segments[sid])
            signal=clean_signal(signal); last_id=sid
        current=signal
        if multiplicity[sid]>1:
            positions=starts[i][starts[i]>=0]
            begin=int(positions.min()); end=int(positions.max()+widths[i])
            assert 0<=begin<end<=signal.shape[-1], (sid,begin,end,signal.shape)
            current=signal[:,begin:end]
        g=compute_global_raw_features(current,fs)
        assert len(g)<=result.shape[1]
        result[i,:len(g)]=g
        if (i+1)%100==0: print('ROW_GLOBAL_FEATURES',store.name,i+1,len(rows),flush=True)
    assert np.isfinite(result[store.cm]).all()
    dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/'global.npy',result)
    write(dest/'manifest.json',dict(signature=signature,shape=list(result.shape)))
    return np.load(dest/'global.npy',mmap_mode='r')


def battery_globals(store):
    """Observed-window aggregate in the same 26-feature physical coordinates."""
    selection=getattr(store,'sensor_selection',None)
    suffix='' if selection is None else '__'+selection['mode']+'_'+'_'.join(map(str,selection['indices']))
    dest=OUT/'global_feature_cache'/(store.name+suffix)
    with np.load(store.path/'arrays.npz') as z:source_rows=z['source_rows']
    source=Path(store.info['signature']['path'])
    signature=dict(store_manifest=file_sha(store.path/'manifest.json'),source=str(source),
                   source_rows_sha=hashlib.sha256(source_rows.tobytes()).hexdigest(),version=1,
                   sensor_selection=selection,domain=store.domain,
                   extraction='observed local-token mean; exact full-cycle duration/capacity replacement for battery when available')
    if (dest/'manifest.json').exists():
        manifest=json.loads((dest/'manifest.json').read_text())
        if manifest['signature']==signature:return np.load(dest/'global.npy',mmap_mode='r')
    fm=np.asarray(getattr(store,'feature_mask',np.broadcast_to((store.tm&store.cm[...,None])[...,None],store.x.shape)))
    observed=fm&(store.tm&store.cm[...,None])[...,None];x=np.asarray(store.x)
    count=observed.sum(2).clip(min=1);result=(x*observed).sum(2)/count
    with np.load(source,allow_pickle=True) as z:
        if store.domain=='battery' and 'cycle_duration_s' in z.files:result[:,0,22]=z['cycle_duration_s'][source_rows]
        if store.domain=='battery' and 'cycle_capacity_ah' in z.files:result[:,0,23]=z['cycle_capacity_ah'][source_rows]
    result=np.asarray(result,np.float32);assert np.isfinite(result).all()
    dest.mkdir(parents=True,exist_ok=True);np.save(dest/'global.npy',result)
    write(dest/'manifest.json',dict(signature=signature,shape=list(result.shape),
          extraction='One 26D whole-snapshot summary per channel from observed local windows; battery additionally uses exact cycle duration/capacity.'))
    return np.load(dest/'global.npy',mmap_mode='r')


class GlobalLocalStore:
    def __init__(self,local,calibrate=True,channels=None):
        self.channels=max(2,local.x.shape[1]) if channels is None else int(channels)
        assert local.x.shape[1]<=self.channels
        self.local=local;self.global_x=battery_globals(local) if local.domain in ('battery','milling') else raw_globals(local);self.x=local.x;self.tm=local.tm;self.cm=local.cm
        self.y=local.y;self.units=local.units;self.order=local.order;self.groups=local.groups
        self.domain=local.domain;self.down=local.down;self.name=local.name;self.path=local.path;self.info=local.info
        fm=np.asarray(getattr(local,'feature_mask',np.broadcast_to((self.tm&self.cm[...,None])[...,None],self.x.shape)))
        self.feature_mask=np.zeros((len(fm),self.channels,*fm.shape[2:]),bool);self.feature_mask[:,:fm.shape[1]]=fm
        self.global_feature_mask=np.zeros((len(fm),self.channels,fm.shape[-1]),bool)
        self.global_feature_mask[:,:fm.shape[1]]=np.any(fm&(self.tm&self.cm[...,None])[...,None],axis=2)
        self.calibration={}
        if calibrate and local.domain=='bearing':self._calibrate()

    def _calibrate(self):
        names=self.info['features'];rms=names.index('rms');g=np.array(self.global_x,copy=True);x=np.array(self.x,copy=True)
        report={}
        for unit,ids in self.groups.items():
            refs=[]
            for c in range(x.shape[1]):
                if not self.cm[ids[0],c]:continue
                scale=float(g[ids[0],c,rms]);assert scale>1e-12
                for f in LINEAR:x[ids,c,:,names.index(f)]/=scale;g[ids,c,names.index(f)]/=scale
                for f in QUADRATIC:x[ids,c,:,names.index(f)]/=scale**2;g[ids,c,names.index(f)]/=scale**2
                refs.append(dict(channel=c,row=int(ids[0]),global_rms=scale))
            report[unit]=refs
        self.x=x;self.global_x=g;self.calibration=report

    def rows(self,units):return self.local.rows(units)
    def split(self):return self.local.split()
    def sequences(self,units,length=10):return self.local.sequences(units,length)
    def sequences_with_mask(self,units,length=10):return self.local.sequences_with_mask(units,length)
    def eligible(self,units):
        if hasattr(self.local,'eligible'):return self.local.eligible(units)
        if not self.down:return {u:self.groups[u][1:] for u in units if len(self.groups[u])>1}
        return {u:self.groups[u] for u in units}

    def normalize(self,train):
        ids=self.rows(train);local=np.asarray(self.x[ids]);valid=self.tm[ids]&self.cm[ids,:,None]
        lc=[];ls=[];gc=[];gs=[]
        for c in range(self.channels):
            if c>=self.x.shape[1]:
                lc.append(np.zeros(self.x.shape[-1]));ls.append(np.ones(self.x.shape[-1]));gc.append(np.zeros(self.x.shape[-1]));gs.append(np.ones(self.x.shape[-1]));continue
            local_center=[];local_scale=[];global_center=[];global_scale=[]
            for f in range(self.x.shape[-1]):
                lv=local[:,c,:,f][valid[:,c]&self.feature_mask[ids,c,:,f]]
                gv=np.asarray(self.global_x[ids,c,f])[self.cm[ids,c]&self.global_feature_mask[ids,c,f]]
                for vals,centers,scales in ((lv,local_center,local_scale),(gv,global_center,global_scale)):
                    if not len(vals):med,spread=0.,1.
                    else:q25,med,q75=np.percentile(vals,[25,50,75]);spread=q75-q25
                    centers.append(med);scales.append(spread if spread>1e-6 else 1.)
            lc.append(local_center);ls.append(local_scale);gc.append(global_center);gs.append(global_scale)
        return tuple(np.asarray(v,np.float32) for v in (lc,ls,gc,gs))

    def transform(self,rows,norm):
        lc,ls,gc,gs=norm;n=len(rows);actual=self.x.shape[1]
        x=np.zeros((n,self.channels,*self.x.shape[2:]),np.float32);g=np.zeros((n,self.channels,self.x.shape[-1]),np.float32)
        cm=np.zeros((n,self.channels),bool);tm=np.zeros((n,self.channels,self.tm.shape[-1]),bool)
        x[:,:actual]=(np.array(self.x[rows])-lc[:actual,None,:])/ls[:actual,None,:]
        g[:,:actual]=(np.array(self.global_x[rows])-gc[:actual])/gs[:actual]
        cm[:,:actual]=self.cm[rows];tm[:,:actual]=self.tm[rows];x=np.arcsinh(x);g=np.arcsinh(g)
        x=np.clip(x,-20,20);g=np.clip(g,-20,20);valid=tm&cm[...,None]
        feature_mask=np.zeros_like(x,dtype=bool);feature_mask[:,:actual]=self.feature_mask[rows,:actual]
        global_mask=np.zeros_like(g,dtype=bool);global_mask[:,:actual]=self.global_feature_mask[rows,:actual]
        x[~(valid[...,None]&feature_mask)]=0;g[~(cm[...,None]&global_mask)]=0;assert np.isfinite(x).all() and np.isfinite(g).all()
        return x,g,cm,tm
