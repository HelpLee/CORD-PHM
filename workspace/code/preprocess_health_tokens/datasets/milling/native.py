"""Read original archives; source wear/RUL is metadata, never an input feature."""
from pathlib import Path
import io
import json
import zipfile
import numpy as np
from preprocess_health_tokens.common.bearing_health_token_utils import extract_local_feature_sequence, FEATURE_NAMES
from preprocess_health_tokens.common.raw_token_contract import validate_raw_health_tokens, raw_contract_metadata

ROOT = Path(__file__).resolve().parents[3] / 'data_phm' / 'raw' / 'Milling'


def feature_blocks(signal, fs):
    """32 nonoverlapping, native-sample windows covering the complete cut.

    Separate rate groups use their own physical time axis, without resampling.
    The final 32 token slots are structural padding.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[1] < 32*8 or not np.isfinite(signal).all():
        raise ValueError('Missing/nonfinite/short native milling waveform')
    edges = np.linspace(0, signal.shape[1], 33, dtype=int)
    x = np.zeros((signal.shape[0],64,26), np.float32)
    for j,(a,b) in enumerate(zip(edges[:-1], edges[1:])):
        features, _ = extract_local_feature_sequence(signal[:,a:b], fs=fs, window_points=b-a, stride=b-a)
        x[:,j] = features[:,0]
    return x


def finish(dataset, records, out_root, audit, partial=False, extra_meta=None):
    if not records:
        raise ValueError(f'No accepted native records for {dataset}')
    x = np.stack([r.pop('x') for r in records])
    # A native source can legitimately omit a recorded sensor group for one
    # observation.  Preserve that fact through both masks; never substitute
    # zero-valued channels as if they were observed.
    cm = np.stack([
        np.asarray(r.pop('c_mask', np.ones(x.shape[1], dtype=bool)), dtype=bool)
        for r in records
    ])
    if cm.shape != x.shape[:2]:
        raise ValueError(f'Invalid native channel-mask shape: {cm.shape}, expected {x.shape[:2]}')
    tm = np.zeros(x.shape[:3], bool)
    tm[:, :, :32] = cm[:, :, None]
    validation = validate_raw_health_tokens(x,tm,cm)
    meta = raw_contract_metadata()
    meta.update(lifecycle_policy='lifecycle_selected_raw_v1', dataset=dataset,
                native_sensor_input=True, window_definition='32 contiguous equal-sample windows per native rate group; no resampling',
                audit=audit, raw_validation=validation, debug_partial=partial)
    if extra_meta:
        meta.update(extra_meta)
    payload = dict(x_health=x,token_mask=tm,c_mask=cm,feature_names=np.asarray(FEATURE_NAMES),
                   feature_median=np.zeros(26,np.float32),feature_iqr=np.ones(26,np.float32),
                   meta_json=np.asarray(json.dumps(meta)))
    for key in records[0]:
        payload[key] = np.asarray([r[key] for r in records])
    if partial:
        payload['endpoint_observed'][:] = False
        payload['y_rul'] = np.full(len(records),np.nan)
    path = Path(out_root) / (dataset+'_health_tokens.npz')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**payload)
    path.with_suffix('.lifecycle.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return dict(dataset=dataset,out_path=str(path),num_samples=len(records),units=len(set(payload['sample_unit_id'])))


def process_luh(cfg=None, *, out_root, limit_segments=None):
    import pandas as pd
    import h5py
    root = ROOT/'luh_tool_wear'
    table = pd.read_csv(root/'filelist.csv')
    audit=[]; records=[]
    with zipfile.ZipFile(root/'luh_tool_wear_v3.zip') as z:
        members={Path(n).name:n for n in z.namelist() if n.endswith('.h5')}
        if len(members)!=6418 or set(table.filename)!=set(members):
            raise ValueError('LUH archive does not match the full official index')
        for (machine,tool), rows in table.groupby(['machine','tool']):
            rows=rows.sort_values(['cumulated_tool_contact_time','run'])
            if not np.array_equal(rows.run.to_numpy(),np.arange(1,len(rows)+1)):
                raise ValueError(f'LUH missing or unordered runs: M{machine}T{tool}')
            # Runs are repeated on different machines; machine+tool is the physical lifecycle.
            ok=len(rows)>1 and rows.wear.iloc[0]<20 and rows.wear.iloc[-1]>=140
            audit.append(dict(unit=f'M{machine}T{tool}',status='accepted_upstream_ssl' if ok else 'excluded',
                              snapshots=len(rows),first_wear=float(rows.wear.iloc[0]),last_wear=float(rows.wear.iloc[-1])))
            if not ok: continue
            end=float(rows.cumulated_tool_contact_time.max())
            for row in rows.itertuples():
                if limit_segments is not None and len(records)>=limit_segments: break
                with h5py.File(io.BytesIO(z.read(members[row.filename])),'r') as f:
                    sig=np.stack([f['signals_sensor/force_sensor_'+a][()].reshape(-1) for a in 'xyz'])
                    t=f['signals_sensor/time_sensor'][()].reshape(-1)
                    dt=np.diff(t)
                    if not np.allclose(dt,1/25000,rtol=.02,atol=1e-8): raise ValueError('LUH sensor time mismatch')
                    if int(f['labels/tool'][()].item())!=int(tool) or int(f['labels/run'][()].item())!=int(row.run):
                        raise ValueError('LUH index/HDF5 identity mismatch')
                    x=feature_blocks(sig,25000.)
                unit=f'M{machine}T{tool}'
                records.append(dict(x=x,sample_unit_id=unit,sample_group_id=unit,sample_run_id=unit,
                    sample_condition_id=f'M{machine}', sample_snapshot_index=int(row.run),
                    sample_source_relpath=members[row.filename],sample_segment_id=row.filename,
                    sample_chunk_id=0,fs=25000.,order_value=float(row.cumulated_tool_contact_time),
                    y_rul=end-float(row.cumulated_tool_contact_time),eol_order=end,endpoint_observed=True,
                    wear_um=float(row.wear),rul_unit='contact_minutes'))
                if len(records)%100==0: print(f'LUH {len(records)} cuts',flush=True)
    return finish('luh_milling',records,out_root,audit,limit_segments is not None,
                  extra_meta=dict(dataset_role='upstream_self_supervised_r2f_sequence',
                                  upstream_eligible=True,
                                  label_usage='lifecycle metadata retained for audit; upstream SSL does not consume RUL or wear labels'))


def process_piecuch(cfg=None, *, out_root, limit_segments=None):
    import pandas as pd
    root=ROOT/'piecuch_2025'
    table=pd.read_csv(root/'FeatureAndMetadata_Milling.csv',sep=';',header=1)
    metadata=pd.read_excel(root/'metadata.xlsx').set_index('ExperimentIndex')
    audit=[];records=[]
    with zipfile.ZipFile(root/'raw_data.zip') as z:
        members={Path(n).stem:n for n in z.namelist() if n.endswith('.csv')}
        if set(members)!=set(table.FileName) or len(members)!=968:
            raise ValueError('Piecuch raw archive/index mismatch')
        for tool,rows in table.groupby('TollIndex'):
            rows=rows.sort_values('NumberOfCycle')
            order=rows.NumberOfCycle.to_numpy(int)
            eols=order+rows.CycleToFailure.to_numpy(int)
            complete=len(order)>1 and np.array_equal(order,np.arange(1,int(eols[0])+1)) and np.all(eols==eols[0])
            audit.append(dict(unit=str(tool),status='accepted_upstream_ssl' if complete else 'excluded',snapshots=len(rows),
                              missing_cycles=sorted(set(range(1,int(eols[0])+1))-set(order.tolist())),
                              reason='complete start-to-failure' if complete else 'missing cycles or single observation'))
            if not complete: continue
            for row in rows.itertuples():
                if limit_segments is not None and len(records)>=limit_segments: break
                label=metadata.loc[row.FileName]
                if int(label.ToolIndex)!=int(tool) or int(label.CycleToFailure)!=int(row.CycleToFailure):
                    raise ValueError('Piecuch label identity mismatch')
                with z.open(members[row.FileName]) as stream:
                    frame=pd.read_csv(stream,dtype=np.float32)
                blocks=[]
                for time_col,columns,fs in [('Timestamps - Acc',frame.columns[1:9],25000.),
                                             ('Timestamps - Current',frame.columns[10:22],500.)]:
                    valid=frame[time_col].notna()
                    t=frame.loc[valid,time_col].to_numpy(dtype=float)/1000.
                    if not np.allclose(np.diff(t),1/fs,rtol=.5,atol=1e-5):
                        raise ValueError(f'Piecuch timestamp mismatch: {row.FileName}/{time_col}')
                    blocks.append(feature_blocks(frame.loc[valid,columns].to_numpy().T,fs))
                unit=str(tool)
                records.append(dict(x=np.concatenate(blocks),sample_unit_id=unit,sample_group_id=unit,
                    sample_run_id=unit,sample_condition_id=str(row.MillingToolType),sample_snapshot_index=int(row.NumberOfCycle),
                    sample_source_relpath=members[row.FileName],sample_segment_id=row.FileName,sample_chunk_id=0,
                    channel_fs=np.array([25000.]*8+[500.]*12),y_rul=float(row.CycleToFailure),
                    eol_order=int(eols[0]),endpoint_observed=True,rul_unit='cut_cycles'))
                print(f'Piecuch {len(records)} cuts ({unit}/{row.NumberOfCycle})',flush=True)
    return finish('piecuch_milling',records,out_root,audit,limit_segments is not None,
                  extra_meta=dict(dataset_role='upstream_self_supervised_r2f_sequence',
                                  upstream_eligible=True,
                                  label_usage='lifecycle metadata retained for audit; upstream SSL does not consume RUL or wear labels'))


def _matwi_sensor_frame(stream):
    """Read one headerless native MATWI sensor CSV and validate its time axis."""
    import pandas as pd
    frame = pd.read_csv(stream, header=None, dtype={0:np.float32, 1:np.float32,
                        2:np.float32, 3:np.float32, 4:np.float32, 5:str})
    if frame.shape[1] != 6 or frame.shape[0] < 32*8:
        raise ValueError(f'Unexpected MATWI sensor shape: {frame.shape}')
    signal = frame.iloc[:, :5].to_numpy(dtype=np.float32).T
    if not np.isfinite(signal).all() or frame.iloc[:, 5].isna().any():
        raise ValueError('Missing/nonfinite MATWI sensor value or timestamp')
    timestamps = frame.iloc[:, 5].to_numpy(dtype=str)
    if not np.all(timestamps[1:] > timestamps[:-1]):
        raise ValueError('Nonchronological MATWI native timestamps')
    edge = pd.to_datetime([timestamps[0], timestamps[1], timestamps[-1]])
    first_dt = float((edge[1]-edge[0]).value)/1e9
    mean_dt = float((edge[-1]-edge[0]).value)/1e9/(len(timestamps)-1)
    if first_dt <= 0 or not np.isclose(first_dt, mean_dt, rtol=1e-5, atol=1e-9):
        raise ValueError(f'Inconsistent MATWI timestamp spacing: first={first_dt}, mean={mean_dt}')
    return signal, 1./mean_dt


def process_matwi(cfg=None, *, out_root, limit_segments=None):
    """Build R2F tokens from all 17 official MATWI tool lifecycles."""
    import pandas as pd
    root = ROOT/'matwi'
    labels = pd.read_csv(root/'labels.csv')
    sets = pd.read_csv(root/'sets.csv', index_col=0)
    sensor_rows = labels[labels.SensorFile.notna()].copy()
    if len(labels) != 1803 or len(sensor_rows) != 1700 or sensor_rows.SensorFile.nunique() != 1700:
        raise ValueError('MATWI labels do not match the complete official index')
    records=[]; audit=[]
    for set_id in range(1,18):
        rows = sensor_rows[sensor_rows.Set == set_id].sort_values('SensorID')
        order = rows.SensorID.to_numpy(dtype=int)
        if not np.array_equal(order, np.arange(len(rows))):
            raise ValueError(f'MATWI Set {set_id}: missing/non-contiguous sensor acquisition IDs')
        archive_path = root/'archives'/f'Set{set_id}.zip'
        with zipfile.ZipFile(archive_path) as archive:
            members = {Path(n).name:n for n in archive.namelist() if '/sensordata/' in n and n.endswith('.csv')}
            expected = set(rows.SensorName.astype(str))
            if set(members) != expected:
                raise ValueError(f'MATWI Set {set_id}: archive sensor inventory differs from labels.csv')
            endpoint = int(order[-1])
            params = sets.loc[f'Set {set_id}']
            condition = '|'.join(f'{k}={params[k]}' for k in ('Vc','n','fz','Vf','Ae','Ap','material'))
            audit.append(dict(unit=f'Set{set_id}', status='accepted_upstream_ssl', snapshots=len(rows),
                              first_sensor_id=0, last_sensor_id=endpoint,
                              archive=archive_path.name, archive_bytes=archive_path.stat().st_size,
                              endpoint_evidence='publisher states each set is one tool run to failure'))
            for row in rows.itertuples():
                if limit_segments is not None and len(records)>=limit_segments: break
                member = members[str(row.SensorName)]
                with archive.open(member) as stream:
                    signal, fs = _matwi_sensor_frame(stream)
                x = feature_blocks(signal, fs)
                snapshot = int(row.SensorID)
                records.append(dict(x=x, sample_unit_id=f'Set{set_id}', sample_group_id=f'Set{set_id}',
                    sample_run_id=f'Set{set_id}', sample_condition_id=condition,
                    sample_snapshot_index=snapshot, sample_source_relpath=f'archives/{archive_path.name}::{member}',
                    sample_segment_id=str(row.SensorName), sample_chunk_id=0,
                    channel_fs=np.full(5,fs), order_value=float(snapshot),
                    y_rul=float(endpoint-snapshot), eol_order=endpoint, endpoint_observed=True,
                    wear_um=float(row.wear) if np.isfinite(row.wear) else np.nan,
                    rul_unit='sensor_acquisition_index'))
                if len(records)%100==0: print(f'MATWI {len(records)} sensor cuts (Set{set_id}/{snapshot})',flush=True)
        if limit_segments is not None and len(records)>=limit_segments: break
    return finish('matwi_milling',records,out_root,audit,limit_segments is not None,
                  extra_meta=dict(dataset_role='upstream_self_supervised_r2f_sequence',
                                  upstream_eligible=True,
                                  label_usage='lifecycle metadata retained for audit; upstream SSL does not consume RUL or wear labels'))


def process_nonastreda_snapshot(cfg=None, *, out_root, limit_segments=None):
    """Tokenize the public Nonastreda raw-force observations.

    This is intentionally an auxiliary self-supervised source, rather than an
    R2F dataset: its 512 observations are ordered by tool/run/block and carry
    measured flank wear, but the release does not document a terminal failure
    event.  The wear is retained only as audit metadata and is never an input
    feature or an RUL target.
    """
    import re
    import pandas as pd
    from scipy.io import loadmat

    root = ROOT / 'nonastreda' / 'extracted' / 'Nonastreda Multimodal Dataset for Identifying Tool Wear Condition'
    labels = pd.read_csv(root / 'labels_reg.csv').set_index('id')
    raw = np.asarray(loadmat(root / 'forces_xyz_raw.mat', squeeze_me=True)['baseDatastore'], dtype=object)
    if raw.shape != (512, 5):
        raise ValueError(f'Unexpected Nonastreda datastore shape: {raw.shape}')
    pattern = re.compile(r'^T(?P<tool>\d+)R(?P<run>\d+)B(?P<block>\d+)$')
    parsed=[]
    for row in raw:
        sample_id = str(row[0]).removesuffix('.jpg')
        match = pattern.fullmatch(sample_id)
        if match is None:
            raise ValueError(f'Nonastreda invalid sample id: {sample_id!r}')
        if sample_id not in labels.index:
            raise ValueError(f'Nonastreda raw/label mismatch: {sample_id}')
        signal = np.asarray(row[3], dtype=np.float32)
        if signal.ndim != 2 or signal.shape[0] != 3 or signal.shape[1] < 32 * 8:
            raise ValueError(f'Nonastreda invalid raw force shape: {sample_id}/{signal.shape}')
        if not np.isfinite(signal).all():
            raise ValueError(f'Nonastreda nonfinite raw force: {sample_id}')
        parsed.append((int(match['tool']), int(match['run']), int(match['block']), sample_id, signal))
    if len({p[3] for p in parsed}) != len(parsed) or set(labels.index) != {p[3] for p in parsed}:
        raise ValueError('Nonastreda labels do not match the complete raw force inventory')

    records=[]; audit=[]
    for tool in sorted({p[0] for p in parsed}):
        rows=sorted((p for p in parsed if p[0] == tool), key=lambda p: (p[1], p[2]))
        order=[(p[1], p[2]) for p in rows]
        monotonic=all(b > a for a,b in zip(order,order[1:]))
        if not monotonic or len(rows) < 2:
            raise ValueError(f'Nonastreda nonchronological source order: T{tool}')
        runs=sorted({p[1] for p in rows})
        audit.append(dict(unit=f'T{tool}', status='accepted_upstream_ssl',
                          snapshots=len(rows), run_ids=runs,
                          endpoint_evidence='not supplied; excluded from strict R2F allowlist',
                          source_sampling_hz=1000.0))
        for _, run, block, sample_id, signal in rows:
            if limit_segments is not None and len(records) >= limit_segments:
                break
            label=labels.loc[sample_id]
            records.append(dict(
                x=feature_blocks(signal, 1000.0), sample_unit_id=f'T{tool}', sample_group_id=f'T{tool}',
                sample_run_id=f'T{tool}', sample_condition_id='nonastreda_industrial_milling',
                sample_snapshot_index=run * 10 + block, sample_source_relpath='extracted/Nonastreda Multimodal Dataset for Identifying Tool Wear Condition/forces_xyz_raw.mat',
                sample_segment_id=sample_id, sample_chunk_id=0, channel_fs=np.full(3, 1000.0),
                order_value=float(run) + float(block) / 10.0, y_rul=np.nan, eol_order=np.nan,
                endpoint_observed=False, wear_um=float(label['flank_wear']), rul_unit='unavailable_auxiliary',
            ))
        if limit_segments is not None and len(records) >= limit_segments:
            break
    return finish('nonastreda_milling', records, out_root, audit,
                  partial=True, extra_meta=dict(
                      source_doi='10.17632/m892d2wtzh.1', source_license='CC BY 4.0',
                      dataset_role='upstream_self_supervised_snapshot_sequence',
                      upstream_eligible=True,
                      raw_channels=['Fx', 'Fy', 'Fz'], source_sampling_hz=1000.0,
                  ))


def _qit_sort_key(member):
    """Publisher's chronological `month-day-observation` filename order."""
    import re
    match=re.fullmatch(r'Force and torque data/(\d+)-(\d+)-(\d+)\.txt', member)
    if match is None:
        raise ValueError(f'Unexpected QIT force filename: {member}')
    return tuple(map(int, match.groups()))


def _qit_active_force_tokens(archive, member):
    """Read one archived 10 kHz cut and remove source-defined idle windows."""
    import subprocess
    import pandas as pd
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import MinMaxScaler

    process=subprocess.Popen(['tar.exe', '-xOf', str(archive), member], stdout=subprocess.PIPE)
    try:
        frame=pd.read_csv(process.stdout, sep='\t', usecols=['Time','Fx','Fy','Fz'],
                          dtype=np.float32)
    finally:
        if process.stdout is not None:
            process.stdout.close()
    if process.wait() != 0:
        raise ValueError(f'Could not extract QIT raw force member: {member}')
    values=frame.to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] < 32 * 8:
        raise ValueError(f'Unexpected QIT raw force shape: {member}/{values.shape}')
    if not np.isfinite(values).all():
        raise ValueError(f'QIT nonfinite raw force: {member}')
    # The accompanying author script clusters 500-sample Fx energies and
    # removes the low-energy class.  Determine that class by its fitted mean
    # instead of relying on an arbitrary GMM label number.
    window=500
    usable=(len(values)//window)*window
    energy=np.square(values[:usable,1], dtype=np.float64).reshape(-1,window).sum(axis=1,keepdims=True)
    scaled=MinMaxScaler().fit_transform(energy)
    labels=GaussianMixture(n_components=2,covariance_type='full',tol=1e-5,random_state=0).fit_predict(scaled)
    low=int(np.argmin([float(energy[labels == label].mean()) for label in range(2)]))
    retained=np.repeat(labels != low, window)
    # Preserve a trailing partial block; it is treated as active only when
    # nonzero signal is present, so no synthetic timestamps are introduced.
    if usable < len(values):
        tail=np.any(values[usable:,1:] != 0,axis=1)
        retained=np.concatenate([retained,tail])
    signal=values[retained,1:].T
    if signal.shape[1] < 32 * 8:
        raise ValueError(f'QIT active force sequence too short: {member}')
    return feature_blocks(signal,10000.0), int(retained.sum()), int(len(values))


def process_qit_cemc_force_only_legacy(cfg=None, *, out_root, limit_segments=None):
    """Legacy three-force-channel QIT builder; excluded from the registry."""
    import subprocess
    import pandas as pd

    root=ROOT/'qit_cemc'
    archive=root/'qit_cemc_v3.rar'
    wear_path=root/'metadata'/'tool wear.xls'
    if not archive.exists() or not wear_path.exists():
        raise FileNotFoundError('QIT-CEMC archive or extracted official wear workbook is absent')
    listing=subprocess.run(['tar.exe','-tf',str(archive)],capture_output=True,text=True,check=True).stdout.splitlines()
    members=sorted((item for item in listing if item.startswith('Force and torque data/')),key=_qit_sort_key)
    if len(members) != 68 or len(set(members)) != 68:
        raise ValueError(f'QIT expected 68 chronological force members, found {len(members)}')
    wear=pd.read_excel(wear_path,header=None).iloc[4:,:13].copy()
    wear=wear[wear.iloc[:,0].notna()]
    cycles=wear.iloc[:,0].to_numpy(dtype=int)
    if not np.array_equal(cycles,np.arange(1,69)):
        raise ValueError('QIT wear workbook does not provide cycle 1..68')
    side_vb=wear.iloc[:,[1,4,7,10]].to_numpy(dtype=float)
    if not np.isfinite(side_vb).all():
        raise ValueError('QIT side-tooth wear contains nonfinite values')

    records=[]; audit=[]
    for cycle,member,vb in zip(cycles,members,side_vb):
        if limit_segments is not None and len(records) >= limit_segments:
            break
        x,kept,total=_qit_active_force_tokens(archive,member)
        records.append(dict(
            x=x,sample_unit_id='QIT_CEMC_T1',sample_group_id='QIT_CEMC_T1',sample_run_id='QIT_CEMC_T1',
            sample_condition_id='Ti6Al4V_circumferential_milling',sample_snapshot_index=int(cycle),
            sample_source_relpath=f'qit_cemc_v3.rar::{member}',sample_segment_id=Path(member).stem,
            sample_chunk_id=0,channel_fs=np.full(3,10000.0),order_value=float(cycle),
            y_rul=np.nan,eol_order=np.nan,endpoint_observed=False,wear_um=float(np.mean(vb)*1000.0),
            rul_unit='unavailable_auxiliary',
        ))
        audit.append(dict(cycle=int(cycle),member=member,raw_samples=total,active_samples=kept,
                          mean_side_vb_mm=float(np.mean(vb))))
        print(f'QIT-CEMC {cycle}/68: {member}',flush=True)
    return finish('qit_cemc_milling_force_only_legacy',records,out_root,audit,
                  partial=True,extra_meta=dict(
                      source_doi='10.6084/m9.figshare.27323346.v3',source_license='CC BY 4.0',
                      dataset_role='auxiliary_self_supervised_snapshot_only',
                      raw_channels=['Fx','Fy','Fz'],source_sampling_hz=10000.0,
                      activity_filter='author-script-derived 500-sample Fx energy GMM; low-energy class removed',
                  ))


def _qit_vibration_member(force_member):
    """Map the publisher's force filename to the matching vibration/sound cut."""
    import re
    match = re.fullmatch(r'Force and torque data/(\d+)-(\d+)-(\d+)\.txt', force_member)
    if match is None:
        raise ValueError(f'Unexpected QIT force filename: {force_member}')
    month, day, cut = map(int, match.groups())
    return f'Vibration and sound data/{month:02d}-{day:02d}-{cut:02d}.csv'


def _qit_read_member(
    archive, member, *, sep, usecols, names, fs_fallback, encoding='utf-8',
    drop_first_column=False,
):
    """Read one QIT archived stream and derive its rate from its time column when possible."""
    import subprocess
    import pandas as pd

    process = subprocess.Popen(['tar.exe', '-xOf', str(archive), member], stdout=subprocess.PIPE)
    try:
        kwargs = dict(sep=sep, usecols=usecols, encoding=encoding)
        if not drop_first_column:
            kwargs['dtype'] = np.float32
        frame = pd.read_csv(process.stdout, **kwargs)
    finally:
        if process.stdout is not None:
            process.stdout.close()
    if process.wait() != 0:
        raise ValueError(f'Could not extract QIT member: {member}')
    if drop_first_column:
        # One publisher file uses a different byte encoding for the unit symbol
        # in its header.  Its first column is always the timestamp, followed by
        # the same four recorded sensor channels in the documented order.
        if frame.shape[1] < 2:
            raise ValueError(f'Unexpected QIT stream schema: {member}/{list(frame.columns)}')
        frame = frame.iloc[:, 1:]
        if names is not None:
            if frame.shape[1] != len(names):
                raise ValueError(f'Unexpected QIT stream schema: {member}/{list(frame.columns)}')
            frame.columns = list(names)
    if names is not None and list(frame.columns) != list(names):
        raise ValueError(f'Unexpected QIT channel schema: {member}/{list(frame.columns)}')
    values = frame.to_numpy(dtype=np.float32)
    if values.ndim != 2 or (names is not None and values.shape[1] != len(names)) or values.shape[0] < 32 * 8:
        raise ValueError(f'Unexpected QIT stream shape: {member}/{values.shape}')
    if not np.isfinite(values).all():
        raise ValueError(f'QIT nonfinite stream: {member}')
    return values.T, float(fs_fallback)


def process_qit_cemc_fullchannels_snapshot(cfg=None, *, out_root, limit_segments=None):
    """Build a separate all-channel QIT-CEMC auxiliary NPZ.

    The fixed physical representation is eight channels: Fx/Fy/Fz/Mz plus
    Vx/Vy/Vz/sound.  ``c_mask`` records an unavailable source stream without
    inventing any replacement values.
    """
    import subprocess
    import pandas as pd

    root = ROOT / 'qit_cemc'
    archive = root / 'qit_cemc_v3.rar'
    wear_path = root / 'metadata' / 'tool wear.xls'
    if not archive.exists() or not wear_path.exists():
        raise FileNotFoundError('QIT-CEMC archive or extracted official wear workbook is absent')
    listing = subprocess.run(['tar.exe', '-tf', str(archive)], capture_output=True, text=True, check=True).stdout.splitlines()
    force_members = sorted((item for item in listing if item.startswith('Force and torque data/')), key=_qit_sort_key)
    vibration_members = {item for item in listing if item.startswith('Vibration and sound data/')}
    if len(force_members) != 68 or len(vibration_members) != 67:
        raise ValueError(f'QIT inventory changed: force={len(force_members)}, vibration={len(vibration_members)}')
    wear = pd.read_excel(wear_path, header=None).iloc[4:, :13].copy()
    wear = wear[wear.iloc[:, 0].notna()]
    cycles = wear.iloc[:, 0].to_numpy(dtype=int)
    if not np.array_equal(cycles, np.arange(1, 69)):
        raise ValueError('QIT wear workbook does not provide cycle 1..68')
    side_vb = wear.iloc[:, [1, 4, 7, 10]].to_numpy(dtype=float)
    if not np.isfinite(side_vb).all():
        raise ValueError('QIT side-tooth wear contains nonfinite values')

    records, audit = [], []
    force_names = ['Fx', 'Fy', 'Fz', 'Mz']
    vibration_names = ['AI1-01[m/s²]', 'AI1-02[m/s²]', 'AI1-03[m/s²]', 'AI1-07[Pa]']
    for cycle, force_member, vb in zip(cycles, force_members, side_vb):
        if limit_segments is not None and len(records) >= limit_segments:
            break
        force, force_fs = _qit_read_member(
            archive, force_member, sep='\t', usecols=['Fx', 'Fy', 'Fz', 'Mz'],
            names=force_names, fs_fallback=10000.0,
        )
        vibration_member = _qit_vibration_member(force_member)
        if vibration_member not in vibration_members:
            xlsx_member = vibration_member.removesuffix('.csv') + '.xlsx'
            if xlsx_member in vibration_members:
                vibration_member = xlsx_member
        has_vibration = vibration_member in vibration_members
        vibration_error = None
        if has_vibration:
            try:
                vibration, vibration_fs = _qit_read_member(
                    archive, vibration_member, sep=',', usecols=None,
                    names=vibration_names, fs_fallback=10000.0, encoding='latin-1',
                    drop_first_column=True,
                )
                x = np.concatenate([feature_blocks(force, force_fs), feature_blocks(vibration, vibration_fs)], axis=0)
                c_mask = np.ones(8, dtype=bool)
                channel_fs = np.asarray([force_fs] * 4 + [vibration_fs] * 4, dtype=np.float32)
            except (UnicodeError, ValueError, pd.errors.ParserError) as error:
                # Preserve the valid force/torque observation.  The publisher
                # archive has malformed bytes/rows in a small number of its
                # vibration streams; treating a partially parsed waveform as
                # observed would be less valid than an explicit mask.
                has_vibration = False
                vibration_error = f'{type(error).__name__}: {error}'
        if not has_vibration:
            x = np.concatenate([feature_blocks(force, force_fs), np.zeros((4, 64, 26), dtype=np.float32)], axis=0)
            c_mask = np.asarray([True] * 4 + [False] * 4, dtype=bool)
            channel_fs = np.asarray([force_fs] * 4 + [np.nan] * 4, dtype=np.float32)
        records.append(dict(
            x=x, c_mask=c_mask, sample_unit_id='QIT_CEMC_T1', sample_group_id='QIT_CEMC_T1',
            sample_run_id='QIT_CEMC_T1', sample_condition_id='Ti6Al4V_circumferential_milling',
            sample_snapshot_index=int(cycle),
            sample_source_relpath=f'qit_cemc_v3.rar::{force_member}' + (f'|{vibration_member}' if has_vibration else ''),
            sample_segment_id=Path(force_member).stem, sample_chunk_id=0, channel_fs=channel_fs,
            order_value=float(cycle), y_rul=np.nan, eol_order=np.nan, endpoint_observed=False,
            wear_um=float(np.mean(vb) * 1000.0), rul_unit='unavailable_auxiliary',
        ))
        audit.append(dict(cycle=int(cycle), status='accepted_upstream_ssl',
                          force_member=force_member, vibration_member=vibration_member,
                          vibration_present=bool(has_vibration), vibration_error=vibration_error,
                          mean_side_vb_mm=float(np.mean(vb))))
        print(f'QIT-CEMC full channels {cycle}/68: {force_member}', flush=True)

    # ``finish`` expects a dense channel mask.  Preserve missing source streams
    # explicitly after it writes the common contract payload.
    # The canonical QIT artifact now is the all-channel representation.  The
    output_name = str(getattr(cfg, 'dataset', '') or 'qit_cemc_milling')
    result = finish(output_name, records, out_root, audit, partial=True, extra_meta=dict(
        source_doi='10.6084/m9.figshare.27323346.v3', source_license='CC BY 4.0',
        dataset_role='upstream_self_supervised_snapshot_sequence',
        upstream_eligible=True,
        raw_channels=['Fx', 'Fy', 'Fz', 'Mz', 'Vx', 'Vy', 'Vz', 'sound'],
        channel_groups={'force_torque': force_names, 'vibration_sound': ['Vx', 'Vy', 'Vz', 'sound']},
        source_sampling_hz={'force_torque': 10000.0, 'vibration_sound': 10000.0},
        missing_channel_policy='unavailable or malformed source stream: c_mask=False; no imputation',
        activity_filter='none; each native stream is tokenized over its complete recorded cut',
    ))
    return result


def _hmotp_wear_table(path):
    """Load one publisher wear table without interpreting a row index as wear."""
    import pandas as pd
    frame=pd.read_csv(path)
    candidates=[]
    for name in frame.columns:
        values=pd.to_numeric(frame[name],errors='coerce')
        if values.notna().all():
            candidates.append((str(name),values.to_numpy(dtype=float)))
    mean_named=[item for item in candidates if item[0].strip().lower() == 'mean']
    named=[item for item in candidates if 'wear' in item[0].lower() or 'vb' in item[0].lower()]
    selected=(mean_named or named or candidates)
    if len(selected) != 1:
        raise ValueError(f'HMoTP ambiguous wear column in {path.name}: {[x[0] for x in selected]}')
    values=selected[0][1]
    if len(values) != 100 or not np.isfinite(values).all():
        raise ValueError(f'HMoTP expected 100 finite wear labels in {path.name}')
    return values,selected[0][0]


def process_hmotp_upstream(cfg=None, *, out_root, limit_segments=None):
    """Build HMoTP upstream SSL snapshots from all three raw tool sequences.

    Measured wear is retained only as audit metadata.  It is neither an input
    nor an upstream training target, and no undocumented RUL is manufactured.
    """
    import re
    import pandas as pd

    root=ROOT/'hmotp'
    records=[]; audit=[]
    expected_columns=['F_x','F_y','F_z','M_z','A_x','A_y','A_z']
    for tool in range(1,4):
        unit=f'T{tool:02d}'
        signal_dir=root/f'CuttingSignals{unit}'
        files=sorted(signal_dir.glob(f'{unit}_*.csv'),key=lambda p:int(re.search(r'_(\d+)$',p.stem).group(1)))
        expected=[f'{unit}_{index:03d}.csv' for index in range(1,101)]
        if [path.name for path in files] != expected:
            raise ValueError(f'HMoTP raw inventory must be complete and ordered: {unit}')
        wear,wear_column=_hmotp_wear_table(root/f'ToolWear{unit}.csv')
        audit.append(dict(unit=unit,status='accepted_upstream_ssl',snapshots=100,
                          raw_channels=expected_columns,source_sampling_hz=50000.0,
                          label_column=wear_column,
            endpoint_evidence='not supplied; measured wear is audit metadata only'))
        for index,path in enumerate(files,start=1):
            if limit_segments is not None and len(records) >= limit_segments:
                break
            frame=pd.read_csv(path,dtype=np.float32)
            if list(frame.columns) != expected_columns:
                raise ValueError(f'HMoTP unexpected channel schema: {path.name}/{list(frame.columns)}')
            signal=frame.to_numpy(dtype=np.float32).T
            if signal.shape[1] < 32*8 or not np.isfinite(signal).all():
                raise ValueError(f'HMoTP invalid raw waveform: {path.name}')
            records.append(dict(
                x=feature_blocks(signal,50000.0),sample_unit_id=unit,sample_group_id=unit,sample_run_id=unit,
                sample_condition_id='Ti6Al4V_thin_walled_high_speed_milling',sample_snapshot_index=index,
                sample_source_relpath=str(path.relative_to(root)).replace('\\','/'),sample_segment_id=path.stem,
                sample_chunk_id=0,channel_fs=np.full(7,50000.0),order_value=float(index),
                y_rul=np.nan,eol_order=np.nan,endpoint_observed=False,wear_um=float(wear[index-1]*1000.0),
                rul_unit='unavailable_upstream_ssl',
            ))
        if limit_segments is not None and len(records) >= limit_segments:
            break
    return finish('hmotp_milling',records,out_root,audit,partial=True,
                  extra_meta=dict(source_doi='10.1016/j.rcim.2024.102723',
                                  dataset_role='upstream_self_supervised_snapshot_sequence',
                                  upstream_eligible=True,
                                  raw_channels=expected_columns,source_sampling_hz=50000.0,
                                  wear_label_note='wear_um retained for audit only; never used by upstream SSL'))


NASA_CHANNELS = ('smcAC','smcDC','vib_table','vib_spindle','AE_table','AE_spindle')
NASA_DOWNSTREAM_CASES = (5,8,9,10,11,13,15,16)


def _nasa_case_admission(rows):
    runs=np.asarray([int(x.run) for x in rows])
    times=np.asarray([float(x.time) for x in rows])
    vb=np.asarray([float(x.VB) for x in rows])
    complete_runs=len(rows)>=5 and np.array_equal(runs,np.arange(1,len(rows)+1))
    chronological=np.isfinite(times).all() and np.all(np.diff(times)>0)
    endpoint=np.isfinite(vb[-1]) and vb[-1]>=.5
    signal_ok=True
    for row in rows:
        lengths=[]
        for name in NASA_CHANNELS:
            values=np.asarray(getattr(row,name)).reshape(-1)
            lengths.append(len(values))
            # The released acquisition is voltage-like and clean records stay
            # within +/-9.996. Case 2/run 1 (~1e34) and Case 12/run 1 (~2771)
            # are source corruption, not large physical degradation responses.
            signal_ok &= (len(values)>=32*8 and np.isfinite(values).all()
                          and float(np.max(np.abs(values)))<=10.1)
        signal_ok &= len(set(lengths))==1
    ok=bool(complete_runs and chronological and endpoint and signal_ok)
    reasons=[]
    if not complete_runs: reasons.append('fewer than five observations or missing run numbers')
    if not chronological: reasons.append('nonchronological source time')
    if not endpoint: reasons.append('terminal measured VB missing or below 0.50 mm')
    if not signal_ok: reasons.append('missing/nonfinite/unaligned or physically invalid sensor sequence')
    return ok,dict(snapshots=len(rows),first_run=int(runs[0]),last_run=int(runs[-1]),
                   first_time=float(times[0]),last_time=float(times[-1]),
                   terminal_vb_mm=float(vb[-1]) if np.isfinite(vb[-1]) else None,
                   reason='; '.join(reasons) if reasons else 'complete case with observed wear endpoint')


def process_nasa_upstream(cfg=None, *, out_root, limit_segments=None):
    """Build the audited NASA/UC Berkeley sequences for upstream SSL."""
    from scipy.io import loadmat
    path=ROOT/'nasa_milling'/'extracted'/'mill.mat'
    mill=np.asarray(loadmat(path,squeeze_me=True,struct_as_record=False)['mill']).reshape(-1)
    if len(mill)!=167 or {int(x.case) for x in mill}!=set(range(1,17)):
        raise ValueError('NASA Milling source is not the documented 167-record/16-case release')
    records=[];audit=[];accepted=[]
    for case in range(1,17):
        rows=sorted((x for x in mill if int(x.case)==case),key=lambda x:int(x.run))
        ok,info=_nasa_case_admission(rows)
        info.update(unit=f'Case{case}',status='accepted_upstream_ssl' if ok else 'excluded')
        audit.append(info)
        if not ok: continue
        accepted.append(case)
        endpoint=float(rows[-1].time)
        for source_index,row in ((i,x) for i,x in enumerate(mill) if int(x.case)==case):
            if limit_segments is not None and len(records)>=limit_segments: break
            signal=np.stack([np.asarray(getattr(row,name),dtype=np.float32).reshape(-1)
                             for name in NASA_CHANNELS])
            x=feature_blocks(signal,250.)
            run=int(row.run); source_time=float(row.time)
            condition=f'DOC={float(row.DOC)}|feed={float(row.feed)}|material={int(row.material)}'
            records.append(dict(x=x,sample_unit_id=f'Case{case}',sample_group_id=f'Case{case}',
                sample_run_id=f'Case{case}',sample_condition_id=condition,sample_snapshot_index=run,
                sample_source_relpath=f'extracted/mill.mat::mill[{source_index}]',
                sample_segment_id=f'Case{case}_run{run}',sample_chunk_id=0,
                channel_fs=np.full(6,250.),order_value=source_time,y_rul=endpoint-source_time,
                eol_order=endpoint,endpoint_observed=True,wear_mm=float(row.VB),
                rul_unit='source_machining_time'))
        if limit_segments is not None and len(records)>=limit_segments: break
    if limit_segments is None and tuple(accepted)!=NASA_DOWNSTREAM_CASES:
        raise ValueError(f'NASA upstream admission changed: {accepted}')
    return finish('nasa_milling',records,out_root,audit,limit_segments is not None,
                  extra_meta=dict(dataset_role='upstream_self_supervised_r2f_sequence',sampling_hz=250.,
                    signal_provenance='released 250 Hz hardware-filtered/RMS sensor sequences',
                    endpoint_rule='contiguous case; terminal measured VB >=0.50 mm',
                    upstream_eligible=True,
                    label_usage='lifecycle metadata retained for audit; upstream SSL does not consume RUL or wear labels'))
