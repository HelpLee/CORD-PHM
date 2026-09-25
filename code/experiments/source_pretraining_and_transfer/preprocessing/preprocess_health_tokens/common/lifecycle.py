"""Explicit lifecycle admission; never equate end of recording with failure."""
import json
import numpy as np

POLICY = 'lifecycle_selected_raw_v1'
BATTERY_DOWNSTREAM = ('calce_cs2_downstream_battery',)
BATTERY_UPSTREAM = ('hust_battery', 'isu_ilcc_battery', 'mich_exp_battery', 'xjtu_battery')
BATTERIES = BATTERY_DOWNSTREAM + BATTERY_UPSTREAM
BEARING_DOWNSTREAM = ('xjtu',)
BEARING_UPSTREAM = ('femto', 'ferrara', 'unsw', 'kaist')
BEARINGS = BEARING_DOWNSTREAM + BEARING_UPSTREAM
MILLING_UPSTREAM = ('luh_milling', 'piecuch_milling', 'matwi_milling')
MILLING_DOWNSTREAM = ('nasa_milling_downstream',)
MILLING = MILLING_UPSTREAM + MILLING_DOWNSTREAM
# Raw snapshot sequences that are eligible for self-supervised masked-token
# pretraining, but are deliberately excluded from the strict R2F/EOL allowlist.
# Their publishers provide ordered observations and raw signals, but not an
# independently evidenced terminal failure boundary.
MILLING_AUXILIARY_SNAPSHOT = ('nonastreda_milling_snapshot', 'qit_cemc_milling_snapshot')
# HMoTP is a downstream wear-regression benchmark.  Its publisher supplies
# longitudinal wear measurements, not an independently defined RUL/EOL rule.
MILLING_DOWNSTREAM_WEAR = ('hmotp_milling_downstream_wear',)


def hust_published_endpoint(unit):
    """Validate the publisher's explicit RUL convention against all raw keys.

    The final recorded cycle has RUL=1; zero is the following cycle boundary.
    Do not create a synthetic zero-RUL snapshot or shift the author's labels.
    """
    import pickle
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'data_phm'/'raw'/'Battery'/'HUST_Battery'/(unit+'.pkl')
    with path.open('rb') as stream:
        record=pickle.load(stream)[unit]
    order=sorted(map(int,record['data']))
    labels={int(k):int(v) for k,v in record['rul'].items()}
    if order!=list(range(1,len(order)+1)) or set(order)!=set(labels):
        raise ValueError(f'Incomplete HUST raw/label keys: {unit}')
    ends={i+labels[i] for i in order}
    if ends!={len(order)+1} or labels[order[-1]]!=1:
        raise ValueError(f'Unexpected HUST publisher endpoint convention: {unit}')
    q={int(k):float(v)/1000. for k,v in record['dq'].items()}
    return len(order)+1,dict(source_last_cycle=order[-1],source_last_rul=1,
        source_first_capacity_ah=q[order[0]],source_last_capacity_ah=q[order[-1]],
        endpoint_evidence='publisher cycle+rul; zero at following cycle boundary')


def select_battery_records(records, dataset):
    """Apply dataset-specific capacity thresholds, independently of last record.

    CALCE: 0.88 Ah; ISU full C/5 RPT: 0.20 Ah; MICH: 50% of initial;
    XJTU: 80% of initial full capacity. HUST: verified publisher RUL labels.
    SDU stages are retained without an invented terminal-failure label.
    This is an observed capacity endpoint, not physical destruction.
    """
    units = {}
    for record in records:
        units.setdefault(record[0].unit_id, []).append(record)
    selected, endpoints, audit = [], {}, []
    for unit, rr in sorted(units.items()):
        rr.sort(key=lambda r: r[0].cycle_index)
        indices = np.array([r[0].cycle_index for r in rr])
        q = np.array([r[1]['cycle_capacity_ah'] for r in rr], dtype=float)
        if len(set(indices)) != len(indices):
            raise ValueError(f'Duplicate cycle identity: {dataset}/{unit}')
        row = dict(unit=unit, observations=len(rr), first_cycle=int(indices[0]),
                   last_cycle=int(indices[-1]), first_capacity=float(q[0]), last_capacity=float(q[-1]))
        row['missing_observation_indices'] = sorted(set(range(int(indices[0]),int(indices[-1])+1))-set(indices.tolist()))
        if dataset == 'sdu_battery':
            row.update(status='stage_only', reason='No verified terminal-failure endpoint')
            selected.extend(rr)
        elif dataset == 'hust_battery':
            end,evidence=hust_published_endpoint(unit)
            row.update(evidence)
            if indices[0]!=1 or indices[-1]!=evidence['source_last_cycle']:
                row.update(status='excluded',reason='First or terminal source snapshot failed quality checks')
            else:
                endpoints[unit]=end
                selected.extend(rr)
                row.update(status='publisher_labelled_eol',eol_cycle=end)
        else:
            ref = float(np.median(q[:5]))
            if dataset == 'xjtu_battery':
                ref = float(q[0])  # Authors explicitly measure initial capacity in cycle 1.
            # ISU nominal 0.25 Ah: only full C/5 RPT can establish capacity EOL.
            # CALCE uses nominal 1.1 Ah. MICH's source terminates at 50% initial.
            threshold = {'isu_ilcc_battery': .2,
                         'mich_exp_battery': .5*float(q[0]),
                         'calce_cs2_downstream_battery': .88}.get(dataset, .8*ref)
            confirmations = 1 if dataset in ('isu_ilcc_battery','mich_exp_battery') else 3
            minimum = 3 if dataset == 'isu_ilcc_battery' else 20
            start = 1 if dataset == 'isu_ilcc_battery' else 5
            row.update(reference_capacity=ref, threshold_capacity=threshold)
            candidates = [i for i in range(start, len(q)-confirmations+1)
                          if np.all(q[i:i+confirmations] <= threshold)
                          and (dataset == 'xjtu_battery' or np.all(np.diff(indices[i:i+confirmations]) == 1))]
            terminal_single = False
            if dataset == 'xjtu_battery':
                if indices[0] != 1 or q[-1] > threshold:
                    candidates = []
                elif not candidates and len(q)>start:
                    candidates = [len(q)-1]
                    terminal_single = True
            if len(rr) < minimum or not np.isfinite(q).all() or ref <= threshold or not candidates:
                row.update(status='excluded', reason='No confirmed observed capacity EOL; do not infer failure from last record')
            else:
                end = candidates[0]
                endpoints[unit] = int(indices[end])
                selected.extend(rr[:end+1])
                row.update(status='observed_capacity_eol', eol_cycle=int(indices[end]),
                           confirmation_cycle=int(indices[end if terminal_single else end+confirmations-1]))
        audit.append(row)
    return selected, endpoints, audit


def check_bearing_segments(dataset, segments):
    expected = {'xjtu': [123,161,158,122,52,491,161,533,42,339,2538,2496,371,1515,114],
                'femto': [2803,871,911,797,515,1637],
                'ferrara': [4917,1985,2386,669,1721,509],
                'unsw': [80,167,79,186], 'kaist': [129]}
    groups = {}
    for seg in segments:
        meta = seg.metadata
        if dataset == 'xjtu':
            meta['snapshot_index'] = int(seg.file_path.stem)
        groups.setdefault(meta['unit_id'], []).append(seg)
    # Fixed lifecycle inventories are available only for the audited R2F
    # subset.  The canonical preprocessing inventory also includes public
    # condition/snapshot datasets; their adapters retain every source segment
    # but do not invent an endpoint or a hidden fixed-count contract.
    if dataset not in expected:
        return groups
    ids = {'xjtu':[f'Bearing{c}_{b}' for c in range(1,4) for b in range(1,6)],
           'femto':[f'Bearing{c}_{b}' for c in range(1,4) for b in range(1,3)],
           'ferrara':[f'E{i}' for i in range(1,7)],
           'unsw':[f'Test {i}' for i in range(1,5)], 'kaist':['run_to_failure']}[dataset]
    if {k:len(v) for k,v in groups.items()} != dict(zip(ids,expected[dataset])):
        raise ValueError(f'Incomplete or unexpected {dataset} lifecycles: {[(k,len(v)) for k,v in groups.items()]}')
    for unit, rr in groups.items():
        order = [r.metadata['snapshot_index'] for r in rr]
        if len(set(order)) != len(order) or any(b <= a for a,b in zip(order,order[1:])):
            raise ValueError(f'Nonunique/nonchronological snapshots: {dataset}/{unit}')
        if dataset in ('femto','xjtu','ferrara') and order != list(range(1, len(rr)+1)):
            raise ValueError(f'Missing numbered snapshots: {dataset}/{unit}')
    return groups
