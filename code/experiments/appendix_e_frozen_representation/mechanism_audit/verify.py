"""Independent numerical checks for the exploratory mechanism audit."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[1]/'source_pretraining_and_transfer'
OUT = HERE/'outputs'


def main():
    reports = [json.loads((OUT/f'{d}.json').read_text()) for d in ('bearing','battery','milling')]
    arrays = list(OUT.glob('*_cross_device.npz'))
    assert len(arrays)==30
    for path in arrays:
        with np.load(path) as data:
            assert not np.isin(data['source_neighbor_ids'], data['target_ids']).any()
            assert data['query_error_k5'].shape==data['target_y'].shape
            assert np.isfinite(data['query_error_k5']).all()
    verified=0
    for package in ('20_downstream_milling_three_domain_adapter','21_downstream_milling_only_adapter'):
        saved = json.loads((SUITE/package/'results.json').read_text())['rows']
        for row in saved:
            if row['arm']!='frozen_probe':
                continue
            folder=SUITE/package/'runtime'/f"adapter_transfer_seed{row['seed']}_{round(row['fraction']*100)}pct"/'frozen_probe'
            with np.load(folder/'predictions.npz') as d:
                error=np.asarray(d['pred'],float)-np.asarray(d['y'],float)
            np.testing.assert_allclose(np.sqrt(np.mean(error**2)),row['metrics']['rmse'],rtol=1e-6,atol=1e-8)
            np.testing.assert_allclose(error.mean(),row['metrics']['bias'],rtol=1e-6,atol=1e-8)
            np.testing.assert_allclose(np.mean(error**2),error.mean()**2+np.mean((error-error.mean())**2),rtol=1e-10,atol=1e-12)
            verified+=1
    output={'cross_device_neighbor_files_checked':len(arrays), 'source_target_disjoint':True,
            'milling_frozen_prediction_files_recomputed':verified,
            'cached_E_embeddings_and_labels_checked_during_extraction':True,
            'embedding_max_difference':{r['domain']:{v:x['E1_embedding_max_absolute_difference'] for v,x in r['variants'].items()} for r in reports}}
    (OUT/'verification.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
