# Canonical non-fuelcell HealthToken preprocessing

This directory is the single source tree for every non-fuelcell NPZ below
`code/data_phm/processed_health_tokens`: 11 bearing datasets, 9 battery
datasets, 7 milling datasets, and N-CMAPSS DS02. Fuel-cell sources are
intentionally excluded from this registry and from the rebuild entry point.

`common/` contains feature extraction, masks, scaling-contract helpers and the
legacy lifecycle validators. `datasets/{bearing,battery,milling,engine}/`
contains one adapter per canonical artifact. `datasets/registry.py` is the
complete inventory and fixes the output filename and modality for each one.

## Rebuild

Never rebuild into the canonical data directory. Use an empty candidate root:

```powershell
cd code
python -m preprocess_health_tokens.main --out-root /tmp/healthtoken_rebuild
python -m preprocess_health_tokens.verify_reproduction `
  --reference-root data_phm/processed_health_tokens `
  --candidate-root /tmp/healthtoken_rebuild
```

For a subset, pass exact registry names, for example:

```powershell
python -m preprocess_health_tokens.main --include xjtu calce_cs2_downstream_battery phm2010_milling_downstream ncmapss_ds02_downstream --out-root /tmp/subset
```

Bearing, battery, and native milling adapters use the existing
`three_domain_strict_raw_v2` contract: raw descriptors are written without
learned scaling or magnitude clipping, and train-fold scaling belongs to
downstream or pretraining code. PHM2010 and N-CMAPSS retain their own complete
historical schemas, likewise without input learned scaling. `verify_reproduction`
compares every generated NPY payload byte-for-byte. It permits only the two
inevitable N-CMAPSS metadata changes, `elapsed_seconds` and `output_sha256`.

PHM2010 uses the complete historical builder restored at
`prepare_phm2010_downstream.py`. N-CMAPSS uses one native flight per row and
the 14 sensor / 64-window / 26-feature contract defined in
`prepare_ncmapss_downstream.py`. Its official dev/test field is provenance only.
All splitting decisions remain downstream responsibilities.
