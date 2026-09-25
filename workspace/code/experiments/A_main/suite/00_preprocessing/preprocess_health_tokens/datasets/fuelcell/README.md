# Fuel-cell HealthToken preprocessing

This directory follows the same separation used by the bearing and battery
preprocessing code:

- each dataset adapter defines native observation boundaries and reads raw
  physical channels;
- `datasets/registry.py` owns window/scaling configuration;
- `preprocess_health_tokens/main.py` owns the shared 26-feature extraction,
  padding, masks, robust scaling and NPZ writer.

## Fixed transfer membership

Upstream-only datasets:

1. `zuo_dynamic_snapshot_fuelcell`: one complete active FC-DLC interval;
2. `realhyfc_stack_durability_upstream_fuelcell`: one non-overlapping dense
   10-hour operating segment;
3. `fraunhofer_catalyst_ast_snapshot_fuelcell`: one uninterrupted block of ten
   native catalyst AST potential cycles (620 samples at 10 Hz). A block never
   crosses an acquisition or ageing-checkpoint boundary. Each physical Cell is
   one independent `sample_run_id`, matching the bearing/battery whole-run
   upstream validation contract.

Downstream-only dataset:

- `ieee_phm_cell_downstream_fuelcell`: one continuous dense four-hour segment
  of one measured cell. It is never eligible for masked pretraining.

Every observation uses voltage, current and electrical power as three physical
channels. Each channel is split into at most 64 local temporal windows and each
window is converted to the shared-within-fuelcell 26 statistical/spectral
descriptors. Missing token positions are right-padded with exact zeros and have
`token_mask=False`.

All three upstream NPZ files independently fit median/IQR on their own valid
tokens and clip scaled valid values to `[-20, 20]`. Strict mode aborts on any
malformed physical observation instead of silently dropping it. The downstream
NPZ remains unscaled so each LOOCV fold can fit median/IQR using only its four
training cells.

Generate all fixed FuelCell files from the `code` directory:

```bash
python -m preprocess_health_tokens.main --fuelcell-all
```

Generate only the Fraunhofer third upstream source:

```bash
python -m preprocess_health_tokens.main \
  --include fraunhofer_catalyst_ast_snapshot_fuelcell
```
