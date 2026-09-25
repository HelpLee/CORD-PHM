# Raw Patch downstream preprocessing

This pipeline creates a controlled input-representation ablation for only the
three evaluated downstream datasets. It never changes or overwrites the
canonical Handcrafted-26 files under `data_phm/processed_health_tokens`.

```powershell
cd code
python -m preprocess_health_tokens.build_downstream_raw_patches --include bearing battery milling
```

Outputs are written beside the Handcrafted directory under
`data_phm/processed_raw_patches/{bearing,battery,milling}`. Each output inherits
the exact sample order, labels, lifecycle identities, `token_mask`, and
`c_mask` from its paired Handcrafted NPZ. The paired file SHA256 is embedded in
the raw artifact.

The raw arrays are deliberately not normalized during preprocessing. Every
downstream seed/budget must fit its variable-wise scaler using its training
fold only. This prevents validation/test leakage and preserves amplitude as a
potential degradation signal.

Capacity-matched array contract:

- `x_raw_local [N,C,64,26]`
- `x_raw_global [N,C,26]`

`C` remains the original physical-channel axis. XJTU and PHM2010 use native
sensor amplitude; CALCE uses its voltage degradation waveform while the
capacity coordinate only defines the exact inherited window boundaries. Every
window is linearly resampled without learned parameters. Downstream then uses
the same source-only channel/position median-IQR, asinh, clipping and
`LayerNorm(26) -> Linear(26,96)` stem as Handcrafted-26.
