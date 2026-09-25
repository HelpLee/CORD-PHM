# E6 — Unknown-device transfer evidence

This package generates four unified figures for Bearing, Battery, and Milling:

1. `F1_frozen_transfer_gain`: paired five-seed RMSE gain of Joint over
   Single-domain pretraining at 10/20/100% labels.
2. `F2_health_conditioned_geometry`: layer-wise cross-device distance at the
   same normalized-health bin (lower) and within-device health-stage separation
   (higher). Five fixed RUL bins are used. Scaling is fitted on source devices.
3. `F3_unseen_device_predictions`: 10%-label Frozen predictions on the held-out
   device, showing five-seed mean and standard deviation.
4. `F4_downstream_interface_geometry`: compact normalized summary at the actual
   archived downstream readout (Snapshot Projector for Bearing/Battery and
   FinalNorm hidden for Milling). It reports Joint/Single device distance,
   health separation, and their ratio.

Run:

```powershell
python make_figures.py
```

The script reads existing checkpoints, cached downstream datasets, five-seed
metrics, and predictions. It does not train or modify any model. Bearing,
Battery, and Milling extraction run in separate processes to isolate legacy
modules with identical Python names. PNG, PDF, and machine-readable geometry
JSON files are written only under `outputs/`.

The geometry plot tests a mechanism hypothesis. Reduced matched-health device
distance is useful only when health-stage separation is retained; either curve
alone is insufficient evidence of a transferable degradation representation.
