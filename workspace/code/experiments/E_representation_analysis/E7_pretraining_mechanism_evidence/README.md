# E7 — Why pretraining helps

This package separates the first paper mechanism question from the Joint versus
Single-domain question in E6. It compares end-to-end Scratch with the frozen
Three-domain pretrained encoder using the existing five paired downstream
seeds, then compares a seed42 random encoder with the self-supervised pretrained
encoder before any RUL-head training.

Outputs:

- `P1`: paired Scratch minus Frozen RMSE at 10/20/100% labels;
- `P2`: reduction in squared systematic bias and centered prediction error;
- `P3`: 10%-label unseen-device prediction trajectories;
- `P4`: frozen representation health-distance correlation, within-device 5-NN
  lifecycle error, and health-conditioned device/health geometry ratio.

The representation audit uses the actual archived downstream readout:
Snapshot Projector for Bearing/Battery and FinalNorm hidden for Milling. For
quadratic pairwise metrics each device is uniformly capped at 250 snapshots.
Scaling is fitted on source devices. No encoder or RUL model is trained.

Run `python make_figures.py`. Results are written only under `outputs/`.
