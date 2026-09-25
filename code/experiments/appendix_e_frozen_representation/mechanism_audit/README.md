# Exploratory mechanism audit (existing E1--E4 preserved)

Run `python analyze.py` and `python verify.py` from this directory. CPU only;
no encoder/head optimization or modification of existing experiments.

## Figures

- `outputs/cross_device_retrieval.png` and `.pdf`: all held-out device snapshots
  query only source-device snapshots (Euclidean k=5; fixed k=1,10 sensitivities
  saved in JSON). Color curves compare single-domain + Adapter and three-domain
  + Adapter. Same source-fitted normalization as the original downstream task.
- `outputs/frozen_paired_gain.png` and `.pdf`: five downstream seeds paired by
  fraction, strategy and seed. Positive gain means three-domain RMSE is lower.
- `outputs/error_decomposition.png` and `.pdf`: per-run decomposition
  `MSE = bias^2 + mean((error-bias)^2)`, averaged over five seeds. This is an
  empirical residual decomposition, NOT the theoretical bias-variance
  decomposition over random datasets or an intervention/calibration method.

All 27 fraction/arm/domain paired comparisons are saved, not only Frozen.
Retrieval uses ALL source labels for post-hoc evaluation and is not a new
label-budget benchmark. RUL is never used to extract or select neighbors.
Target neighbors are excluded by device identity, not merely by row identity.
All snapshot pairs in E3 and all query observations in E4 are dependent;
they are not independent experimental replicates. One upstream checkpoint per
setting and one target device per domain do not establish a population-level
causal mechanism or significance. Target results are exploratory, not used to
select a new encoder or a k value.

## Important architecture correction

The current Milling main experiment is `ToolAdapterTCNModel` in
`20_downstream_milling_three_domain_adapter/code/downstream_interval_val200.py`.
It consumes FinalNorm Global/Local hidden through a newly learned 192->96
token_fusion, then a tool residual adapter, three causal TCN blocks and a GRU
over 20 snapshots. It does NOT use the pretrained 96D snapshot projector.
The old multi-scale delta GRU class in the same code package is not the active
main-experiment head. E3/E4 Final refers to that unused projector for Milling.
Bearing and Battery main heads DO use the pretrained snapshot representation.

Intermediate points here are 192D concatenations; projector is 96D and includes
a nonlinear readout. Block points include domain adapters; this is not an
isolation of the shared Transformer alone. Pooled input already includes the
learned stems, channel attention pooling and positional/type embeddings.

## Current interpretation

- Bearing: cross-device 5-NN RUL discrepancy at the actual snapshot readout
  drops from 0.23263 to 0.14039 (k=1,10 agree in direction). Frozen 10% RMSE
  improvement is +0.02598, with 4/5 paired wins. Its average MSE reduction
  comes from centered residual error, not from a smaller overall offset.
- Battery: cross-device discrepancy improves slightly (0.07080 -> 0.06987),
  while Frozen 10% has 5/5 paired wins. Both offset and centered errors improve
  at 10%/20%; geometric retrieval alone does not explain the full effect.
- Milling: actual hidden readout retrieval is worse (0.10678 -> 0.11607),
  and Frozen 10%/20% has only 2/5 paired wins despite a positive mean gain.
  Mean MSE reduction is driven by bias-squared reduction; centered residual
  error increases. Do not claim uniformly better geometric health information
  or improved temporal modeling on this evidence alone.

Further causal attribution would need paired head interventions (single-state
vs temporal head with the encoder fixed), preferably additional held-out units
and independently pretrained seeds. Matched optimization steps in pretraining
are not matched total exposures: three-domain joint batch=96 vs single=32.
