# Experiment 19: cluster GradNorm downstream comparison

Six independent jobs: target domain (bearing/battery/milling) × pretraining source
(single/three). Each requests one A100 on `gpua100`, 8 CPUs and 32 GB host RAM,
and runs labels 10%, 20%, 100% × seeds 42–46 sequentially (15 fits/job, 90 total).
Walltimes: bearing 1 hour, battery 2 hours, milling 3 hours. Historical local
frozen-probe totals were roughly 8–14 minutes per bearing source, 37–40 minutes
per battery source; Milling seed fits were several minutes. These limits allow
headroom for a different selected encoder and validation-based stopping.

All tasks use BF16 and the original downstream networks and protocols: frozen
encoder, domain-specific head, 200 maximum epochs, validation every 2 epochs,
patience 15, interval source validation, no test-set checkpoint selection. Milling
uses all seven PHM2010 channels. GradNorm is an upstream loss-balancing method,
not an extra downstream architecture. No scratch or full-finetune jobs are added.

Upstream model = experiment 17's `encoder.pt` (validation best), not last.pt or
experiment 18's fixed-update model. Each downstream job copies a stable snapshot
only after its own upstream status is complete, and records its hash and epoch.
Single-domain tasks depend only on their respective upstream, and three-domain
tasks depend only on the shared three-domain upstream. `afterok` dependencies
prevent training on incomplete/failed upstreams. Failed dependencies cancel the
dependent job instead of silently substituting another checkpoint.

Run `python submit_cluster.py` from this directory on cluster. Active/completed jobs
are skipped. `submissions.json` records job IDs/dependencies. Result directories
are isolated under `runs/<domain>/<single|three>`. One CPU-only aggregation job
runs after all six successes and produces `comparison.json` and `comparison.md`
with mean ± sample SD for RMSE, MAE, R2. Best values are bold and underlined.

Portable bearing/milling inputs are exported from existing canonical local
caches by `export_inputs.py`, not recomputed from a different feature recipe.
Each bundle verifies canonical NPZ SHA256, cache file SHA256, train-only scalers,
and normalized reference tensors (rtol 1e-6, atol 1e-7 to allow cross-platform
float32 arcsinh rounding; scalers and masks are checked exactly). Cluster readers do not modify shared
caches. Battery is constructed directly from the existing canonical CALCE NPZ.
Snapshot entrypoints adapt only paths/checkpoint selection; helper modules are
reused from shared code and the already-deployed experiment-15 Milling package.
The local experiment-18 queue and all old result directories remain untouched.
