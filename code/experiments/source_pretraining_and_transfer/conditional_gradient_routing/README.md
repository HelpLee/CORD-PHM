# CONDITIONAL_ROUTING: low-label conditional sharing

Five joint models, each a single encoder checkpoint jointly trained on bearing,
battery and milling (LUH, MATWI, Nonastreda, QIT-CEMC, HMoTP; all channels).
Downstream: all three domains, 10% and 20%, seeds 42–46, full fine-tuning,
encoder LR 3e-4 / head LR 1e-3, identical COMPONENT_ROUTING splits and early stopping.

| Arm | Shared reconstruction fraction (bearing/battery/milling) | Aggregation |
|---|---|---|
| component_pcgrad | 1 / 1 / 1 | PCGrad |
| component_cagrad | 1 / 1 / 1 | CAGrad |
| conditional | .3 / .3 / 1 | Mean |
| conditional_pcgrad | .3 / .3 / 1 | PCGrad |
| conditional_cagrad | .3 / .3 / 1 | CAGrad |

All arms preserve COMPONENT_ROUTING independent loss normalization with equal reconstruction and
dynamics coefficients. Private reconstruction gradients and shared dynamics gradients
stay unchanged. Architecture, validation monitor, patience 30, maximum 500 epochs,
20 updates/domain/epoch and data sampling remain COMPONENT_ROUTING's settings. The 30% coefficient
is a prespecified exploratory intermediate between COMPONENT_ROUTING's tested 10% and 100%.
Domain conditioning is a training-gradient rule, not three independent encoders.

Matched controls: reuse completed COMPONENT_ROUTING component singles for the first two joint arms;
train new .3-routed bearing and battery singles for conditional arms, reusing the
identical COMPONENT_ROUTING milling component single. Also report every joint against the same
fixed COMPONENT_ROUTING component singles so weakening a control cannot masquerade as improvement.

Rationale: COMPONENT_ROUTING component has milling gains at 10%/20% (4/5 seed wins each), battery
10% gains (5/5) and bearing 10% harm. Uniform .1 routing loses milling gains. E33
auxiliary-head/FAMO modifications also fail to uniformly help low-label milling,
so this batch tests conflict handling without severing milling reconstruction.
These observations motivate hypotheses; they do not establish a causal mechanism.

Evaluation: all six low-label cells and all metrics are reported. Macro relative
RMSE change, seed wins and worst-cell degradation are included against both control
sets. No downstream test score selects an upstream checkpoint. This is exploratory
development informed by previous test performance; confirm any chosen method on
untouched devices and multiple upstream seeds before making a general claim.

Runtime: reuse fixed COMPONENT_ROUTING trainer and E33 downstream loader, imported in the CONDITIONAL_ROUTING
configuration context. The COMPONENT_ROUTING prepare-before-domain-restriction fix is required.
Smoke exercises every arm and all downstream domain loaders before production.
Submission: gpua100, 1 GPU/job, 2 pipelines concurrently; 4h joint / 2h single /
1h downstream. COMPONENT_ROUTING measured joint runs were about 1h42, singles about 13–35 min.
Commands: robot Python `submit.py`; `--dry-run` prints without submission.
