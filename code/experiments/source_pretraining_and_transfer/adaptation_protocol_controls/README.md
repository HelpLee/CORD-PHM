# ADAPTATION_CONTROLS: SELECTED_MODEL confirmation, automatic routing, bearing low-label adaptation

Three tasks are submitted in the user's order A -> B -> C. Their GPU jobs can run
independently once their own smoke test passes. Every adaptive downstream job waits
only for its own pretrained model. All jobs use gpua100; no job uses gpu/gpu_test.
All outputs live here; historical COMPONENT_ROUTING/CONDITIONAL_ROUTING/SELECTED_MODEL files are read-only references.

## A. Confirm the SELECTED_MODEL baseline and scratch (15 cells, 75 downstream seeds)

Use SELECTED_MODEL joint_bearingfloor_cagrad, best epoch545, as the fixed joint baseline.
Add 100% downstream for bearing, battery and milling, seeds42--46. Add the same
100% tests for its matching singles: CONDITIONAL_ROUTING single_bearing_conditional (rho=.3),
SELECTED_MODEL continued single_battery_conditional (.3), COMPONENT_ROUTING single_milling_component (1.).
Reuse their existing10%/20% results. The rho=.1 single is NOT the floor arm's control.

Run scratch from random initialization for all3 domains, 10%/20%/100%, five seeds.
Use the same current portable NPZ, seven-channel PHM2010, model/head, normalization,
source/held-out units, labeled rows, interval source validation, FP32, batch sizes,
200-epoch limit and15-epoch patience as the downstream SELECTED_MODEL drivers.
Scratch uses LR1e-3 throughout; pretrained encoder3e-4/head1e-3 (the established
initialization-specific protocol). This LR difference is explicit, not hidden.
Scratch never loads pretrained weights. Prior historical scratch is not reused.

## B. Automatically determine reconstruction sharing (12 upstreams, 36 cells/180 seeds)

Three methods each train one joint model and three matching own-domain single
models. Same network, sources (original4 milling+HMoTP, all channels), normalized
reconstruction/dynamics components, CAGrad and SELECTED_MODEL bearing contribution floor.
The floor is only used when multiple domains are active; private gradients are
unchanged. All methods use the same rules across domains, rho initialized to.5,
common bounds [.05,1], no hand-selected bearing/battery/milling coefficients.

1. `cosine`: EMA(.9) of cosine(g_rec,g_dyn), rho=.05+.95*max(EMA,0).
   This is automatic adaptation by a rule, NOT a learned parameter model.
2. `learned`: one logit per domain. Every5 main updates, compute dynamics gradients
   on a fresh independently sampled TRAIN minibatch of8; learn rho using their
   normalized alignment with reconstruction gradients. The batch is drawn from
   the same upstream TRAIN pool, so samples may overlap; no validation or test
   labels are used. Main training RNG is preserved around the auxiliary pass.
3. `layerwise`: same learned rule, separately for Transformer block0, block1,
   other shared encoder parameters and the shared dynamics head. Parameters with
   zero reconstruction gradient provide no evidence to change that gate.

The learned objective is the first-order surrogate
J(z)=-mean(rho(z)*EMA(cos(g_rec,g_dyn_reference)))+.01*KL(Bernoulli(sigmoid(z))||.5),
rho=.05+.95*sigmoid(z), gate step=.05, logits clipped[-8,8]. Increasing a gate is
rewarded if reconstruction aligns with dynamic-loss reduction on another batch.
This adapts gradient-similarity/meta-reweighting ideas; it is NOT exact bilevel
optimization through CAGrad, clipping and AdamW, and NOT a reproduction of either
paper's guarantees. The weak common prior discourages saturation. Common bounds,
update interval and learning rate remain hyperparameters; only domain-specific
sharing strengths are learned. A dynamics auxiliary task is a proxy for useful
degradation information, not proof of a health coordinate.

No MoE is added in this batch: the parameter-learning question can be isolated
without changing encoder capacity or inference architecture. Gates are training
state and are removed for downstream inference. Save gate trajectories, alignment,
floor diagnostics, optimizer/RNG/gate state for resumable training. Max2000 epochs,
patience30/min_delta1e-4,20 main updates/domain/epoch, BF16 upstream, batch32/micro8,
AdamW1e-4. Extra reference passes are additional compute, not main optimizer steps.
Downstream all3 domains10%/20%,5 seeds, same full-finetuning protocol as A/SELECTED_MODEL.
Compare each against fixed SELECTED_MODEL and its own matched single. Report all cells/metrics.
Success target: preserve SELECTED_MODEL's aggregate transfer benefit and improve its worst
cell; improvement is an experimental goal and is not guaranteed or forced.

## C. Bearing10% adaptation (6 cells, 30 downstream seeds)

Current shortfall is JOINT vs SINGLE, interpreted from the preceding results.
On the fixed SELECTED_MODEL joint and matching CONDITIONAL_ROUTING single, test identical adaptations:
- Partial: last Transformer block, its adapter and final norm plus downstream
  head, from epoch1, encoder LR3e-4/head1e-3.
- L2-SP alpha=.001, full fine-tuning.
- L2-SP alpha=.01, full fine-tuning.

L2-SP adds alpha/2 * sum_encoder ||theta-theta_pretrained||^2 to TRAIN loss.
Use the original unpenalized validation RMSE for early stopping. Record alpha;
all target head parameters remain unconstrained. Compare each joint to the same
adaptation on the single, and each checkpoint to its original full fine-tuning.
This tests whether limited labels allow destructive adaptation of pretrained
features. It does not presume that downstream overfitting caused the gap.

## Execution and limits

`python submit.py` submits75 jobs:3 smoke,12 upstream,57 downstream cells,3 summaries.
Task A is submitted first, then B, then C; submission order is not a promise of
scheduler start order. Smoke1h; joint upstream10h/single4h; downstream low-label2h,
100%3h (five seeds per job); summary10min. Estimated from SELECTED_MODEL where joint models
stopped around575--694 epochs in about2h; ceilings provide headroom for adaptive
routing to continue longer. Finite2000 ceiling and unchanged early stopping apply.
Submission ledger is atomically saved per accepted job. Finished cells are skipped.
Milling incomplete seed directories are retained under interrupted_* before restart.
Reports: reports/baseline.json, adaptive.json, bearing10.json and corresponding.md.

These are development experiments informed by previous test results, one upstream
seed and five downstream seeds. They do not establish independent pretraining-seed
robustness. No test-based automated winner selection or hidden removal of weak cells.

## Primary sources

- Du et al., Adapting Auxiliary Losses Using Gradient Similarity:
  https://arxiv.org/abs/1812.02224
- Ren et al., Learning to Reweight Examples for Robust Deep Learning, ICML2018:
  https://proceedings.mlr.press/v80/ren18a.html
- Li et al., Explicit Inductive Bias for Transfer Learning with Convolutional
  Networks (L2-SP), ICML2018: https://proceedings.mlr.press/v80/li18a.html
