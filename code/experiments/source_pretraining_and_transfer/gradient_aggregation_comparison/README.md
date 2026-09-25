# E29: joint optimization, minimal architectural change

## Question and evidence

The existing all-channel runs show asynchronous per-domain validation minima.
This does NOT establish gradient conflict, and validation loss magnitude is not
gradient magnitude. Exp28 `single5` is milling-only: its bearing/battery transfers
are not own-domain single-pretraining controls. This study supplies those controls.

Use the original four Milling sources (LUH, MATWI, Nonastreda, QIT-CEMC) plus HMoTP,
all available channels, exactly the exp28 five-source runtime; Bearing and Battery
sources/caches/splits remain unchanged. This is NOT the seven-source experiment.
PHM2010 is downstream only, seven channels. No older result is overwritten.

## Prespecified comparisons

| Arm | Training | Checkpoint and early stopping | Hypothesis |
|---|---|---|---|
| joint_raw | original equal-domain raw SSL loss | original raw macro validation | contemporary baseline |
| joint_selection | SAME trajectory as joint_raw, no extra training | normalized best within raw trajectory | checkpoint selection alone |
| joint_norm | fixed initial TRAIN-loss normalization per dataset | normalized macro validation | scale imbalance |
| joint_pcgrad | joint_norm + shared-gradient PCGrad | same as joint_norm | conflicting gradient directions |
| joint_cagrad | joint_norm + shared-gradient CAGrad, alpha=0.4 | same as joint_norm | compromise direction preserving average objective |
| single_{bearing,battery,milling}_{raw,norm} | own domain only, raw/normalized respectively | matching raw/normalized rule | genuine matched single-domain controls |

Four jointly trained trajectories + six own-domain trajectories = ten upstream jobs.
Each joint model is ONE checkpoint used for all three domains, not three independently
chosen domain checkpoints. The selection-only arm intentionally does not test extended
training under normalized stopping; the normalization arm changes both the training
scale and validation rule, so compare it with selection-only cautiously.

Initial per-dataset train scales use 64 deterministic TRAIN rows/windows and fixed masks.
Validation normalization uses initial validation loss only for evaluation/model selection,
never backpropagation or sampling. Macro averaging is dataset then domain. Datasets with
no usable validation sequences are explicitly recorded, not assigned artificial losses.
QIT-CEMC/KAIST missing validation windows remain a limitation of the inherited protocol.
Private stems/adapters/decoders retain ordinary domain-mean gradients. Only truly shared
encoder/GRU/projection gradients undergo surgery. PCGrad randomization has its own RNG.
CAGrad uses the simplex solution, alpha=0.4 and rescale by 1+alpha^2 (fixed in advance).

All upstream arms: seed42, unchanged network, lr1e-4, AdamW wd1e-4, 20 updates/epoch,
32 examples/domain/update, microbatch8, BF16, clip5, maximum500 epochs, patience30,
min_delta1e-4. Exact per-domain stochastic batch/mask seeds are shared across arms.
One joint update consumes all active domains, so equal epoch means equal per-domain
exposure, but unequal total compute. Different early stopping times are reported,
not forced to produce a favorable joint-training duration. Resume uses full optimizer,
model and RNG states; completed runs are not restarted.

## Evaluation and diagnosis

Full finetuning encoder3e-4/head1e-3; exp28 protocol unchanged, 10/20/100%, seeds42–46.
Five joint checkpoints x three domains x three fractions plus six single checkpoints
x own domain x three fractions = 63 cells / 315 downstream runs. All methods reported.
Upstream uses ONE seed in this screening study, so five downstream seeds do not establish
upstream-seed robustness. Existing scratch may be cited only if labels/channels/splits
match; legacy three-channel scratch must not be presented as a seven-channel control.

Primary diagnostic comparisons: joint_selection vs joint_raw (selection); joint_norm vs
joint_raw and selection (scale, with noted stopping confound); PCGrad/CAGrad vs joint_norm
(direction); each joint arm vs own-domain matching raw/norm singles (transfer gain).
Record each update's three shared gradient norms, cosine matrix and alignment with the
applied direction, plus each dataset/domain validation trajectory. Inspect whether
improvements coincide with reduction in harmful directions, not just more epochs.
Report RMSE/MAE/R2 mean +/- sample SD, paired RMSE differences and seed-wise wins.
Do not select methods/hyperparameters on downstream TEST results. For final paper model
selection use development validation only; after exploratory use of old test sets,
confirm with new held-out devices/splits and additional upstream seeds.

## Literature (primary sources)

- PCGrad, Yu et al., NeurIPS 2020: https://arxiv.org/abs/2001.06782
  Projects a task gradient away from another when the dot product is negative.
- CAGrad, Liu et al., NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html
  Balances average descent with worst-task local improvement. The theory does not
  guarantee downstream transfer improvement under Adam/BF16/nonconvex training.
- GradNorm, Chen et al., ICML 2018: https://proceedings.mlr.press/v80/chen18a.html
  Balances gradient magnitudes/training rates; distinct from resolving directions.
- Reference implementations: https://github.com/tianheyu927/PCGrad and https://github.com/Cranial-XIX/CAGrad
  CAGrad rescale convention: https://github.com/Cranial-XIX/CAGrad/blob/main/nyuv2/utils.py

## cluster execution

Submit from this folder with `python submit.py --smoke-only`, then `python submit.py`.
The smoke test executes real data forward/backward, all gradient methods and checkpoint
conversion checks for all downstream domains. All jobs depend on its success.
10 upstream jobs use gpua100, one GPU/job, four bounded lanes, 4h maximum/job.
63 downstream jobs use gpua100, one GPU per model/domain/fraction with five sequential
seeds, 1h maximum/job. Start only after upstream lanes finish, with own-model afterok
dependencies; unrelated cell failures do not cancel the other lanes. A CPU summary runs
after all cells terminate and explicitly reports missing/failed cells.
Walltimes are ceilings, not intentional runtime or fixed-epoch claims. No existing
jobs are cancelled. `submissions.json` is idempotent; read logs before retrying failures.
Use `--resume` to continue an interrupted upstream checkpoint (do not replace its ledger
entry blindly: downstream dependency ids must be updated on resubmission).
