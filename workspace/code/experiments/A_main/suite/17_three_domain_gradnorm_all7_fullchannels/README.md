# Three-domain GradNorm pretraining: seven-source all-channel Milling

This package changes only joint-domain optimization and checkpoint monitoring.
The upstream data, architecture, masking task, dynamics task, sampling schedule,
optimizer, and train/validation unit splits remain those of the seven-source
all-channel Milling refresh.

## Balancing protocol

- Domain loss: masked reconstruction plus `0.2 * dynamics`.
- Each domain loss is divided by that domain's frozen initial validation loss.
- Epochs 1--5 use equal domain weights.
- From epoch 6, GradNorm updates weights every 5 optimizer updates.
- `alpha=0.5`, weight learning rate `0.025`, each weight is clamped to
  `[0.3, 3.0]`, and the three weights are renormalized to sum to 3.
- Gradient norms are measured on the final shared Transformer block.
- Early stopping monitors the macro mean of the three validation losses after
  division by their corresponding initial validation losses.

The history records raw losses, normalized losses, domain weights, shared
gradient norms, and raw and normalized validation scores for diagnosis.

## cluster

```bash
cd /path/to/CORD/code
git pull origin master
cd experiments/A_main/suite/17_three_domain_gradnorm_all7_fullchannels
chmod +x run_gradnorm_cluster.sh
sbatch run_gradnorm_cluster.sh
```

Training can be resubmitted safely: the runner adds `--resume` only when a local
`three_domain/training/last.pt` checkpoint exists.

## Matched single-domain controls

Submit all three controls from this directory with:

```bash
bash submit_single_domains_cluster.sh
```

This creates three independent A100 jobs (bearing, battery, milling), each with
one GPU, 8 CPUs, 64 GB RAM and a ten-hour limit. It skips matching active jobs
and completed controls. Job IDs are recorded in `single_domain/submissions`.
The runner resumes existing checkpoints and never overwrites the joint output.

Each control initializes the exact experiment-17 JointModel with seed 42 but
trains only its selected domain. Inactive domain parameters receive no gradients.
Keeping initialization intact preserves the shared encoder's initial parameters.
The original domain-specific RNG index is retained (bearing=0, battery=1,
milling=2), as are data sampling, masking, optimizer and SSL objectives.

- Domain weight is exactly 1; no GradNorm weight updates for a single domain.
- Loss and validation are divided by the same frozen initial validation loss.
- 20 optimizer updates/epoch, 32 samples/update, microbatch 8.
- Maximum 2000 epochs, patience 30, normalized min_delta 1e-4; no minimum budget.
- Milling reuses experiment 17's existing seven-source all-channel audited cache.
- Bearing uses its historical audited splits/scalers; battery uses the same
  Corpus builder as the joint arm, with isolated cache/output paths.
- Results: `single_domain/<domain>/training/{status.json,history.json,best.pt,encoder.pt,last.pt}`.
- These jobs are upstream-only. No downstream jobs are submitted here.

For one control: `sbatch --job-name=ht_norm_milling run_single_domain_cluster.sh milling`.
Do not interpret the fixed single-domain weight as an adaptive GradNorm method;
it is the matched normalization/early-stopping control for the joint method.
