# C. Transfer and Label Efficiency

This package re-analyzes the completed, canonical A-main results. It performs
no training and never duplicates model checkpoints.

## C1 — Label efficiency

For Bearing, Battery and Milling, compare the shared Scratch baseline against
the Three-domain + Adapter **Frozen Probe** at 10%, 20% and 100% labels. Report
RMSE, MAE and R2 as five-seed mean +/- sample standard deviation.

## C2 — Adaptation strategy

For the same three domains and label budgets, compare only the Three-domain +
Adapter checkpoint under Frozen Probe, Partial Finetune and Full Finetune.
Partial Finetune is the corrected no-warm-up version (unfrozen from epoch 1).

Run `python analyze.py`. Generated figures, CSV/JSON tables, Markdown tables
and the provenance audit are stored in `outputs/`.
