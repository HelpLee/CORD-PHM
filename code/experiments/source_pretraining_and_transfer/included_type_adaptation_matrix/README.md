# Represented-system adaptation matrix

This directory reproduces the manuscript's complete comparison of multi-domain
CORD and matched single-domain checkpoints under Frozen, Partial FT, and Full FT
adaptation. It covers bearings, batteries, and cutting tools at 10%, 20%, and
100% labels with seeds 42--46.

- `config.py` resolves the selected checkpoints and retained result locations.
- `downstream.py` runs one model/adaptation/system/label-budget cell.
- `submit.py` schedules only missing cells on Slurm.
- `summarize.py` validates and aggregates all completed cells.
- `reports/adaptation_matrix.json` is the retained aggregate report.

Partial FT updates the downstream head, the final shared Transformer block,
final normalization, and the system-specific adapter. Full FT updates the
complete encoder and downstream head. Frozen adaptation trains only the
system-specific RUL readout. Pretrained encoder groups use learning rate
`3e-4`; downstream heads use `1e-3`.
