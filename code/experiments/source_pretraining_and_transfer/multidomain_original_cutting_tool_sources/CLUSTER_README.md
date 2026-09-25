# Cluster run: two upstream jobs

This package contains the protocol and source code for the two upstream jobs:

1. `milling_only`
2. `three_domain`

Run them sequentially (and resume checkpoints after an interruption) with:

```bash
python -u run_upstreams_only.py
python -u run_upstreams_only.py --resume
```

The Git repository intentionally excludes raw/processed arrays, runtime caches,
checkpoints, logs, and generated outputs. Stage those large artifacts on the
cluster separately, preserving the relative paths expected by `data.py` and by
the `MILLING_RUNTIME` setting in each `train_joint.py`.
