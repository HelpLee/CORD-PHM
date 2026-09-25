# Package contents

The generated snapshot contains:

- the exact E34, E35, E37, E38, and E39 experiment implementations;
- their inherited preprocessing, shared domain code, optimization, downstream,
  and runtime-preparation dependencies (E00/E01/E14/E15/E17/E19/E28/E29/E33);
- four selected encoder checkpoints with completion records;
- synchronized A-group checkpoints/Scratch results and validated completed C-group
  Full/Partial/Frozen cells currently available in the final-study registry;
- the completed B3 Handcrafted-26 versus parameter-free Raw-Resampled-26 code,
  per-run JSON artifacts, and 90-row aggregate result;
- code and locked protocols, but no legacy outputs, for B1/B2 and D/E/F
  experiments that require rerunning;
- Slurm launchers, environment lock, split/scaler metadata, and provenance;
- a SHA256 manifest for every bundled file;
- a separate SHA256 inventory for every external processed NPZ.

Excluded by design:

- all dataset `.npz` files;
- generated NumPy caches (`.npy`);
- raw datasets;
- transient logs and Python caches.
- development results that do not satisfy the final locked rerun protocol.

Run `python verify_package.py` to validate the bundle. Run it with
`--data-root /path/to/code/data_phm` to additionally validate the external
processed data before launching experiments.
