# CORD double-blind reproducibility package

This directory contains the exact source closure used by the selected E37 model
and E39 downstream matrix, the selected checkpoints, synchronized completed
results, split/scaler metadata, Slurm launchers, and cryptographic manifests.
Dataset arrays are deliberately not bundled.

Results follow `reuse_policy.json`: the complete E39 C matrix includes all
54 validated cells and the exact code used to produce them. Other groups that
still require a final locked-protocol rerun remain code/protocol only.

## Layout

- `workspace/code`: isolated source tree preserving the original relative paths.
- `data_manifest.json`: external NPZ paths, sizes, required flags, and SHA256.
- `package_manifest.json`: SHA256 for every bundled non-NPZ file.
- `build_metadata.json`: anonymized snapshot metadata, copied suites, and file counts.
- `verify_package.py`: validates the bundle and optional external datasets.
- `stage_external_data.py`: links external NPZ files into the isolated workspace.
- `run_e39.sh`: E39 Slurm submission entry point; downstream jobs queue directly.

## Reproduce

From the root of this extracted package:

```bash
python verify_package.py
python verify_package.py --data-root /path/to/code/data_phm
python stage_external_data.py /path/to/code/data_phm
cd workspace/code/experiments/A_main/suite/39_complete_adaptation_matrix
python summarize.py
python submit.py
```

On a Slurm cluster, use the supplied `run_gpu.sh`/`submit.py` scripts after activating the
environment described by
`workspace/code/experiments/A_main/suite/01_shared_dependencies/requirements-lock.txt`.
The four selected encoder checkpoints are materialized at the exact locations
expected by `config.py`; completed result files are likewise restored to their
original paths. Therefore `summarize.py` can reuse completed cells and schedules
only missing cells.

The package intentionally contains no `.npz` file. Raw datasets are also not
redistributed; the external canonical processed NPZs are the experiment inputs.

## Double-blind configuration

This review archive intentionally omits author names, affiliations, repository remotes, Git history, workstation paths, cluster account names, scheduler logs, and raw datasets. Paths such as `/path/to/CORD` and `/home/anonymous` are neutral placeholders; set them for the local environment before launching cluster jobs. The released checkpoints, aggregate results, split metadata, and external-data hashes are retained.

