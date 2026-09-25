# CORD double-blind reproducibility package

This repository contains the anonymized code, selected checkpoints, protocol
records, and retained results used in the CORD manuscript. Dataset arrays are
not redistributed. The required processed files are identified by relative
path, byte count, and SHA256 in `external_data_manifest.json`.

## Start here

1. Read `PAPER_CODE_MAP.md` for the direct mapping from manuscript sections and
   Appendices A--G to code and results.
2. Run `python validate_package.py` to verify all bundled files.
3. Run `python prepare_external_data.py /path/to/data_phm` to stage the external
   processed datasets as symbolic links.
4. Reproduce the represented-system adaptation matrix with
   `bash run_included_type_adaptation.sh`, or use the appendix-specific entry
   points listed in `PAPER_CODE_MAP.md`.

## Repository layout

- `code/preprocess_health_tokens/`: observation construction and dataset
  preprocessing.
- `code/experiments/source_pretraining_and_transfer/`: source pretraining,
  selected CORD checkpoints, matched single-domain controls, downstream
  adaptation, and RUL baselines.
- `code/experiments/appendix_c_included_type_generalization/`: Appendix C
  aggregation and label-efficiency analysis.
- `code/experiments/appendix_d_engine_adaptation/`: engine adaptation code and
  protocol records used for pretraining-excluded-system evaluation.
- `code/experiments/appendix_e_frozen_representation/`: frozen-representation
  diagnostics.
- `code/experiments/appendix_g_efficiency/`: compute and shared-deployment
  measurements.
- `results_availability.json`: machine-readable inventory of bundled evidence.
- `file_checksums.json`: SHA256 inventory for every bundled file other than the
  checksum inventory itself.
- `package_metadata.json`: anonymized package metadata.

## Main represented-system evaluation

```bash
python validate_package.py
python prepare_external_data.py /path/to/data_phm
cd code/experiments/source_pretraining_and_transfer/included_type_adaptation_matrix
python summarize.py
python submit.py
```

`submit.py` schedules only missing cells. Completed cells, split/scaler
metadata, and the four selected encoder checkpoints are retained at the paths
expected by the code. The environment lock is
`code/experiments/source_pretraining_and_transfer/shared_domain_components/requirements-lock.txt`.

## Double-blind configuration

The repository omits author names, affiliations, workstation paths, cluster
accounts, scheduler logs, Git history from the development repository, and raw
datasets. Neutral placeholders such as `/path/to/CORD` and `/home/anonymous`
must be adapted to the execution environment.
