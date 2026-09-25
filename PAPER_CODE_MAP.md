# Paper-to-code map

This file maps the manuscript's reported evidence to the corresponding code,
protocol, checkpoint, and result locations. Development experiment numbers are
intentionally not used in reviewer-facing names.

| Manuscript location | Purpose | Repository location |
|---|---|---|
| Method / Appendix A | Structured descriptors, masks, dataset readers, and processed-data verification | `code/preprocess_health_tokens/` and `code/experiments/source_pretraining_and_transfer/preprocessing/` |
| Appendix B | Shared encoder, system-specific interfaces, ISM/IDM objectives, CAGrad, and bearing-gradient floor | `code/experiments/source_pretraining_and_transfer/shared_domain_components/`, `component_gradient_routing/`, `conditional_gradient_routing/`, and `selected_bearing_floor_cagrad/` |
| Main Tables 2--3 / Appendix C | Frozen, Partial FT, and Full FT comparisons at 10%, 20%, and 100% labels | `code/experiments/source_pretraining_and_transfer/included_type_adaptation_matrix/` |
| Appendix C baselines | MLP, random forest, XGBoost, TCN, PatchTST, iTransformer, and MOMENT evaluation | `code/experiments/source_pretraining_and_transfer/rul_prediction_baselines/` |
| Main Figure 3 / Appendix D | N-CMAPSS engine adaptation for a pretraining-excluded system type | `code/experiments/appendix_d_engine_adaptation/` |
| Main Figures 4--5 / Appendix E | Frozen cross-unit retrieval and lifecycle-organization diagnostics | `code/experiments/appendix_e_frozen_representation/` |
| Main Table 5 / Appendix F | Structured Descriptors versus Raw-Resampled Inputs | `code/preprocess_health_tokens/raw_patch_ablation/` and `code/preprocess_health_tokens/build_downstream_raw_patches.py` |
| Main Table 6 | ISM-only versus ISM+IDM source-pretraining objective comparison | `code/experiments/source_pretraining_and_transfer/transfer_mechanism_comparison/` |
| Appendix G | Parameter count, throughput, memory, and shared-versus-separate deployment | `code/experiments/appendix_g_efficiency/` |

## Selected model and controls

- Multi-domain CORD: `source_pretraining_and_transfer/selected_bearing_floor_cagrad/`
- Matched single-domain controls: `conditional_gradient_routing/` for bearings,
  `selected_bearing_floor_cagrad/` for batteries, and
  `component_gradient_routing/` for cutting tools.
- Complete represented-system matrix:
  `source_pretraining_and_transfer/included_type_adaptation_matrix/`.
- Scratch and adaptation controls:
  `source_pretraining_and_transfer/adaptation_protocol_controls/`.

## Data and integrity files

- `external_data_manifest.json`: required external arrays and their hashes.
- `prepare_external_data.py`: stages those arrays into `code/data_phm/`.
- `file_checksums.json`: hashes all bundled, non-NPZ files.
- `validate_package.py`: validates the code package and optionally the external
  data root.

The repository deliberately contains no `.npz` dataset arrays, generated NumPy
caches, scheduler logs, or Python bytecode.
