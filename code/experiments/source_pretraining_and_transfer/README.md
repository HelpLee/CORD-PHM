# Source pretraining and downstream transfer

This directory contains the selected CORD source-pretraining implementation,
matched single-domain controls, direct optimization dependencies, complete
represented-system adaptation matrix, and external RUL baselines.

## Manuscript-facing entry points

- `selected_bearing_floor_cagrad/`: selected multi-domain CORD source model.
- `included_type_adaptation_matrix/`: Frozen, Partial FT, and Full FT results
  for bearings, batteries, and cutting tools at all label budgets.
- `adaptation_protocol_controls/`: Scratch and matched adaptation controls.
- `rul_prediction_baselines/`: complete predictor baselines.
- `transfer_mechanism_comparison/`: ISM/IDM and gradient-aggregation support
  code used by the selected method.

## Direct implementation dependencies

- `preprocessing/` and `shared_domain_components/`: observation construction,
  system-specific interfaces, and common training code.
- `component_gradient_routing/` and `conditional_gradient_routing/`: matched
  optimization controls from which the selected configuration is constructed.
- Other directories retain source-model and GradNorm controls used to verify
  the final protocol; names describe their function rather than their
  historical experiment order.
