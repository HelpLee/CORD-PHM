# Selected multi-domain CORD model

This directory contains the selected source-pretraining configuration used by
the manuscript. One encoder is trained jointly on bearing, battery, and
cutting-tool source datasets. System-specific observation interfaces and
decoders remain private, while the Transformer backbone is shared.

## Optimization

- ISM coefficients for shared parameters: bearing `0.3`, battery `0.3`,
  cutting tool `1.0`.
- IDM gradients remain full-strength for all three system types.
- Shared gradients are aggregated with CAGrad (`alpha=0.4`, `rescale=1`).
- A minimum Euclidean correction enforces the bearing-gradient directional
  floor before clipping and AdamW.
- Source learning rate and weight decay: `1e-4`.
- Twenty updates per system type per epoch, batch 32, microbatch 8.
- Maximum 2,000 epochs, validation patience 30, minimum improvement `1e-4`.

## Important files

- `config.py`: domains, source controls, routing coefficients, and checkpoint
  selection.
- `methods.py`: gradient aggregation and bearing-floor implementation.
- `train.py`: source-pretraining entry point.
- `downstream.py`: represented-system RUL adaptation.
- `models/joint_bearingfloor_cagrad/training/encoder.pt`: selected multi-domain
  encoder checkpoint.
- `models/single_battery_conditional/training/encoder.pt`: matched battery
  single-domain checkpoint.

The complete Frozen/Partial/Full downstream matrix is kept in the sibling
`included_type_adaptation_matrix/` directory.
