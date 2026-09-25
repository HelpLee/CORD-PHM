# F1 — N-CMAPSS unseen-domain interface calibration

This experiment keeps the canonical **three-domain HealthToken + Domain
Adapter** upstream checkpoint immutable. It adds only an Engine local/global
stem, two Engine adapters, an Engine channel embedding and an Engine masked
reconstruction decoder.

Stage 1 uses only unlabeled flights of `U2/U5/U10/U16/U18/U20`. Shared channel
pooling, both Transformer blocks, FinalNorm, snapshot projector, GRU and
dynamics projection are loaded and frozen. Only the new Engine interface is
optimized with `L_mask + 0.2 L_dyn`. `U11` and all RUL arrays are excluded.

Stage 2 uses the established U11 protocol: six-flight histories, 10%/20%/100%
label budgets, deterministic 20% within-budget validation, 200 epochs maximum,
patience 15, seeds 42–46, batch 32/microbatch 8. It compares `scratch`,
`engine_only_ssl` (random initialization with the identical Engine-only SSL
budget), `pretrained_random_interface`, `pretrained_calibrated_interface`,
and `partial_finetune_calibrated`.

`wait_for_raw_then_run.py` starts this pipeline only after the Raw Patch
Milling→Battery queue and both result files report complete.
