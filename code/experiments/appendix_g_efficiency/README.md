# Appendix G: computational cost and shared deployment

- `compute_efficiency/` contains parameter-count, GPU-time, memory, and
  inference-throughput measurement code.
- `shared_deployment/` compares one shared encoder with system-specific heads
  against separately materialized single-domain models.

These experiments use the same selected checkpoints and input protocol as the
main represented-system evaluation.
