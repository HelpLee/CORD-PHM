# Refreshed three-domain A-main downstream queue

This package waits for `../three_domain/training/status.json` to report
`complete`, then evaluates that exact checkpoint on Milling, Battery, and
Bearing in that order. Each target uses 10%, 20%, and 100% label budgets,
seeds 42--46, and Frozen Probe, Partial Fine-tuning, and Full Fine-tuning.

Existing Scratch and single-domain results are read as comparison controls and
are not overwritten.
