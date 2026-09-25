# Canonical all-channel Milling upstream refresh

This package retrains two upstream controls in a serial queue:

1. `milling_only`: Milling-only pretraining with Adapter.
2. `three_domain`: Bearing + Battery + Milling joint pretraining with Adapter.

The optimization protocol is inherited from the shared CORD source-training
implementation:
seed 42, at most 500 epochs, patience 30, 20 updates/epoch, 32 samples/domain,
microbatch 8, AdamW at 1e-4 with 1e-4 weight decay, gradient clipping at 5,
and Mask reconstruction + 0.2 times six-to-one latent dynamics.

The controlled input revision is limited to Milling: all seven canonical
upstream NPZs and all recorded channels are used. PHM2010 remains downstream
only. Because Piecuch has 20 channels, the Milling reconstruction channel-query
table is expanded from 3 to 20 entries; the shared Transformer, stems,
Adapters, projector, dynamics module, loss, sampling, validation rule, and
training budget are unchanged.

`runtime/milling` is shared by both arms. `queue.py` prepares and validates it,
then runs Milling-only followed by Three-domain, resuming from `last.pt` if an
interrupted stage is restarted.
