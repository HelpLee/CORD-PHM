# Seven-channel PHM2010 transfer from refreshed Milling-only upstream

This downstream experiment compares the existing seven-channel Scratch
control with the refreshed Milling-only encoder trained from all seven
canonical Milling upstream datasets and all recorded channels.

The target protocol is unchanged: C1+C4 train, C6 test, 10/20/100% labels,
seeds 42--46, uniform interval label selection, 20% validation within the
selected source labels, 200 epochs, and no Conv1D stem.  The transferred
checkpoint is evaluated with Frozen Probe, Partial Fine-tuning, and Full
Fine-tuning.  Existing Scratch artifacts are referenced rather than retrained.

`run_transfer_matrix.py` is resumable at completed fraction/seed cells.  The
paused three-domain upstream checkpoint remains separate and is not used.
