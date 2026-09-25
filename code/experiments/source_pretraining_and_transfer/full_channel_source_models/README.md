# Three all-channel upstream models

This experiment trains exactly three upstream encoders with the historical
validation-based early-stopping protocol (seed 42, 20 updates/epoch, maximum
500 epochs, patience 30):

1. `three4`: joint Bearing+Battery+Milling; the original four Milling sources,
   all available channels.
2. `three5`: the same joint model plus HMoTP, all available channels.
3. `single5`: Milling-only SSL on the same five Milling sources and channels.

Each checkpoint is evaluated with full fine-tuning on Bearing, Battery, and
PHM2010 Milling at 10%, 20%, and 100% labels, seeds 42--46.  The pretrained
encoder LR is `3e-4` and the new prediction head LR is `1e-3`.  For the
Milling-only checkpoint transferred to Bearing or Battery, the unseen target
input stem/adapter is deterministically initialized at seed 42 and learned
during full fine-tuning; no three-domain interface weights are borrowed.
