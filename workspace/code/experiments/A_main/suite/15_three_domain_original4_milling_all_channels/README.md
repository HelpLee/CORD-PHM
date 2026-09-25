# Original-four Milling sources with all channels

This isolated experiment changes one Milling upstream factor: the original four
pretraining datasets (LUH, MATWI, Nonastreda, QIT-CEMC) use every available
signal channel instead of the historical three-channel selection. Bearing and
Battery inputs, model dimensions, adapters, mask+dynamics objective, sampling,
optimizer, seed, and stopping protocol remain those of the refreshed A-main
upstream.

Two upstreams are trained from matched initialization and sampling rules:
Milling-only and Three-domain. The all-channel cache prepared and audited by
experiment 14 is reused, but only the original four dataset names are admitted.

Downstream uses seven-channel PHM2010, C1+C4 for training and C6 for testing,
10% labels, seeds 42--46, and Frozen probe. Existing seven-channel Scratch rows
are imported without rerunning. Frozen probe is used so the comparison measures
the pretrained representation rather than encoder adaptation.

The matching historical three-channel Frozen rows for the original-four-source
Milling-only and Three-domain upstreams are imported into the same result table.
This makes the final table a direct three-channel versus all-channel comparison
for both single-domain and joint pretraining.

Run `pipeline.py` with the robot Python environment. It resumes incomplete
upstreams, then automatically launches the ten downstream transfer runs.
