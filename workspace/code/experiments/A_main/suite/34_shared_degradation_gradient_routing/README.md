# E34 — shared-degradation gradient routing

## Motivation

E29 shows substantial cross-domain gradient conflict and a battery shared-gradient
norm about 4.3 times the milling norm. Its SSL objective is reconstruction dominated:
the weighted latent-dynamics term commonly contributes only about 0.4%–6.2% of the
validation objective. A single scalar loss normalization cannot correct this internal
imbalance. Therefore heterogeneous observation reconstruction can dominate the shared
backbone even when domains are externally equally weighted.

## Prespecified intervention

1. Normalize reconstruction and latent-dynamics losses separately at initialization.
2. Give the two normalized components equal objective mass.
3. Private stems, adapters and reconstruction heads retain 100% of reconstruction
   gradients; the shared backbone receives either 100% (component control) or 10%.
4. Dynamics gradients always enter the shared backbone at full normalized strength.
5. Test mean aggregation and CAGrad after routing.

Each proposed joint arm has a matched own-domain single-pretraining control. Downstream
uses the established full-fine-tuning protocol (encoder LR 3e-4, head LR 1e-3), fixed
splits, 10%/20%/100% labels and seeds 42–46. All cells are reported; no test-set model
selection is allowed. One upstream seed means that downstream seeds measure conditional
fine-tuning variability, not pretraining-seed variability.

This is a mechanism test, not a guarantee that all 9 joint/domain/label cells improve.
The key falsifiable prediction is that routing reduces battery negative transfer without
removing the milling gain; CAGrad should additionally protect bearing under conflict.
