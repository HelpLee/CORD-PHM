# E. Representation and Mechanism Analysis

Supplementary exploratory audit: `mechanism_audit/` contains source-to-target
device retrieval, paired five-seed gains and an empirical prediction-error
decomposition. See its README for the correction to the active Milling head
and the limits of the mechanism interpretation. Existing E1--E4 are preserved.

This section contains seven frozen-representation experiments:

1. `E1_health_trajectory`: visualizes each held-out device's frozen 96D health trajectory;
2. `E2_cross_domain_degradation_axis_transfer`: tests whether a health-to-degradation direction fitted on two domains transfers zero-shot to the third.
3. `E3_layerwise_health_information_gain`: measures how strongly frozen layer-wise geometry follows within-device lifecycle distance.
4. `E4_knn_health_consistency`: measures whether each frozen snapshot retrieves neighbors from a similar lifecycle position.
5. `E5_milling_transferable_health_probe`: fits an identical source-only linear
   health readout at each frozen layer and tests it on unseen cutter C6.
6. `E6_unknown_device_transfer_evidence`: unifies Bearing, Battery, and Milling
   evidence through paired transfer gains, health-conditioned representation
   geometry, and five-seed held-out-device prediction trajectories.
7. `E7_pretraining_mechanism_evidence`: separates the effect of self-supervised
   pretraining from the effect of adding domains. It connects paired
   Scratch-versus-Frozen downstream gains to frozen representation geometry
   before any RUL-head training.

The former Adapter-magnitude and health-stage-centroid experiments were deleted
and are not part of the current E protocol.

| Domain | Source-train units used for scaling | Frozen test trajectory |
|---|---|---|
| Bearing | Bearing2_2/2_3/2_4/2_5 | Bearing2_1 post-FPT |
| Battery | CS2_35/36/37 | CS2_38 |
| Milling | C1/C4 | C6 |

E1 compares `Single-domain`, `Three-domain without Adapter`, and
`Three-domain + Adapter` on exactly the same snapshots. PCA is fitted jointly
within each domain and the plot uses continuous normalized RUL color.

E2 uses only the frozen `Three-domain + Adapter` embeddings. In each of three
leave-one-domain-out evaluations, it fits one linear health direction on two
source domains and applies that frozen direction to the third domain. The
orientation is fixed using source chronology (`high = healthy`); it is never
selected or flipped from the target result.

E3 compares `Single-domain` and `Three-domain + Adapter` on all pairwise
snapshot distances at Stem, Block 1, Block 2, and Final. Its health-structure
score is the Spearman correlation between representation distance and
within-device normalized lifecycle distance. No probe is trained.

E4 uses the same frozen representations and compares the normalized-lifecycle
MAE of each snapshot's five nearest neighbors. The query itself is excluded;
lower retrieval error indicates stronger local health consistency.

E5 is a targeted explanation of the Milling result. Ridge regularization and
scaling are selected only by C1/C4 leave-one-cutter-out validation; C6 never
selects a setting. Unlike E3/E4, it is a trained linear probe and is therefore
reported separately. It tests whether health information is more transferable,
not whether three domains share one universal physical degradation axis.

Run a strict two-snapshot checkpoint/forward check first:

```powershell
python run_all.py --verify-only --device cpu
```

Run the complete E1--E4 analysis:

```powershell
python run_all.py --device cuda
```

The scripts read frozen checkpoints and canonical downstream caches and write
only below this E directory. Optional placeholder analyses remain under
`reserved_optional/` and are not part of E1--E4.
