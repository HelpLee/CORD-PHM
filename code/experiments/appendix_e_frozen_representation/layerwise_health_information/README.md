# E3. Layer-wise health information gain

E3 asks whether the frozen encoder progressively organizes snapshots according
to their distance in the same device's lifecycle. It compares a single-domain
checkpoint with the three-domain + Adapter checkpoint on exactly the same
held-out snapshots in Bearing, Battery, and Milling.

At Stem, Block 1, and Block 2, each snapshot is represented by the concatenated
Global token and mean of its valid Local tokens. Final uses the native frozen
96D snapshot embedding. For every layer, all snapshot pairs are evaluated:

`rho_health = Spearman(||z_i-z_j||_2, |t_i/(T-1)-t_j/(T-1)|)`

A larger positive value means snapshots farther apart in that device's
lifecycle are also farther apart in representation space. No probe is trained,
and normalized lifecycle position is used only after frozen extraction.

Run from the E directory:

```powershell
python layerwise_health_information/analyze.py --device cpu
```
