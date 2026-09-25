# E4. Layer-wise kNN health consistency

E4 measures whether nearby snapshots in each frozen latent space occupy nearby
positions in the same device's lifecycle. It uses exactly the same held-out
devices, model variants, layer locations, and intermediate snapshot readout as
E3, but replaces the pairwise correlation metric with direct neighbor retrieval.

For every snapshot, the query itself is excluded and the five nearest frozen
representations are retrieved using Euclidean distance. The reported score is
the mean absolute normalized-lifecycle difference between each query and its
five neighbors. Lower is better. No probe or encoder parameter is trained.

Run from the E directory:

```powershell
python E4_knn_health_consistency/analyze.py --device cpu --k 5
```
