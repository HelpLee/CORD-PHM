# E5 — Milling transferable health probe

Frozen, layer-wise, cross-tool linear diagnostic. It asks whether normalized
health learned from source cutters C1+C4 is linearly decodable on unseen C6.
Only complete 20-snapshot endpoints are used (C6 n=219). Label budgets are
selected uniformly and independently within C1/C4. Ridge penalty and scaler are
selected/fitted using source-only leave-one-cutter-out folds. C6 never selects
the layer, alpha, sign, scaling, or any model setting.

This is an explanatory probe, not a replacement for the nonlinear downstream
benchmark. It compares the same frozen layers and the same probe capacity for
Milling-only+Adapter and Three-domain+Adapter checkpoints. Report target RMSE,
bias, centered RMSE, Spearman ordering, and source LOCO error together.
