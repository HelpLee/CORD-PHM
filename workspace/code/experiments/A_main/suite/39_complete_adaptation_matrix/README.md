# E39 complete downstream adaptation matrix

Final upstream checkpoints only: E37 `joint_bearingfloor_cagrad` and its matched
single-domain controls.  Compare `frozen_probe`, `partial_finetune`, and
`full_finetune` for Bearing, Battery, and Milling at 10/20/100% labels, seeds42--46.

`full` is reused from E38. Bearing10 `partial` is reused from E38. The remaining
34 cells are submitted here. Scratch-full is already in E38 and is not a pretrained
adaptation mode. E38 automatic-routing100% is intentionally excluded.

Frozen trains only the downstream lifecycle/head. Partial trains the downstream
head plus Transformer block1, final norm, and the domain adapter from epoch1.
Full trains the complete encoder and downstream head. Pretrained encoder LR is
3e-4 and head LR1e-3 for partial/full; frozen has no encoder optimizer group.
Inputs, interval validation, early stopping, five seeds and held-out devices are
the same audited drivers inherited by E38.
