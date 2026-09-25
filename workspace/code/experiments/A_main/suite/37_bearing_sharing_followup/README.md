# E37: bearing sharing and contribution follow-up

E35 conditional CAGrad had the best matched macro relative RMSE (-7.58%), but
conditional PCGrad won 5/6 cells instead of 4/6. Neither is universally best.
Battery20 CAGrad .070729 beats its conditional single .071821 but loses against
the original component single .068223. PCGrad .061987 beats both references.
Bearing10 CAGrad .116268 vs matched single .112084 remains the main discrepancy.

All E35 joint models hit max500 epochs, selected499. Bearing's normalized
validation loss under conditional CAGrad falls .35560 (epoch140) -> .34248 (499);
its matched single selected140 (.33271). This does not prove late overfitting.
In the 500 logged first-update gradient samples, bearing's directional progress
ratio dot(g_b,d)/||g_b||^2 is below 1/3 in 29%; median .41745, minimum .14518.
That is a targeted optimization hypothesis, not a proven explanation of RUL error.
E36 MoE has no downstream results at design time and is left running independently.

## Candidates

| Candidate | Bearing reconstruction rho | Aggregation | Contribution floor |
|---|---|---|---|
| joint_lowbearing_pcgrad | .1 | PCGrad | none |
| joint_lowbearing_cagrad | .1 | CAGrad | none |
| joint_bearingfloor_cagrad | .3 | CAGrad | bearing |
| joint_lowbearing_floor_cagrad | .1 | CAGrad | bearing |

Battery rho .3 and milling rho1 remain fixed. Reconstruction remains full-strength
in private stems/adapters/heads; dynamics gradients remain full-strength everywhere.
Floor projects the aggregated shared gradient d onto dot(g_b,d)>=||g_b||^2/D:
d' = d + max(0, ||g_b||^2/D - dot(g_b,d))/max(||g_b||^2,1e-20) * g_b.
It is the minimum Euclidean correction to retain the bearing gradient's own
equal-weight contribution, not a guarantee of non-increasing loss after AdamW,
clipping, other private updates or finite steps, nor of validation/test improvement.
Only one jointly trained backbone/checkpoint is deployed for all domains.

One new matched single_bearing_lowbearing is trained with rho .1. Reuse genuinely
early-stopped E35 bearing (epoch170/best140) and E34 milling (361/best331).
Continue both E35 conditional PCGrad/CAGrad joint parents, E35 conditional battery
and E34 component battery from epoch500 up to2000. Both joint parents and singles
receive the same ceiling/early-stopping rule, isolating training duration from
method changes. Component bearing already early-stopped and is reused unchanged.
Primary comparisons never use a capped500 battery control against a max2000 joint.
Report duration-only (continued vs original500) and method-only (new vs continued
parent) separately. Do not infer benefit merely by changing to a weaker control.

Same original4+HMoTP all-channel milling corpus, bearing/battery sources, E34
component normalization, dataset round-robin, source splits, seed42, AdamW1e-4,
20 updates/domain/epoch, batch32,microbatch8,max2000,patience30,min_delta1e-4,
macro source validation. New methods start fresh; four capped references restore
model, AdamW state, all RNG states, best metric, stale counter and history in new
isolated directories after protocol/fingerprint checks. No early-stopping reset.
Production BF16 matches E35 on gpua100;
downstream FP32 matches E35, encoder3e-4/head1e-3, full fine-tuning,
10%/20% x 3 domains x seeds42..46. 100% is outside this low-label screen.

No downstream label or validation/test score enters upstream gradient routing.
This is exploratory design informed by earlier test comparisons; final claims need
independent devices/upstream seeds. One upstream seed plus five downstream seeds
does not measure pretraining-seed variance. No per-domain best-checkpoint assembly.

Save per-update gradient diagnostics in training/gradient_audit. Diagnostic epoch
140/250/500/1000/1500/2000 encoders are saved without downstream selection. Training continues
under the original validation rule; selected update counts are reported separately.

## Submission

Run python submit.py on cluster. smoke on gpu_test (1h, FP32 for V100 compatibility)
checks real updates, fixed reference completeness and all downstream conversions.
Smoke outputs are isolated and excluded from results. All nine production upstreams
depend only on smoke; each downstream depends only on its own upstream.
Production partition gpua100; 10h joint/4h single/2h per downstream cell (5 sequential
seeds); these are ceilings. E35 500 joint epochs took about6000 seconds (~1.67h),
so2000 is roughly6.7h before overhead;10h provides headroom. The partition limit
was checked as24h. 53 jobs = smoke+9 upstream+42 downstream+summary,210 seeds.
Submission ledger is saved after every accepted job. Old studies remain untouched.

Based on PCGrad (Yu et al., NeurIPS2020) and CAGrad (Liu et al., NeurIPS2021):
https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf
The additional bearing contribution floor is our hypothesis, not a reproduced
published guarantee and not a claim of a universal shared degradation coordinate.
