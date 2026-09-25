# E2. Cross-domain degradation-axis transfer

This experiment freezes the final three-domain + Adapter encoder and asks
whether a linear health direction learned from two physical domains transfers
zero-shot to the third domain.

For each held-out device, chronology is expressed only as its own normalized
lifecycle position `t/(T-1)`. A single linear direction is fitted on two source
domains against `1-t/(T-1)`, fixing the convention `high score = healthy`.
Each source domain receives equal total regression weight so that its number of
snapshots cannot dominate the direction. The fitted direction and sign are
then frozen before application to the target domain.

The three leave-one-domain-out tests are:

1. Bearing + Battery -> Milling;
2. Bearing + Milling -> Battery;
3. Battery + Milling -> Bearing.

The primary metric is zero-shot Spearman correlation between target score and
target lifecycle position. Under the fixed health-score convention, successful
early-to-late ordering produces a negative correlation. The report therefore
also gives `direction agreement = -rho` and a health-oriented concordance index.

No encoder parameter is updated and no target-domain RUL or chronology is used
to fit or flip the transferred direction.
