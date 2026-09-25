#!/bin/bash
# Submit three independent jobs. Re-running this script skips active or completed arms.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p single_domain/submissions
for domain in bearing battery milling; do
    name="ht_norm_${domain}"
    active="$(squeue --noheader --user "$USER" --name "$name" --format='%i')"
    if [ -n "$active" ]; then
        echo "Already queued: ${domain} ${active}"
        continue
    fi
    state="single_domain/${domain}/training/status.json"
    if [ -f "$state" ] && grep -Eq '"state"[[:space:]]*:[[:space:]]*"complete"' "$state"; then
        echo "Already complete: ${domain}"
        continue
    fi
    job="$(sbatch --parsable --job-name="$name" run_single_domain_cluster.sh "$domain")"
    printf '%s\n' "$job" > "single_domain/submissions/${domain}.jobid"
    echo "SUBMITTED ${domain} ${job}"
done
