#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robot
python -u submit_cluster.py
