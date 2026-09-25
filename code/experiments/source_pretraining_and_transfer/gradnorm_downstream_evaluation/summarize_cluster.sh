#!/bin/bash
#SBATCH --job-name=gn19_summary
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/summary_%j.out
#SBATCH --error=logs/summary_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from experiment 19}"
python -u summarize.py
