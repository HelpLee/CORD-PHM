#!/bin/bash
#SBATCH --job-name=ht_transformer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-179%2
#SBATCH --output=/path/to/CORD/code/healthtoken_final_study/A_main/06_fair_baselines/slurm/transformer_%A_%a.out
#SBATCH --error=/path/to/CORD/code/healthtoken_final_study/A_main/06_fair_baselines/slurm/transformer_%A_%a.err

set -euo pipefail
BASE=/path/to/CORD/code/healthtoken_final_study/A_main/06_fair_baselines
cd "$BASE"
export HEALTHTOKEN_BASELINE_PREPARED="$BASE/prepared"
export HEALTHTOKEN_MOMENT_MODEL="$BASE/models/MOMENT-1-base"
export HF_HOME="$BASE/.hf_cache"
export OMP_NUM_THREADS=4
PY=python
srun "$PY" -u slurm_task.py --group transformer --index "$SLURM_ARRAY_TASK_ID"
