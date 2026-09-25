#!/bin/bash
#SBATCH --job-name=gn19_downstream
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=logs/downstream_%j.out
#SBATCH --error=logs/downstream_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from experiment 19}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONUNBUFFERED=1
module purge
module load cuda/12.8.1/none-none
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robot
echo "Job=$SLURM_JOB_ID Domain=$1 Source=$2 Node=$SLURM_JOB_NODELIST"
nvidia-smi
srun python -u run_downstream.py --domain "$1" --initialization "$2"
