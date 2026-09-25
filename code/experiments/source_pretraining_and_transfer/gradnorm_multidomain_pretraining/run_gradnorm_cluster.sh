#!/bin/bash
#SBATCH --job-name=ht_gradnorm7
#SBATCH --output=healthtoken_gradnorm7.o%j
#SBATCH --error=healthtoken_gradnorm7.e%j
#SBATCH --time=10:00:00
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --hint=nomultithread

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from its experiment directory}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

module purge
module load cuda/12.8.1/none-none
module load anaconda3/2023.09-0/none-none

if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate robot
else
    source activate robot
fi

echo "Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Directory: $(pwd)"
python --version
nvidia-smi

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    python -u run_gradnorm.py --resume
