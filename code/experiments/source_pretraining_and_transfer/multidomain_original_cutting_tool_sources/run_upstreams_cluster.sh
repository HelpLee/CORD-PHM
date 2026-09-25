#!/bin/bash
#SBATCH --job-name=healthtoken_three_domain
#SBATCH --output=healthtoken_three_domain.o%j
#SBATCH --error=healthtoken_three_domain.e%j

#SBATCH --time=06:00:00
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --hint=nomultithread

set -euo pipefail
set -x

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

if [[ ! -f run_upstreams_only.py ]]; then
    echo "run_upstreams_only.py not found in $PWD" >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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

echo "======================================"
echo "Running three-domain HealthToken upstream pretraining"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $(python --version 2>&1)"
echo "======================================"

nvidia-smi

# train_joint.py is currently single-process/single-GPU, so one A100 is used.
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    python -u run_upstreams_only.py --resume --job three_domain "$@"
