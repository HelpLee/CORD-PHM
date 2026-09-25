#!/bin/bash
#SBATCH --job-name=ht_single_norm
#SBATCH --output=healthtoken_single_%j.out
#SBATCH --error=healthtoken_single_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --hint=nomultithread

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the experiment 17 directory}"
DOMAIN="${1:?Expected bearing, battery, or milling}"
case "$DOMAIN" in bearing|battery|milling) ;; *) echo "Invalid domain: $DOMAIN" >&2; exit 2 ;; esac
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONUNBUFFERED=1
module purge
module load cuda/12.8.1/none-none
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robot
echo "Job: ${SLURM_JOB_ID}; node: ${SLURM_JOB_NODELIST}; single domain: ${DOMAIN}"
python --version
nvidia-smi
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 python -u run_single_domain.py --domain "$DOMAIN" --resume
