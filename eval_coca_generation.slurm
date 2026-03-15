#!/bin/bash
#SBATCH --job-name=coca-eval-gen
#SBATCH --output=slurm-%j.out
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail
source /dss/dsshome1/0F/ra59ver2/miniforge3/etc/profile.d/conda.sh
conda activate base

cd /dss/dsshome1/0F/ra59ver2/Progects/coca

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SEED="${SEED:-42}"

mkdir -p logs

echo "[INFO] job_id=${SLURM_JOB_ID:-unknown} host=$(hostname) pwd=$PWD"
echo "[INFO] python=$(python --version 2>&1)"
echo "[INFO] date=$(date -Iseconds)"
echo "[INFO] seed=${SEED}"
echo "[INFO] pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
nvidia-smi || true

python -X faulthandler -u eval_coca_generation.py \
	--seed "${SEED}" \
	"$@" 2>&1 | tee "logs/eval_gen_${SLURM_JOB_ID:-manual}.log"