#!/bin/bash
#SBATCH --job-name=coca-best-train
#SBATCH --output=slurm-%j.out
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

set -euo pipefail
source /dss/dsshome1/0F/ra59ver2/miniforge3/etc/profile.d/conda.sh
conda activate base

cd /dss/dsshome1/0F/ra59ver2/Progects/coca

export PYTHONUNBUFFERED=1
mkdir -p logs

echo "[INFO] job_id=${SLURM_JOB_ID:-unknown} host=$(hostname) pwd=$PWD"
echo "[INFO] python=$(python --version 2>&1)"
echo "[INFO] date=$(date -Iseconds)"
nvidia-smi || true

torchrun --nproc_per_node=gpu python -X faulthandler -u run_coca.py \
  --run_name coca_best_lr1e4_bs32_temp007_pseudo_to_report \
  --learning_rate 1e-4 \
  --batch_size 32 \
  --temperature 0.07 \
  --epochs 20 \
  --ts_arch ts2vec \
  --decoder_arch bart \
  --text_source pseudo_report \
  "$@" 2>&1 | tee "logs/train_best_${SLURM_JOB_ID:-manual}.log"
