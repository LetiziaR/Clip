#!/bin/bash
#SBATCH --job-name=eval-perclass
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_perclass_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate base

cd /dss/dsshome1/0F/ra59ver2/Progects/coca

python eval_perclass.py \
    --checkpoint /dss/mcmlscratch/0F/ra59ver2/checkpoints/contrastive_probe_ts2vec_20260403_223245/best.pt \
    --config configs/mimic_classif.yaml \
    --override paths.checkpoint_dir=/dss/mcmlscratch/0F/ra59ver2/checkpoints \
    --batch-size 64 \
    --num-workers 4
