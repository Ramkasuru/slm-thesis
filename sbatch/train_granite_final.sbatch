#!/bin/bash
#SBATCH --job-name=granite_final
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/granite_final_%j.out
#SBATCH --error=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/granite_final_%j.err

set -euo pipefail

module purge
module load python39
source ~/slm_thesis_v1_final/venv/bin/activate

cd ~/slm_thesis_v1_final
mkdir -p logs outputs

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

python scripts/train.py \
  --model-key granite33_2b \
  --train-file data/train_final.jsonl \
  --output-root outputs/granite33_2b_final \
  --num-epochs 2 \
  --batch-size 2 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --max-seq-len 1024 \
  --save-steps 500 \
  --save-total-limit 2 \
  --gradient-checkpointing

echo "Granite training completed at $(date)"
