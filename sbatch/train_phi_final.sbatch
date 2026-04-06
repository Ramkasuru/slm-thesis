#!/bin/bash
#SBATCH --job-name=phi35_final
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/phi35_final_%j.out
#SBATCH --error=logs/phi35_final_%j.err

set -euo pipefail

module purge
module load python39
source ~/venvs/slm_env/bin/activate

cd ~/slm_thesis_v1_final
mkdir -p logs outputs

echo "==== TORCH CHECK ===="
which python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python scripts/train.py \
  --model-key phi35 \
  --train-file data/train_final.jsonl \
  --output-root outputs/phi35_final \
  --num-epochs 2 \
  --batch-size 2 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --max-seq-len 1024 \
  --save-steps 500 \
  --save-total-limit 2 \
  --gradient-checkpointing
EOF

