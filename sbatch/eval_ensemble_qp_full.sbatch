#!/bin/bash
#SBATCH --job-name=ens_qp_full
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/ens_qp_full_%j.out
#SBATCH --error=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/ens_qp_full_%j.err

set -euo pipefail

echo "Starting QP ensemble full eval at $(date)"

module purge
module load python39
source ~/slm_thesis_v1_final/venv/bin/activate

cd ~/slm_thesis_v1_final
mkdir -p logs outputs/ensemble_qp/eval_full

python scripts/eval_ensemble.py \
  --model_keys qwen3b phi35 \
  --adapter_dirs outputs/qwen3b_final/qwen3b_epi_lora outputs/phi35_final/phi35_epi_lora \
  --eval_path data/eval_final.jsonl \
  --pred_path outputs/ensemble_qp/eval_full/preds_full.jsonl \
  --report_path outputs/ensemble_qp/eval_full/report_full.json \
  --voting_mode consensus \
  --device cuda \
  --load_in_4bit

echo "QP ensemble full eval completed at $(date)"
