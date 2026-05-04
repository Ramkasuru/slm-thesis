#!/bin/bash
#SBATCH --job-name=ens_qg_full
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/ens_qg_full_%j.out
#SBATCH --error=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/ens_qg_full_%j.err

set -euo pipefail

echo "Starting QG ensemble full eval at $(date)"

module purge
module load python39
source ~/slm_thesis_v1_final/venv/bin/activate

cd ~/slm_thesis_v1_final
mkdir -p logs outputs/ensemble_qg/eval_full

python scripts/eval_ensemble.py \
  --model_keys qwen3b granite33_2b \
  --adapter_dirs outputs/qwen3b_final/qwen3b_epi_lora outputs/granite33_2b_final/granite33_2b_epi_lora \
  --eval_path data/eval_final.jsonl \
  --pred_path outputs/ensemble_qg/eval_full/preds_full.jsonl \
  --report_path outputs/ensemble_qg/eval_full/report_full.json \
  --voting_mode consensus \
  --device cuda \
  --load_in_4bit

echo "QG ensemble full eval completed at $(date)"
