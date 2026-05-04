#!/bin/bash
#SBATCH --job-name=ens_qpg_smoke
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/ens_qpg_smoke_%j.out
#SBATCH --error=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/ens_qpg_smoke_%j.err

set -euo pipefail

echo "Starting QPG ensemble smoke eval at $(date)"

module purge
module load python39
source ~/slm_thesis_v1_final/venv/bin/activate

cd ~/slm_thesis_v1_final
mkdir -p logs outputs/ensemble_qpg/eval_smoke

python scripts/eval_ensemble.py \
  --model_keys qwen3b phi35 granite33_2b \
  --adapter_dirs outputs/qwen3b_final/qwen3b_epi_lora outputs/phi35_final/phi35_epi_lora outputs/granite33_2b_final/granite33_2b_epi_lora \
  --eval_path data/eval_final.jsonl \
  --pred_path outputs/ensemble_qpg/eval_smoke/preds.jsonl \
  --report_path outputs/ensemble_qpg/eval_smoke/report.json \
  --voting_mode consensus \
  --device cuda \
  --load_in_4bit \
  --max_examples 2 \
  --verbose_examples 2

echo "QPG ensemble smoke eval completed at $(date)"
