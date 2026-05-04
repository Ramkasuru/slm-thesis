#!/bin/bash
#SBATCH --job-name=phi35_eval_full
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/phi35_eval_full_%j.out
#SBATCH --error=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/phi35_eval_full_%j.err

set -euo pipefail

echo "Starting full Phi35 evaluation at $(date)"

module purge
module load python39
source ~/slm_thesis_v1_final/venv/bin/activate

cd ~/slm_thesis_v1_final
mkdir -p logs outputs/phi35_final/eval_full

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

python scripts/eval_single.py \
  --model_key phi35 \
  --adapter_dir outputs/phi35_final/phi35_epi_lora \
  --eval_path data/eval_750.jsonl \
  --pred_path outputs/phi35_final/eval_full/preds_full.jsonl \
  --report_path outputs/phi35_final/eval_full/report_full.json \
  --device cuda \
  --load_in_4bit \
  --max_examples 750

echo "Full Phi35 evaluation completed at $(date)"
