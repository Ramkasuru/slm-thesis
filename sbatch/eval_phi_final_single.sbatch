#!/bin/bash
#SBATCH --job-name=phi35_eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/phi35_eval_%j.out
#SBATCH --error=logs/phi35_eval_%j.err

set -euo pipefail

module purge
module load python39
source ~/venvs/slm_env/bin/activate

cd ~/slm_thesis_v1_final
mkdir -p logs outputs

echo "==== TORCH CHECK ===="
which python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python scripts/eval_single.py \
  --model_key phi35 \
  --adapter_dir outputs/phi35_final/phi35_epi_lora \
  --eval_path data/eval_final.jsonl \
  --pred_path outputs/phi35_final_preds.jsonl \
  --report_path outputs/phi35_final_report.json \
  --max_examples 50 \
  --max_new_tokens 1024 \
  --device cuda \
  --load_in_4bit \
  --fact_threshold 0.2 \
  --verbose_examples 5
