#!/bin/bash
#SBATCH --job-name=ens_pg_full
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/ens_pg_full_%j.out
#SBATCH --error=/mmfs1/home/kkasuru/slm_thesis_v1_final/logs/ens_pg_full_%j.err

set -euo pipefail

module purge
module load python39
module load cuda12.2/toolkit/12.2.2
source ~/slm_thesis_v1_final/venv/bin/activate

cd ~/slm_thesis_v1_final

python scripts/eval_pg.py \
  --eval_path data/eval_750.jsonl  \
  --cache_dir outputs/eval_cache_pg_full \
  --report_path outputs/ensemble_pg_full_report.json \
  --failure_out outputs/ensemble_pg_full_failures.jsonl \
  --max_new_tokens 256 \
  --device cuda \
  --ensemble_mode fact_agreement \
  --agreement_threshold 2 \
  --gpu_hourly_cost 1.80 \
  --load_in_4bit \
  --verbose_examples 3

echo "PG ensemble eval completed at $(date)"
