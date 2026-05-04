# SLM Thesis: Ensemble Learning with Fine-Tuned vs Base Small Language Models

## Overview
This repository contains a reproducible research pipeline for training and evaluating Small Language Models (SLMs) using synthetic datasets, LoRA-based fine-tuning, and ensemble decision strategies.

The central research question explored in this work is:

Do ensembles of fine-tuned SLMs outperform:
1) ensembles of non-fine-tuned SLMs, and  
2) individual SLMs  
in structured explanatory tasks?

---

## Research Motivation
Large Language Models (LLMs) achieve strong performance but are computationally expensive and difficult to deploy at scale. Small Language Models (SLMs) are efficient but often weaker individually.

This project investigates whether combining:
- fine-tuning (LoRA)
- multiple SLMs (ensembles)

can bridge this performance gap while maintaining efficiency.

---

## Experimental Setup

### Models
- Phi
- Qwen
- Granite

### Training Strategy
- Parameter-efficient fine-tuning using LoRA
- Synthetic dataset of structured tutor-style explanations

### Comparison Groups
1. Single SLM (baseline)
2. Ensemble of base (non-fine-tuned) SLMs
3. Ensemble of fine-tuned SLMs (proposed approach)

---

## Key Contributions

- End-to-End Research Pipeline  
  Data generation → training → evaluation → comparison  

- Synthetic Data Generation Framework  
  Structured tutor-style outputs with filtering and deduplication  

- Efficient Fine-Tuning  
  LoRA-based training for scalable adaptation  

- Evaluation Framework  
  Token F1  
  ROUGE-L  
  Schema validation  
  Latency and output length  

- Comparative Analysis  
  Single vs ensemble models  
  Fine-tuned vs non-fine-tuned systems  

- HPC-Ready Implementation  
  SLURM-based execution  
  Dependency chaining  
  Checkpointing for long-running jobs  

---

## Project Structure

slm-thesis/
│
├── scripts/        # Training, evaluation, data processing
├── sbatch/         # SLURM job scripts (train + eval)
├── logs/           # Job outputs (ignored in git)
├── outputs/        # Model checkpoints (ignored in git)
├── data/           # Datasets (ignored in git)
└── README.md

---

## Methodology

### 1. Dataset Generation
- Synthetic tutor-style question-answer pairs
- Structured bullet-based explanations
- Filtering and deduplication applied

### 2. Data Split
- ~90% training
- ~10% evaluation

### 3. Fine-Tuning
- LoRA-based parameter-efficient training
- GPU execution using SLURM

### 4. Evaluation Metrics

- Token F1
- ROUGE-L
- Schema compliance
- Latency
- Output length

Note: Exact Match (EM) is not used due to the open-ended nature of explanatory tasks.

---

## Pipeline

Synthetic Data Generation
        ↓
Dataset Cleaning & Deduplication
        ↓
Train/Test Split
        ↓
Model Training (Phi, Qwen, Granite)
        ↓
Evaluation (Single Models)
        ↓
Ensemble Evaluation
        ↓
Comparative Analysis

---

## Current Status

- Dataset generation (~7.5k samples)
- Train/eval split
- LoRA training pipeline
- Single-model evaluation setup

Pending:
- Base vs fine-tuned comparison
- Ensemble evaluation
- Final analysis

---

## Future Work

- Implement semantic key-fact scoring
- Improve ensemble voting strategies
- Analyze efficiency vs performance trade-offs
- Extend to larger datasets

---

## Reproducibility

This repository is designed to be reproducible:
- Modular scripts
- SLURM-based job execution
- Clean separation of code, data, and outputs

---

## Author

Ram Kasuru  
Master’s Student, Applied AI  
Stevens Institute of Technology

---

## License

To be added

## Thesis Pipeline Overview

This repository contains the experimental pipeline for evaluating fine-tuned small language models and ensemble voting strategies for structured explanation generation in population dynamics and computational epidemiology.

### Main Components

- `scripts/`: training and evaluation scripts
- `sbatch/`: SLURM job scripts for Jarvis L40S runs
- `data/`: train/evaluation JSONL files when available
- `results/final/`: final metric reports used in the thesis
- `outputs/`: generated experiment artifacts; large checkpoints are excluded from Git

### Final Reported Ensemble Results

The final results include:
- Phi + Qwen consensus ensemble
- Granite + Qwen consensus ensemble
- Qwen + Phi + Granite consensus ensemble
- Phi + Granite fact-agreement ensemble

Large model checkpoints and cache files are intentionally excluded from version control.

## Thesis Pipeline Overview

This repository contains the experimental pipeline for evaluating fine-tuned small language models and ensemble voting strategies for structured explanation generation in population dynamics and computational epidemiology.

### Main Components

- `scripts/`: training and evaluation scripts
- `sbatch/`: SLURM job scripts for Jarvis L40S runs
- `results/final/`: final metric reports used in the thesis
- `outputs/`: generated experiment artifacts; large checkpoints are excluded from Git

### Final Reported Ensemble Results

The final results include:
- Phi + Qwen consensus ensemble
- Granite + Qwen consensus ensemble
- Qwen + Phi + Granite consensus ensemble
- Phi + Granite fact-agreement ensemble

Large model checkpoints, virtual environments, caches, and logs are intentionally excluded from version control.
