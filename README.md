# SLM Thesis: Ensemble Learning with Fine-Tuned vs Base Small Language Models

## Overview

This repository contains a reproducible research pipeline for training and evaluating Small Language Models (SLMs) using synthetic datasets, LoRA-based fine-tuning, and ensemble decision strategies.

### Central Research Question

**Do ensembles of fine-tuned SLMs outperform:**
1. **Ensembles of non-fine-tuned SLMs**
2. **Individual SLMs**

In structured explanatory tasks?

---

## Research Motivation

Large Language Models (LLMs) achieve strong performance, but they are computationally expensive and harder to deploy at scale. Small Language Models (SLMs) are more efficient, but they often underperform when used individually.

This project investigates whether combining:

- **parameter-efficient fine-tuning (LoRA)**
- **multiple SLMs through ensemble voting**

can improve answer quality while preserving efficiency.

---

## Comparison Groups

This work compares three experimental settings:

1. **Single SLM baseline**
2. **Ensemble of base (non-fine-tuned) SLMs**
3. **Ensemble of fine-tuned SLMs**

---

## Models

The current pipeline is designed around the following SLMs:

- **Phi**
- **Qwen**
- **Granite**

---

## Key Contributions

### End-to-End Research Pipeline
A full workflow from synthetic data generation to training, evaluation, and final comparison.

### Synthetic Data Generation Framework
Structured tutor-style outputs with filtering and deduplication.

### Efficient Fine-Tuning
LoRA-based training for scalable adaptation of SLMs.

### Evaluation Framework
Includes both quality and efficiency metrics:
- Token F1
- ROUGE-L
- Schema validation
- Latency
- Output length

### Comparative Analysis
Direct comparison across:
- single vs ensemble systems
- base vs fine-tuned systems

### HPC-Ready Implementation
Built for reproducible execution with:
- SLURM job scripts
- dependency chaining
- checkpointing for long-running jobs

---

## Project Structure

```text
slm-thesis/
├── scripts/        # Training, evaluation, and data processing
├── sbatch/         # SLURM job scripts
├── logs/           # Job outputs (ignored in git)
├── outputs/        # Model checkpoints (ignored in git)
├── data/           # Datasets (ignored in git)
└── README.md
```
# Methodology

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
```text 
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
```
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
