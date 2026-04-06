#!/usr/bin/env python
import argparse
import os
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


MODEL_REGISTRY = {
    "qwen3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "max_length": 2048,
    },
    "phi35": {
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "max_length": 2048,
    },
    "granite33_2b": {
        "hf_id": "ibm-granite/granite-3.3-2b-instruct",
        "max_length": 2048,
    },
}

TRAIN_FILE = "epi_train.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for thesis runtime SLMs")
    parser.add_argument(
        "--model-key",
        type=str,
        default="qwen3b",
        choices=list(MODEL_REGISTRY.keys()),
        help="Which runtime SLM to fine-tune.",
    )
    parser.add_argument("--train-file", type=str, default=TRAIN_FILE)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def format_prompt(example: Dict[str, str]) -> str:
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    if inp:
        return f"Instruction: {instruction}\n\nInput: {inp}\n\nAnswer:\n"
    return f"Instruction: {instruction}\n\nAnswer:\n"


def build_training_text(example: Dict[str, str]) -> Dict[str, str]:
    prompt = format_prompt(example)
    output = (example.get("output") or "").strip()
    return {
        "prompt": prompt,
        "text": prompt + output,
    }


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def _tok(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        tokenized_full = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized_prompt = tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        labels = []
        for input_ids, prompt_ids in zip(tokenized_full["input_ids"], tokenized_prompt["input_ids"]):
            masked = input_ids.copy()
            prompt_len = min(len(prompt_ids), len(masked))
            masked[:prompt_len] = [-100] * prompt_len
            labels.append(masked)

        tokenized_full["labels"] = labels
        return tokenized_full

    return dataset.map(
        _tok,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing with prompt masking",
    )


def find_target_modules(model) -> List[str]:
    linear_names = sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }
    )
    preferred_endings = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "qkv_proj",
        "gate_up_proj",
    ]
    target_modules = [name for name in linear_names if name in preferred_endings]
    return target_modules or linear_names


def main():
    args = parse_args()
    model_info = MODEL_REGISTRY[args.model_key]
    model_name = model_info["hf_id"]
    max_length = args.max_seq_len or model_info["max_length"]
    output_dir = os.path.join(args.output_root, f"{args.model_key}_epi_lora")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Model key      : {args.model_key}")
    print(f"Hugging Face ID: {model_name}")
    print(f"Train file     : {args.train_file}")
    print(f"Output dir     : {output_dir}")
    print(f"Max seq len    : {max_length}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    target_modules = find_target_modules(model)
    print(f"Using LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": args.train_file})
    train_dataset = dataset["train"].map(build_training_text, desc="Formatting examples")
    print(f"Train samples: {len(train_dataset)}")

    tokenized_dataset = tokenize_dataset(train_dataset, tokenizer, max_length)
    print("Tokenized columns:", tokenized_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        ),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving LoRA adapter + tokenizer to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Training complete.")


if __name__ == "__main__":
    main()
