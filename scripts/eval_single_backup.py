import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_MAP = {
    "phi35": "microsoft/Phi-3.5-mini-instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "granite33_2b": "ibm-granite/granite-3.3-2b-instruct",
}


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no} of {path}: {e}") from e
    return rows


def save_jsonl(path: str, rows: List[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", normalize_text(text))


def token_f1(pred: str, ref: str) -> float:
    p = simple_tokens(pred)
    r = simple_tokens(ref)

    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0

    pc = Counter(p)
    rc = Counter(r)
    overlap = sum((pc & rc).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def lcs_len(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def rouge_l_f1(pred: str, ref: str) -> float:
    p = simple_tokens(pred)
    r = simple_tokens(ref)

    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0

    lcs = lcs_len(p, r)
    if lcs == 0:
        return 0.0

    precision = lcs / len(p)
    recall = lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_bullets(text: str) -> List[str]:
    bullets = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            bullets.append(line[2:].strip())
    return bullets


def valid_schema(text: str) -> bool:
    bullets = extract_bullets(text)
    if not (4 <= len(bullets) <= 6):
        return False

    for bullet in bullets:
        if not bullet:
            return False
        if bullet.count(".") + bullet.count("!") + bullet.count("?") > 1:
            return False

    return True


def key_fact_f1(pred: str, ref: str, threshold: float = 0.2) -> float:
    ref_facts = extract_bullets(ref)
    pred_facts = extract_bullets(pred)

    if not ref_facts and not pred_facts:
        return 1.0
    if not ref_facts or not pred_facts:
        return 0.0

    matched_ref = 0
    used_pred = set()

    for rf in ref_facts:
        best_j = None
        best_score = 0.0
        for j, pf in enumerate(pred_facts):
            if j in used_pred:
                continue
            score = token_f1(pf, rf)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None and best_score >= threshold:
            matched_ref += 1
            used_pred.add(best_j)

    precision = matched_ref / len(pred_facts) if pred_facts else 0.0
    recall = matched_ref / len(ref_facts) if ref_facts else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_key: str,
    adapter_dir: Optional[str],
    device: str,
    load_in_4bit: bool,
):
    hf_id = MODEL_MAP[model_key]
    dtype = resolve_dtype()

    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map="auto" if device in {"auto", "cuda"} and torch.cuda.is_available() else None,
        quantization_config=quantization_config,
    )

    if adapter_dir and os.path.isdir(adapter_dir):
        model = PeftModel.from_pretrained(model, adapter_dir)

    if not load_in_4bit:
        if device in {"auto", "cuda"} and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

    model.eval()

    print("CUDA available:", torch.cuda.is_available())
    try:
        print("Model device:", next(model.parameters()).device)
    except Exception as e:
        print("Could not inspect model device:", e)

    return model, tokenizer, hf_id


def build_prompt(tokenizer, instruction: str, user_input: str) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{user_input}\n\n"
        f"### Response:\n"
    )


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[str, float, int]:
    inputs = tokenizer(prompt, return_tensors="pt")

    if hasattr(model, "device"):
        target_device = model.device
        if str(target_device) != "cpu":
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
    elif torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return decoded, elapsed, len(new_tokens)


def main():
    parser = argparse.ArgumentParser(description="Single-model eval for smoke tests")
    parser.add_argument("--model_key", required=True, choices=sorted(MODEL_MAP.keys()))
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--fact_threshold", type=float, default=0.2)
    parser.add_argument("--verbose_examples", type=int, default=3)
    args = parser.parse_args()

    rows = load_jsonl(args.eval_path)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    model, tokenizer, hf_id = load_model_and_tokenizer(
        model_key=args.model_key,
        adapter_dir=args.adapter_dir,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )

    pred_rows = []
    schema_scores = []
    fact_scores = []
    rouge_scores = []
    token_f1_scores = []
    latencies = []
    new_token_counts = []

    print(f"\nEvaluating {args.model_key}")
    print(f"HF model: {hf_id}")
    print(f"Adapter: {args.adapter_dir}\n")

    for idx, ex in enumerate(rows, start=1):
        instruction = ex["instruction"]
        user_input = ex["input"]
        reference = ex["output"]

        prompt = build_prompt(tokenizer, instruction, user_input)
        prediction, latency_s, new_tokens = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )

        schema_ok = valid_schema(prediction)
        fact_f1 = key_fact_f1(prediction, reference, threshold=args.fact_threshold)
        rouge = rouge_l_f1(prediction, reference)
        tf1 = token_f1(prediction, reference)

        schema_scores.append(1.0 if schema_ok else 0.0)
        fact_scores.append(fact_f1)
        rouge_scores.append(rouge)
        token_f1_scores.append(tf1)
        latencies.append(latency_s)
        new_token_counts.append(new_tokens)

        pred_rows.append({
            "index": idx,
            "instruction": instruction,
            "input": user_input,
            "reference": reference,
            "prediction": prediction,
            "valid_schema": schema_ok,
            "key_fact_f1": fact_f1,
            "rouge_l_f1": rouge,
            "token_f1": tf1,
            "latency_s": latency_s,
            "new_tokens": new_tokens,
        })

        if idx <= args.verbose_examples:
            print("-" * 60)
            print(f"Example {idx}")
            print(f"Question: {user_input}")
            print(f"Prediction:\n{prediction}\n")
            print(f"Valid schema: {schema_ok}")
            print(f"Key-Fact F1: {fact_f1:.3f}")
            print(f"ROUGE-L F1: {rouge:.3f}")
            print(f"Token F1: {tf1:.3f}")

    report = {
        "model_key": args.model_key,
        "hf_id": hf_id,
        "adapter_dir": args.adapter_dir,
        "eval_path": args.eval_path,
        "num_examples": len(pred_rows),
        "avg_valid_schema": sum(schema_scores) / len(schema_scores) if schema_scores else 0.0,
        "avg_key_fact_f1": sum(fact_scores) / len(fact_scores) if fact_scores else 0.0,
        "avg_rouge_l_f1": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "avg_token_f1": sum(token_f1_scores) / len(token_f1_scores) if token_f1_scores else 0.0,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "avg_new_tokens": sum(new_token_counts) / len(new_token_counts) if new_token_counts else 0.0,
    }

    save_jsonl(args.pred_path, pred_rows)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
