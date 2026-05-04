#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_CONFIGS = [
    {
        "name": "phi35_ft",
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "adapter_dir": "outputs/phi35_final/phi35_epi_lora",
    },
    {
        "name": "granite33_2b_ft",
        "hf_id": "ibm-granite/granite-3.3-2b-instruct",
        "adapter_dir": "outputs/granite33_2b_final/granite33_2b_epi_lora",
    },
]

ENSEMBLE_MEMBERS = ["phi35_ft", "granite33_2b_ft"]

def parse_args():
    parser = argparse.ArgumentParser(description="Structured evaluation for the thesis SLM ensemble")
    parser.add_argument("--eval_path", type=str, default="epi_eval.jsonl")
    parser.add_argument("--cache_dir", type=str, default="outputs/eval_cache")
    parser.add_argument("--report_path", type=str, default="outputs/eval_report.json")
    parser.add_argument("--failure_out", type=str, default="outputs/eval_failures.jsonl")
    parser.add_argument("--human_eval_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fact_threshold", type=float, default=0.60)
    parser.add_argument("--agreement_threshold", type=int, default=2)
    parser.add_argument("--ensemble_mode", choices=["fact_agreement", "simple_majority"], default="fact_agreement")
    parser.add_argument("--gpu_hourly_cost", type=float, default=1.80)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--verbose_examples", type=int, default=3)
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_set(s: str) -> set:
    return set(normalize_text(s).split())


def fact_similarity(a: str, b: str) -> float:
    a_tokens = token_set(a)
    b_tokens = token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(ref_tokens) + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def extract_bullets(answer: str) -> List[str]:
    bullets = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if re.match(r"^[-*•]\s+", line):
            bullets.append(re.sub(r"^[-*•]\s+", "", line).strip())
    return bullets


def validate_prediction_schema(obj: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, int], Optional[Dict[str, Any]]]:
    flags = {
        "valid_json": 0,
        "valid_schema": 0,
        "valid_bullet_count": 0,
        "valid_key_fact_count": 0,
    }
    if not isinstance(obj, dict):
        return False, flags, None

    flags["valid_json"] = 1
    answer = obj.get("answer")
    key_facts = obj.get("key_facts")
    confidence = obj.get("confidence")
    if not isinstance(answer, str) or not answer.strip():
        return False, flags, None
    if not isinstance(key_facts, list) or not all(isinstance(x, str) and x.strip() for x in key_facts):
        return False, flags, None
    if not isinstance(confidence, int) or not (1 <= confidence <= 5):
        return False, flags, None

    bullets = extract_bullets(answer)
    if 4 <= len(bullets) <= 6:
        flags["valid_bullet_count"] = 1
    if 3 <= len(key_facts) <= 5:
        flags["valid_key_fact_count"] = 1

    if flags["valid_bullet_count"] and flags["valid_key_fact_count"]:
        flags["valid_schema"] = 1
        return True, flags, {
            "answer": answer.strip(),
            "key_facts": [x.strip() for x in key_facts],
            "confidence": confidence,
        }
    return False, flags, None


def get_reference_facts(example: Dict[str, Any]) -> List[str]:
    key_facts = example.get("key_facts")
    if isinstance(key_facts, list) and key_facts:
        return [str(x).strip() for x in key_facts if str(x).strip()]
    bullets = extract_bullets(example.get("output", ""))
    return bullets[:5]


def greedy_fact_match(predicted: List[str], reference: List[str], threshold: float) -> Tuple[int, List[Tuple[str, str, float]]]:
    matches = []
    used = set()
    correct = 0
    for pred_fact in predicted:
        best_idx = None
        best_score = 0.0
        for idx, ref_fact in enumerate(reference):
            if idx in used:
                continue
            score = fact_similarity(pred_fact, ref_fact)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= threshold:
            used.add(best_idx)
            correct += 1
            matches.append((pred_fact, reference[best_idx], best_score))
    return correct, matches


def key_fact_metrics(predicted: List[str], reference: List[str], threshold: float) -> Dict[str, Any]:
    correct, matches = greedy_fact_match(predicted, reference, threshold)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(reference) if reference else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "matches": matches,
    }


def build_eval_prompt(instruction: str, inp: str) -> str:
    task = inp.strip() if inp else instruction.strip()
    return (
        "You are a population dynamics and computational epidemiology tutor.\n"
        "Return ONLY valid JSON matching this schema:\n"
        "{\"answer\":\"- bullet\\n- bullet\\n- bullet\\n- bullet\","
        "\"key_facts\":[\"fact 1\",\"fact 2\",\"fact 3\"],\"confidence\":1}\n"
        "Rules:\n"
        "- answer must contain 4 to 6 bullets\n"
        "- key_facts must contain 3 to 5 short atomic facts\n"
        "- confidence must be an integer from 1 to 5\n"
        "- do not include markdown fences or extra commentary\n\n"
        f"Question: {task}\n"
    )


def load_model_and_tokenizer(hf_id: str, adapter_dir: Optional[str], device: str, load_in_4bit: bool):
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    if not load_in_4bit:
        if adapter_dir and os.path.isdir(adapter_dir):
            adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

            if len(adapter_tokenizer) != len(tokenizer):
                tokenizer = adapter_tokenizer
                model.resize_token_embeddings(len(tokenizer))

            model = PeftModel.from_pretrained(model, adapter_dir)
      
        if device in {"auto","cuda"} and torch.cuda.is_available():
           model = model.to("cuda")
        else:
             model = model.to("cpu")

    model.eval()
 
    #  Debug 
   
    print("CUDA available:", torch.cuda.is_available())
    try:
        print("Model device:", next(model.parameters()).device)
    except Exception as e:
        print("Could not inspect model device:", e)
 
    return model, tokenizer


def generate_prediction(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    max_input_tokens = min(int(getattr(tokenizer, "model_max_length", 4096) or 4096), 4096)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    prompt_tokens = int(encoded["input_ids"].shape[1])
    model_device = next(model.parameters()).device
    encoded = {k: v.to(model_device) for k, v in encoded.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_sec = time.time() - start
    gen_ids = outputs[0][encoded["input_ids"].shape[1]:]
    gen_tokens = int(gen_ids.shape[0])
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return {
        "raw_text": raw_text,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "latency_sec": latency_sec,
    }


def evaluate_single_prediction(
    example: Dict[str, Any],
    parsed: Optional[Dict[str, Any]],
    flags: Dict[str, int],
    fact_threshold: float,
) -> Dict[str, Any]:
    reference_answer = example.get("output", "")
    reference_facts = get_reference_facts(example)
    if not parsed:
        return {
            "key_fact_precision": 0.0,
            "key_fact_recall": 0.0,
            "key_fact_f1": 0.0,
            "token_f1": 0.0,
            "rouge_l": 0.0,
            "flags": flags,
            "agreed_facts": 0,
            "reference_fact_count": len(reference_facts),
        }

    fact_scores = key_fact_metrics(parsed["key_facts"], reference_facts, fact_threshold)
    return {
        "key_fact_precision": fact_scores["precision"],
        "key_fact_recall": fact_scores["recall"],
        "key_fact_f1": fact_scores["f1"],
        "token_f1": token_f1(parsed["answer"], reference_answer),
        "rouge_l": rouge_l(parsed["answer"], reference_answer),
        "flags": flags,
        "matches": fact_scores["matches"],
        "reference_fact_count": len(reference_facts),
    }


def cluster_facts(facts_by_model: Dict[str, List[str]], threshold: float) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for model_name, facts in facts_by_model.items():
        for fact in facts:
            placed = False
            for cluster in clusters:
                if fact_similarity(fact, cluster["canonical_fact"]) >= threshold:
                    cluster["facts"].append(fact)
                    cluster["models"].add(model_name)
                    placed = True
                    break
            if not placed:
                clusters.append(
                    {
                        "canonical_fact": fact,
                        "facts": [fact],
                        "models": {model_name},
                    }
                )
    return clusters


def choose_ensemble_prediction(
    model_outputs: Dict[str, Optional[Dict[str, Any]]],
    per_model_dev_scores: Dict[str, float],
    mode: str,
    fact_threshold: float,
    agreement_threshold: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    valid = {k: v for k, v in model_outputs.items() if v is not None}
    if not valid:
        return None, {
            "agreement_rate": 0.0,
            "avg_support": 0.0,
            "agreed_fact_count": 0,
            "total_clusters": 0,
        }

    if mode == "simple_majority":
        grouped = Counter(normalize_text(v["answer"]) for v in valid.values())
        best_norm, _ = grouped.most_common(1)[0]
        candidates = [(name, pred) for name, pred in valid.items() if normalize_text(pred["answer"]) == best_norm]
        candidates.sort(
            key=lambda item: (
                per_model_dev_scores.get(item[0], 0.0),
                item[1]["confidence"],
                -len(item[1]["answer"]),
                item[0] == "phi35_ft"
            ),
            reverse=True,
        )
        return candidates[0][1], {
            "agreement_rate": 0.0,
            "avg_support": 0.0,
            "agreed_fact_count": 0,
            "total_clusters": 0,
        }

    facts_by_model = {model_name: pred["key_facts"] for model_name, pred in valid.items()}
    clusters = cluster_facts(facts_by_model, fact_threshold)
    support_values = [len(cluster["models"]) for cluster in clusters]
    agreed_clusters = [cluster for cluster in clusters if len(cluster["models"]) >= agreement_threshold]
    agreed_fact_set = {cluster["canonical_fact"] for cluster in agreed_clusters}

    best_name = None
    best_score = None
    for model_name, pred in valid.items():
        agreed = sum(
            1 for fact in pred["key_facts"]
            if any(fact_similarity(fact, agreed_fact) >= fact_threshold for agreed_fact in agreed_fact_set)
        )
        support_scores = []
        for fact in pred["key_facts"]:
            best_support = 1
            for cluster in clusters:
                if fact_similarity(fact, cluster["canonical_fact"]) >= fact_threshold:
                    best_support = max(best_support, len(cluster["models"]))
            support_scores.append(best_support)

        score_tuple = (
            agreed / max(len(pred["key_facts"]), 1),
            mean(support_scores) if support_scores else 0.0,
            pred["confidence"] / 5.0,
            per_model_dev_scores.get(model_name, 0.0),
            -len(pred["answer"]),
            1 if model_name == "phi35_ft" else 0,
        )
        if best_score is None or score_tuple > best_score:
            best_score = score_tuple
            best_name = model_name

    return valid[best_name], {
        "agreement_rate": len(agreed_clusters) / max(len(clusters), 1),
        "avg_support": mean(support_values) if support_values else 0.0,
        "agreed_fact_count": len(agreed_clusters),
        "total_clusters": len(clusters),
    }


def summarize_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    def avg(key: str) -> float:
        return sum(row.get(key, 0.0) for row in rows) / len(rows)

    key_fact_scores = [row.get("key_fact_f1", 0.0) for row in rows]
    return {
        "key_fact_f1": avg("key_fact_f1"),
        "key_fact_precision": avg("key_fact_precision"),
        "key_fact_recall": avg("key_fact_recall"),
        "token_f1": avg("token_f1"),
        "rouge_l": avg("rouge_l"),
        "valid_json_rate": avg("valid_json"),
        "structure_compliance_rate": avg("valid_schema"),
        "valid_bullet_count_rate": avg("valid_bullet_count"),
        "valid_key_fact_count_rate": avg("valid_key_fact_count"),
        "agreement_rate": avg("agreement_rate"),
        "avg_agreement_per_example": avg("avg_support"),
        "key_fact_f1_variance": mean([(score - avg("key_fact_f1")) ** 2 for score in key_fact_scores]) if key_fact_scores else 0.0,
        "examples": len(rows),
    }


def load_human_eval(path: Optional[str]) -> Optional[Dict[str, Dict[str, float]]]:
    if not path:
        return None
    rows = []
    if path.endswith(".jsonl"):
        rows = load_jsonl(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["system"]].append(row)

    summary = {}
    for system, system_rows in grouped.items():
        metrics = {}
        for key in ("correctness", "clarity", "completeness", "usefulness"):
            values = [float(row[key]) for row in system_rows if row.get(key) not in (None, "")]
            metrics[key] = sum(values) / len(values) if values else 0.0
        summary[system] = metrics
    return summary


def cache_predictions(path: str, rows: List[Dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def main():
    args = parse_args()
    ensure_parent(args.report_path)
    ensure_parent(args.failure_out)
    os.makedirs(args.cache_dir, exist_ok=True)

    data = load_jsonl(args.eval_path)
    if args.max_examples is not None:
        data = data[:args.max_examples]
    for idx, example in enumerate(data):
        example.setdefault("example_id", idx)

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"

    all_predictions: Dict[str, List[Dict[str, Any]]] = {}
    per_model_rows: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Any] = {
        "eval_path": args.eval_path,
        "fact_threshold": args.fact_threshold,
        "agreement_threshold": args.agreement_threshold,
        "ensemble_mode": args.ensemble_mode,
        "models": {},
    }

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        print(f"\nEvaluating {name}")
        model, tokenizer = load_model_and_tokenizer(
            cfg["hf_id"],
            cfg["adapter_dir"],
            requested_device,
            load_in_4bit=args.load_in_4bit,
        )

        cache_rows = []
        metric_rows = []
        for idx, example in enumerate(data):
            prompt = build_eval_prompt(example.get("instruction", ""), example.get("input", ""))
            pred = generate_prediction(model, tokenizer, prompt, args.max_new_tokens)
            raw_obj = extract_json_obj(pred["raw_text"])
            is_valid, flags, parsed = validate_prediction_schema(raw_obj)
            metrics = evaluate_single_prediction(example, parsed, flags, args.fact_threshold)
            row = {
                "example_id": example["example_id"],
                "model_name": name,
                "raw_text": pred["raw_text"],
                "parsed": parsed,
                "latency_sec": pred["latency_sec"],
                "prompt_tokens": pred["prompt_tokens"],
                "gen_tokens": pred["gen_tokens"],
                "key_fact_f1": metrics["key_fact_f1"],
                "key_fact_precision": metrics["key_fact_precision"],
                "key_fact_recall": metrics["key_fact_recall"],
                "token_f1": metrics["token_f1"],
                "rouge_l": metrics["rouge_l"],
                "valid_json": flags["valid_json"],
                "valid_schema": flags["valid_schema"],
                "valid_bullet_count": flags["valid_bullet_count"],
                "valid_key_fact_count": flags["valid_key_fact_count"],
                "reference_fact_count": metrics["reference_fact_count"],
            }
            cache_rows.append(row)
            metric_rows.append(row)

            if idx < args.verbose_examples:
                print("-" * 60)
                print(f"Example {idx + 1}")
                print(f"Question: {example.get('input', '')}")
                print(f"Raw output: {pred['raw_text']}")
                print(f"Valid schema: {is_valid}")
                if parsed:
                    print(f"Key-Fact F1: {metrics['key_fact_f1']:.3f}")

        cache_path = os.path.join(args.cache_dir, f"{name}.jsonl")
        cache_predictions(cache_path, cache_rows)
        all_predictions[name] = cache_rows
        per_model_rows[name] = metric_rows

        model_summary = summarize_metrics(metric_rows)
        total_latency = sum(row["latency_sec"] for row in metric_rows)
        total_prompt_tokens = sum(row["prompt_tokens"] for row in metric_rows)
        total_gen_tokens = sum(row["gen_tokens"] for row in metric_rows)
        model_summary.update(
            {
                "latency_sec_total": total_latency,
                "examples_per_sec": len(metric_rows) / max(total_latency, 1e-9),
                "prompt_tokens_per_example": total_prompt_tokens / max(len(metric_rows), 1),
                "tokens_per_example": total_gen_tokens / max(len(metric_rows), 1),
                "total_generated_tokens": total_gen_tokens,
                "gpu_hours": total_latency / 3600.0,
                "estimated_cost_usd": (total_latency / 3600.0) * args.gpu_hourly_cost,
                "cache_path": cache_path,
            }
        )
        summary["models"][name] = model_summary

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dev_scores = {name: metrics["key_fact_f1"] for name, metrics in summary["models"].items()}
    ensemble_rows = []
    failure_rows = []
    for example in data:
        example_id = example["example_id"]
        model_outputs = {}
        for member in ENSEMBLE_MEMBERS:
            pred_row = next((row for row in all_predictions.get(member, []) if row["example_id"] == example_id), None)
            model_outputs[member] = pred_row["parsed"] if pred_row else None

        chosen, agreement_stats = choose_ensemble_prediction(
            model_outputs,
            dev_scores,
            args.ensemble_mode,
            args.fact_threshold,
            args.agreement_threshold,
        )

        flags = {
            "valid_json": 1 if chosen else 0,
            "valid_schema": 1 if chosen else 0,
            "valid_bullet_count": 1 if chosen else 0,
            "valid_key_fact_count": 1 if chosen else 0,
        }
        metrics = evaluate_single_prediction(example, chosen, flags, args.fact_threshold)
        row = {
            "example_id": example_id,
            "system": "ensemble_ft",
            "parsed": chosen,
            "key_fact_f1": metrics["key_fact_f1"],
            "key_fact_precision": metrics["key_fact_precision"],
            "key_fact_recall": metrics["key_fact_recall"],
            "token_f1": metrics["token_f1"],
            "rouge_l": metrics["rouge_l"],
            "valid_json": flags["valid_json"],
            "valid_schema": flags["valid_schema"],
            "valid_bullet_count": flags["valid_bullet_count"],
            "valid_key_fact_count": flags["valid_key_fact_count"],
            "agreement_rate": agreement_stats["agreement_rate"],
            "avg_support": agreement_stats["avg_support"],
            "agreed_fact_count": agreement_stats["agreed_fact_count"],
            "total_clusters": agreement_stats["total_clusters"],
        }
        ensemble_rows.append(row)

        best_single = max(
            (
                next((pred for pred in all_predictions.get(member, []) if pred["example_id"] == example_id), None)
                for member in ENSEMBLE_MEMBERS
            ),
            key=lambda pred: pred["key_fact_f1"] if pred else -1.0,
        )
        if best_single and row["key_fact_f1"] < best_single["key_fact_f1"]:
            failure_rows.append(
                {
                    "example_id": example_id,
                    "question": example.get("input", ""),
                    "reference_facts": get_reference_facts(example),
                    "ensemble_key_fact_f1": row["key_fact_f1"],
                    "best_single_model": best_single["model_name"],
                    "best_single_key_fact_f1": best_single["key_fact_f1"],
                    "ensemble_prediction": chosen,
                    "best_single_raw": best_single["raw_text"],
                }
            )

    write_jsonl(args.failure_out, failure_rows)
    ensemble_summary = summarize_metrics(ensemble_rows)
    ensemble_latency = sum(summary["models"].get(member, {}).get("latency_sec_total", 0.0) for member in ENSEMBLE_MEMBERS)
    ensemble_generated_tokens = sum(summary["models"].get(member, {}).get("total_generated_tokens", 0) for member in ENSEMBLE_MEMBERS)
    ensemble_summary["failure_case_count"] = len(failure_rows)
    ensemble_summary["latency_sec_total"] = ensemble_latency
    ensemble_summary["examples_per_sec"] = len(ensemble_rows) / max(ensemble_latency, 1e-9)
    ensemble_summary["tokens_per_example"] = ensemble_generated_tokens / max(len(ensemble_rows), 1)
    ensemble_summary["total_generated_tokens"] = ensemble_generated_tokens
    ensemble_summary["gpu_hours"] = ensemble_latency / 3600.0
    ensemble_summary["estimated_cost_usd"] = (ensemble_latency / 3600.0) * args.gpu_hourly_cost
    summary["ensemble_ft"] = ensemble_summary

    base_names = [cfg["name"] for cfg in MODEL_CONFIGS if cfg["name"].endswith("_base")]
    ft_names = [cfg["name"] for cfg in MODEL_CONFIGS if cfg["name"].endswith("_ft")]
    summary["baselines"] = {
        "base_single_slms": {name: summary["models"].get(name, {}) for name in base_names},
        "finetuned_single_slms": {name: summary["models"].get(name, {}) for name in ft_names},
        "ensemble_ft": summary["ensemble_ft"],
    }

    human_eval_summary = load_human_eval(args.human_eval_path)
    if human_eval_summary is not None:
        summary["human_eval"] = human_eval_summary

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== SUMMARY ==========")
    for name, metrics in summary["models"].items():
        print(
            f"{name:18s} | KF1={metrics.get('key_fact_f1', 0.0):.3f} | "
            f"TF1={metrics.get('token_f1', 0.0):.3f} | "
            f"ROUGE-L={metrics.get('rouge_l', 0.0):.3f} | "
            f"schema={metrics.get('structure_compliance_rate', 0.0):.3f}"
        )
    print(
        "ensemble_ft         | KF1={:.3f} | TF1={:.3f} | ROUGE-L={:.3f} | schema={:.3f}".format(
            summary["ensemble_ft"].get("key_fact_f1", 0.0),
            summary["ensemble_ft"].get("token_f1", 0.0),
            summary["ensemble_ft"].get("rouge_l", 0.0),
            summary["ensemble_ft"].get("structure_compliance_rate", 0.0),
        )
    )
    print(f"Failure cases written to: {args.failure_out}")
    print(f"Full report written to: {args.report_path}")


if __name__ == "__main__":
    main()
