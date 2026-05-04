import argparse
import json
from pathlib import Path
from typing import Dict, List

from ensemble_utils import ensemble_vote
from eval_single import (
    load_model_and_tokenizer,
    build_prompt,
    generate_answer,
    valid_schema,
    key_fact_f1,
    rouge_l_f1,
    token_f1,
    save_jsonl,
)


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ensemble with voting")
    parser.add_argument("--model_keys", nargs="+", required=True, choices=["qwen3b", "phi35", "granite33_2b"])
    parser.add_argument("--adapter_dirs", nargs="+", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--voting_mode", choices=["consensus", "schema_first"], default="consensus")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--fact_threshold", type=float, default=0.2)
    parser.add_argument("--verbose_examples", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.model_keys) != len(args.adapter_dirs):
        raise ValueError("model_keys and adapter_dirs must have the same length")

    eval_rows = load_jsonl(args.eval_path)
    if args.max_examples is not None:
        eval_rows = eval_rows[: args.max_examples]

    print(f"[info] loaded {len(eval_rows)} eval examples from {args.eval_path}")

    loaded = {}
    for model_key, adapter_dir in zip(args.model_keys, args.adapter_dirs):
        print(f"[info] loading model={model_key} adapter={adapter_dir}")
        model, tokenizer, hf_id = load_model_and_tokenizer(
            model_key=model_key,
            adapter_dir=adapter_dir,
            device=args.device,
            load_in_4bit=args.load_in_4bit,
        )
        loaded[model_key] = {
            "model": model,
            "tokenizer": tokenizer,
            "hf_id": hf_id,
            "adapter_dir": adapter_dir,
        }

    pred_rows = []
    schema_scores = []
    fact_scores = []
    rouge_scores = []
    token_f1_scores = []
    latencies = []
    new_token_counts = []

    for idx, row in enumerate(eval_rows, start=1):
        instruction = row["instruction"]
        user_input = row["input"]
        reference = row["output"]

        model_outputs = {}
        per_model_meta = {}
        total_latency_s = 0.0
        total_new_tokens = 0

        for model_key in args.model_keys:
            pack = loaded[model_key]
            prompt = build_prompt(pack["tokenizer"], instruction, user_input)

            prediction, latency_s, new_tokens = generate_answer(
                model=pack["model"],
                tokenizer=pack["tokenizer"],
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )

            model_outputs[model_key] = prediction
            per_model_meta[model_key] = {
                "latency_s": latency_s,
                "new_tokens": new_tokens,
            }
            total_latency_s += latency_s
            total_new_tokens += new_tokens

        ensemble_prediction, vote_debug = ensemble_vote(
            model_outputs=model_outputs,
            voting_mode=args.voting_mode,
            preferred_tiebreak_order=["qwen3b", "phi35", "granite33_2b"],
        )

        schema_ok = valid_schema(ensemble_prediction)
        fact_f1 = key_fact_f1(ensemble_prediction, reference, threshold=args.fact_threshold)
        rouge = rouge_l_f1(ensemble_prediction, reference)
        tf1 = token_f1(ensemble_prediction, reference)

        schema_scores.append(1.0 if schema_ok else 0.0)
        fact_scores.append(fact_f1)
        rouge_scores.append(rouge)
        token_f1_scores.append(tf1)
        latencies.append(total_latency_s)
        new_token_counts.append(total_new_tokens)

        pred_rows.append(
            {
                "index": idx,
                "instruction": instruction,
                "input": user_input,
                "reference": reference,
                "model_outputs": model_outputs,
                "per_model_meta": per_model_meta,
                "ensemble_prediction": ensemble_prediction,
                "vote_debug": vote_debug,
                "valid_schema": schema_ok,
                "key_fact_f1": fact_f1,
                "rouge_l_f1": rouge,
                "token_f1": tf1,
                "latency_s": total_latency_s,
                "new_tokens": total_new_tokens,
            }
        )

        if idx <= args.verbose_examples:
            print("-" * 60)
            print(f"Example {idx}")
            print(f"Question: {user_input}")
            print(f"Winner model: {vote_debug['winner_model']}")
            print(f"Ensemble prediction:\n{ensemble_prediction}\n")
            print(f"Valid schema: {schema_ok}")
            print(f"Key-Fact F1: {fact_f1:.3f}")
            print(f"ROUGE-L F1: {rouge:.3f}")
            print(f"Token F1: {tf1:.3f}")
            print(f"Latency (sum): {total_latency_s:.3f}s")
            print(f"New tokens (sum): {total_new_tokens}")

        if idx % 25 == 0:
            print(f"[progress] {idx}/{len(eval_rows)} done")

    report = {
        "model_keys": args.model_keys,
        "adapter_dirs": args.adapter_dirs,
        "voting_mode": args.voting_mode,
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
