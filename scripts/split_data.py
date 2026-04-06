import json, random

inp = "data/synth_1k.jsonl"
train_out = "data/train_900.jsonl"
eval_out = "data/eval_100.jsonl"

with open(inp) as f:
    rows = [json.loads(line) for line in f]

random.seed(42)
random.shuffle(rows)

train_rows = rows[:900]
eval_rows = rows[900:]

for path, data in [(train_out, train_rows), (eval_out, eval_rows)]:
    with open(path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")

print(f"train={len(train_rows)} eval={len(eval_rows)}")
