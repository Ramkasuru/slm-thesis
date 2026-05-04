import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def simple_tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"\b\w+\b", text)


def token_f1_pair(a: str, b: str) -> float:
    a_tokens = simple_tokenize(a)
    b_tokens = simple_tokenize(b)

    if not a_tokens or not b_tokens:
        return 0.0

    a_count = Counter(a_tokens)
    b_count = Counter(b_tokens)
    common = sum((a_count & b_count).values())

    if common == 0:
        return 0.0

    precision = common / max(1, len(a_tokens))
    recall = common / max(1, len(b_tokens))
    return 2 * precision * recall / (precision + recall + 1e-12)


def extract_bullets(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = []
    for ln in lines:
        if re.match(r"^[-*•]\s+", ln):
            bullets.append(re.sub(r"^[-*•]\s+", "", ln).strip())
    return bullets


def schema_score(text: str) -> float:
    score = 0.0
    bullets = extract_bullets(text)

    if len(bullets) >= 3:
        score += 0.5
    if 4 <= len(bullets) <= 8:
        score += 0.3

    avg_words = 0.0
    if bullets:
        avg_words = sum(len(simple_tokenize(b)) for b in bullets) / len(bullets)

    if 6 <= avg_words <= 40:
        score += 0.2

    return min(score, 1.0)


def brevity_score(text: str, target_min: int = 60, target_max: int = 220) -> float:
    n = len(simple_tokenize(text))
    if target_min <= n <= target_max:
        return 1.0
    if n < target_min:
        return max(0.0, n / max(1, target_min))
    overflow = n - target_max
    return max(0.0, 1.0 - overflow / max(1, target_max))


def pairwise_consensus_scores(candidates: List[str]) -> List[float]:
    n = len(candidates)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    scores = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sims.append(token_f1_pair(candidates[i], candidates[j]))
        scores.append(sum(sims) / max(1, len(sims)))
    return scores


def select_by_consensus(
    model_outputs: Dict[str, str],
    preferred_tiebreak_order: Optional[List[str]] = None,
) -> Tuple[str, Dict]:
    preferred_tiebreak_order = preferred_tiebreak_order or ["qwen3b", "phi35", "granite33_2b"]

    model_names = list(model_outputs.keys())
    texts = [model_outputs[m] for m in model_names]

    consensus = pairwise_consensus_scores(texts)
    schema = [schema_score(t) for t in texts]
    brevity = [brevity_score(t) for t in texts]

    combined = []
    for i, name in enumerate(model_names):
        score = 0.6 * consensus[i] + 0.25 * schema[i] + 0.15 * brevity[i]
        combined.append(
            {
                "model": name,
                "text": texts[i],
                "consensus": consensus[i],
                "schema": schema[i],
                "brevity": brevity[i],
                "combined": score,
            }
        )

    combined.sort(key=lambda x: x["combined"], reverse=True)

    top = combined[0]
    tied = [x for x in combined if abs(x["combined"] - top["combined"]) < 1e-9]

    if len(tied) > 1:
        rank = {m: i for i, m in enumerate(preferred_tiebreak_order)}
        tied.sort(key=lambda x: rank.get(x["model"], 999))
        top = tied[0]

    debug = {
        "candidates": combined,
        "winner_model": top["model"],
        "winner_score": top["combined"],
        "voting_mode": "consensus",
    }
    return top["text"], debug


def select_by_schema_first(
    model_outputs: Dict[str, str],
    preferred_tiebreak_order: Optional[List[str]] = None,
) -> Tuple[str, Dict]:
    preferred_tiebreak_order = preferred_tiebreak_order or ["qwen3b", "phi35", "granite33_2b"]

    items = []
    for name, text in model_outputs.items():
        items.append(
            {
                "model": name,
                "text": text,
                "schema": schema_score(text),
                "brevity": brevity_score(text),
            }
        )

    items.sort(key=lambda x: (x["schema"], x["brevity"]), reverse=True)
    top = items[0]
    tied = [x for x in items if abs(x["schema"] - top["schema"]) < 1e-9 and abs(x["brevity"] - top["brevity"]) < 1e-9]

    if len(tied) > 1:
        rank = {m: i for i, m in enumerate(preferred_tiebreak_order)}
        tied.sort(key=lambda x: rank.get(x["model"], 999))
        top = tied[0]

    debug = {
        "candidates": items,
        "winner_model": top["model"],
        "winner_score": top["schema"],
        "voting_mode": "schema_first",
    }
    return top["text"], debug


def ensemble_vote(
    model_outputs: Dict[str, str],
    voting_mode: str = "consensus",
    preferred_tiebreak_order: Optional[List[str]] = None,
) -> Tuple[str, Dict]:
    if voting_mode == "schema_first":
        return select_by_schema_first(model_outputs, preferred_tiebreak_order)
    return select_by_consensus(model_outputs, preferred_tiebreak_order)
