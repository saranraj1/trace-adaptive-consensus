import re

HEDGE_WORDS = ("maybe", "perhaps", "probably", "might", "i think", "not sure", "uncertain")
REVERSAL_WORDS = ("actually", "wait", "correction", "instead", "on second thought")
CONTRADICTION_WORDS = ("but that", "however", "this contradicts", "does not follow")

def response_features(question, response):
    low = response.lower()
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", response)
    sentences = [s for s in re.split(r"[.!?\n]+", response) if s.strip()]
    return {
        "question_chars": len(question),
        "question_words": len(question.split()),
        "response_chars": len(response),
        "response_words": len(response.split()),
        "sentence_count": len(sentences),
        "numeric_token_count": len(nums),
        "hedge_count": sum(low.count(w) for w in HEDGE_WORDS),
        "reversal_count": sum(low.count(w) for w in REVERSAL_WORDS),
        "contradiction_phrase_count": sum(low.count(w) for w in CONTRADICTION_WORDS),
        "distinct_numeric_values": len(set(n.replace(",", "") for n in nums)),
    }

def agreement_features(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return {"agreement_fraction": 0.0, "unique_answer_count": 0, "valid_answer_fraction": 0.0}
    counts = {}
    for a in valid:
        counts[a] = counts.get(a, 0) + 1
    return {
        "agreement_fraction": max(counts.values()) / len(answers),
        "unique_answer_count": len(counts),
        "valid_answer_fraction": len(valid) / len(answers),
    }
