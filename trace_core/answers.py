import re
from collections import Counter
from typing import Optional, Sequence

NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"

def normalize_number(value: str) -> str:
    value = value.replace(",", "").strip()
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, ".12g")
    except (ValueError, OverflowError):
        return value

def extract_answer(text: str):
    """Return (normalized_answer, extraction_method)."""
    marked = re.findall(r"####\s*(" + NUMBER + r")", text)
    if marked:
        return normalize_number(marked[-1]), "gsm8k_marker"
    # Deliberately labeled fallback; it can be wrong and must be audited.
    nums = re.findall(NUMBER, text)
    if nums:
        return normalize_number(nums[-1]), "last_number_fallback"
    return None, "no_number"

def gold_from_gsm8k(answer_text: str) -> Optional[str]:
    match = re.search(r"####\s*(" + NUMBER + r")", answer_text)
    return normalize_number(match.group(1)) if match else None

def is_correct(prediction, gold) -> int:
    return int(prediction is not None and gold is not None and prediction == gold)

def select_majority(answers: Sequence[Optional[str]]):
    """Majority vote; ties are broken by first occurrence among tied answers."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    highest = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == highest}
    return next(answer for answer in valid if answer in tied)
