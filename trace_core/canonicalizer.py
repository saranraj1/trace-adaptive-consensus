import re
from typing import Optional, Tuple

NUMBER_PATTERN = r"[-+]?\$?\d[\d,]*(?:\.\d+)?"

def normalize_number(value: str) -> str:
    """Normalize numeric strings: strip currency symbols, commas, and handle float decimals."""
    if value is None:
        return ""
    clean = value.replace("$", "").replace("€", "").replace("£", "").replace(",", "").strip()
    try:
        number = float(clean)
        return str(int(number)) if number.is_integer() else format(number, ".12g")
    except (ValueError, OverflowError):
        return clean

def canonicalize_action(action: str) -> str:
    """
    Appendix A Action Canonicaliser:
    1. Strip leading/trailing whitespace.
    2. Convert to lowercase.
    3. Remove trailing punctuation (periods, exclamation marks).
    4. Collapse consecutive whitespace to a single space.
    """
    if not action:
        return ""
    # Strip whitespace and lowercase
    text = action.strip().lower()
    # Strip common conversational/bullet prefixes
    text = re.sub(r"^(?:action|chosen action|next action)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[-*•>]\s*", "", text).strip()
    # Remove trailing punctuation
    text = re.sub(r"[.!?]+$", "", text).strip()
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text)
    return text

def extract_gsm8k_answer(text: str) -> Tuple[Optional[str], str]:
    """
    GSM8K extraction pipeline from paper (Section 4.1 & Appendix A):
    1. Search for 'Answer: N'
    2. Search for '#### N'
    3. Search for LaTeX '\\boxed{N}'
    4. Fall back to the last numeric token in the response.
    Returns (normalized_answer, extraction_method).
    """
    if not text:
        return None, "empty_text"

    # 1. Search for "Answer:\s*N"
    ans_match = re.findall(r"(?:answer|the answer is|final answer)\s*[:=]?\s*(" + NUMBER_PATTERN + r")", text, re.IGNORECASE)
    if ans_match:
        return normalize_number(ans_match[-1]), "answer_marker"

    # 2. Search for "#### N" (standard GSM8K delimiter)
    gsm_match = re.findall(r"####\s*(" + NUMBER_PATTERN + r")", text)
    if gsm_match:
        return normalize_number(gsm_match[-1]), "gsm8k_marker"

    # 3. Search for "\boxed{N}"
    boxed_match = re.findall(r"\\boxed\{\s*([^{}]+)\s*\}", text)
    if boxed_match:
        inner = boxed_match[-1].strip()
        nums = re.findall(NUMBER_PATTERN, inner)
        if nums:
            return normalize_number(nums[-1]), "boxed_marker"

    # 4. Fall back to last numeric token in the response
    nums = re.findall(NUMBER_PATTERN, text)
    if nums:
        return normalize_number(nums[-1]), "last_number_fallback"

    return None, "no_number"

def extract_gold_gsm8k(gold_text: str) -> Optional[str]:
    """Extract and normalize gold answer from GSM8K ground truth."""
    if not gold_text:
        return None
    match = re.search(r"####\s*(" + NUMBER_PATTERN + r")", gold_text)
    if match:
        return normalize_number(match.group(1))
    nums = re.findall(NUMBER_PATTERN, gold_text)
    return normalize_number(nums[-1]) if nums else None
