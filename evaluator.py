# evaluator.py

def evaluate_answer(answer: str) -> bool:
    """
    Simple heuristic evaluator.
    Returns True if answer is acceptable, False otherwise.
    """
    if not answer:
        return False

    weak_phrases = [
        "i don't know",
        "not sure",
        "cannot",
        "unknown"
    ]

    for phrase in weak_phrases:
        if phrase in answer.lower():
            return False

    # too short = probably low quality
    if len(answer.split()) < 6:
        return False

    return True
