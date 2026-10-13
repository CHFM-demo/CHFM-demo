"""Accuracy metrics used by CHFM evaluation."""

import string


def normalize_answer(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return " ".join(text.split())


def compute_metrics(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not references:
        return {"exact_match": 0.0}
    correct = sum(
        normalize_answer(prediction) == normalize_answer(reference)
        for prediction, reference in zip(predictions, references)
    )
    return {"exact_match": round(100.0 * correct / len(references), 4)}


def compute_grouped_metrics(predictions, references, groups):
    if not (len(predictions) == len(references) == len(groups)):
        raise ValueError("predictions, references, and groups must have the same length")
    grouped = {}
    for prediction, reference, group in zip(predictions, references, groups):
        grouped.setdefault(group, ([], []))
        grouped[group][0].append(prediction)
        grouped[group][1].append(reference)
    return {
        f"exact_match_for_{group}": compute_metrics(group_predictions, group_references)["exact_match"]
        for group, (group_predictions, group_references) in grouped.items()
    }
