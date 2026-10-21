"""
Standard KGQA evaluation metrics, matching the RoG / GNN-RAG / WebQSP-CWQ
literature (Luo et al., ICLR 2024; GNN-RAG 2024) so numbers are directly
comparable to published results and the code is open-sourceable.

Reference: RoG `src/qa_prediction/evaluate_results.py`.

Definitions
-----------
normalize : lowercase -> strip punctuation -> strip articles (a/an/the) ->
            strip <pad> -> collapse whitespace.
match     : a gold answer matches if normalize(gold) is a SUBSTRING of
            normalize(prediction)  (one direction only: gold in pred).
Hit       : 1 if ANY gold answer matches the prediction, else 0.  (primary metric,
            == RoG "Hit"/"Hits@1")
Acc       : (# gold answers matched) / (# gold answers).            (RoG eval_acc)
F1        : prediction is split into an answer SET; precision = matched/|pred set|,
            recall = matched/|gold set|, F1 = 2PR/(P+R).             (RoG eval_f1)
"""

import re
import string
from typing import Dict, List, Union


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.replace("<pad>", " ")
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _match(prediction_norm: str, gold: str) -> bool:
    """RoG match: normalized gold answer is a substring of the normalized prediction."""
    g = normalize_answer(gold)
    return bool(g) and g in prediction_norm


def _split_prediction(prediction: str) -> List[str]:
    """Split a prediction string into a set of candidate answers (RoG splits on
    newlines; models here also emit comma / 'and' / semicolon separated lists)."""
    parts = re.split(r"[\n,;]| and ", prediction)
    out = [normalize_answer(p) for p in parts]
    return [p for p in out if p]


# ---------------------------------------------------------------------------
# Per-prediction scorers (RoG-standard)
# ---------------------------------------------------------------------------

def hit_single(prediction: str, golds: List[str]) -> int:
    pred_norm = normalize_answer(prediction)
    return int(any(_match(pred_norm, g) for g in golds))


def acc_single(prediction: str, golds: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    if not golds:
        return 0.0
    matched = sum(1 for g in golds if _match(pred_norm, g))
    return matched / len(golds)


def f1_single(prediction: str, golds: List[str]) -> float:
    pred_set = _split_prediction(prediction)
    if not pred_set or not golds:
        return 0.0
    golds_norm = [normalize_answer(g) for g in golds]
    matched = sum(1 for g in golds_norm if g and any(g in p for p in pred_set))
    if matched == 0:
        return 0.0
    precision = matched / len(pred_set)
    recall = matched / len(golds_norm)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_metrics(
    predictions: List[str],
    ground_truths: List[Union[str, List[str]]],
) -> Dict[str, float]:
    """RoG-standard KGQA metrics over a batch.

    Returns (keys kept stable for the existing harness):
        hits_at_1 -> Hit   (primary; == RoG "Hit": any gold answer in prediction)
        f1        -> entity-set F1 (RoG eval_f1)
        accuracy  -> Acc   (RoG eval_acc: fraction of gold answers found)
    """
    hit_sum = acc_sum = f1_sum = 0.0
    for pred, truth in zip(predictions, ground_truths):
        golds: List[str] = [truth] if isinstance(truth, str) else list(truth)
        hit_sum += hit_single(pred, golds)
        acc_sum += acc_single(pred, golds)
        f1_sum += f1_single(pred, golds)
    n = len(predictions) or 1
    return {"hits_at_1": hit_sum / n, "f1": f1_sum / n, "accuracy": acc_sum / n}
