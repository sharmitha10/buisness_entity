"""
Evaluation Module for Business Entity Resolution.

Computes challenge-standard metrics:
- Precision, Recall
- F0.5 score (precision weighted twice as heavily as recall: beta = 0.5)
- Singleton performance (accuracy on non-matching entities)
"""

from typing import Dict, Set, Tuple


def compute_metrics(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Computes global Precision, Recall, F0.5, and Singleton accuracy.
    """
    tp = 0
    fp = 0
    fn = 0
    
    total_singletons = 0
    correct_singletons = 0
    
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        
        is_singleton = (len(true_matches) == 0)
        if is_singleton:
            total_singletons += 1
            if len(pred_matches) == 0:
                correct_singletons += 1
                
        tp_i = len(true_matches & pred_matches)
        fp_i = len(pred_matches - true_matches)
        fn_i = len(true_matches - pred_matches)
        
        tp += tp_i
        fp += fp_i
        fn += fn_i
        
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # F0.5: beta = 0.5 -> (1 + 0.25) * P * R / (0.25 * P + R)
    beta_sq = 0.25
    denom = (beta_sq * precision + recall)
    f05 = ((1.0 + beta_sq) * precision * recall / denom) if denom > 0 else 0.0
    
    singleton_acc = (correct_singletons / total_singletons) if total_singletons > 0 else 1.0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "total_singletons": total_singletons,
        "singleton_accuracy": singleton_acc,
    }


if __name__ == "__main__":
    # Quick sanity test
    gt = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),  # singleton
        "S1-3": {"S2-30"},
    }
    preds = {
        "S1-1": {"S2-10"},           # 1 TP, 1 FN
        "S1-2": set(),                # correct singleton
        "S1-3": {"S2-30", "S3-99"},   # 1 TP, 1 FP
    }
    res = compute_metrics(gt, preds)
    print("Sanity Test Evaluation Metrics:", res)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["singleton_accuracy"] == 1.0
    print("[SUCCESS] Evaluation module verified.")
